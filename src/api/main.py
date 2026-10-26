"""FastAPI app: predict, model promote and rollback, drift status, health."""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

from src import db
from src.alerts import fire_drift_alert
from src.api.schemas import (
    BatchRequest,
    BatchResponse,
    DriftStatus,
    MachineReading,
    ModelsResponse,
    PredictionResponse,
    PromoteRequest,
    VersionRow,
)
from src.config import get_settings
from src.drift import DriftDetector
from src.predictor import Prediction, Predictor
from src.registry import Registry, RegistryError

logger = logging.getLogger(__name__)


def _load_serving_model(app: FastAPI) -> None:
    reference = app.state.predictor.load()
    app.state.detector.set_reference(reference)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    db.create_tables()
    app.state.registry = Registry(settings.mlflow_tracking_uri)
    app.state.predictor = Predictor(app.state.registry)
    app.state.detector = DriftDetector(settings.drift_threshold, settings.drift_window_size)
    try:
        _load_serving_model(app)
    except Exception as exc:  # the API stays up and answers 503 until a model exists
        logger.error("model not loaded at startup: %s", exc)
    yield


app = FastAPI(
    title="ML Inference Pipeline",
    description="Machine failure classifier trained on the AI4I 2020 dataset.",
    version="2.0.0",
    lifespan=lifespan,
)


def _log_prediction(request_id: str, machine_id: str, pred: Prediction) -> None:
    """Runs after the response is sent. A database error is logged, never raised."""
    try:
        with db.session_scope() as session:
            db.log_prediction(
                session,
                {
                    "request_id": request_id,
                    "machine_id": machine_id,
                    "features": pred.features,
                    "prediction": pred.label,
                    "failure_probability": pred.failure_probability,
                    "model_name": pred.model_name,
                    "model_version": pred.model_version,
                    "latency_ms": pred.latency_ms,
                },
            )
    except Exception as exc:
        logger.error("prediction logging failed: %s", exc)


def _watch_drift(detector: DriftDetector, pred: Prediction) -> None:
    try:
        result = detector.observe(pred.features)
        if result is not None and result.drift_detected:
            with db.session_scope() as session:
                fire_drift_alert(session, pred.model_name, pred.model_version, result)
    except Exception as exc:
        logger.error("drift check failed: %s", exc)


def _predict_one(
    app_: FastAPI, reading: MachineReading, tasks: BackgroundTasks
) -> PredictionResponse:
    settings = get_settings()
    predictor: Predictor = app_.state.predictor
    if not predictor.is_loaded():
        raise HTTPException(503, "No model loaded. Train one with `make train`.")
    pred = predictor.predict(reading.model_dump(exclude={"machine_id"}))
    request_id = str(uuid.uuid4())
    if settings.enable_prediction_logging:
        tasks.add_task(_log_prediction, request_id, reading.machine_id, pred)
    if settings.enable_drift_detection:
        tasks.add_task(_watch_drift, app_.state.detector, pred)
    return PredictionResponse(
        request_id=request_id,
        machine_id=reading.machine_id,
        prediction=pred.label,  # type: ignore[arg-type]
        failure_probability=pred.failure_probability,
        model_name=pred.model_name,
        model_version=pred.model_version,
        latency_ms=pred.latency_ms,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(
    reading: MachineReading, request: Request, tasks: BackgroundTasks
) -> PredictionResponse:
    return _predict_one(request.app, reading, tasks)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(body: BatchRequest, request: Request, tasks: BackgroundTasks) -> BatchResponse:
    return BatchResponse(results=[_predict_one(request.app, r, tasks) for r in body.readings])


def _models_response(request: Request) -> ModelsResponse:
    registry: Registry = request.app.state.registry
    name = get_settings().model_name
    champion = registry.champion(name)
    return ModelsResponse(
        model_name=name,
        champion_version=champion.version if champion else None,
        versions=[VersionRow(**row) for row in registry.list_versions(name)],
    )


@app.get("/models", response_model=ModelsResponse)
def list_models(request: Request) -> ModelsResponse:
    return _models_response(request)


@app.post("/models/promote", response_model=ModelsResponse)
def promote(body: PromoteRequest, request: Request) -> ModelsResponse:
    try:
        request.app.state.registry.promote(get_settings().model_name, body.version)
    except RegistryError as exc:
        raise HTTPException(404, str(exc)) from exc
    _load_serving_model(request.app)
    return _models_response(request)


@app.post("/models/rollback", response_model=ModelsResponse)
def rollback(request: Request) -> ModelsResponse:
    try:
        request.app.state.registry.rollback(get_settings().model_name)
    except RegistryError as exc:
        raise HTTPException(409, str(exc)) from exc
    _load_serving_model(request.app)
    return _models_response(request)


@app.get("/drift", response_model=DriftStatus)
def drift_status(request: Request) -> DriftStatus:
    detector: DriftDetector = request.app.state.detector
    last = detector.last_result
    return DriftStatus(
        has_reference=detector.has_reference,
        last_window_checked=last is not None,
        drift_detected=bool(last and last.drift_detected),
        drifted_features=last.drifted_features if last else [],
        p_values=last.p_values if last else {},
    )


@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    info = request.app.state.predictor.info
    return {
        "status": "ok",
        "model_loaded": request.app.state.predictor.is_loaded(),
        "model_version": info.version if info else None,
        "database": db.db_ok(),
    }
