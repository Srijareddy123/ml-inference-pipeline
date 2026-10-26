from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MachineReading(BaseModel):
    machine_id: str = Field(min_length=1, max_length=64, examples=["M-0001"])
    product_type: Literal["L", "M", "H"] = Field(description="Quality variant of the product")
    air_temperature_k: float = Field(ge=250, le=400, examples=[298.1])
    process_temperature_k: float = Field(ge=250, le=400, examples=[308.6])
    rotational_speed_rpm: float = Field(ge=0, le=10000, examples=[1551])
    torque_nm: float = Field(ge=0, le=200, examples=[42.8])
    tool_wear_min: float = Field(ge=0, le=1000, examples=[0])


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    request_id: str
    machine_id: str
    prediction: Literal["normal", "failure"]
    failure_probability: float
    model_name: str
    model_version: str
    latency_ms: float


class BatchRequest(BaseModel):
    readings: list[MachineReading] = Field(min_length=1, max_length=500)


class BatchResponse(BaseModel):
    results: list[PredictionResponse]


class PromoteRequest(BaseModel):
    version: str


class VersionRow(BaseModel):
    version: str
    alias: str | None
    run_id: str


class ModelsResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    champion_version: str | None
    versions: list[VersionRow]


class DriftStatus(BaseModel):
    has_reference: bool
    last_window_checked: bool
    drift_detected: bool
    drifted_features: list[str]
    p_values: dict[str, float]
