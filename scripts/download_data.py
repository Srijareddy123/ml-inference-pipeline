"""Download the AI4I 2020 Predictive Maintenance dataset from UCI and verify it.

The same file is committed at data/raw/ai4i2020.csv, so this is only needed to
refresh or check it. Dataset: Matzka, S. (2020), CC BY 4.0,
https://archive.ics.uci.edu/dataset/601
"""

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
SHA256 = "dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e"
TARGET = Path("data/raw/ai4i2020.csv")


def main() -> int:
    with urllib.request.urlopen(URL, timeout=60) as resp:  # noqa: S310 (fixed https URL)
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    data = archive.read("ai4i2020.csv")
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        print(f"checksum mismatch: {digest}", file=sys.stderr)
        return 1
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_bytes(data)
    print(f"wrote {TARGET} ({len(data)} bytes, sha256 ok)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
