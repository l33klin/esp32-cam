#!/usr/bin/env python3
"""Download official Coral models, validating every byte against pinned hashes."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]/"models"


def main():
    manifest = json.loads((ROOT/"manifest.json").read_text())
    for name, item in manifest.items():
        target = ROOT/name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == item["sha256"]:
            print(f"verified {name}")
            continue
        with urlopen(item["url"], timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise RuntimeError(f"SHA256 mismatch for {name}; downloaded bytes were not saved")
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(target)
        print(f"downloaded {name}")


if __name__ == "__main__":
    main()
