#!/usr/bin/env python3
"""Capture ESP32 UART logs, keeping setup passwords out of terminal output."""
import argparse
import json
from pathlib import Path
import re
import time
import serial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("port")
    parser.add_argument("--reset", action="store_true", help="Reset ESP32 via the verified UART RTS/EN circuit")
    parser.add_argument("--seconds", type=float, default=0, help="0 monitors until Ctrl-C")
    parser.add_argument("--log", type=Path, default=Path("artifacts/hardware/serial-after.txt"))
    parser.add_argument("--setup-file", type=Path, default=Path("artifacts/hardware/wifi-setup.json"))
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.setup_file.parent.mkdir(parents=True, exist_ok=True)
    # Contains the temporary AP password, never the household Wi-Fi password.
    with args.setup_file.open("w") as handle:
        args.setup_file.chmod(0o600)
        handle.write("{}\n")
    state = {}
    uart = serial.Serial(port=None, baudrate=115200, timeout=0.2)
    uart.dtr = False
    uart.rts = False
    uart.port = args.port
    uart.open()
    if args.reset:
        uart.reset_input_buffer()
        uart.rts = True
        time.sleep(0.1)
        uart.rts = False
    started = time.monotonic()
    pending = bytearray()
    try:
        with args.log.open("wb") as log:
            args.log.chmod(0o600)
            while not args.seconds or time.monotonic()-started < args.seconds:
                chunk = uart.read(max(1, uart.in_waiting))
                if not chunk:
                    continue
                log.write(chunk)
                log.flush()
                pending.extend(chunk)
                while b"\n" in pending:
                    line, _, rest = pending.partition(b"\n")
                    pending = bytearray(rest)
                    text = re.sub(r"\x1b\[[0-9;]*m", "", line.decode("utf-8", "replace")).strip()
                    if text.startswith(("WIFI_SETUP_", "WIFI_CONNECTED_IP=")) and "=" in text:
                        key, value = text.split("=", 1)
                        state[key] = value
                        args.setup_file.write_text(json.dumps(state, indent=2))
                    if re.search(r"password|passwd|\bpwd\b", text, re.I):
                        print(f"[setup password saved in {args.setup_file}]", flush=True)
                    else:
                        print(text, flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        uart.close()


if __name__ == "__main__":
    main()
