#!/usr/bin/env python3
"""Run the golden MNIST vectors through the FPGA and diff every response byte. This is for WSL.
Windows side, admin PowerShell:

    usbipd attach --wsl --busid 2-2
    
Then in WSL:

    ~/miniforge3/envs/dwn/bin/python3 verification/hw_check.py
    
"""
import json
import sys
import time
from pathlib import Path

from serial import serial_for_url

PORT = "/dev/ttyUSB0"
BAUD = 115200
TIMEOUT = 2.0
GAP = 0.002
NUM_VECTORS = 100

REPO = Path(__file__).resolve().parents[1]

def main():
    model = REPO / (REPO / ".model-stamp").read_text().strip()
    meta = json.loads(model.read_text())["meta"]
    in_bytes = meta["input_size"] // 8
    resp_bytes = 1 + meta["k"]

    words = (REPO / "stim.hex").read_text().split()[:NUM_VECTORS]
    vectors = [int(w, 16).to_bytes(in_bytes, "little") for w in words]

    lines = (REPO / "expected.txt").read_text().splitlines()
    expected = [[int(f) for f in line.split()]
                for line in lines if line.strip()][:NUM_VECTORS]

    passed = 0
    time.sleep(0.005)
    with serial_for_url(PORT, baudrate=BAUD, timeout=TIMEOUT, write_timeout=TIMEOUT) as ser:
        for i, (vector, want) in enumerate(zip(vectors, expected)):
            ser.reset_input_buffer()
            ser.write(vector)
            got = list(ser.read(resp_bytes))
            if got == want:
                passed += 1
            else:
                print(f"vector {i}: got {got}, expected {want}")
            time.sleep(GAP)

    total = len(vectors)
    print(f"{passed}/{total} vectors correct")
    if passed == total:
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
