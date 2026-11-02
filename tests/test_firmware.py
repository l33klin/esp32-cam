"""Exercise exactly the JPEG parser and packetizer compiled into ESP firmware."""
from pathlib import Path
import shutil
import struct
import subprocess
import cv2
import numpy as np
import pytest


@pytest.mark.parametrize("sampling,rtp_type", [(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420, 1), (cv2.IMWRITE_JPEG_SAMPLING_FACTOR_422, 0)])
def test_firmware_rtp_packetization(tmp_path, sampling, rtp_type):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler is required for host firmware tests")
    root = Path(__file__).resolve().parents[1]
    helper = tmp_path/"packets"
    subprocess.run([compiler, "-std=c++11", "-I", str(root/"firmware/include"),
                    str(root/"tests/jpeg_packets.cpp"), "-o", str(helper)], check=True)
    # Random texture ensures fragmentation over multiple packets.
    rng = np.random.default_rng(42)
    image = rng.integers(0, 255, (480,640,3), dtype=np.uint8)
    path = tmp_path/"frame.jpg"
    assert cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, sampling])
    stream = subprocess.check_output([str(helper), str(path)])
    at, expected_offset, expected_sequence = 0, 0, 65534
    scan, quant, marker = bytearray(), None, False
    while at < len(stream):
        assert stream[at:at+2] == b"$\x00"
        length = struct.unpack_from("!H", stream, at+2)[0]
        packet = stream[at+4:at+4+length]
        assert packet[0] == 0x80
        assert struct.unpack_from("!H", packet, 2)[0] == expected_sequence % 65536
        assert struct.unpack_from("!I", packet, 4)[0] == 90000
        assert int.from_bytes(packet[13:16], "big") == expected_offset
        assert packet[16:20] == bytes([rtp_type,255,80,60])
        header = 20
        if expected_offset == 0:
            assert packet[20:24] == b"\x00\x00\x00\x80"
            quant = packet[24:152]
            header = 152
        scan.extend(packet[header:])
        expected_offset += len(packet)-header
        expected_sequence += 1
        marker = bool(packet[1] & 0x80)
        at += 4+length
        assert marker == (at == len(stream))
    assert quant and marker and len(scan) > 1400
    # Reconstruct using the source's JPEG headers; scan bytes must decode identically.
    source = path.read_bytes()
    sos = source.index(b"\xff\xda")
    start = sos+2+int.from_bytes(source[sos+2:sos+4], "big")
    rebuilt = source[:start]+scan+b"\xff\xd9"
    original = cv2.imdecode(np.frombuffer(source, np.uint8), cv2.IMREAD_COLOR)
    decoded = cv2.imdecode(np.frombuffer(rebuilt, np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(original, decoded)
