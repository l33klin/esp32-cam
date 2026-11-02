import os
import threading

from indoorcam.video import read_pipe_frame


def test_pipe_preserves_frame_boundaries_across_partial_writes():
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    payload = bytes(range(256))*512

    def write():
        try:
            for at in range(0, len(payload), 173):
                os.write(write_fd, payload[at:at+173])
        finally:
            os.close(write_fd)

    writer = threading.Thread(target=write)
    writer.start()
    try:
        stop = threading.Event()
        first = read_pipe_frame(read_fd, 65536, stop, 3)
        second = read_pipe_frame(read_fd, 65536, stop, 3)
        assert first == payload[:65536]
        assert second == payload[65536:]
        assert read_pipe_frame(read_fd, 65536, stop, 3) is None
    finally:
        os.close(read_fd)
        writer.join(timeout=3)


def test_incomplete_frame_is_discarded_on_disconnect():
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b"partial")
        os.close(write_fd)
        assert read_pipe_frame(read_fd, 100, threading.Event(), 1) is None
    finally:
        os.close(read_fd)


def test_idle_capture_and_shutdown_are_bounded():
    read_fd, write_fd = os.pipe()
    try:
        stop = threading.Event()
        assert read_pipe_frame(read_fd, 100, stop, 0.02) is None
        stop.set()
        assert read_pipe_frame(read_fd, 100, stop, 10) is None
    finally:
        os.close(read_fd)
        os.close(write_fd)
