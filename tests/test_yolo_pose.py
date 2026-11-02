import numpy as np
import pytest

from scripts.yolo_pose_adapter import decode, letterbox


def test_yolo_coordinates_roundtrip_nms_and_padding_confidence():
    frame = np.full((540, 960, 3), [10, 20, 30], np.uint8)
    image, transform = letterbox(frame, 320)
    assert image[0, 0].tolist() == [114]*3
    assert image[70, 0].tolist() == [30, 20, 10]
    rows = np.zeros((2, 56), np.float32)
    # Original box [96,54,288,324] -> padded input [32,88,96,178].
    rows[:, :4] = [.2, 133/320, .2, 90/320]
    rows[:, 4] = [.8, .9]
    points = np.tile([.2, .5, .8], (17, 1))
    points[0, 1] = 0  # lies in top padding, not in the source image
    rows[:, 5:] = points.reshape(-1)
    people = decode(rows.T[None], 320, transform, frame.shape)
    assert len(people) == 1
    assert people[0]['score'] == pytest.approx(.9)
    assert people[0]['bbox'] == pytest.approx([96, 54, 288, 324], abs=1e-4)
    assert people[0]['keypoints'][1].tolist() == pytest.approx([192, 270, .8])
    assert people[0]['keypoints'][0, 2] == 0


def test_yolo_end_to_end_head_is_not_silently_misparsed():
    with pytest.raises(ValueError, match='Expected raw'):
        decode(np.zeros((1, 300, 57)), 320, (0, 0, 3, 3), (540, 960, 3))
