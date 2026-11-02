from unittest.mock import patch

import pytest

from indoorcam.config import Config
from indoorcam.inference import Inference


@pytest.mark.parametrize("model,backend,filename", [
    ("lightning", "coral", "pose_edgetpu.tflite"),
    ("thunder", "coral", "pose_thunder_edgetpu.tflite"),
    ("lightning", "cpu", "pose.tflite"),
    ("thunder", "cpu", "pose_thunder.tflite"),
])
def test_pose_model_selection(model, backend, filename):
    cfg = Config(source="test.mkv", pose_model=model, pose_backend=backend,
                 coral_device="pci:0", pose_device="pci:1")
    cfg.validate()
    with patch("indoorcam.inference.make_interpreter") as make:
        Inference(cfg)
    detector, pose = make.call_args_list
    assert detector.args[0].name == "mobiledet_edgetpu.tflite"
    assert detector.args[1] == "pci:0"
    assert pose.args[0].name == filename
    assert pose.args[1] == ("pci:1" if backend == "coral" else None)


def test_unknown_pose_model_rejected():
    with pytest.raises(ValueError, match="pose_model"):
        Config(source="test.mkv", pose_model="typo").validate()
