import numpy as np
import pytest
from indoorcam.analytics import Analytics, Track, posture
from indoorcam.config import Config


def skeleton(kind):
    p = np.zeros((17, 3))
    p[:, 2] = 0.95
    if kind == "standing":
        for left, right, y in ((5, 6, 50), (11, 12, 130), (13, 14, 210), (15, 16, 290)):
            p[left, :2], p[right, :2] = (90, y), (110, y)
    elif kind == "sitting":
        for i, pos in {5:(90,50), 6:(110,50), 11:(90,130), 12:(110,130),
                       13:(170,140), 14:(190,140), 15:(170,220), 16:(190,220)}.items():
            p[i, :2] = pos
    else:
        for left, right, x in ((5, 6, 80), (11, 12, 160), (13, 14, 230), (15, 16, 300)):
            p[left, :2], p[right, :2] = (x, 220), (x, 240)
    return p


@pytest.mark.parametrize("kind", ["standing", "sitting", "lying"])
def test_posture_geometry(kind):
    assert posture(skeleton(kind), 0.3)[0] == kind


def test_occluded_legs_are_unknown_not_guessed():
    points = skeleton("standing")
    points[13:17, 2] = 0
    assert posture(points, 0.3)[0] == "unknown"


def test_bending_with_bent_knees_is_not_lying():
    points = skeleton("standing")
    for a, b, x, y in ((5,6,50,200), (11,12,130,180), (13,14,140,260), (15,16,220,260)):
        points[a,:2], points[b,:2] = (x,y), (x+10,y+10)
    assert posture(points, 0.3)[0] == "unknown"


def test_slow_lie_down_and_already_lying_do_not_alert():
    cfg = Config()
    track = Track(1, [0, 0, 100, 300], 0)
    for now in np.arange(0, 5, 0.1):
        assert not track.update("lying", {"hip_y": 200, "torso": 80}, float(now), cfg)
    track = Track(2, [0, 0, 100, 300], 0)
    assert not track.update("standing", {"hip_y": 100, "torso": 80}, 0, cfg)
    for now in np.arange(0.1, 3, 0.1):
        assert not track.update("unknown", {"hip_y": 100+now*20, "torso": 80}, float(now), cfg)
    for now in np.arange(3, 6, 0.1):
        assert not track.update("lying", {"hip_y": 200, "torso": 80}, float(now), cfg)


def test_fast_drop_needs_hold_and_emits_once():
    cfg = Config()
    track = Track(1, [0, 0, 100, 300], 0)
    events = []
    for now in np.arange(0, 6, 0.1):
        raw, y = ("standing", 100) if now < 0.3 else ("lying", 200)
        if track.update(raw, {"hip_y": y, "torso": 80}, float(now), cfg):
            events.append(now)
    assert len(events) == 1
    assert events[0] >= 1.8


def test_tracking_gap_cannot_trigger_a_fall():
    cfg = Config()
    track = Track(1, [0, 0, 100, 300], 0)
    track.update("standing", {"hip_y": 100, "torso": 80}, 0, cfg)
    for now in np.arange(0.7, 3, 0.1):
        assert not track.update("lying", {"hip_y": 200, "torso": 80}, float(now), cfg)


def person(box, kind="standing"):
    return {"bbox": box, "score": 0.9, "keypoints": skeleton(kind)}


def test_roi_count_disappearance_and_independent_tracks():
    cfg = Config(roi=[[0,0], [0.5,0], [0.5,1], [0,1]], posture_hold=0.1)
    analytics = Analytics(cfg)
    dets = [person([10,20,80,300]), person([300,20,400,300], "sitting")]
    first = analytics.update(dets, 0, (400,500,3))
    second = analytics.update(dets, 0.2, (400,500,3))
    assert first["count"] == 1
    assert second["postures"]["standing"] == 1
    assert [p["id"] for p in first["people"]] == [p["id"] for p in second["people"]]
    assert analytics.update([], 0.3, (400,500,3))["count"] == 0
    assert analytics.update(dets, 2, (400,500,3))["people"][0]["id"] != first["people"][0]["id"]


def test_posture_debounce():
    cfg = Config(posture_hold=0.5)
    track = Track(1, [], 0)
    for now, raw in [(0, "standing"), (0.2, "standing"), (0.4, "standing"), (0.6, "standing")]:
        track.update(raw, None, now, cfg)
    assert track.posture == "standing"
    track.update("lying", None, 0.7, cfg)
    track.update("standing", None, 0.8, cfg)
    assert track.posture == "standing"


def test_outside_roi_cannot_arm_later_alarm():
    cfg = Config(roi=[[0.5,0],[1,0],[1,1],[0.5,1]])
    analytics = Analytics(cfg)
    for now in np.arange(0, 2.5, 0.1):
        kind = "standing" if now < 0.3 else "lying"
        state = analytics.update([person([20,20,120,300], kind)], float(now), (400,500,3))
        assert not state["events"] and not state["fall_candidate"]
    assert not next(iter(analytics.tracks.values())).history
    assert next(iter(analytics.tracks.values())).alarm_until == 0


@pytest.mark.parametrize("values", [{"roi": [[-1,0],[0,1],[1,1]]}, {"fps": 0}, {"width": 961}, {"backend": "magic"}])
def test_bad_config_rejected(values):
    with pytest.raises(ValueError):
        Config(source="test.mkv", **values).validate()
