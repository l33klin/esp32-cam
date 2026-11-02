import numpy as np

from scripts.temporal_models import Alarm, History, mcfd_features, tsstg_inputs


def pose(value=0):
    points = np.zeros((17,3), np.float32)
    points[:,0] = np.arange(17)*5+value
    points[:,1] = np.arange(17)*8+value
    points[:,2] = .8
    return points


def test_history_waits_for_evidence_and_resets_after_gap():
    history=History(1.2,8,30)
    for n in range(9):
        assert history.append(n/8,pose(n)) is None
    sequence=history.append(9/8,pose(9))
    assert sequence.shape==(30,17,3)
    assert sequence[-1,0,0]==9
    assert history.append(3,pose(30)) is None


def test_causal_prefix_does_not_depend_on_future():
    first,second=History(1.2,8,30),History(1.2,8,30)
    a=b=None
    for n in range(10):
        a=first.append(n/8,pose(n));b=second.append(n/8,pose(n))
    first.append(10/8,pose(99999))
    assert np.array_equal(a,b)
    assert a[...,0].max()<100


def test_alarm_needs_persistence_and_does_not_bridge_gap():
    alarm=Alarm()
    assert not alarm.update(0,True,.5)
    assert not alarm.update(.125,True,.5)
    assert alarm.update(.25,True,.5)
    assert not alarm.update(.5,True,.5)
    assert not alarm.update(8,True,.5)
    assert not alarm.update(8.125,False,.5)
    assert not alarm.update(8.25,True,.5)


def test_joint_mapping_and_feature_shapes():
    seq=np.stack([pose(n) for n in range(32)])
    graph=tsstg_inputs(seq[:30])
    assert graph['points'].shape==(1,3,30,14)
    assert graph['motion'].shape==(1,2,29,14)
    assert np.allclose(graph['points'][0,:,:,13],
        (graph['points'][0,:,:,1]+graph['points'][0,:,:,2])/2)
    assert mcfd_features(seq,True)['features'].shape==(1,32,85)
    assert mcfd_features(seq,False)['features'].shape==(1,32,51)
