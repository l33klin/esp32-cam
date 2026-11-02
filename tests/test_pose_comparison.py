import numpy as np
from scripts.compare_pose_models import letterbox_rgb, unletterbox, skeleton_box
from scripts.score_pose_comparison import match_audits, masked


def test_letterbox_roundtrip_and_padding_confidence():
    image=np.zeros((540,960,3),np.uint8)
    _,transform=letterbox_rgb(image,481,353)
    left,top,sx,sy=transform
    original=np.array([[100,100],[400,300],[800,500]],float)
    model_yx=np.stack([original[:,1]/sy+top,original[:,0]/sx+left],axis=1)
    points=unletterbox(model_yx,np.ones(3),transform,image.shape)
    assert np.allclose(points[:,:2],original)
    padding=unletterbox(np.array([[0,240]]),np.ones(1),transform,image.shape)
    assert padding[0,2]==0


def test_low_confidence_outlier_cannot_expand_skeleton_box():
    points=np.array([[100,100,.9],[120,150,.8],[110,200,.9],[900,500,.01]])
    box=skeleton_box(points,(540,960,3))
    assert box[0]<100 and box[2]<150 and box[3]<240


def test_top_left_detector_letterbox_preserves_rgb_and_maps_back():
    image = np.full((540, 960, 3), [10, 20, 30], np.uint8)
    resized, transform = letterbox_rgb(image, 384, 384, centered=False)
    assert transform[:2] == (0, 0)
    assert resized[0, 0].tolist() == [30, 20, 10]
    assert np.all(resized[216:] == 0)
    points = unletterbox(np.array([[108, 192]]), np.ones(1), transform, image.shape)
    assert np.allclose(points[0], [480, 270, 1])


def test_pose_subject_matching_is_one_to_one_and_mask_is_explicit():
    audits=[{'bbox_normalized':[.1,.1,.3,.4]},{'bbox_normalized':[.12,.1,.32,.4]}]
    person={'bbox':[96,54,288,216]}
    assert len(match_audits(audits,[person]))==1
    assert masked(person,[{'bbox_normalized':[0,0,.4,.5]}])
    assert not masked(person,[])
