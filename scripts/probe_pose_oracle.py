"""Diagnostic only: replace detector boxes with reviewed coarse subject crops.

This measures neither deployable detection accuracy nor end-to-end performance.
It isolates MoveNet keypoints/common posture rules from missing detector boxes.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from indoorcam.analytics import posture
from indoorcam.inference import Inference,make_interpreter


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels',type=Path,default=Path('config/pose-review-labels.json'))
    parser.add_argument('--model-root',type=Path,default=Path('evaluations/pose-comparison'))
    parser.add_argument('--images-root',type=Path,default=Path('evaluations/pose-comparison/review-frames'))
    parser.add_argument('--out',type=Path,default=Path('evaluations/pose-comparison/oracle.json'))
    args=parser.parse_args();labels=json.loads(args.labels.read_text());results={}
    cv2.setNumThreads(2)
    for name,path in [('lightning',Path('models/pose_edgetpu.tflite')),
                      ('thunder',args.model_root/'models/movenet_single_pose_thunder_ptq_edgetpu.tflite')]:
        engine=Inference.__new__(Inference);engine.pose=make_interpreter(path,'pci:0')
        rows=[]
        for f in labels['frames']:
            if not f['pose_audit']:continue
            image=cv2.imread(str(args.images_root/Path(f['image']).name))
            if image is None:raise RuntimeError('Review frame missing')
            for gt in f['pose_audit']:
                box=(np.asarray(gt['bbox_normalized'])*[960,540,960,540]).tolist()
                tick=time.perf_counter();points=engine.keypoints(image,box);elapsed=(time.perf_counter()-tick)*1000
                raw,features=posture(points,.3)
                rows.append({'video':f['video'],'time':f['time'],'split':f['split'],'bbox':box,'expected':gt['posture'],
                    'raw_posture':raw,'features':features,'keypoints':points.tolist(),'pose_ms':elapsed,'note':gt['note']})
        results[name]=rows
        del engine
    args.out.write_text(json.dumps({'purpose':__doc__,'results':results},indent=2)+'\n')
    print(json.dumps({name:{'targets':len(rows),'correct_core':sum(r['raw_posture']==r['expected'] for r in rows if r['expected']!='other'),
                          'lying_correct':sum(r['raw_posture']=='lying' and r['expected']=='lying' for r in rows)} for name,rows in results.items()}))


if __name__=='__main__':main()
