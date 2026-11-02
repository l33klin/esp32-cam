"""Paired image-only quality probe; hardware timing is not ranked across hosts."""
import argparse
import json
from pathlib import Path

import cv2

from indoorcam.analytics import posture
from indoorcam.config import Config
from scripts.score_pose_comparison import score_frames
from scripts.yolo_pose_adapter import YoloPose


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',required=True)
    p.add_argument('--backend',choices=['coral','cpu-fp32','cpu-int8'],required=True)
    p.add_argument('--model-root',type=Path,default=Path('artifacts/yolo-comparison/models'))
    p.add_argument('--images-root',type=Path,default=Path('artifacts/pose-comparison/review-frames'))
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();labels=json.loads(Path('config/pose-review-labels.json').read_text());results={}
    cv2.setNumThreads(2)
    suffix={'coral':'int8_edgetpu','cpu-fp32':'fp32','cpu-int8':'int8'}[args.backend]
    cfg=Config(backend='coral' if args.backend=='coral' else 'cpu',threshold=.25)
    for name in args.models:
        engine=YoloPose(cfg,args.model_root/f'{name}_{suffix}.tflite',cfg.backend)
        rows=[];index={}
        for f in labels['frames']:
            image=cv2.imread(str(args.images_root/Path(f['image']).name))
            if image is None:raise RuntimeError(f['image'])
            people,extra=engine.predict(image)
            people=[{**a,'raw_posture':posture(a['keypoints'],.3)[0],
                     'keypoints':a['keypoints'].tolist()} for a in people]
            row={'video':f['video'],'time':f['time'],'people':people,'count':len(people),
                 'candidates':extra['candidates']}
            rows.append(row);index[(f['video'],round(f['time']*8))]=row
        results[name]={'rows':rows,'metrics':score_frames(labels,index,name)[0],
                       'review_holdout':score_frames(labels,index,name,split='review')[0],
                       'calibration':score_frames(labels,index,name,split='calibration')[0],
                       'input':{'shape':engine.input['shape'].tolist(),'dtype':str(engine.input['dtype']),
                                'quantization':[float(engine.input['quantization'][0]),int(engine.input['quantization'][1])]},
                       'output':{'shape':engine.outputs[0]['shape'].tolist(),
                                 'quantization':[float(engine.outputs[0]['quantization'][0]),int(engine.outputs[0]['quantization'][1])]}}
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps({'backend':args.backend,'purpose':__doc__,'results':results},indent=2)+'\n')
        print(name,results[name]['review_holdout'],flush=True)
        del engine


if __name__=='__main__':main()
