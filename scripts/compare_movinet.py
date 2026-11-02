"""Official MoViNet-A0 streaming activity diagnostic; CPU, never Coral/MQTT.

Kinetics600 has no general fall class. The chair/bike scores below are only
diagnostics and must not be interpreted as calibrated generic fall likelihood.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np


def interpreter_class():
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter
    return Interpreter


def export_metadata(root):
    interpreter = interpreter_class()(model_path=str(root/'models/movinet_a0_stream.tflite'),num_threads=2)
    interpreter.allocate_tensors()
    # This is export-time metadata. Runtime can use old TFLite without signatures.
    signature = interpreter._get_full_signature_list()['serving_default']
    details = {d['index']:d for d in interpreter.get_input_details()}
    metadata = {'inputs':{name:{'index':index,'shape':details[index]['shape'].tolist(),
        'dtype':np.dtype(details[index]['dtype']).name} for name,index in signature['inputs'].items()},
        'outputs':signature['outputs']}
    (root/'models/movinet-signature.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'input_tensors':len(metadata['inputs']),'output_tensors':len(metadata['outputs'])}))


class Movinet:
    def __init__(self,root):
        self.root=Path(root)
        self.meta=json.loads((self.root/'models/movinet-signature.json').read_text())
        self.interpreter=interpreter_class()(model_path=str(self.root/'models/movinet_a0_stream.tflite'),num_threads=2)
        self.interpreter.allocate_tensors()
        self.reset()

    def reset(self):
        self.states={k:np.zeros(v['shape'],dtype=v['dtype']) for k,v in self.meta['inputs'].items() if k!='image'}

    def predict(self,bgr):
        # Preserve the complete room rather than center-cropping away people.
        # Letterbox is explicitly an evaluation adaptation, not a trained recipe.
        h,w=bgr.shape[:2];scale=min(172/w,172/h);nw,nh=round(w*scale),round(h*scale)
        image=np.zeros((172,172,3),np.float32);x,y=(172-nw)//2,(172-nh)//2
        image[y:y+nh,x:x+nw]=cv2.cvtColor(cv2.resize(bgr,(nw,nh)),cv2.COLOR_BGR2RGB)/255.
        return self.predict_image(image[None,None])

    def predict_image(self,image):
        for name,value in dict(self.states,image=image).items():
            self.interpreter.set_tensor(self.meta['inputs'][name]['index'],value)
        self.interpreter.invoke()
        outputs={name:self.interpreter.get_tensor(index) for name,index in self.meta['outputs'].items()}
        logits=outputs.pop('logits').reshape(-1)
        self.states=outputs
        scores=np.exp(logits-logits.max());scores/=scores.sum()
        return scores


def run(root,source,video,fps=5,reset_seconds=0):
    model=Movinet(root)
    labels=(root/'research/kinetics_600_labels.txt').read_text().splitlines()
    chair,bike,crawl=(labels.index(s) for s in ('falling off chair','falling off bike','crawling baby'))
    cuts=json.loads((root/'scene-cuts.json').read_text())['cuts'] if video=='falls' else []
    cuts=iter(cuts);next_cut=next(cuts,float('inf'))
    last_reset=0.;resets=[]
    out=root/'movinet'/f'{reset_seconds:g}s'/video;out.mkdir(parents=True,exist_ok=False)
    cap=cv2.VideoCapture(str(source));source_fps=cap.get(cv2.CAP_PROP_FPS)
    frame_index,sampled=0,0;timings=[];started=time.perf_counter()
    with (out/'predictions.jsonl').open('w') as record:
        while True:
            ok,image=cap.read()
            if not ok:break
            now=frame_index/source_fps;frame_index+=1
            if now+1e-6<sampled/fps:continue
            if now>=next_cut or (reset_seconds and now-last_reset>=reset_seconds):
                model.reset();last_reset=now;resets.append(now)
            while now>=next_cut:next_cut=next(cuts,float('inf'))
            tick=time.perf_counter();scores=model.predict(image);timings.append((time.perf_counter()-tick)*1000)
            top=np.argsort(scores)[-5:][::-1]
            record.write(json.dumps({'time':sampled/fps,'source_time':now,'top5':[{ 'action':labels[i],'score':float(scores[i])} for i in top],
                'chair_score':float(scores[chair]),'bike_score':float(scores[bike]),'crawl_score':float(scores[crawl])})+'\n')
            sampled+=1
    cap.release()
    summary={'model':'official_movinet_a0_stream','video':video,'frames':sampled,'input_fps':fps,
        'elapsed_s':time.perf_counter()-started,'cpu_ms_mean':float(np.mean(timings)),
        'cpu_ms_p95':float(np.percentile(timings,95)),'reset_seconds':reset_seconds,'state_resets':resets,
        'warning':'Generic Kinetics600 activity diagnostic, not general fall detection. Full-room RGB letterbox to 172; no person boxes/posture outputs.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('artifacts/temporal-comparison'))
    p.add_argument('--metadata-only',action='store_true')
    p.add_argument('--reset-seconds',type=float,default=0)
    args=p.parse_args()
    if args.metadata_only:
        export_metadata(args.root);return
    sources={'ring':next((Path.home()/'Downloads').glob('*R6IfcTpU2Qk*.mkv')),
             'falls':Path('artifacts/pose-comparison/falls-full.mp4')}
    for name,path in sources.items():run(args.root,path,name,reset_seconds=args.reset_seconds)


if __name__=='__main__':main()
