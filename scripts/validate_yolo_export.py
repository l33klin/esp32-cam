"""Compare TFLite FP32 raw outputs with the native PyTorch checkpoint."""
import argparse
import json
from pathlib import Path
import os


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',nargs='+',required=True);args=p.parse_args()
    root=Path('artifacts/yolo-comparison').resolve()
    os.environ.setdefault('YOLO_CONFIG_DIR',str(root/'config'));os.environ.setdefault('MPLCONFIGDIR',str(root/'mplconfig'))
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
    import cv2
    import numpy as np
    import torch
    import tensorflow as tf
    from ultralytics import YOLO
    from scripts.yolo_pose_adapter import letterbox
    torch.set_num_threads(2);cv2.setNumThreads(2);rows=[]
    for name in args.models:
        family,size=name.rsplit('_',1);size=int(size)
        y=YOLO(str(root/'models'/(family+'-pose.pt'))).model.eval();y.end2end=False;y.fuse()
        interpreter=tf.lite.Interpreter(model_path=str(root/'models'/f'{name}_fp32.tflite'),num_threads=2)
        interpreter.allocate_tensors();inp=interpreter.get_input_details()[0];out=interpreter.get_output_details()[0]
        for filename in ['ring-001.00.jpg','falls-005.50.jpg','falls-070.00.jpg']:
            frame=cv2.imread(str(Path('artifacts/pose-comparison/review-frames')/filename))
            image,_=letterbox(frame,size);real=image.astype(np.float32)/255
            with torch.inference_mode():native=y(torch.from_numpy(real.transpose(2,0,1)[None].copy()))
            if isinstance(native,(tuple,list)):native=native[0]
            native=native.detach().cpu().numpy().copy();native[:,:4]/=size
            points=native[:,5:].reshape(1,17,3,-1);points[:,:,:2]/=size
            interpreter.set_tensor(inp['index'],real[None]);interpreter.invoke();converted=interpreter.get_tensor(out['index'])
            if native.shape!=converted.shape:raise ValueError(f'{name}: shape mismatch {native.shape} vs {converted.shape}')
            error=np.abs(native-converted)
            row={'model':name,'image':filename,'max_normalized_error':float(error.max()),
                 'mean_absolute_error':float(error.mean()),'max_person_conf_error':float(error[:,4].max())}
            if error.max()>.01:raise ValueError(f'Export mismatch: {row}')
            rows.append(row);print(row,flush=True)
        del y,interpreter
    (root/'conversion-validation.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
