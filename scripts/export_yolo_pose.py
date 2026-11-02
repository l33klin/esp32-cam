"""Export normalized raw YOLO Pose heads on macOS; compile separately on pn41.

Uses Ultralytics' own tf_wrapper and onnx2saved_model implementation. The explicit
two stages permit ARM64 export while the Edge TPU compiler runs on Linux x86_64.
No evaluation video frames are used for INT8 calibration.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['yolov8n-pose', 'yolo11n-pose', 'yolo26n-pose'], required=True)
    parser.add_argument('--size', type=int, choices=[320, 480, 512], required=True)
    parser.add_argument('--root', type=Path, default=Path('artifacts/yolo-comparison'))
    args = parser.parse_args()
    root = args.root.resolve()
    os.environ.setdefault('YOLO_CONFIG_DIR', str(root/'config'))
    os.environ.setdefault('MPLCONFIGDIR', str(root/'mplconfig'))
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
    os.environ.setdefault('YOLO_AUTOINSTALL', 'False')
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.export.tensorflow import tf_wrapper, onnx2saved_model

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    cv2.setNumThreads(2)
    name = args.model.replace('-pose', '')+'_'+str(args.size)
    out = root/'exports'/name
    out.mkdir(parents=True, exist_ok=True)
    weight = root/'models'/(args.model+'.pt')
    model = YOLO(str(weight))
    copied = out/weight.name
    shutil.copy2(weight, copied)
    model = YOLO(str(copied))
    # Edge TPU's official export disables the TopK/end-to-end branch too.
    model.model.end2end = False
    model.model = tf_wrapper(model.model)
    def tensorflow_c2f_graph(exporter):
        # The generic ONNX exporter swaps C2f.chunk for split. Official TF/Edge
        # TPU export deliberately keeps chunk to avoid FlexSplitV/layout barriers.
        # Restore that same graph in this two-stage ARM64 -> Linux workflow.
        from ultralytics.nn.modules import C2f
        for layer in exporter.model.modules():
            if isinstance(layer,C2f):
                layer.forward=type(layer).forward.__get__(layer,type(layer))
    model.add_callback('on_export_start',tensorflow_c2f_graph)
    started = time.time()
    onnx_path = model.export(format='onnx', imgsz=args.size, opset=13,
                             dynamic=False, simplify=True, nms=None, device='cpu')
    images, calibration_paths = [], []
    for path in sorted((root/'calibration/coco128/images/train2017').glob('*.jpg')):
        image = cv2.imread(str(path))
        h, w = image.shape[:2]
        scale = min(args.size/w, args.size/h)
        rw, rh = round(w*scale), round(h*scale)
        padded = np.full((args.size, args.size, 3), 114, np.uint8)
        left, top = (args.size-rw)//2, (args.size-rh)//2
        padded[top:top+rh, left:left+rw] = cv2.cvtColor(cv2.resize(image, (rw, rh)), cv2.COLOR_BGR2RGB)
        images.append(padded)
        calibration_paths.append(path.name)
    if len(images) != 128:
        raise RuntimeError(f'Expected 128 independent COCO images, got {len(images)}')
    # Pin the optimization used by official edgetpu export; ordinary SavedModel
    # export otherwise enables group convolution optimizations on some models.
    saved = out/'saved_model'
    onnx_path=str(Path(onnx_path).resolve())
    previous_cwd=Path.cwd()
    try:
        # The upstream helper keeps a public auxiliary sample in its cwd.
        # Keep that download with calibration assets, outside the repo root.
        os.chdir(root/'calibration')
        onnx2saved_model(onnx_path, saved, quantize=None,
                         disable_group_convolution=True, prefix=name)
    finally:
        os.chdir(previous_cwd)
    # Produce only the INT8 variant we need, rather than onnx2tf's additional
    # dynamic-range and INT16-activation calibrations of the same graph.
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    def representative():
        for image in images:
            yield [image[None].astype(np.float32)/255]
    converter.representative_dataset = representative
    (saved/'model_full_integer_quant.tflite').write_bytes(converter.convert())
    outputs = {}
    for suffix, kind in [('float32', 'fp32'), ('full_integer_quant', 'int8')]:
        files = [saved/'model_full_integer_quant.tflite'] if kind=='int8' else list(saved.glob('*_'+suffix+'.tflite'))
        if len(files) != 1:
            raise RuntimeError(f'Expected one {suffix} model: {files}')
        dest = root/'models'/f'{name}_{kind}.tflite'
        shutil.copy2(files[0], dest)
        outputs[kind] = {'file': str(dest.relative_to(root)), 'bytes': dest.stat().st_size,
                          'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}
    manifest = {'name': name, 'source_model': args.model, 'size': args.size,
                'source_weight_sha256': hashlib.sha256(weight.read_bytes()).hexdigest(),
                'versions': {p: importlib.metadata.version(p) for p in
                    ['ultralytics', 'torch', 'tensorflow', 'onnx', 'onnx2tf', 'numpy']},
                'calibration': {'source': 'Ultralytics assets coco128.zip', 'images': calibration_paths,
                    'archive_sha256': hashlib.sha256((root/'calibration/coco128.zip').read_bytes()).hexdigest(),
                    'evaluation_images_used': False},
                'head': 'one-to-many raw head; xywh/keypoints normalized by input dimensions; CPU NMS',
                'preprocessing': 'centered RGB letterbox, pad 114, float input 0..1',
                'opset': 13, 'disable_group_convolution': True, 'outputs': outputs,
                'c2f_export': 'TensorFlow chunk path; generic ONNX forward_split reverted',
                'quantizer': 'TensorFlow Lite representative-dataset full INT8; int8 input/output',
                'started': started, 'finished': time.time()}
    (out/'export.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
