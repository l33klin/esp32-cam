"""Check EfficientDet tensor semantics and resize sensitivity on review frames.

Diagnostic only: not a timed full-pipeline configuration. Named outputs follow
google-coral/pycoral's signature mapping: output_0=count, 1=scores, 2=classes,
3=boxes. The existing ordered adapter must agree exactly on these models.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from indoorcam.inference import make_interpreter


def main():
    root = Path('evaluations/pose-comparison')
    labels = json.loads(Path('config/pose-review-labels.json').read_text())
    cv2.setNumThreads(2)
    report = {}
    for name, size in [('efficientdet0', 320), ('efficientdet1', 384)]:
        model = root / f'models/efficientdet_lite{name[-1]}_{size}_ptq_edgetpu.tflite'
        engine = make_interpreter(model, 'pci:0')
        inp, outputs = engine.get_input_details()[0], engine.get_output_details()
        record = {'input': {'shape': inp['shape'].tolist(), 'dtype': str(inp['dtype']),
                            'quantization': list(inp['quantization'])},
                  'outputs': [{'name': d['name'], 'shape': d['shape'].tolist()} for d in outputs],
                  'rows': []}
        for frame in labels['frames']:
            image = cv2.imread(str(root/'review-frames'/Path(frame['image']).name))
            if image is None:
                raise RuntimeError('Review frame missing')
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            for mode in ('stretch', 'letterbox_top_left'):
                if mode == 'stretch':
                    resized = cv2.resize(rgb, (size, size))
                    sx, sy = w/size, h/size
                else:
                    scale = min(size/w, size/h)
                    rw, rh = round(w*scale), round(h*scale)
                    resized = np.zeros((size, size, 3), np.uint8)
                    resized[:rh, :rw] = cv2.resize(rgb, (rw, rh))
                    sx, sy = w/rw, h/rh
                engine.set_tensor(inp['index'], resized[None].astype(inp['dtype']))
                engine.invoke()
                values = [engine.get_tensor(d['index']) for d in outputs]
                by_suffix = {d['name'].rsplit(':', 1)[1]: v for d, v in zip(outputs, values)}
                boxes, classes, scores, count = values
                for a, b in zip(values, [by_suffix[k] for k in ('3', '2', '1', '0')]):
                    if not np.array_equal(a, b):
                        raise RuntimeError('Positional outputs disagree with official named mapping')
                n = min(int(count.flat[0]), boxes.shape[1])
                if not (np.all((scores >= 0) & (scores <= 1)) and
                        np.allclose(classes, np.round(classes)) and np.all(np.isfinite(boxes))):
                    raise RuntimeError('Invalid detector output ranges')
                detections = []
                for i in range(n):
                    y1, x1, y2, x2 = boxes[0, i]
                    detections.append({'class': int(classes[0, i]), 'score': float(scores[0, i]),
                        'bbox': [float(x1*size*sx), float(y1*size*sy),
                                 float(x2*size*sx), float(y2*size*sy)]})
                record['rows'].append({'video': frame['video'], 'time': frame['time'],
                                       'mode': mode, 'detections': detections})
        report[name] = record
        del engine
    out = root/'detector-sanity.json'
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'output': str(out), 'models': list(report),
                      'positional_matches_official_mapping': True,
                      'scores_and_classes_valid': True}))


if __name__ == '__main__':
    main()
