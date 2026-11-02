"""Export research checkpoints to CPU ONNX and verify against PyTorch."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

from scripts.temporal_models import load_torch_model, mcfd_features, tsstg_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/temporal-comparison'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    np.random.seed(42)
    result = {}
    for kind in ('tsstg', 'mcfd_tcn', 'mcfd_lstm'):
        model, config, checkpoint = load_torch_model(kind, args.root)
        frames = config['frames']
        sequences = []
        states = [json.loads(s) for s in Path('artifacts/yolo-comparison/results/thunder/falls/detections.jsonl').read_text().splitlines()]
        for start in (0, 280, 600):
            poses = [s['people'][0]['keypoints'] for s in states[start:start+50] if s['people']]
            if len(poses) >= frames:
                sequences.append(np.asarray(poses[:frames], np.float32))
        sequences.append(np.random.default_rng(42).uniform(0, 1, (frames,17,3)).astype(np.float32)*[960,540,1])
        feeds = tsstg_inputs(sequences[0]) if kind == 'tsstg' else mcfd_features(sequences[0], config['velocity'])
        tensors = tuple(torch.from_numpy(x) for x in feeds.values())
        path = args.root / 'models' / (kind+'.onnx')
        torch.onnx.export(model, (tensors,) if kind == 'tsstg' else tensors,
            str(path), input_names=list(feeds), output_names=['output'], opset_version=17,
            dynamo=False)
        opts = ort.SessionOptions(); opts.intra_op_num_threads = 2; opts.inter_op_num_threads = 1
        session = ort.InferenceSession(str(path), sess_options=opts, providers=['CPUExecutionProvider'])
        checks = []
        for seq in sequences:
            feeds = tsstg_inputs(seq) if kind == 'tsstg' else mcfd_features(seq, config['velocity'])
            tensors = tuple(torch.from_numpy(x) for x in feeds.values())
            with torch.inference_mode():
                expected = model(tensors if kind == 'tsstg' else tensors[0]).numpy()
            actual = session.run(None, feeds)[0]
            error = float(np.max(np.abs(expected-actual)))
            assert np.allclose(expected, actual, atol=2e-5, rtol=1e-4), (kind, error)
            checks.append(error)
        result[kind] = {'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'onnx_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'onnx_bytes': path.stat().st_size,
            'parameters': sum(p.numel() for p in model.parameters()), 'config': config,
            'input_shapes': {k:list(v.shape) for k,v in feeds.items()},
            'export_checks': len(checks), 'max_absolute_error': max(checks),
            'torch': torch.__version__, 'onnxruntime': ort.__version__}
        print(json.dumps({kind:result[kind]}), flush=True)
    (args.root/'export-validation.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
