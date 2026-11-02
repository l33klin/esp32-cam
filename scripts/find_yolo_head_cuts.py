"""Find three-scale learned head tensors before CPU layout/coordinate decoding."""
import json
from pathlib import Path


def main():
    import tensorflow as tf
    root=Path('artifacts/yolo-comparison');result={}
    for path in sorted((root/'models').glob('*_int8.tflite')):
        engine=tf.lite.Interpreter(model_path=str(path),num_threads=2);engine.allocate_tensors()
        details={d['index']:d for d in engine.get_tensor_details()};ops=[op for op in engine._get_ops_details() if op['op_name']!='DELEGATE']
        size=int(engine.get_input_details()[0]['shape'][1])
        producers={int(i):op for op in ops for i in op['outputs']};selected=[]
        for op in ops:
            if op['op_name']=='TRANSPOSE':
                index=int(op['inputs'][0]);d=details[index]
                if producers[index]['op_name']=='CONV_2D' and len(d['shape'])==4 and d['shape'][3] in (4,51,64):
                    selected.append(index)
            if op['op_name']=='LOGISTIC':
                d=details[int(op['outputs'][0])]
                # Trace the single-class score vector back to its three learned
                # convolutions; attention unrolling also creates 1-channel convs.
                n=sum((size//stride)**2 for stride in (8,16,32))
                if len(d['shape'])==3 and sorted(d['shape'])==[1,1,n]:
                    stack=[int(op['inputs'][0])];seen=set()
                    while stack:
                        index=stack.pop()
                        if index in seen:continue
                        seen.add(index);parent=producers.get(index)
                        if parent is None:continue
                        if parent['op_name']=='CONV_2D':
                            selected.append(index)
                        else:
                            stack.extend(int(i) for i in parent['inputs'] if int(i)>=0)
        names=[details[i]['name'] for i in dict.fromkeys(selected)]
        if len(names)!=9:raise RuntimeError(f'{path.name}: expected nine 3-scale head cuts, got {len(names)}')
        result[path.stem.removesuffix('_int8')]=names
    (root/'head-cuts.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:len(v) for k,v in result.items()})


if __name__=='__main__':main()
