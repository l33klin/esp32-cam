"""Score explicitly reviewed frames; labels are preliminary, not benchmark truth.

Count metrics do not need boxes. Rough pose-audit boxes are only for matching
selected subjects, never for reporting detection mAP. Ambiguous counts excluded.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from indoorcam.analytics import iou, posture


def masked(person, regions, width=960, height=540):
    b = person["bbox"]
    x, y = (b[0]+b[2])/2/width, (b[1]+b[3])/2/height
    return any(r["bbox_normalized"][0] <= x <= r["bbox_normalized"][2]
               and r["bbox_normalized"][1] <= y <= r["bbox_normalized"][3] for r in regions)


def match_audits(audits, people, threshold=0.2):
    pairs = []
    for g, gt in enumerate(audits):
        box = np.array(gt["bbox_normalized"])*[960,540,960,540]
        for p, person in enumerate(people):
            overlap = iou(box, person["bbox"])
            if overlap >= threshold:
                pairs.append((overlap,g,p))
    matches, used = {}, set()
    for overlap,g,p in sorted(pairs,reverse=True):
        if g not in matches and p not in used:
            matches[g] = (people[p],overlap)
            used.add(p)
    return matches


def score_frames(labels, state_index, model, threshold=None, split=None, keypoint_threshold=None):
    rows, audit_rows = [], []
    confusion = Counter()
    per_class = {k:Counter(total=0,matched=0,correct=0) for k in ('standing','sitting','lying','other')}
    for f in labels["frames"]:
        if split and f["split"] != split:
            continue
        key = (f["video"],round(f["time"]*8))
        if key not in state_index:
            continue
        s = state_index[key]
        people = s["people"]
        if keypoint_threshold is not None:
            # Classification-layer diagnostic only: keep the detected subjects
            # fixed so this does not masquerade as a newly measured detector.
            people=[{**p,'raw_posture':posture(np.asarray(p['keypoints']),keypoint_threshold)[0]} for p in people]
        if threshold is not None:
            people = []
            for p in s["candidates"]:
                points = np.asarray(p["keypoints"])
                if p["score"] >= threshold and p["bbox"][3]-p["bbox"][1] >= 540*.06 and np.count_nonzero(points[:,2]>=.3)>=3:
                    people.append({**p,"raw_posture":posture(points,.3)[0]})
            people = sorted(people,key=lambda p:p["score"],reverse=True)[:10]
        unmasked = [p for p in people if not masked(p,f["ignore_regions"])]
        if f["count"] is not None:
            rows.append({"model":model,"video":f["video"],"time":f["time"],"split":f["split"],
                "truth_count":f["count"],"pred_count_raw":len(people),"pred_count_masked":len(unmasked),
                "ignored_screen_mirror":len(people)-len(unmasked)})
        matches = match_audits(f["pose_audit"],unmasked,labels["matching_iou"])
        for g,gt in enumerate(f["pose_audit"]):
            predicted,overlap = matches.get(g,(None,0))
            expected = gt["posture"] if gt["posture"]!='other' else 'unknown'
            actual = predicted["raw_posture"] if predicted else 'miss'
            per_class[gt["posture"]].update(total=1,matched=int(predicted is not None),correct=int(actual==expected))
            confusion[(gt["posture"],actual)] += 1
            audit_rows.append({"model":model,"video":f["video"],"time":f["time"],"split":f["split"],
                "expected":gt["posture"],"predicted":actual,"match_iou":overlap,"note":gt["note"],
                "visible_keypoints":int(np.count_nonzero(np.asarray(predicted['keypoints'])[:,2]>=.3)) if predicted else 0})
    def mean_error(field):
        return float(np.mean([abs(r[field]-r['truth_count']) for r in rows])) if rows else None
    core = [per_class[k] for k in ('standing','sitting','lying')]
    n,matched_count,correct = (sum(c[k] for c in core) for k in ('total','matched','correct'))
    metrics = {"count_frames":len(rows),"count_mae_raw":mean_error('pred_count_raw'),"count_mae_masked":mean_error('pred_count_masked'),
        "count_exact_masked":sum(r['pred_count_masked']==r['truth_count'] for r in rows),
        "pose_core_targets":n,"pose_core_matched":matched_count,"pose_core_correct":correct,
        "pose_e2e_accuracy":correct/n if n else None,"pose_conditional_accuracy":correct/matched_count if matched_count else None,
        "pose_macro_recall":float(np.mean([c['correct']/c['total'] for c in core if c['total']])) if n else None,
        "pose_per_class":per_class,"confusion":{f'{a}->{b}':v for (a,b),v in confusion.items()}}
    return metrics,rows,audit_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels',type=Path,default=Path('config/pose-review-labels.json'))
    parser.add_argument('--results',type=Path,default=Path('artifacts/pose-comparison/results'))
    parser.add_argument('--out',type=Path,default=Path('artifacts/pose-comparison/scores'))
    args = parser.parse_args()
    labels = json.loads(args.labels.read_text())
    args.out.mkdir(parents=True,exist_ok=True)
    all_metrics,all_counts,all_audits,sweeps,class_sweeps = {},[],[],{},{}
    for directory in sorted(args.results.iterdir()):
        if not directory.is_dir():
            continue
        state_index = {}
        summaries = {}
        for video in ('ring','falls'):
            p = directory/video/'summary.json'
            if not p.exists():
                continue
            summaries[video] = json.loads(p.read_text())
            for line in (p.parent/'detections.jsonl').open():
                s=json.loads(line);state_index[(video,round(s['time']*8))]=s
        if not state_index:
            continue
        model = directory.name
        metrics,rows,audits = score_frames(labels,state_index,model)
        all_metrics[model]={'metrics':metrics,'summaries':summaries,
                            'calibration':score_frames(labels,state_index,model,split='calibration')[0],
                            'review_holdout':score_frames(labels,state_index,model,split='review')[0]}
        all_counts += rows;all_audits += audits
        class_sweeps[model]={str(t):{split:score_frames(labels,state_index,model,split=split,keypoint_threshold=t)[0]
                                   for split in ('calibration','review')} for t in (.15,.2,.25,.3,.4,.5)}
        if model in ('posenet353','posenet481','posenet721','resnet416','resnet640'):
            sweeps[model]={}
            for threshold in (.10,.15,.20,.25,.30,.40,.50):
                sweeps[model][str(threshold)]={split:score_frames(labels,state_index,model,threshold,split)[0]
                                              for split in ('calibration','review')}
    (args.out/'comparison.json').write_text(json.dumps(all_metrics,ensure_ascii=False,indent=2)+'\n')
    (args.out/'threshold-sweeps.json').write_text(json.dumps(sweeps,ensure_ascii=False,indent=2)+'\n')
    (args.out/'classification-thresholds.json').write_text(json.dumps(class_sweeps,ensure_ascii=False,indent=2)+'\n')
    for name,rows in [('count-audit.csv',all_counts),('pose-audit.csv',all_audits)]:
        if rows:
            with (args.out/name).open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({m:v['metrics'] for m,v in all_metrics.items()},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
