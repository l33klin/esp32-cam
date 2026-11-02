"""Causal, CPU-only temporal replay of cached person skeletons. No MQTT/RTSP."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from indoorcam.analytics import Analytics
from indoorcam.config import Config
from scripts.temporal_models import Alarm, History, TemporalModel


def run(root, backbone, kind, video, fps, seconds, tsstg_min_score=0):
    source = Path('artifacts/yolo-comparison/results') / backbone / video / 'detections.jsonl'
    if not source.exists():
        source = root/'cache'/backbone/video/'detections.jsonl'
    model = TemporalModel(kind, root)
    frames = 30 if kind == 'tsstg' else 32
    cfg = Config(source='cached_skeletons', track_timeout=max(.8,1.5/fps))
    analytics = Analytics(cfg)
    histories, alarms = {}, {}
    cuts = json.loads((root/'scene-cuts.json').read_text())['cuts'] if video == 'falls' else []
    cuts = iter(cuts); next_cut = next(cuts, float('inf'))
    name = f'{backbone}__{kind}__{seconds:g}s__{fps:g}fps'
    if kind == 'tsstg' and tsstg_min_score == 0:
        name += '__argmax'
    out = root/'results'/name/video
    out.mkdir(parents=True, exist_ok=False)
    durations, events, scores = [], [], Counter()
    baseline_events, baseline_postures = [], Counter()
    index, previous_bucket, predictions, observations = 0, -1, 0, 0
    started = time.perf_counter()
    with source.open() as stream, (out/'detections.jsonl').open('w') as writer:
        for line in stream:
            original = json.loads(line)
            now = original['time']
            bucket = int(now*fps+1e-5)
            if bucket == previous_bucket:
                continue
            previous_bucket = bucket
            while now >= next_cut:
                analytics.reset(); histories.clear(); alarms.clear()
                next_cut = next(cuts, float('inf'))
            detections = [{**p, 'keypoints':np.asarray(p['keypoints'])} for p in original['people']]
            state = analytics.update(detections, now, (540,960,3))
            baseline_events.extend(state['events'])
            baseline_postures.update(state['postures'])
            frame_events = []
            for person in state['people']:
                ident = person['id']
                history = histories.setdefault(ident, History(seconds, fps, frames))
                alarm = alarms.setdefault(ident, Alarm())
                sequence = history.append(now, person['keypoints'])
                person.update(action='warming_up', fall_score=None, temporal_posture='unknown')
                observations += 1
                active = False
                if sequence is not None:
                    tick = time.perf_counter()
                    try:
                        prediction = model.predict(sequence)
                    except ValueError:
                        person['action'] = 'invalid_skeleton'
                    else:
                        durations.append((time.perf_counter()-tick)*1000)
                        person.update(prediction)
                        predictions += 1
                        scores[prediction['action']] += 1
                        cutoff = tsstg_min_score if kind == 'tsstg' else .5
                        active = prediction['fall_top1'] and prediction['fall_score'] >= cutoff
                        if kind == 'tsstg' and prediction['confidence'] >= tsstg_min_score:
                            person['temporal_posture'] = {'Standing':'standing', 'Walking':'standing',
                                'Sitting':'sitting', 'Lying Down':'lying'}.get(prediction['action'], 'unknown')
                person['temporal_fall_candidate'] = active
                if alarm.update(now, active, history.max_gap):
                    event = {'time':now, 'track_id':ident, 'bbox':person['bbox'],
                             'action':person['action'], 'score':person['fall_score']}
                    events.append(event); frame_events.append(event)
            # Expire dead tracks so a long recording uses bounded memory.
            live = set(analytics.tracks)
            histories = {k:v for k,v in histories.items() if k in live}
            alarms = {k:v for k,v in alarms.items() if k in live}
            state['temporal_events'] = frame_events
            state['original_baseline_events'] = original['events']
            writer.write(json.dumps(state)+'\n')
            index += 1
    elapsed = time.perf_counter()-started
    summary = {'name':name, 'backbone':backbone, 'model':kind, 'video':video,
        'input_fps':fps, 'window_seconds':seconds, 'window_samples':frames,
        'tsstg_min_score':tsstg_min_score if kind == 'tsstg' else None,
        'frames':index, 'person_observations':observations, 'classified_observations':predictions,
        'coverage':predictions/observations if observations else 0,
        'classifier_ms_mean':float(np.mean(durations)) if durations else None,
        'classifier_ms_p95':float(np.percentile(durations,95)) if durations else None,
        'elapsed_s':elapsed, 'cached_replay_fps':index/elapsed, 'actions':dict(scores),
        'events':events, 'baseline_events':baseline_events,
        'baseline_postures':dict(baseline_postures), 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'timing_note':'Replay excludes person/pose inference. Classifier timing includes input preparation and CPU ONNX. Not end-to-end camera FPS.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('events','baseline_events','baseline_postures')}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('artifacts/temporal-comparison'))
    parser.add_argument('--backbones',nargs='+',default=['thunder','yolov8n_480_cal010'])
    parser.add_argument('--models',nargs='+',default=['tsstg','mcfd_tcn','mcfd_lstm'])
    parser.add_argument('--fps',nargs='+',type=float,default=[8,3,2,1])
    parser.add_argument('--windows',nargs='+',type=float,default=[1.2,3.625])
    parser.add_argument('--videos',nargs='+',default=['ring','falls'])
    parser.add_argument('--tsstg-min-score',type=float,default=0,
                        help='Upstream uses argmax without a .5 sigmoid gate; 0 preserves that behavior')
    args=parser.parse_args()
    for backbone in args.backbones:
        for kind in args.models:
            for fps in args.fps:
                for seconds in args.windows:
                    for video in args.videos:
                        run(args.root,backbone,kind,video,fps,seconds,args.tsstg_min_score)


if __name__ == '__main__':
    main()
