"""Small, explicitly bounded audit of temporal-model replay results."""
import argparse
import json
from pathlib import Path

from scripts.score_pose_comparison import score_frames


def inside_region(box, region):
    x,y=(box[0]+box[2])/2/960,(box[1]+box[3])/2/540
    return region[0]<=x<=region[2] and region[1]<=y<=region[3]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('artifacts/temporal-comparison'))
    args=p.parse_args();root=args.root
    labels=json.loads(Path('config/temporal-review-labels.json').read_text())
    pose_labels=json.loads(Path('config/pose-review-labels.json').read_text())
    results={}
    for directory in sorted((root/'results').iterdir()):
        summaries={};states={};index={}
        for video in ['ring','falls']:
            if not (directory/video/'summary.json').exists():continue
            summaries[video]=json.loads((directory/video/'summary.json').read_text())
            states[video]=[json.loads(x) for x in (directory/video/'detections.jsonl').read_text().splitlines()]
            for state in states[video]:
                for person in state['people']:
                    person['raw_posture']=person['temporal_posture']
                index[(video,round(state['time']*8))]=state
        if len(summaries)!=2:continue
        first=summaries['falls']
        audited=[]
        for window in labels['positive_windows']:
            video=window['video'];start,end=window['window'];region=window['target_region']
            events=[e for e in summaries[video]['events'] if start<=e['time']<=end and inside_region(e['bbox'],region)]
            people=[(s['time'],p) for s in states[video] if start<=s['time']<=end for p in s['people'] if inside_region(p['bbox'],region)]
            classified=[(t,p) for t,p in people if p['fall_score'] is not None]
            peak=max(classified,key=lambda x:x[1]['fall_score']) if classified else None
            audited.append({**window,'hit':bool(events),'alarm_times':[e['time'] for e in events],
                'onset_or_later_alarms':[e for e in events if e['time']>=window.get('reviewed_onset_approx',start)],
                'alarms':events,'target_observations':len(people),'classified_observations':len(classified),
                'peak_score':peak[1]['fall_score'] if peak else None,'peak_time':peak[0] if peak else None,
                'peak_action':peak[1]['action'] if peak else None})
        negatives=[]
        for window in labels['negative_windows']:
            start,end=window['window']
            events=[e for e in summaries[window['video']]['events'] if start<=e['time']<end]
            negatives.append({**window,'alarms':events,'count':len(events)})
        pose=score_frames(pose_labels,index,directory.name,split='review')[0] if first['model']=='tsstg' else None
        results[directory.name]={'backbone':first['backbone'],'model':first['model'],'fps':first['input_fps'],
            'window_seconds':first['window_seconds'],'reviewed_positive_hits':sum(w['hit'] for w in audited),
            'onset_or_later_window_hits':sum(bool(w['onset_or_later_alarms']) for w in audited),
            'reviewed_positive_windows':len(audited),'negative_alarm_count':sum(w['count'] for w in negatives),
            'negative_seconds':sum(w['window'][1]-w['window'][0] for w in negatives),
            'all_video_alarm_count':sum(len(s['events']) for s in summaries.values()),
            'classified_observations':sum(s['classified_observations'] for s in summaries.values()),
            'person_observations':sum(s['person_observations'] for s in summaries.values()),
            'positive_windows':audited,'negative_windows':negatives,'posture_review':pose,'summaries':summaries}
    (root/'scores.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    for name,v in results.items():
        print(json.dumps({'name':name,'hits':v['reviewed_positive_hits'],'negative_alarms':v['negative_alarm_count'],
            'all_alarms':v['all_video_alarm_count'],'pose_correct':v['posture_review']['pose_core_correct'] if v['posture_review'] else None}))
    movinet={}
    for directory in (root/'movinet').glob('*'):
        videos={}
        for video in ('ring','falls'):
            if not (directory/video/'summary.json').exists():continue
            states=[json.loads(x) for x in (directory/video/'predictions.jsonl').read_text().splitlines()]
            videos[video]={'summary':json.loads((directory/video/'summary.json').read_text()),
                'max_chair_score':max(s['chair_score'] for s in states),
                'max_bike_score':max(s['bike_score'] for s in states),
                'top1_actions':sorted({s['top5'][0]['action'] for s in states}),
                'windows':[{**w,'max_chair_score':max(s['chair_score'] for s in states if w['window'][0]<=s['time']<=w['window'][1]),
                            'max_bike_score':max(s['bike_score'] for s in states if w['window'][0]<=s['time']<=w['window'][1])}
                           for w in labels['positive_windows'] if w['video']==video]}
        movinet[directory.name]=videos
    (root/'movinet-scores.json').write_text(json.dumps(movinet,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
