"""Restore upstream TSSTG argmax decisions using saved network outputs only."""
import json
from pathlib import Path

from scripts.temporal_models import Alarm


def main():
    root=Path('artifacts/temporal-comparison')
    for directory in sorted((root/'results').glob('*__tsstg__*')):
        if directory.name.endswith('__argmax'):continue
        for video in ('ring','falls'):
            source=directory/video
            summary=json.loads((source/'summary.json').read_text())
            summary['tsstg_min_score']=.5
            (source/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            out=root/'results'/(directory.name+'__argmax')/video
            out.mkdir(parents=True,exist_ok=False)
            alarms={};events=[]
            max_gap=max(.5,1.5/summary['input_fps'])
            cuts=iter(json.loads((root/'scene-cuts.json').read_text())['cuts'] if video=='falls' else [])
            next_cut=next(cuts,float('inf'))
            with (source/'detections.jsonl').open() as stream,(out/'detections.jsonl').open('w') as writer:
                for line in stream:
                    state=json.loads(line);now=state['time'];frame_events=[]
                    while now>=next_cut:
                        alarms.clear();next_cut=next(cuts,float('inf'))
                    current=set()
                    for person in state['people']:
                        ident=person['id'];current.add(ident)
                        person['temporal_posture']={'Standing':'standing','Walking':'standing','Sitting':'sitting',
                            'Lying Down':'lying'}.get(person['action'],'unknown')
                        active=person['action']=='Fall Down'
                        person['temporal_fall_candidate']=active
                        alarm=alarms.setdefault(ident,Alarm())
                        if alarm.update(now,active,max_gap):
                            event={'time':now,'track_id':ident,'bbox':person['bbox'],'action':person['action'],'score':person['fall_score']}
                            events.append(event);frame_events.append(event)
                    alarms={k:v for k,v in alarms.items() if k in current or v.last is not None and now-v.last<=max_gap}
                    state['temporal_events']=frame_events;writer.write(json.dumps(state)+'\n')
            result={**summary,'name':directory.name+'__argmax','events':events,'tsstg_min_score':0,
                    'decision_replay':'Exact saved network outputs; restores upstream argmax without .5 sigmoid gating. Timing inherited; inference not repeated.'}
            (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Upstream argmax decisions replayed without rerunning inference')


if __name__=='__main__':main()
