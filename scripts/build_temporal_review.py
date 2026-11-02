"""Create inspectable temporal overlays and a local synchronized review page."""
import json
from pathlib import Path
import subprocess

import cv2

from indoorcam.video import fit

ROOT=Path('artifacts/temporal-comparison')


def annotate_video(name,video):
    directory=ROOT/'results'/name/video
    output=directory/'annotated.mp4'
    if output.exists():return
    source=next((Path.home()/'Downloads').glob('*R6IfcTpU2Qk*.mkv')) if video=='ring' else Path('artifacts/pose-comparison/falls-full.mp4')
    states=[json.loads(x) for x in (directory/'detections.jsonl').read_text().splitlines()]
    cap=cv2.VideoCapture(str(source));fps=cap.get(cv2.CAP_PROP_FPS);index=0;sampled=0
    encoder=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','bgr24',
        '-s','960x540','-r','8','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','veryfast',
        '-crf','24','-pix_fmt','yuv420p','-movflags','+faststart','-n',str(output)],stdin=subprocess.PIPE)
    try:
        while sampled<len(states):
            ok,image=cap.read()
            if not ok:raise RuntimeError('Video ended before cached state sequence')
            now=index/fps;index+=1
            if now+1e-6<states[sampled]['time']:continue
            image=fit(image,960,540);state=states[sampled]
            for person in state['people']:
                x1,y1,x2,y2=map(round,person['bbox'])
                red=person['temporal_fall_candidate'];color=(70,70,240) if red else (100,230,160)
                cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
                score=person.get('fall_score')
                label=f"#{person['id']} {person['action']}"
                if score is not None:label+=f" / fall {score:.2f}"
                cv2.putText(image,label,(max(0,x1),max(55,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),3,cv2.LINE_AA)
                cv2.putText(image,label,(max(0,x1),max(55,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
                for x,y,confidence in person['keypoints']:
                    if confidence>=.3 and 0<=x<960 and 0<=y<540:cv2.circle(image,(round(x),round(y)),2,color,-1)
            cv2.rectangle(image,(0,0),(960,44),(12,12,12),-1)
            title=f"{name.replace('thunder__','')} | {video} {state['time']:.3f}s"
            cv2.putText(image,title,(8,18),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
            status='EVENT' if state['temporal_events'] else 'No new event'
            cv2.putText(image,status,(8,36),cv2.FONT_HERSHEY_SIMPLEX,.43,(80,140,255) if state['temporal_events'] else (170,170,170),1,cv2.LINE_AA)
            encoder.stdin.write(image.tobytes());sampled+=1
    finally:
        cap.release();encoder.stdin.close()
        if encoder.wait(timeout=30):raise RuntimeError('H264 encoding failed')
    print(name,video,sampled,flush=True)


def main():
    names=[f'thunder__{kind}__{window}s__8fps'+('__argmax' if kind=='tsstg' else '')
        for kind in ['tsstg','mcfd_tcn','mcfd_lstm'] for window in ['1.2','3.625']]
    for name in names:
        for video in ['ring','falls']:annotate_video(name,video)
    scores=json.loads((ROOT/'scores.json').read_text())
    labels=json.loads(Path('config/temporal-review-labels.json').read_text())
    options=[{'name':name,'model':scores[name]['model'],'window':scores[name]['window_seconds'],
        'hits':scores[name]['reviewed_positive_hits'],'false':scores[name]['negative_alarm_count'],
        'afterOnset':scores[name]['onset_or_later_window_hits'],
        'pose':scores[name]['posture_review']['pose_core_correct'] if scores[name]['posture_review'] else None,
        'events':{video:scores[name]['summaries'][video]['events'] for video in ('ring','falls')}} for name in names]
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>连续视频时序识别 · 离线评审</title>
<style>body{font:16px/1.6 system-ui,sans-serif;max-width:1400px;margin:30px auto;padding:0 24px;background:#f6f5f1;color:#202522}h1{font-size:27px}small,.note{color:#616862}select,button{font:inherit;padding:6px 10px;margin:4px;border:1px solid #abb5ac;border-radius:4px;background:white}video{width:100%;background:#111}section{display:grid;grid-template-columns:1fr 1fr;gap:16px}article{min-width:0}table{border-collapse:collapse;width:100%;background:white}th,td{padding:8px 10px;border-bottom:1px solid #d9ded7;text-align:left}a{color:#215f4b}.events{max-height:230px;overflow:auto} .warn{border-left:4px solid #b56a3e;padding:8px 15px;background:#f4eadf}@media(max-width:800px){section{grid-template-columns:1fr}}</style>
<h1>连续视频时序识别 · 离线评审</h1><p>相同人体骨架，比较原有几何规则与预训练时序模型。两个视频只用于评估，没有用于训练。</p>
<p class="warn">本轮不建议替换线上算法。TCN/LSTM 仅输出 fall / non_fall；TSSTG 支持动作类别。表中“命中”只是 4 个初步核对窗口内、目标区域中出现告警，需检查告警是否发生在真实倒地过程中。47 秒负例中的告警按误报计；其他时段尚未完整标注。</p>
<label>视频 <select id="video"><option value="falls">宝宝摔倒合集 · 完整 210.8 秒</option><option value="ring">Ring · 完整 37.9 秒</option></select></label>
<label>时序模型 <select id="model"></select></label><button id="play">同步播放 / 暂停</button>
<section><article><h3>基线：Thunder + 几何规则</h3><video id="baseline" controls preload="metadata"></video></article><article><h3 id="heading">时序模型</h3><video id="temporal" controls preload="metadata"></video></article></section>
<p id="metrics"></p><h3>已核对的摔倒窗口</h3><div id="windows"></div><h3>当前模型的全部候选告警</h3><p class="note">点击时间同步跳转；数量多不代表识别更好。请特别检查成年人弯腰、桌边遮挡和电视中的人。</p><div class="events" id="events"></div>
<h3>8 fps 对比</h3><table><thead><tr><th>模型 / 窗口</th><th>宽窗口覆盖 / 4</th><th>失衡后触发 / 4</th><th>47 秒负例误报</th><th>姿态正确 / 20</th></tr></thead><tbody id="table"></tbody></table>
<p class="note">TSSTG 使用原项目 argmax 决策；分数是未校准的 sigmoid 输出。3.625 秒与 1.2 秒窗口均为固定对照，不按测试结果选阈值。插值到 30/32 点不会增加真实观测信息。页内视频 8 fps 是回放帧率，非推理性能。</p>
<p><a href="../../../docs/temporal-algorithm-comparison.md">完整实验报告（Markdown）</a></p>
<script>const options=OPTIONS,labels=LABELS;const $=s=>document.getElementById(s),a=$('baseline'),b=$('temporal');let guard=false;
function title(x){return x.model+' / '+x.window+' 秒'}
for(const x of options){const o=document.createElement('option');o.value=x.name;o.textContent=title(x);$('model').append(o)}
function jump(t){a.currentTime=t;b.currentTime=t}function link(t,label){const e=document.createElement('button');e.textContent=label;e.onclick=()=>jump(t);return e}
function load(){const x=options.find(x=>x.name===$('model').value),v=$('video').value,t=Number.isFinite(a.currentTime)?a.currentTime:0;a.pause();b.pause();a.src='../../yolo-comparison/results/thunder/'+v+'/annotated.mp4';b.src='../results/'+x.name+'/'+v+'/annotated.mp4';for(const e of[a,b])e.addEventListener('loadedmetadata',()=>{e.currentTime=Math.min(t,e.duration-.2)},{once:true});$('heading').textContent=title(x);$('metrics').textContent='宽窗口覆盖 '+x.hits+'/4 · 失衡后触发 '+x.afterOnset+'/4 · 47 秒负例中 '+x.false+' 次误报 · 姿态 '+(x.pose===null?'该模型不输出姿态':x.pose+'/20 正确');$('windows').replaceChildren();for(const w of labels.positive_windows.filter(w=>w.video===v))$('windows').append(link(w.window[0],w.window[0]+'–'+w.window[1]+'s '+w.note));$('events').replaceChildren();for(const e of x.events[v])$('events').append(link(e.time, e.time.toFixed(3)+'s #'+e.track_id+' '+e.score.toFixed(3)));}
$('video').onchange=()=>{jump(0);load()};$('model').onchange=load;$('play').onclick=async()=>{if(a.paused){b.currentTime=a.currentTime;await Promise.all([a.play(),b.play()])}else{a.pause();b.pause()}};
for(const [src,dst] of[[a,b],[b,a]]){src.addEventListener('seeking',()=>{if(!guard&&Math.abs(src.currentTime-dst.currentTime)>.15){guard=true;dst.currentTime=src.currentTime;setTimeout(()=>guard=false,200)}});src.addEventListener('pause',()=>dst.pause());src.addEventListener('play',()=>{if(dst.paused)dst.play().catch(()=>{})})}
for(const x of options){const tr=document.createElement('tr');for(const value of[title(x),x.hits+'/4',x.afterOnset+'/4',x.false,x.pose===null?'—':x.pose+'/20']){const td=document.createElement('td');td.textContent=value;tr.append(td)}$('table').append(tr)}load();</script></html>'''
    page=page.replace('OPTIONS',json.dumps(options,ensure_ascii=False)).replace('LABELS',json.dumps(labels,ensure_ascii=False))
    (ROOT/'review/index.html').write_text(page)
    (ROOT/'review/options.json').write_text(json.dumps(options,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
