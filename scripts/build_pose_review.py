"""Build local-only videos, review frames and a self-contained HTML index."""
import argparse
import html
import json
from pathlib import Path

import cv2
import numpy as np

from indoorcam.config import Config
from indoorcam.video import annotate
from scripts.score_pose_comparison import masked,match_audits


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('artifacts/pose-comparison'))
    parser.add_argument('--labels',type=Path,default=Path('config/pose-review-labels.json'))
    parser.add_argument('--recommended',default='thunder')
    parser.add_argument('--baseline',default='lightning')
    parser.add_argument('--study',choices=['pose','yolo'],default='pose')
    parser.add_argument('--recommendation',help='Review-specific conclusion, rendered as plain text')
    args=parser.parse_args();root=args.root;out=root/'review';out.mkdir(exist_ok=True)
    frames_dir=out/'frames';frames_dir.mkdir(exist_ok=True)
    labels=json.loads(args.labels.read_text())
    comparison=json.loads((root/'scores/comparison.json').read_text())
    complete={m:d for m,d in comparison.items() if set(d['summaries'])=={'ring','falls'}}
    preferred=[args.recommended,'lightning','posenet721','posenet481','posenet353',
               'posenet721_thunder','resnet416','resnet640','efficientdet1_letterbox_thunder']
    names=list(dict.fromkeys(m for m in preferred+list(complete) if m in complete))
    data={'models':names,'recommended':args.recommended,'baseline':args.baseline,'cases':[],'metrics':{}}
    states={};table=[];markdown=[]
    for model in names:
        record=complete[model]
        latencies=[]
        for video in ('ring','falls'):
            rows=[json.loads(l) for l in (root/'results'/model/video/'detections.jsonl').open()]
            for s in rows:states[(model,video,round(s['time']*8))]=s
            latencies += [s['inference_ms'] for s in rows]
        summaries=list(record['summaries'].values());n=sum(s['frames'] for s in summaries)
        perf={'inference_mean_ms':float(np.mean(latencies)),'inference_p95_ms':float(np.percentile(latencies,95)),
              'pipeline_fps':n/sum(s['elapsed_s'] for s in summaries),
              'parent_cpu_ms_frame':sum(s['parent_cpu_seconds'] for s in summaries)/n*1000,
              'peak_parent_rss_mb':max(s['peak_rss_mb'] for s in summaries),
              'tpu_devices':summaries[0]['coral_devices'],'frames':n,
              'first90_falls_candidates':sum(e['time']<90 for e in record['summaries']['falls']['fall_candidates'])}
        quality=record['review_holdout'];data['metrics'][model]={'performance':perf,'quality':quality,'all_frames_quality':record['metrics']}
        values=[model,f"{perf['inference_mean_ms']:.1f} / {perf['inference_p95_ms']:.1f}",f"{perf['pipeline_fps']:.1f}",
                f"{quality['count_mae_raw']:.2f} / {quality['count_mae_masked']:.2f}",
                f"{quality['count_exact_masked']}/{quality['count_frames']}",
                f"{quality['pose_core_correct']}/{quality['pose_core_targets']}",
                str(perf['tpu_devices']),f"{perf['peak_parent_rss_mb']:.0f}",str(perf['first90_falls_candidates'])]
        table.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>')
        markdown.append('| '+' | '.join(values)+' |')
    for i,f in enumerate(labels['frames']):
        image=cv2.imread(f['image'])
        if image is None:raise RuntimeError(f['image'])
        reference=image.copy()
        for region in f['ignore_regions']:
            x1,y1,x2,y2=(np.asarray(region['bbox_normalized'])*[960,540,960,540]).astype(int)
            cv2.rectangle(reference,(x1,y1),(x2,y2),(170,170,170),2)
            cv2.putText(reference,region['kind'],(x1+2,y1+16),cv2.FONT_HERSHEY_SIMPLEX,.5,(200,200,200),1)
        for g in f['pose_audit']:
            x1,y1,x2,y2=(np.asarray(g['bbox_normalized'])*[960,540,960,540]).astype(int)
            cv2.rectangle(reference,(x1,y1),(x2,y2),(255,80,220),2)
            cv2.putText(reference,'AUDIT '+g['posture'],(x1,max(16,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,80,220),1)
        ref_name=f'{i:02d}-reference.jpg';cv2.imwrite(str(frames_dir/ref_name),reference)
        case={'id':i,'video':f['video'],'time':f['time'],'truth_count':f['count'],'split':f['split'],
              'reference':'frames/'+ref_name,'expected':f['pose_audit'],'predictions':{}}
        for model in names:
            state=states[(model,f['video'],round(f['time']*8))]
            drawn=annotate(image,state,Config())
            cv2.putText(drawn,f"{model}  {f['video']} {f['time']}s",(8,525),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            filename=f'{i:02d}-{model}.jpg';cv2.imwrite(str(frames_dir/filename),drawn)
            people=state['people'];unmasked=[p for p in people if not masked(p,f['ignore_regions'])]
            matches=match_audits(f['pose_audit'],unmasked,labels['matching_iou'])
            case['predictions'][model]={'image':'frames/'+filename,'count':len(people),'masked_count':len(unmasked),
                'audit_predictions':[matches[g][0]['raw_posture'] if g in matches else 'miss' for g in range(len(f['pose_audit']))]}
        data['cases'].append(case)
    (out/'data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    (out/'labels.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2)+'\n')
    (root/'metrics-table.md').write_text('| 配置 | 推理均值/p95 ms | 完整处理 fps | 人数 MAE 原始/掩膜 | 人数完全一致 | 姿态正确 | TPU数 | RSS MB | 前90秒候选事件 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'+'\n'.join(markdown)+'\n')
    template=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Coral 姿态算法对比 Review</title><style>
body{font:16px/1.6 system-ui,-apple-system,sans-serif;margin:32px auto;padding:0 24px;max-width:1440px;color:#202620;background:#fafbf9}h1{font-size:30px;margin-bottom:4px}h2{margin-top:32px}p{max-width:1000px}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{border-bottom:1px solid #d6ddd6;text-align:left;padding:10px;white-space:nowrap}th{background:#eaf0e9}.scroll{overflow:auto}select,button{font:inherit;padding:8px 12px;border:1px solid #a6b3a5;border-radius:4px;background:white;color:inherit}button{cursor:pointer}.players{display:grid;grid-template-columns:1fr 1fr;gap:16px}video{width:100%;background:#111}label{display:block;margin-bottom:8px}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:16px}.card{background:white;border:1px solid #d8dfd6;padding:10px}.card img{width:100%}.note{background:#fff5d9;border-left:4px solid #bd8d2b;padding:12px 16px}small{color:#526052}a{color:#175b3e}code{background:#eaf0e9;padding:2px 5px}footer{margin:40px 0;border-top:1px solid #d6ddd6;padding-top:16px}@media(max-width:800px){.players{grid-template-columns:1fr}body{padding:0 12px}}
</style><h1>Coral 姿态算法对比</h1><p>pn41 · 双 PCIe Coral · Intel N6005 · Python 3.9 / TFLite 2.5 · 2026-10-31</p>
<div class="note">这是本次两个视频的实测与初步视觉核对。43 个抽查帧中，30 帧人数明确；预设其中 10 帧用于阈值校准，表格使用另外 20 帧。36 个姿态目标包含 30 个站/坐/躺目标和 6 个其他动作；表格使用保留复核部分的 20 个站/坐/躺目标。标签由助手视觉核对、尚待你复核，不属于 COCO mAP 或正式跌倒数据集成绩。基线开发曾使用这些视频，同片相邻帧也有关联，不能当作独立泛化测试。各模型共用现有姿态/跌倒规则。</div>
<h2>本轮选择</h2><p>__RECOMMENDATION__</p>
<h2>实测结果</h2><p>每组均测试 Ring 全片和完整 210.826 秒合集，每秒抽 8 帧。8 fps 是文件采样/播放速度；下面“完整处理 fps”才是硬件实际离线处理速度，包含解码、缩放、规则、标注和 H.264 写出。“人数 MAE”越低越好；掩膜版排除灰框内电视/镜面重复人形。“姿态正确”核对平滑前的 raw_posture，把漏检和 unknown 视为未成功分类；视频展示的是按现有 0.5 秒保持规则平滑后的姿态。各方案置信度未做统一概率校准，这里比较的是列明配置的实际管线，不代表各架构经过充分调参后的上限。</p>
<div class="scroll"><table><thead><tr><th>配置</th><th>推理均值/p95 ms</th><th>完整处理 fps</th><th>人数 MAE 原始/掩膜</th><th>人数完全一致</th><th>姿态正确</th><th>TPU 数</th><th>主进程 RSS MB</th><th>前90秒候选事件</th></tr></thead><tbody>__TABLE__</tbody></table></div>
<p><a href="../scores/count-audit.csv">逐帧人数 CSV</a> · <a href="../scores/pose-audit.csv">逐目标姿态 CSV</a> · <a href="../scores/comparison.json">原始摘要 JSON</a> · <a href="labels.json">原始视觉核对标签</a> · <a href="data.json">此页面数据</a></p>
<h2>同步视频对比</h2><label>视频 <select id="clip"><option value="ring">Ring 全片</option><option value="falls">完整摔倒合集</option></select> <button id="toggle">同步播放 / 暂停</button> <button id="reset">回到开头</button></label>
<div class="players"><div><label>左侧配置 <select id="left"></select></label><video id="a" controls muted playsinline preload="metadata"></video><a id="da" download>下载左侧视频</a></div><div><label>右侧配置 <select id="right"></select></label><video id="b" controls muted playsinline preload="metadata"></video><a id="db" download>下载右侧视频</a></div></div>
<small>左侧视频控制两侧时间。视频无音轨。前 90 秒是主要跌倒核对段；后续片段参与完整速度测试和部分视觉核对。零候选事件不等于没有摔倒。</small>
<h2>原图与模型逐帧核对</h2><label>固定复核帧 <select id="case"></select> <button id="jump">跳转两侧视频到此帧</button></label><p id="case-info"></p><div id="gallery" class="gallery"></div>
<details><summary>原始视频中的倒地、跌坐与歧义动作连拍</summary><p>以下只做动作核对，不把剪辑后已躺地、被桌子遮挡或主动蹲下自动当成确定摔倒。<a href="../fall-event-review/notes.json">逐段观察说明</a>。</p><ul><li><a href="../fall-event-review/event-1.jpg">3–6.5 秒：站立后倒地</a></li><li><a href="../fall-event-review/event-2.jpg">19–22.5 秒：低姿态前扑，很快被成人遮挡</a></li><li><a href="../fall-event-review/event-3.jpg">29–32.5 秒：椅旁失衡、跌坐</a></li><li><a href="../fall-event-review/event-4.jpg">40–43.5 秒：椅子倾倒、儿童被遮挡</a></li><li><a href="../fall-event-review/event-5.jpg">58–61.5 秒：快速失衡，桌沿挡住身体</a></li><li><a href="../fall-event-review/event-6.jpg">68–71.5 秒：切镜后已躺卧，未看到起始动作</a></li><li><a href="../fall-event-review/event-7.jpg">80–83.5 秒：低姿态动作存在歧义</a></li></ul></details>
<h2>怎么读这些结果</h2><ul><li>紫框是少量姿态复核目标的粗标框，仅用于匹配，不是完整人体框真值；未用它报告检测 mAP。</li><li>gray/灰框是电视或镜面排除区域。原始算法仍可能计入其中人形，表格同时保留原始和掩膜后的误差。</li><li>standing / sitting / lying 是关节点之后的现有几何规则输出；unknown 是证据不足或其他动作，miss 表示复核目标未匹配到人体。</li><li>模糊、遮挡严重的帧仍提供图片和视频，但不强行纳入精确人数 MAE。每类姿态样本很少，结果需用实际机位再验证。</li><li>主进程 RSS 与 CPU 统计不包含 H.264 编码子进程；本次模型调用计时包含 CPU 后处理，不能全部解释为 TPU 内核运算。</li><li>本次没有向 HA 发布测试状态或事件，也没有把新模型部署到现有监控服务。</li></ul>
<footer><strong>来源</strong>：<a href="https://github.com/google-coral/test_data">Google Coral 官方模型</a> · <a href="https://github.com/google-coral/project-posenet">官方 PoseNet 与 CPU 解码器</a> · <a href="https://blog.tensorflow.org/2021/08/pose-estimation-and-classification-on-edge-devices-with-MoveNet-and-TensorFlow-Lite.html">TensorFlow MoveNet 与姿态分类说明</a> · <a href="https://docs.ultralytics.com/integrations/edge-tpu/">YOLO Edge TPU 导出限制（调研，未作为实测候选）</a></footer>
<script>const D=__PAYLOAD__;const $=id=>document.getElementById(id);const a=$('a'),b=$('b');
for(const id of ['left','right']){for(const m of D.models){const o=document.createElement('option');o.value=m;o.textContent=m;$(id).append(o)}}$('left').value=D.models.includes(D.baseline)?D.baseline:D.models[0];$('right').value=D.models.includes(D.recommended)?D.recommended:D.models[0];
function source(v,id,dl,t){v.src='../results/'+$(id).value+'/'+$('clip').value+'/annotated.mp4';$(dl).href=v.src;v.onloadedmetadata=()=>{if(t<v.duration)v.currentTime=t}}
function update(){const t=a.currentTime||0;a.pause();b.pause();source(a,'left','da',t);source(b,'right','db',t)}for(const id of ['left','right','clip'])$(id).onchange=update;
function seek(t){if(Number.isFinite(a.duration))a.currentTime=Math.min(t,a.duration-.01);if(Number.isFinite(b.duration))b.currentTime=Math.min(t,b.duration-.01)}
$('reset').onclick=()=>seek(0);$('toggle').onclick=()=>{if(a.paused){a.play().catch(()=>{});b.play().catch(()=>{})}else{a.pause();b.pause()}};
a.onplay=()=>b.play().catch(()=>{});a.onpause=()=>b.pause();a.onseeked=()=>{if(Math.abs(a.currentTime-b.currentTime)>.2)b.currentTime=a.currentTime};a.ontimeupdate=()=>{if(!a.paused&&Math.abs(a.currentTime-b.currentTime)>.3)b.currentTime=a.currentTime};
for(const c of D.cases){const o=document.createElement('option');o.value=c.id;o.textContent=c.video+' '+c.time+'s · '+c.split;$('case').append(o)}
function gallery(){const c=D.cases[Number($('case').value)];$('case-info').textContent='人数核对：'+(c.truth_count===null?'不确定，不纳入人数误差':c.truth_count)+'；姿态目标：'+(c.expected.map(e=>e.posture+' ('+e.note+')').join('；')||'无')+'.';const g=$('gallery');g.textContent='';function card(title,url,text){const d=document.createElement('div');d.className='card';const h=document.createElement('strong');h.textContent=title;const link=document.createElement('a');link.href=url;link.target='_blank';const im=document.createElement('img');im.src=url;im.loading='lazy';link.append(im);const p=document.createElement('small');p.textContent=text;d.append(h,link,p);g.append(d)}card('原始画面与复核标注',c.reference,'灰框：排除区域；紫框：粗标姿态目标');for(const m of D.models){const p=c.predictions[m];card(m,p.image,'人数 '+p.count+' / 排除后 '+p.masked_count+'；目标预测 '+(p.audit_predictions.join(', ')||'无'))}}
$('case').onchange=gallery;$('jump').onclick=()=>{const c=D.cases[Number($('case').value)];if($('clip').value!==c.video){$('clip').value=c.video;update();a.addEventListener('loadedmetadata',()=>seek(c.time),{once:true});b.addEventListener('loadedmetadata',()=>seek(c.time),{once:true})}else seek(c.time)};
$('case').value=String(D.cases.findIndex(c=>c.video==='falls'&&c.time===5.5));gallery();update();
</script></html>'''
    payload=json.dumps(data,ensure_ascii=False).replace('</','<\\/')
    vga_path=root/'vga-scores/comparison.json'
    vga_html=''
    if vga_path.exists():
        vga=json.loads(vga_path.read_text());vga_rows=[]
        for name in ('lightning','thunder'):
            record=vga.get(name,{})
            if set(record.get('summaries',{}))!={'ring','falls'}:continue
            ss=list(record['summaries'].values());quality=record['review_holdout']
            fps=sum(s['frames'] for s in ss)/sum(s['elapsed_s'] for s in ss)
            vga_rows.append(f'<tr><td>{name}</td><td>{fps:.2f}</td><td>{quality["count_mae_masked"]:.2f}</td>'
                            f'<td>{quality["pose_core_correct"]}/20</td><td>'
                            f'<a href="../vga-results/{name}/ring/annotated.mp4">Ring</a> · '
                            f'<a href="../vga-results/{name}/falls/annotated.mp4">合集</a></td></tr>')
        if vga_rows:
            vga_html=('<h2>640 像素宽输入补测</h2><p>保持视野，先把源图宽度降为 640，再进入同样的管线。'
                      '该补测没有模拟真实镜头、JPEG、Wi-Fi 或采集时延，不能视为 ESP32 实机成绩。'
                      '下面仍使用同一组保留复核标签，与上方主表分开统计。</p><table><thead><tr><th>配置</th>'
                      '<th>完整 fps</th><th>人数 MAE 排除后</th><th>核心姿态正确</th><th>补测视频</th></tr></thead>'
                      '<tbody>'+''.join(vga_rows)+'</tbody></table>')
    template=template.replace('<h2>怎么读这些结果</h2>',vga_html+'<h2>怎么读这些结果</h2>')
    recommendation=args.recommendation or ('MobileDet + MoveNet Thunder 是本轮优先继续验证的姿态管线，Lightning 作为速度优先方案。'
                    'Thunder 的复核优势只有一个目标，不能据此证明统计优势。PoseNet 的快速配置适合进一步探索人数/骨架场景，'
                    '但配合当前姿态规则的分类效果较差。本轮没有通过婴幼儿摔倒验证的方案；零候选事件存在明显漏报，不能作为安全告警验收。'
                    '新模型尚未替换线上服务。')
    if args.study=='yolo':
        template=template.replace('Coral 姿态算法对比','YOLO Pose / Coral 实测对比')
        template=template.replace('<h2>实测结果</h2>',
            '<h2>完整视频实测结果</h2><p>主表只列两个全片均完成的配置。编译失败或过慢的配置另见文字报告，'
            '不把短测帧率当作完整视频成绩。YOLO 均使用原始多人头与 CPU NMS；YOLO26 使用 one-to-many 分支。'
            'cal010 表示人体置信度 0.10；thunder 后缀表示再用 Thunder 精修。单独的 thunder / lightning 是 MobileDet 基线。</p>')
        start=template.index('<footer>');end=template.index('</footer>',start)+len('</footer>')
        template=template[:start]+('<footer><strong>来源</strong>：'
            '<a href="https://docs.ultralytics.com/integrations/edge-tpu/">Ultralytics Edge TPU</a> · '
            '<a href="https://docs.ultralytics.com/tasks/pose/">YOLO Pose 输出说明</a> · '
            '<a href="https://github.com/ultralytics/ultralytics">Ultralytics 源码</a> · '
            '<a href="https://github.com/google-coral/test_data">Coral MoveNet 基线权重</a></footer>')+template[end:]
    (out/'index.html').write_text(template.replace('__TABLE__','\n'.join(table)).replace('__PAYLOAD__',payload)
                                .replace('__RECOMMENDATION__',html.escape(recommendation)))
    print(json.dumps({'models':names,'cases':len(data['cases']),'html':str(out/'index.html')},ensure_ascii=False))


if __name__=='__main__':main()
