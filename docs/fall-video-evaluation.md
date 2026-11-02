# 婴儿摔倒合集测试（2026-10-30）

## 格式转换

输入为用户提供的 `All the babies getting hurt [k3WKIgPuaBY].webm`，原始视频 AV1 + Opus、1280×716、30 fps、约 210.841 秒。

截取前 **90.000 秒**，转换为 **H.264 / yuv420p + AAC，MP4 faststart**，保持原始 1280×716 和 30 fps。输出为本地 `artifacts/falls/input-90s.mp4`，约 16 MB，已用 ffprobe 校验时长、视频与音频编码。原文件未覆盖。

可复用的转换命令（输出必须是新文件）：

```bash
python scripts/prepare_video.py \
  "$HOME/Downloads/All the babies getting hurt [k3WKIgPuaBY].webm" \
  artifacts/falls/input-90s.mp4 --seconds 90
```

转换脚本还支持 `--start 秒数`，可以提取某次倒地前后的独立片段，避免每次都处理整个合集。

## 回放结果

在 pn41 以原有 MobileDet + MoveNet 双 TPU 配置离线处理，MQTT/RTSP/API 全部关闭，评估容器使用 `--network none`。Coral 设备被现有服务独占，测试期间短暂停止本项目 analytics，退出后自动恢复。

| 项目 | 结果 |
|---|---|
| 基线输入 | 完整前 90 秒 |
| 基线采样 | 8 fps，共 720 帧 |
| 平均/p95 处理耗时 | 180.46 / 241.37 ms |
| 总处理时间 | 135.97 秒，约 5.3 处理帧/秒 |
| 稳定姿态观察 | standing 247、sitting 158、lying 0、unknown 1407 |
| 原规则的候选跌倒事件 | **0** |
| 按剪辑分界重置轨迹后 | **仍为 0** |
| 更敏感的前 12 秒对照 | 96 帧，**仍为 0**；出现大量假人体框 |

这里的 8 fps 是离线输出文件的时间基准，不代表推理计算达到 8 fps。标注 MP4 的时长仍为 90 秒。

这段视频明确包含可见倒地过程，现有版本却没有产生事件，说明存在明显漏报，**不能把目前的规则当作可靠的婴儿摔倒告警**。没有完整逐帧人工真值标注，因此没有报告精确率/召回率百分比。

## 核对到的问题

1. **首次倒地时目标丢失。** 约 3.5 秒仍有宝宝的站立检测和髋部关键点；约 4 秒起在连续抽查的倒地/躺地帧中没有对应人体框。没有连续目标证据，后续时序规则无法判断。
2. **没有识别出躺卧。** 基线的原始和稳定姿态均没有 `lying`，而现有告警要求近期站/坐后转为持续躺卧。缩短 `fall_hold` 不能修复前面的检测/姿态缺失。
3. **部分失衡来自爬行/跪姿。** 约 20 秒的宝宝从低姿态倒地，不符合原先“站/坐后倒地”的前提；同时很快被成人遮挡。
4. **电视中的人被计数。** 合集约 35–54 秒的电视里有真人节目，会出现人体框。这种机位需要按实际场景设置 ROI 或排除电视区域，不能将全画面人数当作室内真人真值。
5. **合集并非连续监控。** 不同日期/视角之间存在转场。参考重置位置为 17.5、28.1、35.2、54.1、68.7、79.45 秒；它们由画面核对与 FFmpeg scene score 辅助确定，并非人工逐帧精确的分镜标注。重放时清空轨迹后事件仍为 0，所以本轮零告警不是转场重置造成的。

## 敏感度对照

单独提取前 12 秒，使用 `config/fall-diagnostic.json`：检测阈值从 0.45 降至 0.25，关键点阈值降至 0.2，最小人体高度从 6% 降至 3%，关闭肩部校验，增加为 3×3 分块，躺卧持续要求从 1.5 秒降至 0.5 秒。

这一对照没有产生告警，单帧假人体框可达到 `max_people=10`，平均处理约 345.57 ms、p95 505.75 ms。该配置仅用于定位问题，**没有部署到实时服务**。需要改进倒地过程中的目标检测/跟踪和姿态证据，再评估告警逻辑；仅调低阈值会同时增加误检和计算量。

## 输出文件

- `artifacts/falls/input-90s.mp4`：转换后的原画面，包含 AAC 音频。
- `artifacts/falls/baseline/annotated-h264.mp4`：当前程序的完整标注结果，H.264、960×540、8 fps，无音频。
- `artifacts/falls/baseline/contact-sheet.jpg`：抽帧检查图。
- `artifacts/falls/baseline/detections.jsonl`：逐帧框、骨架、姿态、轨迹与事件。
- `artifacts/falls/baseline/reset-detections.jsonl`：在剪辑分界重置轨迹的规则重放。
- `artifacts/falls/diagnostic/`：前 12 秒敏感度对照的配置、逐帧记录和耗时。

服务器结果保存在 `~/esp32-indoorcam/evaluations/falls-baseline` 与 `evaluations/falls-diagnostic`。测试结束后原循环流已恢复。既有自动检查再次执行为 19 项通过、1 项按需 MQTT 测试跳过。

## 复现

在 pn41 中使用全新的输出目录；运行器复制现有配置后强制切到离线模式，避免向 HA 发布测试告警：

```bash
cd ~/esp32-indoorcam
python3 scripts/run_offline_pn41.py falls-90s.mp4 evaluations/another-baseline
python3 scripts/run_offline_pn41.py falls-first12s.mp4 evaluations/another-diagnostic \
  --overrides config/fall-diagnostic.json
```

规则重放不重新运行模型，无法找回当时漏掉的人体。它适合隔离“模型漏检”与“规则参数”问题。下面命令在本地已安装项目依赖的 Python 虚拟环境中执行：

```bash
python -m scripts.replay_analytics \
  artifacts/falls/baseline/detections.jsonl artifacts/falls/another-reset.jsonl \
  --reset-at 17.5 28.1 35.2 54.1 68.7 79.45
```
