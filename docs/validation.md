# 验证记录（2026-10-30）

## 实物接入追加记录（2026-10-31）

- 鹿小班 ESP32-S3 N16R8 CAM / WROOM-1 板载天线，用户接口图确认为 ESP32-S3-EYE 引脚；已烧录并实测 16 MB Flash、约 8 MB PSRAM、OV3660 PID `0x3660`。
- 配网后地址 `192.168.50.39`，重启自动连接保存的 Wi-Fi。pn41 现已从 Ring 回放切换为真实摄像头 RTSP，`loop=false`、`roi=[]`（用户选择全画面）。
- 默认 OpenCV RTSP 探测在低帧率 MJPEG 上反复超时；加入独立 FFmpeg 采集路径，保留最新完整帧、重连代数、断流过期与有界停止。新增管道分段读取、帧边界、半帧 EOF 和停止/超时测试。
- 本机完整测试：22 passed、1 项需真实 broker 的检查默认 skipped；随后在 pn41 单独执行真实 MQTT 检查：1 passed。`esp32s3_uart` 和临时 `wifi_diag` 均实际构建、烧录成功；最后已恢复摄像头固件。
- `scripts/check_stack.py` 通过：HTTP 状态、认证 H.264 RTSP 960×540 / 6 fps、真实 MQTT。HA 浏览器实测 readyState=4、未暂停，初次读数约 116 秒/699 帧，结束验收时累计约 389 秒/2273 帧；实物截图 `artifacts/hardware/ha-live-camera.png`，播放器及实体状态 `artifacts/hardware/ha-live-status.json`。
- 45 秒取样包含重启后的前 19 秒等待；首次在线后 26/26 次状态检查在线，抽到 10 个不同的分析时间戳。记录 `artifacts/hardware/live-acceptance.json`。此窗口不是长期可用性测试。
- 固件当前 VGA、JPEG quality 20、10 MHz XCLK、单个内部 RAM 帧缓冲、32 KiB 内部 RAM RTP 发送缓冲。实际新帧约 1–3 fps，仍可能停顿；服务器 6 fps 输出会重复最近画面，不代表采集/推理 6 fps。Wi-Fi 未启动相机的吞吐对照为约 157/176/184 KB/s，仍需继续排查相机工作时的传输下降。
- HA“实时监控说明”已替换原“测试回放”，保留对卡顿、离线和跌倒漏报的说明。候选事件仍只验证了带 `test=true` 的测试消息，未启用通知。最终机位、真人站/坐/躺、多人计数、长期供电/温升尚未完成现场校准。
- 用户随后要求旋转 180°，已在 OV3660 传感器端设置水平/垂直同时翻转并烧录验证；`/health` 返回 `rotation: 180`，旋转前后标注截图为 `artifacts/hardware/before-rotation.jpg` 与 `artifacts/hardware/after-rotation.jpg`。文字和框在旋转后正常绘制。用户确认 USB 直连 Mac，未使用扩展坞/Hub。

同日新增的真实婴儿摔倒合集测试见 [摔倒视频测试](fall-video-evaluation.md)：现有配置在前 90 秒输出 0 次告警，暴露了明显漏报。以下为此前 Ring 日常活动视频的验证记录，不能用来证明跌倒功能已通过实测。

## 已完成的实测

在 `pn41` 的两颗 PCIe Coral 上测试，输入为用户提供的 `Baby's First Steps Are Captured Via Indoor Cam ｜ RingTV [R6IfcTpU2Qk].mkv`，1920×1080、24000/1001 fps、约 37.901 秒。文件通过 SSH 复制到服务器的独立项目目录，无云端推理。

最终选用 **MobileDet + MoveNet 的 Edge TPU 模型**，检测使用全图及 2×2 重叠裁剪，过滤分块截断目标，合并框，并以肩部/关键点证据辅助排除较弱的假人体框。输出尺寸 960×540。

| 检查 | 结果 |
|---|---|
| 完整离线回放 | 303 帧，按媒体时间抽帧到 8 fps |
| 标注视频时长 | 37.875 秒；H.264 预览无音频 |
| 最终规则/模型配置的处理耗时 | 平均 121.14 ms，p95 171.67 ms；包括标注/文件输出 |
| 本段候选跌倒事件 | 0；不能推导跌倒召回率 |
| 姿态观察次数 | standing 129、sitting 17、lying 0、unknown 362；非准确率 |
| 每帧人数分布 | 0 人 17 帧、1 人 90 帧、2 人 171 帧、3 人 24 帧、4 人 1 帧 |
| 自动检查 | 本机 19 项通过；真实 MQTT 测试另行执行通过 |
| 固件 | `esp32s3` 和启用相机代码的 `compile_check` 均编译成功 |
| 固件 RTP/JPEG | 实际固件 C++ 代码在主机测试，YUV420/422、分片偏移、序号回绕、量化表、末片标记、重建 JPEG 一致性通过 |
| 实际 RTSP 输出 | ffprobe 读到 H.264、960×540、6/1 fps |
| 实际 MQTT | 认证、Discovery、状态、事件、晚到订阅者不接收历史事件、离线状态通过 |
| CPU 备用路径 | 无 TPU 设备、TFLite2.14 下完成 2 帧；约 3.44 秒/帧，仅作功能回放用途 |
| 部署状态 | analytics / mediamtx / mqtt 均运行；analytics healthcheck healthy |

离线默认 8 fps 是抽帧/文件时间基准，并不保证持续计算始终达到 8 fps。实际部署选择 6 fps 编码输出，为检测和其他服务器服务留出余量；实时输入保留最新帧，编码可重复最近一次标注。CPU 负载和人数变化会影响实际标注更新率。

预览文件位于本地项目 `artifacts/annotated-h264.mp4`，抽帧图为 `artifacts/contact-sheet.jpg`，完整逐帧记录为 `artifacts/detections.jsonl`。原始 mp4v 标注文件另存 `artifacts/annotated.mp4`。这些测试输出和模型二进制被 Git 忽略。

## 视觉检查发现与处理

本片存在婴儿、小目标、广角畸变、沙发遮挡和大量画面边缘截断。最初 SSD MobileNet 全图检测漏检严重，未采用其较快的耗时作为最终性能数字。MobileDet 提升了小孩检测，但也把椅子/毯子作为人体；已加入低置信度目标的肩部与关键点验证、最小目标高度和分块边缘过滤。

同样模型只做全图检测时，平均耗时约 55.74 ms，但本片有 72 帧输出 0 人，站立观察仅 10 次；最终分块方案为 17 帧 0 人，站立观察 129 次，因此保留分块方案并降低实时编码目标帧率。

**仍有漏检、姿态未知和偶发重复/误检，未做逐帧人工真值标注，没有可报告的准确率。** 例如约 6 秒的画面仍可能漏掉被遮挡的小孩；前景成人经常只有上半身，无法可靠区分坐/站/躺。默认肩部验证会减少假框，也可能漏掉严重遮挡的真人。`verify_person_pose: false` 可关闭该过滤，用于机位对照；`min_person_height` 默认限制人体高度至少为画面的 6%，远处更小目标需要调低或扩大监控区域的成像比例。

本片没有足够的真实躺卧/跌倒样本。站/坐/躺几何分类、快速下降后持续躺卧、慢速躺下、已处于躺卧、冷却、跟踪中断、ROI 外不收集跌倒证据，均有合成关键点/时序测试。它们验证程序规则，不能替代实际倒地片段和现场验收。

## 运行库与部署

服务器原有 Coral 镜像中的 TFLite 2.5.0.post1 可以稳定运行。尝试 TFLite 2.14 后出现原生 segmentation fault，未采用该组合。最终单独构建 `indoorcam:pn41`，保留旧镜像及其他程序不变，并将 MoveNet 也移到第二颗 TPU。

pn41 的 Docker Hub 访问返回 EOF。Mosquitto 2.0.22、MediaMTX 1.12.3 通过本机下载 linux/amd64 镜像后 `docker save/load` 离线导入；analytics 基于服务器已有镜像加离线 wheels 构建。`deploy/Dockerfile.pn41` 是此已验证路径；标准 `deploy/Dockerfile` 为可联网 Linux x86_64 的构建配方，本轮没有执行该标准镜像的在线完整构建。

测试服务目录为 `~/esp32-indoorcam`，Compose 项目 `indoorcam`，仅添加本项目容器/网络/数据卷。现有 HA、Immich、Nextcloud 等未修改。私密连接信息保存在服务器 `deploy/runtime/ha-connection.txt`，未写入仓库或聊天。

```bash
cd ~/esp32-indoorcam
python3 scripts/check_stack.py
sudo docker compose --env-file deploy/.env -f deploy/compose.yaml -f deploy/compose.pn41.yaml ps
```

## 尚未完成的现场验收

- 2026-10-31 鹿小班 ESP32-S3 N16R8 + OV3660 已连接。用户接口图确认 ESP32-S3-EYE 引脚；固件通过 WCH UART 以 230400 baud 烧录成功，16 MB Flash、约 8 MB PSRAM 与 Sensor PID `0x3660` 均已实测确认，配网热点已启动。
- 实物画面、RTSP、重启恢复 Wi-Fi、Coral 到 HA 视频和 MQTT 已通过短时链路检查。持续码率/卡顿、温升、供电和真实机位仍需验证。
- HA 接入现已完成（2026-10-31）：在用户登录会话中添加了 MQTT 与通用摄像头，8 个 MQTT 实体正常接收数据，标注视频在 HA 中连续解码播放，并已新增“室内监控”仪表盘。已验证一条 `test=true` 的模拟候选事件进入 HA；未启用自动通知。实际机位与跌倒识别效果仍需现场验证。详见部署文档的 HA 验收记录。
- 用户已选择先监控整个画面，因此 `roi=[]`；镜头固定摆正后复测真人姿态。需要局部区域时再按真实截图填写多边形。
