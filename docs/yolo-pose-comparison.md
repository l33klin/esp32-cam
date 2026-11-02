# YOLO Pose × Coral 实测对比（2026-10-31）

**YOLO 已是可用备选：人数与速度优先，建议下一步实机验证 YOLOv8n-pose 480、人体置信度 0.10；姿态优先仍保留 MobileDet + Thunder。** 前者单颗 Coral 完整处理 18.47 fps，人数 MAE 原始 1.00、人工排除电视/镜面后 0.80，核心姿态正确 8/20；同轮 Thunder 为双 TPU、7.78 fps、1.15 / 1.05、10/20。仅差两个姿态目标，不能宣称泛化精度优势。

YOLO 提案再用 Thunder 精修没有收益：8.85 fps、姿态 7/20、人数 MAE 1.20。YOLO11/26 的 512 可以运行，但本轮没有比上述候选更好的综合成绩。**全部十组完整配置均未通过婴幼儿摔倒验证**，新模型未替换线上服务。

[交互 Review：同步视频与 43 帧原图对照](../artifacts/yolo-comparison/review/index.html) · [完整评分 JSON](../artifacts/yolo-comparison/scores/comparison.json) · [人数 CSV](../artifacts/yolo-comparison/scores/count-audit.csv) · [姿态 CSV](../artifacts/yolo-comparison/scores/pose-audit.csv)

## 测试范围与口径

本轮实际导出并测试 YOLOv8n-pose、YOLO11n-pose、YOLO26n-pose 的 320/480 输入，针对 480 的编译问题补测 YOLO11n/26n 的 512 输入；另测低置信度控制配置及“YOLOv8n Pose 提案框 → Thunder 精修”组合。**编译、兼容性、短测和全片测试分开报告**，没有把未跑完两个全片的配置混入完整处理排名。

推理硬件仍为 pn41 的 Intel N6005、双 PCIe Coral；纯 YOLO 使用 `pci:0` 一颗，MobileDet + MoveNet 和实验精修组合使用两颗。所有硬件测试使用原 `indoorcam:pn41` 镜像、Python 3.9、TFLite Runtime 2.5，未升级生产运行库。导出在 Mac 的独立 `.venv-yolo` 进行；部分编译在 Mac 的 Linux x86_64 Docker 中复核，**所有表中硬件推理速度均来自 pn41**。

输入仍是 Ring 全片约 37.901 秒，以及 AV1 WebM 转为 H.264 后的完整摔倒合集约 210.826 秒。完整回放每秒选 8 帧，分别 303/1687 帧，每个配置共 1990 帧；统一 960×540 等比输出、全画面 ROI、相同媒体时间和主要切镜重置点，预热三次后计时。MP4 按 8 fps 播放，**完整处理 fps 才是离线硬件吞吐**，包括解码、缩放、推理、规则、JSON、画框与 H.264 写出，不是 RTSP/HA 实时延迟。

沿用预设 43 个视觉复核帧，其中 30 帧人数明确；10 帧用于检测置信度校准，另外 20 帧用于本报告主表。姿态主表为保留复核的 20 个站/坐/躺目标（站13、坐3、躺4），另有 5 个其他动作目标单列。粗框采用 IoU≥0.2 一对一匹配，**不报告检测 mAP、关键点 PCK 或 COCO 精度**。漏检、unknown 均计为未完成所需分类；视频显示平滑后姿态，评分使用平滑前 `raw_posture`。

这些标签由助手查看原图初步标注，尚待用户复核；基线开发曾用过这两个视频，同片相邻帧也有关联，因此不是专业独立盲测或独立泛化测试。坐/躺样本很少，不能按一两个目标的差别宣称统计优势。人数 MAE 同时保留原始值与人工排除电视/镜面区域后的值；后者不表示算法已自动识别电视。

各网络共用现有几何姿态、跟踪与跌倒规则，但模型适配、NMS 和人体门控按各自配置执行。YOLO 默认人体置信度 0.25、NMS IoU 0.70、至少三个关节点≥0.30、最小人体高度为画面 6%；映射到原图之外的关节点置信度置零。MobileDet/MoveNet 基线沿用原先的分块检测与裁剪门控。**这是当前完整管线的业务表现比较，不是单独骨架网络的学术精度排名。**

## 完整双视频结果

| 配置 | TPU 数 | 完整 fps | 推理均值 / p95 ms | 人数 MAE 原始 / 人工排除后 | 人数完全一致 | 姿态正确 | 姿态目标匹配 |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n Pose 480 / 0.10 | 1 | 18.47 | 38.3 / 49.5 | 1.00 / 0.80 | 9/20 | 8/20 | 12/20 |
| MobileDet + Lightning | 2 | 9.96 | 80.0 / 129.7 | 1.30 / 1.20 | 3/20 | 9/20 | 15/20 |
| MobileDet + Thunder | 2 | 7.78 | 108.0 / 200.7 | 1.15 / 1.05 | 4/20 | 10/20 | 15/20 |
| YOLO11n Pose 320 | 1 | 27.95 | 20.8 / 27.7 | 1.60 / 1.60 | 1/20 | 6/20 | 9/20 |
| YOLO11n Pose 512 | 1 | 13.58 | 57.4 / 76.7 | 1.40 / 1.30 | 3/20 | 7/20 | 9/20 |
| YOLO26n Pose 320 | 1 | 29.89 | 18.9 / 21.9 | 1.65 / 1.70 | 2/20 | 5/20 | 7/20 |
| YOLO26n Pose 512 | 1 | 13.98 | 51.8 / 66.4 | 1.35 / 1.30 | 5/20 | 7/20 | 9/20 |
| YOLOv8n Pose 320 | 1 | 31.17 | 17.5 / 21.4 | 1.65 / 1.65 | 2/20 | 3/20 | 6/20 |
| YOLOv8n Pose 480 / 0.25 | 1 | 18.70 | 38.3 / 49.4 | 1.30 / 1.30 | 4/20 | 7/20 | 8/20 |
| YOLOv8n Pose 480 → Thunder | 2 | 8.85 | 94.0 / 211.0 | 1.20 / 1.20 | 4/20 | 7/20 | 12/20 |

人数误差可能因漏检与误检互相抵消而变小，不等于检测召回率提高；需要结合目标匹配与原图复核。

| 配置 | Ring fps | 合集 fps | 主进程 CPU ms/帧 | 主进程 peak RSS MB |
|---|---:|---:|---:|---:|
| YOLOv8n Pose 480 / 0.10 | 18.13 | 18.53 | 54.4 | 149.0 |
| MobileDet + Lightning | 8.94 | 10.17 | 86.0 | 159.9 |
| MobileDet + Thunder | 6.80 | 7.99 | 105.8 | 169.1 |
| YOLO11n Pose 320 | 26.96 | 28.14 | 39.7 | 150.6 |
| YOLO11n Pose 512 | 13.53 | 13.59 | 67.9 | 151.8 |
| YOLO26n Pose 320 | 26.36 | 30.63 | 33.1 | 151.3 |
| YOLO26n Pose 512 | 14.39 | 13.91 | 61.3 | 152.0 |
| YOLOv8n Pose 320 | 28.24 | 31.77 | 38.6 | 149.4 |
| YOLOv8n Pose 480 / 0.25 | 17.82 | 18.87 | 53.8 | 151.2 |
| YOLOv8n Pose 480 → Thunder | 8.53 | 8.92 | 100.3 | 167.8 |

主进程 CPU 与 RSS 不包含编码子进程或 Coral 私有内存。测试时其他服务器应用仍在运行，属于单次顺序回放；小幅速度差异不宜当作稳定统计优势。温度只有点检记录，没有连续温度或降频监控。

| 配置 | 站立正确 / 13 | 坐姿正确 / 3 | 躺卧正确 / 4 | 其他动作保留 unknown / 5 |
|---|---:|---:|---:|---:|
| YOLOv8n Pose 480 / 0.10 | 5 | 3 | 0 | 2 |
| MobileDet + Lightning | 6 | 3 | 0 | 3 |
| MobileDet + Thunder | 7 | 3 | 0 | 3 |
| YOLO11n Pose 320 | 5 | 1 | 0 | 0 |
| YOLO11n Pose 512 | 5 | 2 | 0 | 0 |
| YOLO26n Pose 320 | 4 | 1 | 0 | 0 |
| YOLO26n Pose 512 | 5 | 2 | 0 | 1 |
| YOLOv8n Pose 320 | 3 | 0 | 0 | 0 |
| YOLOv8n Pose 480 / 0.25 | 4 | 3 | 0 | 2 |
| YOLOv8n Pose 480 → Thunder | 4 | 3 | 0 | 3 |

## 编译、短测与输入尺寸限制

完整回放前，对每个能生成 Coral 模型的配置分别在两个视频上短测 16 帧，均先预热三次。两段短测都达到 2 fps 的方案才进入全片评估；这是预设的可用性筛选线，**短测 fps 不能替代全片吞吐**。

| 模型尺寸 | 最终编译路线 | TPU / CPU 算子 | Ring / 合集短测 fps | 是否全片回放 |
|---|---|---:|---:|---|
| YOLOv8n Pose 320 | 显式九头边界 | 247 / 45 | 26.75 / 35.01 | 是 |
| YOLOv8n Pose 480 / 0.25 | 显式九头边界 | 247 / 45 | 17.73 / 18.51 | 是 |
| YOLO11n Pose 320 | 显式九头边界 | 468 / 45 | 28.38 / 27.23 | 是 |
| YOLO11n Pose 480 | 默认分段回退 | 44 / 469 | 0.33 / 0.33 | 否，低于2 fps |
| YOLO11n Pose 512 | 显式九头边界 | 468 / 45 | 12.90 / 12.56 | 是 |
| YOLO26n Pose 320 | 显式九头边界 | 653 / 38 | 26.60 / 27.70 | 是 |
| YOLO26n Pose 480 | 两条路线均失败 | — | — | 否，未生成Coral版本 |
| YOLO26n Pose 512 | 显式九头边界 | 653 / 38 | 11.88 / 13.52 | 是 |

YOLO11n/26n 的 480 配置在显式头部边界编译时失败。在 pn41 的 768 MB 编译容器及 Mac x86_64 的 3 GB 容器中均复核过；日志仅给出 `Compilation failed`，未定位具体原因，不能据此断言是模型精度差或物理内存不足。512 的结果单列，不把它冒充 480 成绩，也不将此观察推广为其他版本工具链的永久限制。

YOLO26 使用原始 one-to-many 分支及 CPU NMS，没有使用官方默认 NMS-free 路径的速度数字。

## 为什么默认导出很慢，而本轮优化后较快

同一个 YOLOv8n-320 INT8 输入文件，默认编译只有 108 个算子上 TPU、184 个留在 CPU，Ring 16 帧短测约 0.88 fps。显式指定三层尺度的九个学习头输出作为编译边界后，247 个算子上 TPU、45 个留在 CPU，72 个卷积算子进入 TPU；相同 Ring 短测约 26.75 fps。最终完整视频速度见主表。

模型的 INT8 权重未因这项编译优化而改变。余下布局变换、DFL/坐标解码等仍由 CPU 执行，Python 再做 NMS、姿态规则和跟踪；**TPU 算子个数比例不是计算量或耗时比例**。也尝试过实验性的多子图开关，但此模型未编译成功，最终没有采用。

官方明确支持 YOLOv8/11/26 的 Pose Edge TPU 导出，同时说明不支持的运算会留给 CPU。因此“能导出 `.tflite`”与“在现有 Coral 上有足够吞吐”必须分别验证。[Ultralytics Edge TPU 文档](https://docs.ultralytics.com/integrations/edge-tpu/)

## 量化前后与真实 Coral 的同图复核

导出环境为 Ultralytics 8.4.194、Torch 2.15.0、TensorFlow 2.19、onnx2tf 1.28.8。INT8 激活范围校准使用独立公开 COCO128 的 128 张图片，**没有使用两个评估视频的帧作量化校准**。这与后面的“检测置信度校准子集”是两件事。权重、量化模型、编译模型和 COCO 压缩包均保留哈希；未训练或微调网络。

先把 TFLite FP32 的原始输出与同权重的原生 PyTorch 输出核对，再在同一组 43 张 JPEG 上对比 Mac FP32、Mac 未编译 INT8 和 pn41 编译后 INT8。浮点核对覆盖每种尺寸的三张图，具体误差见 [conversion-validation.json](../artifacts/yolo-comparison/conversion-validation.json)。Mac 只提供质量参考，没有把其 CPU 速度拿来与 pn41 排名。

| 模型尺寸 | FP32 核心姿态正确 | 未编译 INT8 CPU 正确 | 编译后 Coral INT8 正确 | 人数 MAE：FP32 / INT8 CPU / Coral（人工排除后） |
|---|---:|---:|---:|---:|
| YOLOv8n Pose 320 | 4/20 | 3/20 | 4/20 | 1.45 / 1.45 / 1.50 |
| YOLOv8n Pose 480 / 0.25 | 8/20 | 6/20 | 7/20 | 1.10 / 1.20 / 1.35 |
| YOLO11n Pose 320 | 5/20 | 5/20 | 5/20 | 1.55 / 1.65 / 1.70 |
| YOLO11n Pose 480 | 8/20 | 6/20 | 7/20 | 1.40 / 1.30 / 1.45 |
| YOLO11n Pose 512 | 8/20 | 8/20 | 7/20 | 1.00 / 1.25 / 1.30 |
| YOLO26n Pose 320 | 4/20 | 3/20 | 3/20 | 1.60 / 1.95 / 1.95 |
| YOLO26n Pose 480 | 7/20 | 6/20 | — | 1.40 / 1.25 / — |
| YOLO26n Pose 512 | 6/20 | 7/20 | 7/20 | 1.40 / 1.20 / 1.20 |

此表与主表分开解读：主表处理完整视频解码帧，这里三方均使用完全相同的保存 JPEG，专门观察转换、量化和硬件运算差异。JPEG 编码及抽帧路径的像素差异，在小目标和阈值附近可能改变结果；不能跨这两张表把分数差归因于硬件，也不能仅凭这些数字确定泛化量化损失。

## 校准阈值与实验精修管线

`yolov8n_480_cal010` 的人体置信度 0.10 依据预设校准子集的**人数 MAE**选择，误差并列时取较高阈值；没有依据保留复核部分选它。它仍使用 YOLO 自身的关节点，提供与后续精修组合相同提案阈值的控制配置。

`yolov8n_480_thunder` 使用同一个 YOLOv8n Pose 模型的人体框，再在第二颗 Coral 上运行 Thunder；YOLO 自身的关节点不用于最终姿态。它沿用 Thunder 的人体验证门控，没有要求提案先通过 YOLO 的三个关节点门控。阈值取自基础 YOLO 的上述校准诊断，未针对精修组合独立优化。**这是实验组合，尚不是检测专用 YOLO 权重的性能测试**。

几何规则中的关节点阈值另有离线重评分诊断，但没有把它当作新训练模型或完整时序性能结果。阈值和门控影响业务数字，不应把本轮结果解读为通用 COCO 骨架精度结论。

## 躺卧与摔倒结果

十组配置完成 20 次全片回放，共 19,900 次采样帧处理；这是对同样 1990 个媒体时间点的重复比较，不是 19,900 个独立真值样本。所有候选事件为零，保留复核的四个初步躺卧目标也均未正确分类。少量全片 lying 输出不能当作正确识别或告警验收：

| 配置 | 前90秒原始 / 稳定躺卧帧 | 合集全片原始 / 稳定躺卧帧 | Ring / 合集候选事件 |
|---|---:|---:|---:|
| YOLOv8n Pose 480 / 0.10 | 6 / 0 | 7 / 0 | 0 / 0 |
| MobileDet + Lightning | 0 / 0 | 2 / 0 | 0 / 0 |
| MobileDet + Thunder | 0 / 0 | 11 / 0 | 0 / 0 |
| YOLO11n Pose 320 | 2 / 0 | 4 / 0 | 0 / 0 |
| YOLO11n Pose 512 | 1 / 0 | 2 / 0 | 0 / 0 |
| YOLO26n Pose 320 | 0 / 0 | 3 / 0 | 0 / 0 |
| YOLO26n Pose 512 | 0 / 0 | 2 / 0 | 0 / 0 |
| YOLOv8n Pose 320 | 0 / 0 | 1 / 0 | 0 / 0 |
| YOLOv8n Pose 480 / 0.25 | 1 / 0 | 2 / 0 | 0 / 0 |
| YOLOv8n Pose 480 → Thunder | 0 / 0 | 0 / 0 | 0 / 0 |

YOLO Pose 输出人体框与 17 个关节点，不直接输出站/坐/躺或摔倒类别。[官方输出说明](https://docs.ultralytics.com/tasks/pose/)

当前规则需要较可信的肩、髋、膝、踝，且躺姿要求身体横向、腿部接近伸展；屈腿侧躺、遮挡、爬行前扑、跌坐和倒地后迅速被抱起都可能不满足条件。原始视频中 3–6.5 秒可见明确倒地过程，其余逐段观察见 [连拍说明](../artifacts/yolo-comparison/fall-event-review/notes.json)。其中包含遮挡或动作歧义，未把七个观察窗视为七个确认跌倒真值，不据此计算正式召回率。

本轮两段公开视频与少量视觉复核不足以验收婴幼儿摔倒预警。下一步应针对实际机位补充独立姿态/时序标注，区分“失衡”“跌坐”“持续躺地”等事件，并验证日常动作误报；更换骨架网络本身不等于完成这层分类。

## 代码、复现与部署状态

- [导出器](../scripts/export_yolo_pose.py)、[批量导出](../scripts/run_yolo_exports.py)、[浮点转换核对](../scripts/validate_yolo_export.py)。
- [头部边界查找](../scripts/find_yolo_head_cuts.py)、[资源受限编译](../scripts/compile_yolo_models_pn41.py)、[YOLO 适配器](../scripts/yolo_pose_adapter.py)。
- [兼容性短测](../scripts/run_yolo_smoke_pn41.py)、[第一阶段编译/回放](../scripts/run_yolo_benchmark_pn41.py)、[补充测试](../scripts/run_yolo_followup_pn41.py)、[同图质量核对](../scripts/probe_yolo_pose.py)。
- [共用视频比较器](../scripts/compare_pose_models.py)、[评分器](../scripts/score_pose_comparison.py)、[Review 生成器](../scripts/build_pose_review.py)。

首轮同图脚本因远端评分模块未同步而失败，已补齐依赖并重跑；同图表采用补测阶段全部成功的 JSON，旧失败日志保留但未参与计分。

本地模型、视频、日志、manifest、包版本和哈希保存在 `artifacts/yolo-comparison/`；远端对应 `~/esp32-indoorcam/evaluations/yolo-comparison/`。大文件被 Git 忽略，复制交互报告需同时保留其图片、视频和数据子目录。模型权重来自 Ultralytics 官方发布，MoveNet 基线来自 Coral 官方仓库。

已准备好本次输入与模型的 pn41 项目根目录可以通过下列命令重新回放；它会暂时停止本项目 analytics，以免争用 Coral：

```bash
python3 -m scripts.run_pose_comparison_pn41 \
  --models lightning thunder yolov8n_320 yolov8n_480 yolo11n_320 yolo26n_320 \
  --model-root evaluations/yolo-comparison \
  --output evaluations/yolo-comparison/new-results

python3 -m scripts.run_pose_comparison_pn41 \
  --models yolov8n_480_cal010 yolov8n_480_thunder yolo11n_512 yolo26n_512 \
  --model-root evaluations/yolo-comparison \
  --output evaluations/yolo-comparison/new-followup-results
```

若重新导出，使用 [requirements-export.txt](../artifacts/yolo-comparison/requirements-export.txt) 中的隔离环境版本；仅有普通默认导出模型时，不应直接期待达到本轮优化后的速度。

最终校验：20 次全片运行均正常退出；20 段 H.264/yuv420p 标注 MP4 均完成全片解码和帧数校验。8 个模型尺寸的 24 次浮点输出核对通过，最大归一化差约 0.0000313。代码测试 **28 passed、1 skipped**（MQTT 联网测试默认跳过）。Review 页在浏览器验证了跳转、同步播放/暂停和更换模型后保持时间，见 [validation.json](../artifacts/yolo-comparison/validation.json)。

HTML 可直接用浏览器打开，也可用本地字节范围预览服务：

```bash
.venv311/bin/python -m scripts.serve_pose_review --root artifacts/yolo-comparison --port 18591
# http://127.0.0.1:18591/review/index.html
```

本轮未向 HA 发布测试人数或事件，未替换生产模型或修改 ESP32 固件；原 analytics 服务、镜像和 RTSP 输入已恢复。ESP32 仍按用户说明处于拔出状态。本轮文件测试不能代替实际机位、JPEG、Wi-Fi 和 HA 播放延迟验证。
