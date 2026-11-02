# Coral 姿态算法实测对比（2026-10-31）

本轮建议把 **MobileDet + MoveNet Thunder** 作为下一步实景验证的姿态管线，保留 **MobileDet + MoveNet Lightning** 作为速度优先方案。Thunder 在本次小样本复核中只比 Lightning 多正确一个姿态目标，不能据此宣称统计意义上的优势。**没有一个方案通过婴幼儿摔倒验证**，现阶段不能把替换关节点模型等同于修复摔倒预警。

所有推理在 pn41 的真实 PCIe Coral 上完成，使用用户提供的两个本地视频；未使用已拔出的 ESP32，未上传视频到云服务，未向 HA 发布测试人数或事件，也未替换线上模型。

## Review 入口

- [本地交互报告](../artifacts/pose-comparison/review/index.html)：完整比较表、两路同步视频、43 个固定帧的原图与各模型结果、逐段倒地连拍。
- [人数逐帧 CSV](../artifacts/pose-comparison/scores/count-audit.csv)、[姿态逐目标 CSV](../artifacts/pose-comparison/scores/pose-audit.csv)、[完整摘要 JSON](../artifacts/pose-comparison/scores/comparison.json)。
- [初步视觉标签](../config/pose-review-labels.json)、[详细方法](../artifacts/pose-comparison/methodology.md)、[模型来源与哈希](../artifacts/pose-comparison/model-provenance.json)、[视频来源与哈希](../artifacts/pose-comparison/video-provenance.json)。

## 网上调研与实际测试范围

| 算法/组成 | Coral 路径 | 本轮处理 |
|---|---|---|
| MoveNet SinglePose Lightning，192×192 | Google Coral 官方量化编译模型；逐人裁剪 | 与 MobileDet 组合，基线 |
| MoveNet SinglePose Thunder，256×256 | 同上，较大关节点网络 | 与 MobileDet、EfficientDet、PoseNet 提案分别组合 |
| PoseNet MobileNet V1 0.75 | 官方 Edge TPU 模型 + x86_64 CPU 多人解码 delegate | 353×481、481×641、721×1281 三种输入 |
| PoseNet ResNet50 | 同上 | 416×288、640×480 两种输入 |
| EfficientDet-Lite0 / Lite1 | 官方 Edge TPU 人体检测，再用 Thunder 估计关节点 | 320×320 / 384×384；Lite1 另测仅全帧检测 |
| PoseNet721 + Thunder | 本项目实验组合：多人骨架导出粗框，再逐人精修 | 真实双 TPU 测试，不是上游发布的独立算法 |
| YOLO Pose 的 Edge TPU 导出 | 官方导出支持 pose；能映射的算子上 TPU，其余部分可能在 CPU | 只调研，未完成导出、编译与此运行库验证，不给出实测排名 |

主要依据是 [Coral 官方模型仓库](https://github.com/google-coral/test_data)、[Coral PoseNet 项目](https://github.com/google-coral/project-posenet)、[TensorFlow 的 MoveNet 与姿态分类说明](https://blog.tensorflow.org/2021/08/pose-estimation-and-classification-on-edge-devices-with-MoveNet-and-TensorFlow-Lite.html) 和 [Ultralytics Edge TPU 导出文档](https://docs.ultralytics.com/integrations/edge-tpu/)。PoseNet 的网络在 TPU 上运行，多人解码在 CPU；MoveNet 输出关节点，站/坐/躺仍由本项目规则判断。上游通用基准的准确率不能替代本次婴幼儿场景的实际效果。

EfficientDet-Lite2 权重仅下载备用，未进行完整视频测试；更大的 PoseNet ResNet 权重也未全部测试。没有把“已下载”算成“已实测”。PoseNet 使用的上游 commit 为 `37cfb84eaff9fa7a02bbc8b60a36d6d422859c5f`。

## 环境、输入与评分口径

pn41 为 Intel Pentium Silver N6005、4 核、两个 PCIe Coral（`/dev/apex_0`、`/dev/apex_1`）。采用现有兼容镜像 `indoorcam:pn41`，Python 3.9.25、TFLite Runtime 2.5.0.post1、OpenCV 4.11；没有为了新模型升级线上运行库。单 PoseNet 使用一颗 TPU；检测器 + MoveNet 使用两颗，按检测、裁剪推理的顺序执行，未实现两阶段流水并行。

| 输入 | 原始格式 | 本轮处理范围 | 每个配置采样数 |
|---|---|---|---:|
| Baby's First Steps…RingTV | H.264 MKV，1920×1080，约 37.901 秒 | 全片 | 303 |
| All the babies getting hurt | AV1 WebM，1280×716，约 210.841 秒 | 转成 H.264 MP4 后全片约 210.826 秒；前 90 秒重点核对跌倒 | 1687 |

统一每秒选 8 帧、960×540 等比补边、全画面 ROI、同一套姿态/跟踪/跌倒规则；每段先预热三次。主表每个配置共 1990 帧。摔倒合集在已确认的主要切镜处清空轨迹，避免跨片段拼接；后半片段分镜标注并不完整，不据此计算正式全片事件召回率。

MobileDet 与默认 EfficientDet 配置每帧执行一次全图检测及四次重叠裁剪检测，帮助远处小目标；`full` 与 `letterbox` 配置只做一次全图。单 PoseNet 一次处理多人，MoveNet 则逐人运行，人数会影响耗时。漏检也会减少后续逐人推理次数，因此高 fps 可能伴随更差的检测覆盖，不能脱离质量单独排名。

**完整处理 fps** 包含读取/解码、缩放、推理、规则、JSON、画框和 H.264 输出，是离线实测吞吐。**推理 ms** 包含输入处理、模型调用和输出处理；delegate invoke 时间包含 CPU 后处理，不能称为纯 TPU 内核耗时。输出 MP4 本身以 8 fps 播放，不能把播放帧率当作硬件吞吐。主进程 CPU 和 RSS 不包含编码子进程；服务器还运行其他应用，结果为单次顺序运行，不是隔离实验室的多轮统计基准。

视觉复核预设 43 个时间点：30 帧人数明确，13 帧因严重遮挡或边缘人体等原因不强行计入人数 MAE。姿态粗标共 36 个目标：站立 22、坐姿 3、躺卧 5、其他动作 6。预设每三帧取一帧做校准，主表采用其余 **20 个明确人数帧、20 个站/坐/躺目标（站13、坐3、躺4）**。人工粗框仅用于 IoU≥0.2 的一对一目标关联，不是完整检测框真值，不报告检测 mAP。

标签由助手查看原图后初步标注，尚待用户 review；不是专业独立盲标。基线开发曾使用这两个视频，相邻帧也有关联，因此“保留复核”不等于独立泛化测试集。各类别数量很少，单个目标即可改变排名。

部分语义上能辨认姿态的目标仍有关键关节被遮挡；unknown 是允许的保留输出。在此评分里它表示未完成所需的站/坐/躺分类，并不据此断言每个 unknown 都是错误的骨架估计。

人数同时报告原始误差与人工排除电视/镜面区域后的误差；后者不表示模型自动识别了电视或镜子。姿态评分使用平滑前的 `raw_posture`，漏检、unknown 均视为未成功分类；视频框上显示按 0.5 秒保持规则平滑后的姿态。默认检测阈值 0.45、关键点阈值 0.30、PoseNet pose score 0.25；实验组合提案阈值 0.20 只依据校准子集选择。不同模型分数未做概率校准，表格比较的是这些实际配置，不是各架构调参后的能力上限。

## 完整视频对比

| 配置 | 完整 fps | 推理均值 / p95 ms | 人数 MAE 原始 / 排除后 | 人数完全一致 | 姿态正确 | 姿态目标匹配 | TPU 数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MobileDet + Thunder | 7.26 | 115.3 / 211.5 | 1.15 / 1.05 | 4/20 | 10/20 | 15/20 | 2 |
| MobileDet + Lightning | 9.19 | 86.4 / 140.9 | 1.30 / 1.20 | 3/20 | 9/20 | 15/20 | 2 |
| PoseNet MobileNet 721×1281 | 14.43 | 41.9 / 46.8 | 1.15 / 0.95 | 6/20 | 3/20 | 12/20 | 1 |
| PoseNet MobileNet 481×641 | 37.89 | 11.5 / 12.2 | 1.10 / 1.05 | 3/20 | 5/20 | 12/20 | 1 |
| PoseNet MobileNet 353×481 | 49.45 | 6.5 / 6.9 | 1.30 / 1.30 | 1/20 | 3/20 | 9/20 | 1 |
| PoseNet721 → Thunder | 7.51 | 108.9 / 211.9 | 1.20 / 1.10 | 5/20 | 7/20 | 12/20 | 2 |
| PoseNet ResNet50 416×288 | 11.82 | 53.1 / 55.4 | 1.70 / 1.70 | 1/20 | 3/20 | 8/20 | 1 |
| PoseNet ResNet50 640×480 | 2.92 | 322.3 / 329.3 | 1.15 / 1.15 | 3/20 | 9/20 | 12/20 | 1 |
| EfficientDet-Lite1 补边/0.20 + Thunder | 7.94 | 106.4 / 197.3 | 1.60 / 1.50 | 2/20 | 7/20 | 11/20 | 2 |
| EfficientDet-Lite0 分块 + Thunder | 3.44 | 271.7 / 356.7 | 2.10 / 2.10 | 0/20 | 1/20 | 3/20 | 2 |
| EfficientDet-Lite1 全帧 + Thunder | 11.20 | 73.0 / 118.9 | 2.20 / 2.20 | 0/20 | 2/20 | 4/20 | 2 |
| EfficientDet-Lite1 分块 + Thunder | 2.42 | 394.8 / 498.7 | 1.95 / 1.95 | 0/20 | 4/20 | 6/20 | 2 |

“匹配”表示该姿态目标找到可关联的检测，不代表分类正确；“正确/匹配”能帮助区分漏框和后续分类问题，但不能取代把漏检计入的完整成绩。人数 MAE 的单位为人/帧，越低越好。

| 配置 | 站立正确 / 13 | 坐姿正确 / 3 | 躺卧正确 / 4 | 其他动作保留为 unknown / 5 | 已匹配核心目标中正确 |
|---|---:|---:|---:|---:|---:|
| MobileDet + Thunder | 7 | 3 | 0 | 3 | 10/15 |
| MobileDet + Lightning | 6 | 3 | 0 | 3 | 9/15 |
| PoseNet MobileNet 721×1281 | 2 | 1 | 0 | 2 | 3/12 |
| PoseNet MobileNet 481×641 | 4 | 1 | 0 | 2 | 5/12 |
| PoseNet MobileNet 353×481 | 3 | 0 | 0 | 1 | 3/9 |
| PoseNet721 → Thunder | 4 | 3 | 0 | 1 | 7/12 |
| PoseNet ResNet50 416×288 | 2 | 1 | 0 | 1 | 3/8 |
| PoseNet ResNet50 640×480 | 6 | 3 | 0 | 2 | 9/12 |
| EfficientDet-Lite1 补边/0.20 + Thunder | 4 | 3 | 0 | 1 | 7/11 |
| EfficientDet-Lite0 分块 + Thunder | 1 | 0 | 0 | 0 | 1/3 |
| EfficientDet-Lite1 全帧 + Thunder | 1 | 1 | 0 | 0 | 2/4 |
| EfficientDet-Lite1 分块 + Thunder | 1 | 3 | 0 | 1 | 4/6 |

速度与资源的逐视频数据：

| 配置 | Ring fps | Ring 推理均值 / p95 ms | 合集 fps | 合集推理均值 / p95 ms | 主进程 CPU ms/帧 | 主进程 peak RSS MB |
|---|---:|---:|---:|---:|---:|---:|
| MobileDet + Thunder | 6.26 | 131.9 / 191.5 | 7.47 | 112.3 / 214.4 | 117.4 | 168.2 |
| MobileDet + Lightning | 8.43 | 91.5 / 123.8 | 9.34 | 85.5 / 144.5 | 97.0 | 157.6 |
| PoseNet MobileNet 721×1281 | 14.59 | 38.5 / 41.3 | 14.40 | 42.5 / 47.1 | 62.1 | 149.7 |
| PoseNet MobileNet 481×641 | 32.75 | 11.5 / 12.0 | 38.99 | 11.5 / 12.2 | 29.0 | 145.2 |
| PoseNet MobileNet 353×481 | 39.40 | 6.5 / 6.9 | 51.83 | 6.5 / 6.9 | 26.1 | 141.7 |
| PoseNet721 → Thunder | 6.56 | 125.0 / 193.4 | 7.71 | 106.0 / 217.5 | 109.1 | 165.9 |
| PoseNet ResNet50 416×288 | 11.27 | 53.0 / 55.7 | 11.93 | 53.1 / 55.4 | 58.4 | 186.3 |
| PoseNet ResNet50 640×480 | 2.88 | 322.4 / 330.0 | 2.93 | 322.3 / 329.1 | 37.8 | 198.4 |
| EfficientDet-Lite1 补边/0.20 + Thunder | 9.25 | 84.8 / 123.9 | 7.74 | 110.3 / 204.2 | 109.9 | 192.1 |
| EfficientDet-Lite0 分块 + Thunder | 3.35 | 274.1 / 362.9 | 3.46 | 271.2 / 355.6 | 248.4 | 187.9 |
| EfficientDet-Lite1 全帧 + Thunder | 11.02 | 70.2 / 109.6 | 11.23 | 73.5 / 120.5 | 81.7 | 194.9 |
| EfficientDet-Lite1 分块 + Thunder | 2.52 | 374.5 / 480.6 | 2.41 | 398.5 / 501.4 | 333.8 | 195.0 |

主进程 CPU ms/帧适合比较算法开销，不能直接当作在线 CPU 百分比。TPU 温度只有点检记录，没有连续温度/降频监测，不能宣称观测到的值是全程最高温度。

## 推荐如何选择

1. **继续验证站/坐/躺：MobileDet + Thunder。** 本轮姿态正确数最高，但只比 Lightning 多一个；人数仍会漏检，尤其是沙发上的儿童、远处小孩和倒地目标。建议后续实景以 4–5 次新推理/秒起步，给多人裁剪、编码和其他服务留余量；这只是部署建议，未更改现有配置。
2. **优先速度与余量：MobileDet + Lightning。** 精度接近 Thunder，完整处理更快；人数/姿态需求不高或服务器有其他负载时更合适。
3. **只关心较快多人骨架或人数探索：PoseNet481/721。** 一颗 TPU 即可运行；721 在本次人数误差上较好，481 更快。但当前几何规则下站/坐/躺分类较差，不能用高 fps 代替姿态质量。
4. **本轮不优先采用 ResNet50、EfficientDet 已测配置和 PoseNet→Thunder 组合。** 表中这些配置没有提供足以抵消成本的综合收益；此结论限定于本轮输入、缩放和阈值，不是对架构本身作永久否定。

## 640 像素宽输入补测

| 配置 | 细节宽度 | 完整 fps | 人数 MAE 原始/排除后 | 核心姿态正确 | 站 / 坐 / 躺正确 |
|---|---:|---:|---:|---:|---|
| lightning | 原始→960 | 9.19 | 1.30 / 1.20 | 9/20 | 6 / 3 / 0 |
| lightning | 640→960 | 9.17 | 1.15 / 1.05 | 5/20 | 2 / 3 / 0 |
| thunder | 原始→960 | 7.26 | 1.15 / 1.05 | 10/20 | 7 / 3 / 0 |
| thunder | 640→960 | 7.21 | 1.10 / 1.00 | 7/20 | 4 / 3 / 0 |

完整数据见 [640 宽补测评分](../artifacts/pose-comparison/vga-scores/comparison.json)。这次人数误差没有随细节减少而一致变差，但站姿判断减少，说明人数指标不能代替姿态质量。

补测先把源图像宽度降到 640，再进入同样的 960×540 输出与推理流程；Ring 内容相当于 640×360，不是额外裁切。这用于观察细节减少的影响，**没有模拟 OV3660 的真实光学、JPEG 质量、Wi-Fi 丢帧、采集时延和机位**，也不是 ESP32 实机性能测试。主表完整输入测试与此补测分开保存、评分。

## EfficientDet 与姿态规则诊断

对 EfficientDet 的真实输出做了额外检查，位置顺序 `[boxes, classes, scores, count]` 与官方命名 `output_3 / output_2 / output_1 / output_0` 完全一致，分数在 0–1、类别为整数，COCO person 为 0；没有发现把类别当分数或类别偏移的错误。输入为 uint8 RGB，Lite0/1 量化参数均为 scale=0.0078125、zero=127。[官方输出适配器](https://github.com/google-coral/pycoral/blob/master/pycoral/adapters/detect.py)和[补边方法](https://github.com/google-coral/pycoral/blob/master/pycoral/adapters/common.py)作为核对依据。原始记录见[检测器诊断 JSON](../artifacts/pose-comparison/detector-sanity.json)。

默认检测器测试沿用基线的逐裁剪拉伸方式。另对 43 帧试了全帧等比补边与 0.15/0.20/0.30/0.45 检测阈值：Lite1 在校准 10 帧上的检测器单独人数 MAE 从拉伸、0.45 的 2.1 降到补边、0.20 的 1.2（此诊断未做姿态验证、排除区域或完整后处理）。0.20 是校准误差并列最低时较高的阈值，因此追加 `efficientdet1_letterbox_thunder` 完整双视频测试；其主表成绩才代表完整管线，不能拿这个单独检测诊断成绩代替。

人工粗框直接裁剪 MoveNet 的结果如下，使用相同姿态规则和关键点阈值 0.30：

| 模型 | 保留复核核心姿态正确 | 全部核心目标正确 | 全部躺卧目标正确 |
|---|---:|---:|---:|
| Lightning | 5/20 | 9/30 | 0/5 |
| Thunder | 7/20 | 11/30 | 0/5 |

这是受控裁剪诊断，见[逐目标 JSON](../artifacts/pose-comparison/oracle.json)与[五个躺卧目标对照图](../artifacts/pose-comparison/overlay-qa/oracle-lying-contact.jpg)。人工粗框有的较紧，会改变上下文，结果不能当作理想检测器的准确率上限，也不能直接以分数下降推断检测器有益。它支持的有限结论是：即使明确提供这些目标的位置，当前 MoveNet + 几何规则仍未成功处理这些躺卧姿态，问题不限于漏框。

另在 PoseNet 的保存候选上按校准人数误差选择 pose score：721 配置选 0.20 后，保留复核人数 MAE 为 0.75，核心姿态仍仅 3/20；481 选 0.20 后分别为 1.05、5/20。这只是阈值重评分，不是重新测过完整时序/吞吐的独立配置，未把它混进主表或声称解决姿态分类。

关键点阈值另做了 0.15–0.50 的离线重评分，保持检测目标固定，只重新运行姿态规则，未伪装成新的完整推理/性能结果。Thunder 的低阈值没有改善本轮保留复核核心目标的正确数；降低阈值也没有恢复复核躺卧目标。因此不能仅靠调低关键点阈值宣称解决跌倒。

## 为什么本轮仍没有可验收的摔倒方案

12 组配置各自完成两个全片，共 23,880 个采样帧；加上两组 640 宽补测，共 28 次完整视频运行、27,860 个采样帧。**12 组标准配置在两个视频中均为零候选跌倒事件**。前 90 秒没有稳定躺卧输出；PoseNet721 偶发一帧 raw lying，未形成稳定姿态或告警。完整合集后半段有少量 raw lying，ResNet640 也曾短暂输出稳定 lying，但仍没有候选事件；不能笼统说全片从未输出过躺卧。以下仅统计输出出现次数，不表示这些躺卧预测已被验证为正确：

| 配置 | 前90秒 raw lying 帧 | 前90秒稳定 lying 帧 | 全片 raw / 稳定 lying 帧 | 全片候选事件 |
|---|---:|---:|---:|---:|
| MobileDet + Thunder | 0 | 0 | 11 / 0 | 0 |
| MobileDet + Lightning | 0 | 0 | 2 / 0 | 0 |
| PoseNet MobileNet 721×1281 | 1 | 0 | 1 / 0 | 0 |
| PoseNet MobileNet 481×641 | 0 | 0 | 0 / 0 | 0 |
| PoseNet MobileNet 353×481 | 0 | 0 | 1 / 0 | 0 |
| PoseNet721 → Thunder | 0 | 0 | 0 / 0 | 0 |
| PoseNet ResNet50 416×288 | 0 | 0 | 2 / 0 | 0 |
| PoseNet ResNet50 640×480 | 0 | 0 | 11 / 7 | 0 |
| EfficientDet-Lite1 补边/0.20 + Thunder | 0 | 0 | 2 / 0 | 0 |
| EfficientDet-Lite0 分块 + Thunder | 0 | 0 | 0 / 0 | 0 |
| EfficientDet-Lite1 全帧 + Thunder | 0 | 0 | 0 / 0 | 0 |
| EfficientDet-Lite1 分块 + Thunder | 0 | 0 | 2 / 0 | 0 |

前 90 秒逐段图见[倒地连拍目录](../artifacts/pose-comparison/fall-event-review/notes.json)，重点包括：

| 时间窗 | 原图可见动作 | 对告警规则的含义 |
|---|---|---|
| 3–6.5 秒 | 有清楚的站立到倒地过程 | 关键漏报案例，不能用零事件表示正常 |
| 19–22.5 秒 | 低姿态前扑/塌落，很快被成人遮挡 | 不一定先有站/坐，且难保持可见轨迹 |
| 29–32.5 秒 | 椅旁失衡、跌坐 | 不一定以躺卧结束 |
| 40–43.5 秒 | 椅子倾倒，成人遮挡儿童 | 落地姿态证据不足 |
| 58–61.5 秒 | 快速失衡，桌沿遮挡部分身体 | 小目标和遮挡损害框与关节点 |
| 68–71.5 秒 | 切镜进入时已经躺着 | 没有看到发生过程，不能直接算起始跌倒漏报 |
| 80–83.5 秒 | 降低身体、桌子遮挡 | 可能主动蹲下，存在歧义 |

这七个观察窗不是七个已确认跌倒真值；没有报告貌似精确的“0/7 召回率”。现有规则要求近期站/坐、髋部下降、转躺并连续维持 1.5 秒，难覆盖跌坐、爬行前扑和倒地后立即被抱起。检测漏框、关节点不确定和时序定义都有贡献，不能把问题全部归到某一个网络。

下一步应该先补充实际机位下的站、坐、躺、爬、蹲、弯腰、失衡、被抱起等标注，增加独立的姿态/短时动作分类层；将“疑似失衡”和“持续躺地”分开作为可复核事件，验证漏报及日常动作误报。仅凭本次两个视频不训练并宣称可靠的婴儿摔倒模型。HA 接入链路与算法检测质量需要分别验收。

## 复现与保留文件

- [离线比较器](../scripts/compare_pose_models.py)：所有候选适配与逐帧输出；不会发布 MQTT、RTSP 或 HTTP。
- [顺序批测 runner](../scripts/run_pose_comparison_pn41.py)：独占 Coral，结束恢复原 analytics。
- [检测器核对](../scripts/probe_detector_sanity.py)、[人工裁剪诊断](../scripts/probe_pose_oracle.py)、[诊断/VGA runner](../scripts/run_pose_diagnostics_pn41.py)。
- [评分脚本](../scripts/score_pose_comparison.py)、[交互报告生成器](../scripts/build_pose_review.py)。
- 原始批次 manifest、主表结果、额外配置、VGA 结果、诊断 JSON、H.264 标注视频均保存在 `artifacts/pose-comparison/`；服务器对应 `~/esp32-indoorcam/evaluations/pose-comparison/`。大型本地视频和模型权重按项目规则忽略 Git，复制报告时需连同该目录，不能只复制 HTML。

pn41 已有本次模型和输入文件时，可在该项目根目录复现（会暂时停止本项目 analytics，避免 TPU 冲突；应在维护窗口运行）：

```bash
python3 -m scripts.run_pose_comparison_pn41 \
  --models lightning thunder posenet353 posenet481 posenet721 resnet416 resnet640 \
  --output evaluations/pose-comparison/new-results

python3 -m scripts.run_pose_comparison_pn41 \
  --models efficientdet0_thunder efficientdet1_thunder efficientdet1_full_thunder efficientdet1_letterbox_thunder posenet721_thunder \
  --output evaluations/pose-comparison/new-extra-results

python3 -m scripts.run_pose_comparison_pn41 \
  --models lightning thunder --input-max-width 640 \
  --output evaluations/pose-comparison/new-vga-results
```

将远端 summary、detections 和 annotated 视频同步到本地统一目录后：

```bash
.venv311/bin/python -m scripts.score_pose_comparison
.venv311/bin/python -m scripts.build_pose_review --recommended thunder
```

最终校验：28 次视频运行均正常退出；28 段标注 MP4 均通过完整解码、H.264/yuv420p 格式和帧数校验，HTML 通过脚本语法和静态链接检查，记录见 [validation.json](../artifacts/pose-comparison/validation.json)。本轮未完成浏览器交互复测，可直接用浏览器打开 HTML 进行 review。

规则、几何变换与视频接口等测试 **26 passed、1 skipped**（MQTT 联网测试按默认设置跳过）；比较器使用真实 Coral，未静默切换成纯 CPU 推理。原 `analytics` 服务恢复为 `indoorcam:pn41` 和 ESP32 RTSP 输入，未部署候选模型。ESP32 已拔出，摄像头源不可用符合当前状态。
