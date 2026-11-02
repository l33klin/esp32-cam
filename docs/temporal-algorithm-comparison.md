# 连续视频时序识别验证

2026-10-31。本轮为离线验证，**没有训练或微调，没有更换线上 Thunder，没有发布 HA/MQTT 测试状态或事件**。测试期间未映射或占用 Coral；pn41 性能测量使用独立、无网络、限 CPU 的临时容器。结束后线上镜像、摄像头源和健康检查均保持正常。

## 结论

**时序方案在工程上可运行，但本轮公开预训练权重不适合直接上线。** TCN/LSTM 在 pn41 上的额外网络开销约 3–4 ms/人/窗口，适合继续做定向训练；计算速度不是这两种模型的主要问题。当前主要问题是婴幼儿/遮挡场景与训练数据差异、倒地过程的人体漏检及跟踪中断。

TSSTG 的站/坐/躺小样本结果从基线 10/20 变为 11/20；只多对一个目标，不能据此断言更好。四个躺卧目标仍然 0/4。摔倒的几个“宽窗口命中”经检查发生在真正失衡之前，不能当作成功识别。正常活动样本中同时出现了误报。

建议保留线上 Thunder。本轮最值得保留的技术路线是 **Thunder 骨架 + 轻量 TCN**，下一步需要婴幼儿动作和实际机位的数据，而不是直接部署此次下载的 TCN 权重。

## 可直接查看的结果

- [同步视频评审页](../artifacts/temporal-comparison/review/index.html)：左侧基线，右侧选择 TSSTG/TCN/LSTM 和两种窗口；点击事件时间跳转。
- 本机预览：<http://127.0.0.1:18592/artifacts/temporal-comparison/review/index.html>。
- [完整结构化评分](../artifacts/temporal-comparison/scores.json)、[MoViNet 活动结果](../artifacts/temporal-comparison/movinet-scores.json)。
- [pn41 骨架模型性能](../artifacts/temporal-comparison/benchmark-pn41.json)、[MoViNet 性能](../artifacts/temporal-comparison/benchmark-movinet-pn41.json)。
- [失衡过程一](../artifacts/temporal-comparison/review/first-incident.jpg)、[失衡过程二](../artifacts/temporal-comparison/review/second-incident.jpg)：0.25 秒间隔人工核对图，裁剪只用于观察。

## 比较了什么

| 方法 | 训练来源与输入 | 输出 | 本轮运行位置 |
|---|---|---|---|
| 原有基线 | MobileDet + MoveNet Thunder；几何与跟踪规则 | 人数、站/坐/躺、候选事件 | 复用上一轮完整双视频 Coral 输出 |
| TSSTG / 双流 ST-GCN | 公开 Le2i Coffee/Home 权重；30 点骨架序列与运动差分 | 站、走、坐、躺、起立、坐下、跌倒 | Mac CPU 完整回放；pn41 CPU 测速 |
| Pose TCN-Attention | 作者公开的 MCFD 权重；32 点骨架序列 | fall / non_fall | 同上 |
| Pose LSTM | 同一作者公开的 MCFD 权重；32 点骨架序列 | fall / non_fall | 同上 |
| 官方 MoViNet-A0 Stream | Kinetics-600；连续 RGB 图像及 43 个状态张量 | 全画面活动类别 | Mac CPU 完整视频；pn41 TFLite CPU 测速 |

**TCN/LSTM 是二分类跌倒模型，不输出站/坐/躺。** 不能把它们的 non_fall 当作站立。TSSTG 将 Standing/Walking 映射为站立，Sitting 映射为坐，Lying Down 映射为躺；三个过渡动作映射为 unknown。

骨架模型在过去的滑动窗口上重算；其中发布的 LSTM 为双向模型，但本轮窗口终点是当前时刻，没有读取当前时刻之后的帧。MoViNet 则保留内部流式状态。两者都利用了连续视频，推理方式与延迟不能视为相同。

MoViNet 的公开权重没有通用“跌倒”类别，只有 falling off chair / falling off bike 等特定动作，因而只作为连续 RGB 活动识别对照，不能与专门的跌倒模型计算同口径准确率。

本轮另外对上一轮 **YOLOv8n Pose 480 / conf .10** 的骨架做了相同回放；没有发现其与上述时序权重组合后更适合上线。

## 数据与评估边界

使用用户提供的两个完整视频：Ring 约 37.9 秒，摔倒合集约 210.8 秒。后者使用先前转换的 H.264 文件，未改动原始下载文件。人体/骨架记录复用上一轮的 8 fps 全片输出：每个骨架来源为 303 + 1687 = 1990 帧。这样隔离了时序分类的作用，也避免暂停当前生产服务。

两个骨架来源 × 三个模型 × 四种输入帧率 × 两种窗口 × 两个完整视频，共 **96 次网络回放**。此外，从已保存的 TSSTG 输出重放了 32 组原项目 argmax 决策，没有为此重新跑网络。MoViNet 对两个完整视频以 5 fps 各运行两次，分别只在切镜时重置状态、以及额外每 10 秒重置状态。

动作窗口固定为 **1.2 秒、3.625 秒**，分别插值到网络所需的 30/32 个点；所有窗口仅使用当前和过去的观测。至少三次实际观测、覆盖足够的历史后才分类。插值不会创造新的动作证据。

降帧测试保留 8 fps 缓存中对应的 3/2/1 fps 检测，**仅用保留帧重新关联轨迹**；没有利用被丢弃帧提供的原跟踪 ID。跟踪超时为 `max(0.8, 1.5 / fps)`，时序断点阈值为 `max(0.5, 1.5 / fps)`，切镜清空全部历史。这是抽帧模拟，没有模拟 Wi-Fi 卡顿、曝光变化和不规则丢帧，因此真实 ESP 效果可能更差。

评分来自以下有限标注，均是助手初步视觉核对，需用户 review：

- 复用原有 **20 个姿态 review 目标**：站 13、坐 3、躺 4。未检出、冷启动和 unknown 均算未成功；不是 COCO mAP。
- 摔倒合集前 90 秒中的 **4 个已核对失衡窗口**，分别约 4、20、31、60 秒。以原有粗略区域匹配告警目标，区域内可能有多个或被遮挡的人，不能替代精确目标标注。
- 共 **47 秒已核对的无新跌倒片段**，包括 Ring 的走路练习、成年人照顾孩子、婴儿已经在地上等。这里出现的告警按端到端误报计；包括电视中的人导致的告警。全画面运行没有自动排除电视。

这些视频相邻帧高度相关，负例时长很短，事件真值也不完整，**不报告正式召回率、精确率、每小时误报率或统计显著性**。完整视频里的其他告警保留供 review，不自动全部判为误报。

## 主要效果：Thunder 骨架，8 fps

每次事件要求同一个人持续满足动作条件至少 0.25 秒，随后按轨迹冷却 5 秒。TCN/LSTM 使用固定 0.5 阈值。TSSTG 主表遵循原项目 argmax；其 sigmoid 分数未校准，不与 TCN 的概率直接比较。

| 方法 | 窗口 | 宽窗口触发 / 4 | 失衡开始后触发 / 4 | 47 秒负例误报 | 姿态正确 / 20 |
|---|---:|---:|---:|---:|---:|
| 原几何规则 | 原设置 | 0 | 0 | 0 | 10 |
| TSSTG 原项目决策 | 1.2 s | 1 | **0** | 6 | 11 |
| TCN-Attention | 1.2 s | 0 | **0** | 7 | 不输出姿态 |
| LSTM | 1.2 s | 0 | **0** | 10 | 不输出姿态 |
| TSSTG 原项目决策 | 3.625 s | 0 | **0** | 2 | 7 |
| TCN-Attention | 3.625 s | 0 | **0** | 4 | 不输出姿态 |
| LSTM | 3.625 s | 1 | **0** | 5 | 不输出姿态 |

“宽窗口触发”不能当作识别成功：

- TSSTG 1.2 秒在 **19.125 秒**触发，而约 **20 秒**才开始明显失衡；19.75/20 秒输出已变回 Walking，随后倒地阶段丢失或重建轨迹。
- LSTM 3.625 秒在 **3.750 秒**触发；0.25 秒连拍中约 **4.125 秒**才开始失衡。随后倒地阶段没有新的正确告警。

因此主表中的 1/4 只说明宽窗口覆盖，**不是确认的有效摔倒识别**。失衡开始点也是近似视觉标注，不用于报告毫秒级延迟。

短窗口下，TSSTG 的姿态分项为站 9/13、坐 2/3、躺 0/4；原几何规则为站 7/13、坐 3/3、躺 0/4。它增加了部分站立判断，同时损失了一次坐姿判断，仍没有解决躺卧。

最初另测过 TSSTG argmax 再加 0.5 分数门槛：1.2 秒窗口为 0 告警、姿态 2/20。核对原项目后发现它直接使用最高分动作，故保留原项目决策作为主表，并将 0.5 门槛版本保留为对照；没有按测试集寻找“最佳阈值”。零告警不代表识别可靠。

## 低帧率会怎样

下表固定为 Thunder、1.2 秒窗口；TSSTG 使用原项目决策。

| 新画面帧率 | 可分类人体观测占比 | TSSTG 宽窗口触发 / 负例误报 | TCN 宽窗口触发 / 负例误报 | LSTM 宽窗口触发 / 负例误报 |
|---|---:|---:|---:|---:|
| 8 fps | 72.3% | 1 / 6 | 0 / 7 | 0 / 10 |
| 3 fps | 62.9% | 0 / 0 | 0 / 6 | 1 / 9 |
| 2 fps | 65.7% | 0 / 1 | 0 / 5 | 0 / 8 |
| 1 fps | 53.6% | 1 / 0 | 0 / 3 | 0 / 3 |

低帧率结果没有单调改善，不能将误报减少理解为模型更好：可判断的历史更少、动作容易错过、轨迹也更容易断。1 fps 的 TSSTG 告警在约 22 秒，正值照顾者介入，粗区域不能确认目标是跌倒的孩子，仍不算已验证成功。

服务器把视频编码为 6 fps 并不能补足 ESP 实际约 1–3 fps 的新画面。要让时序方案在真实场景有效，仍需要改善实际采集连续性。

## YOLO 骨架与 MoViNet 对照

YOLOv8n Pose 480/conf .10 + 1.2 秒窗口，在四个宽摔倒窗口均未触发：TSSTG/TCN/LSTM 的 47 秒负例告警分别为 **13 / 8 / 12**。TSSTG 姿态为 4/20。它没有修复宝宝倒地时的漏检与连续历史不足。

MoViNet 保留并更新流式状态，确实使用了视频历史，但其通用活动权重与室内婴儿场景不匹配：完整摔倒合集里 falling off chair 的最高分约 **0.057**；四个已核对窗口中的峰值分别约 **0.00024、0.00111、0.00155、0.00247**。Ring 中该类别最高约 0.222，表明该类别也不能直接解释成真实摔倒概率。每 10 秒额外重置状态的对照同样没有提供可用的通用跌倒识别。

本轮 MoViNet 采用全画面 RGB 等比补边至 172×172，保留画面两侧的人；这是明确记录的输入适配，不是任务微调。它也不提供人数、人体框和个人姿态，因此不能独立替换当前整条链路。

## pn41 性能实测

Intel N6005，原有服务继续运行。临时容器无网络、无 Coral 设备、最多 1.5 个 CPU、1 GB 内存。每个模型预热 5 次、测 120 次、两个线程、batch=1。骨架模型用 ONNX Runtime 1.16.3 + NumPy 1.26.4，安装在容器内临时目录；生产镜像和宿主机没有安装或升级这些包。MoViNet 使用已有兼容 TFLite 的 CPU 路径。

| 网络 | 平均 | p95 | 计量单位 |
|---|---:|---:|---|
| TSSTG | **90.28 ms** | 122.23 ms | 每个人的一个 30 点窗口 |
| TCN-Attention | **3.72 ms** | 5.20 ms | 每个人的一个 32 点窗口 |
| LSTM | **3.29 ms** | 4.00 ms | 每个人的一个 32 点窗口 |
| MoViNet-A0 Stream | **25.27 ms** | 31.42 ms | 一张 172×172 图像及流式状态更新 |

骨架表测量网络 session，不含检测、Thunder、解码、历史插值、特征预处理、绘图或 H.264 编码；MoViNet 包含状态张量搬运和 softmax，不含视频解码和图像缩放。**不能把这些毫秒直接换算成摄像头整条管线的 FPS。** TSSTG 多人时开销会按人数增加，明显高于 TCN/LSTM。

## 复现与校验

代码只在 `scripts/` 中用于研究，没有接入生产 `Inference` 工厂。外部源码和权重均留在 Git 忽略的 `artifacts/temporal-comparison/`；[下载清单](../config/temporal-models.json)固定每个文件的 URL 和 SHA256。

```bash
# 项目根目录；使用已具备 PyTorch / ONNX Runtime 的独立 .venv-yolo。
.venv-yolo/bin/python -m scripts.download_temporal_models
.venv-yolo/bin/python -m scripts.export_temporal_models

# 上一轮缓存需要保留在 artifacts/yolo-comparison/results/。
# out 必须为空；首次重放时复制切镜清单。
cp artifacts/pose-comparison/scene-cuts.json artifacts/temporal-comparison/scene-cuts.json
.venv-yolo/bin/python -m scripts.compare_temporal_models
# 额外的 TSSTG 0.5 门槛对照，仍不调参：
.venv-yolo/bin/python -m scripts.compare_temporal_models --models tsstg --tsstg-min-score .5

.venv-yolo/bin/python -m scripts.compare_movinet --metadata-only
.venv-yolo/bin/python -m scripts.compare_movinet
.venv-yolo/bin/python -m scripts.compare_movinet --reset-seconds 10
.venv311/bin/python -m scripts.score_temporal_comparison
.venv311/bin/python -m scripts.build_temporal_review
.venv311/bin/python -m scripts.serve_pose_review --root . --port 18592
```

评审入口为 `/artifacts/temporal-comparison/review/index.html`。MoViNet 元数据导出需要 TensorFlow；完整版本记录在导出校验中。模型权重使用 `torch.load(..., weights_only=True)`，严格匹配网络参数；没有回退到执行不受限 pickle。

已完成以下校验：

- 10 个原始 PyTorch / 导出 ONNX 对照，最大绝对误差小于 4×10⁻⁷。
- 三个骨架预处理对照，TCN 两种特征与发布实现完全一致，TSSTG 差异小于 3×10⁻⁷。
- 相同保存输入在 Mac ORT 1.31.1 与 pn41 ORT 1.16.3 上对照，最大绝对误差约 6×10⁻⁸。
- 12 个完整标注 MP4 全部解码通过；H.264/yuv420p，960×540，303 或 1687 帧。
- 37 项回归测试通过，1 项需主动启用的 MQTT 测试跳过；没有为了本轮实验发送模拟事件。
- 浏览器验证两路视频解码、同步跳转、播放/暂停、模型切换保持时间及 Ring/合集切换。

[完整验收记录](../artifacts/temporal-comparison/validation.json)、[导出校验](../artifacts/temporal-comparison/export-validation.json)、[线上状态](../artifacts/temporal-comparison/production-verified.json)。原项目下载链接失效时使用了公开社区 TSSTG 镜像，其 SHA256 与公开 Git LFS 指针一致：`1cac9a6d…750a149`，完整哈希见清单。

## 下一步应做什么

优先解决倒地过程的漏检和历史断裂，再训练轻量 TCN。训练输入应考虑骨架、置信度/缺失掩码、实际时间间隔、人体中心下降速度和框的变化。当前公开特征会逐帧居中缩放，可能丢掉房间坐标中的整体下降信息；这属于值得验证的改进方向，并非本轮已经证实的原因。

训练与验收应按不同摄像头/家庭/完整事件划分数据，包含主动坐下、慢躺、爬行、弯腰照顾、被抱起、遮挡和电视等负例；不能把相邻帧随机切到训练和测试中。用户这两个视频继续保留为回归样本，避免在它们上反复挑阈值后声称泛化提升。

## 来源

- [TSSTG 原项目](https://github.com/GajuuzZ/Human-Falling-Detect-Tracks)，[模型与输入处理](https://github.com/GajuuzZ/Human-Falling-Detect-Tracks/blob/master/ActionsEstLoader.py)。
- [MCFD TCN/LSTM 作者公开代码与权重](https://huggingface.co/datasets/minhy112/pose-guided-fall-detection-icta2026)。本轮只验证其公开权重在用户视频上的表现，没有独立复现作者训练集成绩。
- [TensorFlow 官方 MoViNet 实现与流式 TFLite 示例](https://github.com/tensorflow/models/tree/master/official/projects/movinet)，[官方模型说明](https://www.tensorflow.org/hub/tutorials/movinet)。
