# pn41 部署与 Home Assistant 接入

## 环境与路径

检测到 pn41 为 Ubuntu x86_64，LAN 有线地址 `192.168.50.102`，有 `/dev/apex_0`、`/dev/apex_1`，设备组 `apex` 的 GID 为 `1004`，已安装 libedgetpu。HA 是 root 用户启动的 VirtualBox `HomeAssistant` 虚拟机，使用桥接网络；控制台显示 IP 为 `192.168.50.88`，已验证局域网前端为 **http://192.168.50.88/**（HTTP、80 端口）。该 IP 的 HTTPS/80 TLS 握手失败、443 和 8123 均拒绝连接；不要沿用默认 8123 端口。新服务放在 `~/esp32-indoorcam`，不替换现有 HA 或其他容器。

## 标准 Compose 部署

把整个项目（包括下载后的模型）同步到服务器。不要同步本地 `.venv*`、`.git`、固件 Wi-Fi 密码。下面命令均在项目根目录执行，pn41 上 Docker 需要 `sudo`。

```bash
python3 scripts/download_models.py
getent group apex
# 首次初始化；重复运行会拒绝覆盖已有账号
python3 scripts/provision.py --lan-ip 192.168.50.102 --apex-gid 1004 \
  --source /media/test.mkv --loop
cp /absolute/path/test.mkv deploy/runtime/media/test.mkv
sudo chown 1883:1883 deploy/runtime/passwords
sudo chmod 600 deploy/runtime/passwords

sudo docker compose --env-file deploy/.env -f deploy/compose.yaml up -d --build
sudo docker compose --env-file deploy/.env -f deploy/compose.yaml ps
sudo docker compose --env-file deploy/.env -f deploy/compose.yaml logs -f analytics
```

`provision.py` 创建随机 MQTT、RTSP 发布和观看密码，写入被 Git 忽略的 `deploy/runtime/` 和 `deploy/.env`。只有这个项目使用这些账号。`.env` 不要发到聊天/版本库。查看连接方式：

```bash
cat deploy/runtime/ha-connection.txt
```

标准 Dockerfile 使用 Python 3.9、Coral 官方 TFLite 2.5.0.post1、libedgetpu。人体检测和 MoveNet 分别使用两颗 TPU，预处理/部分不支持的算子/规则/编码仍在 CPU 上执行。Compose 映射 `/dev/apex_0` 和 `/dev/apex_1` 并附加设备组，不使用 `--privileged`。这台机器升级到 TFLite 2.14 后原生崩溃，已回到实测兼容版本；Python3.9 是旧 Coral 运行库的兼容约束，后续升级必须整体测试。宿主机上的驱动必须先安装，容器内装 libedgetpu 无法替代内核驱动。

镜像拉取受限时，pn41 已有的 `human-detect:media-validation-20261021` 可作为离线基础。`deploy/Dockerfile.pn41` 只适用于这台服务器，要求 `downloads/indoorcam-wheels` 中存在匹配 CPython3.9/Linux 的 wheels。它不影响原镜像或原程序；build 出独立 `indoorcam:pn41` 镜像。正式环境推荐标准 Dockerfile，避免长期依赖旧基础镜像。

## 服务地址与认证

| 服务 | 地址 | 用途 |
|---|---|---|
| 标注视频 | `rtsp://viewer:PASSWORD@192.168.50.102:18554/annotated` | HA、VLC、ffplay；H.264、TCP |
| MQTT | `192.168.50.102:18883` | HA 集成；账号 indoorcam |
| 当前截图 | `http://192.168.50.102:18081/snapshot.jpg` | HA Generic Camera 静态图 |
| 状态 | `http://192.168.50.102:18081/state` | 当前人数、框、骨架与事件 |
| 健康 | `http://192.168.50.102:18081/health` | 新鲜画面返回 200，断流 >5 秒为 503 |

只绑定该 LAN 地址，不开放公网。RTSP 使用独立观看账号，MQTT 禁止匿名。状态/截图 HTTP 是 LAN 内无认证的便捷接口，含室内画面，应通过本地防火墙限制到 HA 和管理员设备；若不需要截图，可删除 Compose 的 `18081` 端口映射，内部 healthcheck 仍可用。摄像头本身的 8554 与服务器的 18554 是两个不同端点。

## HA：MQTT 实体和事件

在 HA 的 **设置 → 设备与服务 → 添加集成 → MQTT**，填 `ha-connection.txt` 中的 broker、18883 端口、用户名和密码。MQTT 是一个独立 broker，不需要安装 HA Mosquitto add-on。如果 HA 已配置其他 MQTT broker，通常应复用原 broker：把 analytics 的 `mqtt_host`、`mqtt_port` 和账号改为该 broker，停用本项目 `mqtt` 服务及它的 `depends_on`，不要破坏原 MQTT 集成。

连接后 MQTT Discovery 会创建设备 `Indoor Camera` 及以下实体（实体 ID 以 HA 实际生成为准）：

- 人数、站立/坐/躺/未知数量。
- 有人二进制传感器。
- 疑似跌倒二进制传感器（短时显示）。
- `Fall candidate event` 事件实体。

默认 topic：

```text
homeassistant/{sensor|binary_sensor|event}/indoor_cam/.../config
indoorcam/indoor_cam/state
indoorcam/indoor_cam/availability
indoorcam/indoor_cam/event
```

状态和 Discovery retained；事件 **不 retained**，避免重启后当成新跌倒。事件 JSON 包含 `event_type`、UTC `timestamp`、`track_id` 和框。UTC 时间用于通知新鲜度，`time` 用于模型时序，在实时模式是单调时间，不是 Unix 时间。HA 启动会重新发送 Discovery。断流、推理卡住超过 5 秒或程序正常退出时实体离线；进程崩溃由 MQTT LWT 发布离线。断流时不伪造“人数为 0”。

`deploy/ha-automation.yaml` 可复制到自动化 YAML 编辑器。替换为实际事件实体 ID；先用 HA 持久通知验证，再启用手机通知。示例只接受带新鲜原始 UTC 时间戳的事件，忽略不可用状态；第一次真实事件也可以触发。MQTT QoS1 允许重复交付，通知接收方若需要严格去重可用 `timestamp + track_id` 作键。

## HA：视频

在 **设置 → 设备与服务 → 添加集成 → Generic Camera（通用摄像头）**：

1. 静态图 URL 填 `http://192.168.50.102:18081/snapshot.jpg`（可选）。
2. Stream Source 填完整的 `rtsp://viewer:密码@192.168.50.102:18554/annotated`。
3. RTSP transport 选 TCP；如果界面有“认证”字段，应区分截图 HTTP 与 RTSP 的认证。
4. 确认 HA 启用了 `stream` 集成（默认 `default_config` 已包含）。

把生成的 camera 实体加入仪表盘，选择实时视图：

```yaml
type: picture-entity
entity: camera.indoor_camera
camera_view: live
show_name: true
show_state: false
```

先在 pn41 用 `ffprobe -rtsp_transport tcp 'rtsp://viewer:...@192.168.50.102:18554/annotated'` 确认能解码 H.264，再排查 HA。HA 浏览器通常通过 HLS 播放，会比 RTSP 原流多几秒延迟；可以后续增加 WebRTC/go2rtc，但当前链路不依赖它。不要让 HA 直接读取 ESP32 的 MJPEG RTSP，HA 应读取 pn41 转码后的 H.264。

## 接入或更换摄像头输入

先按 [硬件文档](hardware.md) 确认摄像头正常，再编辑 `deploy/runtime/config.json`：

```json
{
  "source": "rtsp://CAMERA_LAN_IP:8554/camera",
  "realtime": true,
  "loop": false,
  "roi": []
}
```

这里只展示要改的字段，**不要用这个片段覆盖整个配置**，否则会丢掉 MQTT、输出 RTSP 和模型路径。变更后：

```bash
sudo docker compose --env-file deploy/.env -f deploy/compose.yaml restart analytics
```

实时读取线程持续排空视频，只保留最新帧，避免推理速度不足后延迟累积。RTSP 输入使用独立 FFmpeg、较小的探测缓冲并按完整原始帧读取，避免 OpenCV 在稀疏 MJPEG 流上反复探测超时；文件回放仍使用 OpenCV。编码线程按固定帧率输出最近一次标注。因此标注更新率受采集和推理共同限制，重复的画面不代表新增推理。源重连/文件循环会重置跟踪。断流超过 5 秒会关闭旧的视频发布，避免无限播放看似正常的冻结画面。服务恢复后会重新推流。

## 运维与回滚

```bash
sudo docker compose --env-file deploy/.env -f deploy/compose.yaml logs --tail=100 analytics mediamtx mqtt
curl -f http://192.168.50.102:18081/health
ls -l /dev/apex_0
sudo docker compose --env-file deploy/.env -f deploy/compose.yaml down
```

`down` 只停止本项目；不使用 `down -v` 可以保留 MQTT 数据。密码轮换需同时改 broker password 文件、analytics 环境、MediaMTX 配置和 HA 连接。为排障单独运行回放时先停止 analytics，避免同一个 TPU 和 `camera_id` 被多个进程争用。

单 TPU 机器需从 Compose 的 `devices` 删除 `/dev/apex_1`，并将 `pose_device` 改为 `pci:0` 或将 `pose_backend` 设为 `cpu`；不能照搬 pn41 的双 TPU 性能结果。

## 姿态模型与 Thunder 上线（2026-10-31）

生产代码已支持 `pose_model: lightning`（默认、192×192）和 `pose_model: thunder`（256×256），适用于 Coral 和 CPU 两种姿态后端，不自动降级。`scripts/download_models.py` 的固定哈希清单同时包含两者；Thunder 文件名为 `pose_thunder_edgetpu.tflite` / `pose_thunder.tflite`，不会覆盖 Lightning 文件。

按用户要求，pn41 当前已切换为 **MobileDet + MoveNet Thunder**：仅在原私有配置中添加 `"pose_model": "thunder"`，人体检测仍使用 `pci:0`，姿态使用 `pci:1`。保持相机 RTSP、全画面 ROI、6 fps 输出、HA/MQTT/RTSP 认证参数。旧 Coral 运行库未升级。

已通过本地 33 项测试（MQTT 独立集成测试未启用）、新镜像中 5 项模型选择测试及线上健康、实时 HTTP、H.264 960×540/6 fps 和实际 MQTT 更新检查。MQTT 验证只订阅真实状态，没有注入测试人数或事件。6 fps 是编码帧率，实际新画面更新率仍受 ESP 无线采集限制。

新镜像标记为 `indoorcam:thunder-20261031`，也已标记为 Compose 使用的 `indoorcam:pn41`。私有配置/原源码备份为服务器 `evaluations/thunder-rollout-20261031/backup-20261031T114343Z/`；原镜像保留为 `indoorcam:before-thunder-20261031t114343z`。部署元数据在同目录 `rollout.json`。必要时在 pn41 回滚：

```bash
cd ~/esp32-indoorcam
cp evaluations/thunder-rollout-20261031/backup-20261031T114343Z/deploy/runtime/config.json deploy/runtime/config.json
sudo docker tag indoorcam:before-thunder-20261031t114343z indoorcam:pn41
sudo docker compose --env-file deploy/.env -f deploy/compose.yaml \
  -f deploy/compose.pn41.yaml up -d --no-build --no-deps --force-recreate analytics
```

Thunder 在离线比较中姿态表现较好，但仍未可靠识别躺卧及跌倒；切换模型不表示跌倒预警已达到可依赖的水平。左右镜像建议在 ESP 传感器端校正，详见 [镜像校正与烧录说明](hardware.md#左右镜像校正)。

## 此次 pn41 的已部署实例

已构建 `indoorcam:pn41` 并导入上游 broker/relay 镜像。2026-10-31 已将输入切换到 `rtsp://192.168.50.39:8554/camera`，关闭文件循环，保留双 TPU、全画面 ROI、6 fps 编码和原 MQTT/RTSP 认证。运行库与全部实测见 [验证记录](validation.md)。原配置保存在服务器私有的 `deploy/runtime/config.before-camera.json`，更新前镜像保留为 `indoorcam:before-camera-20261031`。

```bash
cd ~/esp32-indoorcam
python3 scripts/check_stack.py
sudo docker compose --env-file deploy/.env -f deploy/compose.yaml -f deploy/compose.pn41.yaml ps
# 更新源码后使用同样的离线构建方式
sudo docker build --network=none -f deploy/Dockerfile.pn41 -t indoorcam:pn41 .
sudo docker compose --env-file deploy/.env -f deploy/compose.yaml -f deploy/compose.pn41.yaml up -d --no-build
```

标准 Compose 命令中的 `up --build` 会尝试标准 Dockerfile；在当前 pn41 上更新时使用上面的离线构建命令。只修改运行配置后可直接 `restart analytics`。测试视频循环会重置轨迹但保持推流连续。

## HA 接入验收（2026-10-31）

已在用户的 HA 登录会话中完成以下配置：

- MQTT 集成连接 `192.168.50.102:18883`，发现设备 `Indoor Camera`。
- 通用摄像头使用项目的认证 RTSP 和截图端点，TCP 传输。实际实体为 `camera.192_168_50_102`。
- 侧边栏新增 **室内监控**，视图路径 `/indoor-monitor/live`。页面包含标注视频、人数/各姿态数量、有人状态、疑似跌倒状态和事件时间。
- 浏览器实际视频解码为 960×540，readyState=4、未暂停；验收时已解码 1149 帧、播放时间 191.4 秒。
- 已发送一条非 retained、`test=true`、`source=ha_integration_check` 的模拟事件，HA 事件实体接收成功。页面会对最近的测试事件明确标注。它不是算法在测试视频中识别出的跌倒。

实际 MQTT 实体：

```text
sensor.indoor_camera_people
sensor.indoor_camera_standing
sensor.indoor_camera_sitting
sensor.indoor_camera_lying
sensor.indoor_camera_unknown
binary_sensor.indoor_camera_occupied
binary_sensor.indoor_camera_possible_fall
event.indoor_camera_fall_candidate_event
```

LAN 内可打开 `http://192.168.50.88/indoor-monitor/live`。当前电脑通过仅监听 127.0.0.1 的 SSH 转发访问 `http://127.0.0.1:18480/indoor-monitor/live`；HA 配置中使用 pn41 的 LAN 地址，HA 与推理服务之间的连接独立于此浏览器转发。需要手动恢复本机转发时：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
  -L 127.0.0.1:18480:192.168.50.88:80 pn41
```

本地验收截图和无凭据状态记录为 `artifacts/ha-dashboard.png`、`artifacts/ha-integration-status.json`，服务器备份位于 `evaluations/ha-acceptance/`。仪表盘 YAML 已保存为 `deploy/ha-dashboard.yaml`。

实物摄像头接入后再次验收：HA 视频连续解码约 116 秒/699 帧，960×540；`scripts/check_stack.py` 的 HTTP、认证 H.264 RTSP（6 fps）和真实 MQTT 检查通过。实物截图保存在 `artifacts/hardware/ha-live-camera.png`。仪表盘说明已更新为实物、全画面监控，并注明无线传输仍可能卡顿或离线。摄像头采集仍约 1–3 fps，不能将 6 fps 编码当作新画面更新率。

跌倒识别仍有明显漏报。已接通候选事件通道，尚未启用自动通知。`deploy/ha-automation.yaml` 的示例额外过滤 `test=true` 事件；启用提醒前先改进并复测识别效果。

复测 HA 事件接收链路时，可在 pn41 发送一条带明确测试标记的非 retained 事件：

```bash
cd ~/esp32-indoorcam
sudo docker run --rm --network indoorcam_default --env-file deploy/.env \
  -v "$PWD:/work:ro" -w /work --entrypoint python indoorcam:pn41 \
  -m scripts.emit_ha_test_event
```
