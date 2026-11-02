# ESP32-S3 N16R8 + OV3660 硬件配置

## 当前确认程度

用户的板型为鹿小班 ESP32-S3-N16R8 CAM，摄像头已确认为 OV3660。2026-10-31 已收到用户提供的摄像头接口图，图中明确标注 `CAMERA_MODEL_ESP32S3_EYE`，GPIO 与本项目原先准备的表一致。该板已连接本机 UART USB 接口，WCH VID:PID 为 `1A86:55D3`，串口为 `/dev/cu.usbmodem5D300054671`。

串口工具已确认 ESP32-S3 revision v0.2、40 MHz 晶振、16 MB Quad Flash，MAC 为 `d8:85:ac:d6:ff:1c`。用户确认模块是 WROOM-1、使用板载天线。出厂程序是 `led_strip` / TinyUSB 示例，启动头配置为 2 MB，但物理 Flash 实际为 16 MB。2026-10-31 新固件已成功烧录并启动：PSRAM 报告约 8 MB（初始化前 8,386,279 字节），相机初始化成功、Sensor PID 为 `0x3660`。已配网，当前 LAN 地址 `192.168.50.39`；重启后自动恢复保存的 Wi-Fi，无需重新填写密码。

实物 JPEG、RTSP、Coral 标注和 HA 播放已接通。HA 实测连续解码约 116 秒、699 帧；这对应服务器 6 fps 编码，摄像头新帧仍约 1–3 fps，低于目标上限 10 fps，且仍有停顿。安装位置、供电、长期稳定性与真人姿态尚需现场验收，不把当前状态描述为稳定 10 fps。

`firmware/include/camera_pins.h` 已按该图核实，`INDOORCAM_PINS_CONFIRMED=1`。这个确认只适用于当前板型；更换 PCB 时重新核对，不能仅凭 N16R8 内存规格推断引脚。固件使用包含 OV3660 支持的 Espressif `esp32-camera`。

## 接线核对表

摄像头通过板载排线插座连接，通常无需杜邦线。断电后打开卡扣，确认排线触点朝向和 Pin 1，再插到底并锁紧；不能带电插拔。

| 信号 | 固件符号 | 核对要点 |
|---|---|---|
| D0…D7 / Y2…Y9 | CAM_D0…CAM_D7 | D0 对应传感器 Y2，顺序不能反 |
| XCLK | CAM_XCLK | 当前默认 10 MHz，可用 INDOORCAM_XCLK_HZ 调整 |
| SIOD / SDA | CAM_SIOD | SCCB 控制数据线 |
| SIOC / SCL | CAM_SIOC | SCCB 控制时钟线 |
| VSYNC | CAM_VSYNC | 帧同步 |
| HREF | CAM_HREF | 行同步 |
| PCLK | CAM_PCLK | 像素时钟 |
| PWDN / RESET | CAM_PWDN / CAM_RESET | 未连接填 -1，不可凭猜测占用 GPIO |
| 电源/地 | 板载插座 | 确认模块电压，由开发板供电；不要给信号脚接 5 V |

当前已确认 GPIO：XCLK=15、SDA=4、SCL=5、Y2…Y9（固件 D0…D7）依次为 `11,9,8,10,12,18,17,16`，VSYNC=6、HREF=7、PCLK=13。PWDN 由 10K 电阻下拉，RESET 接 ESP_RST，因此两个软件 GPIO 都设为 -1。

使用质量可靠的 USB 5 V/2 A 电源与短线。电源额定 2 A 是余量，并非板子会持续耗电 2 A。不要用 ESP32 GPIO 给摄像头供电，也不要把 GPIO35/36/37 等可能被 OPI PSRAM 占用的引脚作为任意扩展 IO。

用户已确认当前 USB 直连 Mac，未经过扩展坞/USB Hub；这不能直接证明供电电压和瞬态电流没有问题，但已排除“经扩展坞供电”这一接线情况。

## 固件编译与烧录

```bash
python3 -m pip install platformio
# 当前这块板接 WCH UART 口，日志也应走 UART
export PLATFORMIO_CORE_DIR="$PWD/.pio-core"
pio run -d firmware -e esp32s3_uart
pio device list
pio run -d firmware -e esp32s3_uart -t upload --upload-port /dev/cu.usbmodem5D300054671
pio device monitor -d firmware -e esp32s3_uart --port /dev/cu.usbmodem5D300054671 --baud 115200
```

PlatformIO 设置：ESP32-S3、16 MB Flash、QIO Flash + OPI PSRAM。`esp32s3_uart` 关闭启动 USB CDC，使 `Serial` 输出到板载 UART；`esp32s3` 用于另一个原生 USB 口并启用 USB CDC。当前 WCH 链路在 460800/921600 baud 下出现包错误，已以 230400 验证读取，UART 上传使用 230400。

`partitions.csv` 使用单个 factory app，无应用 OTA 更新。NVS 为 0x9000–0xDFFF，0xE000–0xFFFF 留给 Arduino 上传器的 `boot_app0.bin`，避免覆盖保存的 Wi-Fi 设置。若无法进入下载模式，按住 BOOT、短按 RESET、松开 BOOT 后再烧录。首次串口信息应包含 PSRAM 大小、相机 PID（OV3660 通常 0x3660）、Wi-Fi IP 与 RTSP 地址。

## 首次配网

未保存 Wi-Fi 时，固件会建立 `IndoorCam-XXXXXX` 的 WPA2 配网热点，随机密码与 `http://192.168.4.1/` 地址通过串口输出。用手机连接这个热点并打开页面，填写可到达 pn41 的 2.4 GHz Wi-Fi。连上后页面和串口显示设备 LAN IP，配网热点随后关闭。

网络凭据保存在开发板 NVS 的 `indoorcam` 命名空间，不需要写入代码。后续上电自动连接，连接失败约 20 秒后会再次进入配网模式。仍支持可选的、被 Git 忽略的 `firmware/include/secrets.h` 作为首次默认配置；串口不会输出家庭 Wi-Fi 密码。

配网完成后继续以下验收：

首次启动按顺序确认：

1. 串口没有 PSRAM/相机初始化错误，传感器型号正确。
2. `http://CAMERA_IP/health` 可读，RSSI 建议优于约 -65 dBm。
3. `http://CAMERA_IP/snapshot.jpg` 能看到正常颜色、方向和对焦。
4. 在路由器里给摄像头做 DHCP 地址保留。
5. 在 pn41 验证 RTSP：

```bash
ffprobe -rtsp_transport tcp rtsp://CAMERA_IP:8554/camera
ffmpeg -rtsp_transport tcp -i rtsp://CAMERA_IP:8554/camera -t 20 -an -c:v libx264 camera-check.mp4
```

固件只支持 **单个 RTSP TCP 读取者**，建议让 analytics 独占；HA/VLC 从 pn41 的标注流读取，避免抢摄像头连接。RTSP 是 RFC2435 JPEG，默认 VGA 640×480、调度上限 10 fps、JPEG quality 20（数值越小质量越高，可用 `INDOORCAM_JPEG_QUALITY` 调整）。当前使用单个内部 RAM 帧缓冲和 32 KiB 内部 RAM 发送缓冲，将多个完整 RTP 包批量写入 TCP；启动日志会报告相机初始化后的内部空闲内存。PSRAM 仍在启动时检查，但当前 VGA 帧缓冲不依赖 PSRAM。

实物曾出现一帧发送数秒的停顿。相机关闭时，独立 Wi-Fi 基准连续三次达到约 157/176/184 KB/s；启用相机后仍明显偏慢。降低时钟、单帧缓冲、内部 RAM 与较高压缩率未消除全部停顿，不能据此认定最终根因。服务器多网卡路由和 ARP 的临时对照设置均已撤销。Wi-Fi 目前恢复 802.11b/g/n（启动打印 protocol=7）、关闭省电；`INDOORCAM_WIFI_HT=0` 仅用于兼容性对照。

仅支持 baseline YUV422/YUV420 JPEG，无 progressive/restart 标记；不兼容会在串口报告。串口每约 10 秒打印实际发送帧数、JPEG 字节数、取帧/发送耗时和 RSSI，便于区分采集与传输瓶颈。

需要重新隔离无线与相机影响时，可临时烧录 `wifi_diag` 环境，访问 `http://CAMERA_IP/benchmark.bin` 下载 256 KiB。该固件不启动相机、不输出 RTSP；测完必须用 `esp32s3_uart` 恢复监控固件，保存的 Wi-Fi 不会被清除。诊断代码位于 `firmware/src/wifi_diag.cpp`。

ESP32-S3 + OV3660 不提供硬件 H.264 编码，不应承诺 1080p H.264 实时输出。VGA 起步验证稳定性，再决定是否改 SVGA；增加分辨率同时增大 Wi-Fi 带宽和时延。没有音频。

## 安装机位

固定相机，尽量让指定区域内的人从头到脚可见，避免沙发/桌沿遮挡髋部和膝盖。不要逆着大窗户拍摄；OV3660 普通模组没有夜视补光能力，暗处需要室内照明。镜头如果带手动调焦环，按实际监控距离调焦。

按用户要求，固件默认 `INDOORCAM_ROTATION=180`：在传感器端同时翻转水平和垂直方向，再编码 JPEG。原始截图、RTSP、Coral 的推理输入及 HA 标注流使用同一方向，人数文字和框由服务器在旋转后绘制。`/health` 返回 `rotation: 180`。需要恢复原方向时将该编译宏设为 `0`，重新烧录；不应再在 HA 中叠加一次旋转。更改方向后已有局部 ROI 需重新标注；当前按用户选择使用全画面，不受多边形坐标变化影响。

### 左右镜像校正

画面上下已正确、但左右镜像时，推荐在 ESP 的 OV3660 传感器端切换水平翻转，保留当前垂直翻转。该操作发生在 JPEG 编码前，不需要解码重编码或服务器逐帧翻转，也不增加传输带宽。不能简单把 180° 改回 0°，因为那会同时改变上下方向。

新增 `INDOORCAM_MIRROR_CORRECTION`（默认 `0`），设为 `1` 会在原方向基础上额外水平翻转一次。已准备专用环境，保留默认 180° 安装方向：

```bash
# 在项目根目录执行；先连接开发板 UART 口。
PLATFORMIO_CORE_DIR="$PWD/.pio-core" .venv311/bin/pio run -d firmware \
  -e esp32s3_uart_unmirrored -t upload --upload-port /dev/cu.YOUR_UART_PORT
```

此环境的编译结果位于 `firmware/.pio/build/esp32s3_uart_unmirrored/firmware.bin`。烧录不擦除已保存的 Wi-Fi。`/health` 新增 `mirror_correction`、实际传感器 `hmirror`、`vflip`，便于确认状态。用带文字的纸核对左右方向和上下方向；不要仅凭人物举手判断镜像。

2026-10-31 用户重新连接 USB 后，已通过 `/dev/cu.usbmodem5D300054671` 烧录并校验成功；设备自动恢复 Wi-Fi。实际返回 `rotation:180, mirror_correction:1, hmirror:0, vflip:1`，前后截图确认只改变左右方向。服务器 Thunder 实时状态与认证 H.264 标注流（960×540、6 fps 编码）已自动恢复。校验截图与状态保存在 `artifacts/hardware/mirror-before.jpg`、`mirror-after.jpg`、`mirror-after-health.json`。后续更新固件继续使用 `esp32s3_uart_unmirrored` 环境，避免恢复旧的镜像方向。

服务器也可在推理前水平翻转作为临时方案，但当前服务没有启用此功能；若以后添加，必须在检测、ROI 和标注之前处理。只翻转最终标注视频会反转文字且让输出坐标与原始画面不一致。只在 HA 卡片上做显示镜像也不会校正原始流。

摄像头端 HTTP/RTSP 没有认证，只用于可信局域网；不配置公网端口转发。pn41 的标注 RTSP 和 MQTT 使用独立账号。推荐在路由器上限制摄像头只能访问 pn41/DNS/NTP，手机从 HA 观看。

当前已通过引脚图核对、USB 烧录、PSRAM 与 OV3660 初始化、实际 JPEG/RTSP、重启后恢复 Wi-Fi 和 HA 短时播放检查；持续帧率、温升、供电与最终机位仍需现场验收，软件编译和模拟 RTP 测试无法替代这些检查。

`compile_check` 环境专供无硬件编译验证；本机烧录选 `esp32s3_uart`。PlatformIO 的板名仍显示 DevKitC N8，这是基础 board 名称；flash image 和分区已明确覆盖为 16 MB，PSRAM 以启动时实测为准。
