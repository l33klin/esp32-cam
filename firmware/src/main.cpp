#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include "esp_camera.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "camera_pins.h"
#include "jpeg_rtp.h"
#include "wifi_setup.h"
#if __has_include("secrets.h")
#include "secrets.h"
#else
#include "secrets.example.h"
#endif
#ifndef INDOORCAM_JPEG_QUALITY
#define INDOORCAM_JPEG_QUALITY 20
#endif
#ifndef INDOORCAM_XCLK_HZ
#define INDOORCAM_XCLK_HZ 10000000
#endif
#ifndef INDOORCAM_ROTATION
#define INDOORCAM_ROTATION 180
#endif
#ifndef INDOORCAM_MIRROR_CORRECTION
#define INDOORCAM_MIRROR_CORRECTION 0
#endif
static_assert(INDOORCAM_JPEG_QUALITY >= 0 && INDOORCAM_JPEG_QUALITY <= 63, "JPEG quality must be 0..63 (lower is higher quality)");
static_assert(INDOORCAM_ROTATION == 0 || INDOORCAM_ROTATION == 180, "Sensor rotation must be 0 or 180 degrees");
static_assert(INDOORCAM_MIRROR_CORRECTION == 0 || INDOORCAM_MIRROR_CORRECTION == 1, "Mirror correction must be 0 or 1");

// One RTSP TCP reader (the server), JPEG 640x480 @ up to 10fps.
// No internet exposure: firmware RTSP/HTTP are unauthenticated LAN endpoints.
WiFiServer rtsp(8554);
WebServer http(80);
WiFiClient peer;
String requests;
bool playing = false;
uint8_t rtpChannel = 0;
uint16_t sequence = 0;
uint32_t ssrc = 0;
uint64_t lastFrame = 0;
uint32_t lastActivity = 0;
uint32_t framesSent = 0, frameBytes = 0, captureMicros = 0, sendMicros = 0;
uint32_t lastStats = 0;
constexpr size_t SEND_BUFFER_SIZE = 32768;
uint8_t* sendBuffer = nullptr;
constexpr const char* SESSION = "5343414d";

bool writeAll(const uint8_t* data, size_t size) {
  uint32_t started = millis();
  while (size && peer.connected() && millis()-started < 1000) {
    size_t written = peer.write(data, size);
    data += written;
    size -= written;
    if (!written) delay(1);
  }
  return size == 0;
}

void reply(const String& cseq, const String& headers = "", const String& body = "", int status = 200) {
  String message = "RTSP/1.0 " + String(status) + (status == 200 ? " OK\r\n" : " Error\r\n");
  message += "CSeq: " + cseq + "\r\nServer: IndoorCam/1\r\n" + headers;
  if (body.length()) message += "Content-Length: " + String(body.length()) + "\r\n";
  message += "\r\n" + body;
  if (!writeAll(reinterpret_cast<const uint8_t*>(message.c_str()), message.length())) peer.stop();
}

String header(const String& req, const String& name) {
  String lower = req;
  lower.toLowerCase();
  String key = "\r\n" + name + ":";
  key.toLowerCase();
  int at = lower.indexOf(key);
  if (at < 0) return "";
  int end = req.indexOf("\r\n", at+2);
  String value = req.substring(at+key.length(), end);
  value.trim();
  return value;
}

void handleRequest(const String& request) {
  String cseq = header(request, "CSeq");
  String method = request.substring(0, request.indexOf(' '));
  if (method == "OPTIONS") {
    reply(cseq, "Public: OPTIONS, DESCRIBE, SETUP, PLAY, GET_PARAMETER, TEARDOWN\r\n");
  } else if (method == "DESCRIBE") {
    String base = "rtsp://" + WiFi.localIP().toString() + ":8554/camera/";
    String sdp = "v=0\r\no=- 1 1 IN IP4 " + WiFi.localIP().toString() +
      "\r\ns=ESP32 Indoor Camera\r\nc=IN IP4 0.0.0.0\r\nt=0 0\r\na=control:*\r\n"
      "m=video 0 RTP/AVP 26\r\na=rtpmap:26 JPEG/90000\r\na=control:trackID=0\r\na=framerate:10\r\n";
    reply(cseq, "Content-Type: application/sdp\r\nContent-Base: " + base + "\r\n", sdp);
  } else if (method == "SETUP") {
    String transport = header(request, "Transport");
    if (transport.indexOf("RTP/AVP/TCP") < 0) { reply(cseq, "", "", 461); return; }
    int at = transport.indexOf("interleaved=");
    rtpChannel = at >= 0 ? transport.substring(at+12).toInt() : 0;
    reply(cseq, "Transport: RTP/AVP/TCP;unicast;interleaved=" + String(rtpChannel) + "-" + String(rtpChannel+1) +
                "\r\nSession: " + SESSION + ";timeout=60\r\n");
  } else if (method == "PLAY") {
    reply(cseq, "Session: " + String(SESSION) + "\r\nRange: npt=0.000-\r\n");
    playing = true;
  } else if (method == "GET_PARAMETER") {
    reply(cseq, "Session: " + String(SESSION) + "\r\n");
  } else if (method == "TEARDOWN") {
    reply(cseq);
    playing = false;
    peer.stop();
  } else {
    reply(cseq, "", "", 405);
  }
}

void sendFrame() {
  uint64_t captureStart = esp_timer_get_time();
  camera_fb_t* fb = esp_camera_fb_get();
  captureMicros += uint32_t(esp_timer_get_time()-captureStart);
  if (!fb) return;
  JpegScan jpeg;
  if (!parseJpeg(fb->buf, fb->len, jpeg)) {
    Serial.println("Unsupported JPEG: require baseline YUV422/420, no restart markers");
    esp_camera_fb_return(fb);
    return;
  }
  uint32_t timestamp = uint32_t(esp_timer_get_time()*9/100);
  uint64_t sendStart = esp_timer_get_time();
  size_t offset = 0;
  size_t buffered = 0;
  while (offset < jpeg.size && peer.connected()) {
    // Batch complete interleaved RTP packets to reduce socket write calls.
    size_t length = jpegPacket(jpeg, offset, sequence, timestamp, ssrc, rtpChannel,
                               sendBuffer+buffered, 1404);
    if (!length) { peer.stop(); break; }
    buffered += length;
    if (offset == jpeg.size || SEND_BUFFER_SIZE-buffered < 1404) {
      if (!writeAll(sendBuffer, buffered)) { peer.stop(); break; }
      buffered = 0;
    }
  }
  sendMicros += uint32_t(esp_timer_get_time()-sendStart);
  if (offset == jpeg.size && !buffered) { ++framesSent; frameBytes += fb->len; }
  if (millis()-lastStats >= 10000) {
    Serial.printf("RTSP stats: frames=%u jpeg_bytes=%u capture_ms=%u send_ms=%u elapsed_ms=%u rssi=%d\n",
      framesSent, frameBytes, captureMicros/1000, sendMicros/1000, millis()-lastStats, WiFi.RSSI());
    framesSent = frameBytes = captureMicros = sendMicros = 0;
    lastStats = millis();
  }
  esp_camera_fb_return(fb);
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.printf("Chip: %s rev %u, flash: %u bytes, PSRAM: %u bytes\n", ESP.getChipModel(),
                ESP.getChipRevision(), ESP.getFlashChipSize(), ESP.getPsramSize());
  if (!CAMERA_PINS_CONFIRMED) {
    Serial.println("STOP: verify camera pinout and set INDOORCAM_PINS_CONFIRMED=1");
    while (true) delay(1000);
  }
  if (!psramFound()) {
    Serial.println("STOP: PSRAM unavailable; N16R8 requires OPI PSRAM");
    while (true) delay(1000);
  }
  sendBuffer = static_cast<uint8_t*>(heap_caps_malloc(SEND_BUFFER_SIZE, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
  if (!sendBuffer) { Serial.println("STOP: RTSP send buffer allocation failed"); while (true) delay(1000); }
  camera_config_t c = {};
  c.pin_pwdn=CAM_PWDN; c.pin_reset=CAM_RESET; c.pin_xclk=CAM_XCLK;
  c.pin_sccb_sda=CAM_SIOD; c.pin_sccb_scl=CAM_SIOC;
  c.pin_d0=CAM_D0; c.pin_d1=CAM_D1; c.pin_d2=CAM_D2; c.pin_d3=CAM_D3;
  c.pin_d4=CAM_D4; c.pin_d5=CAM_D5; c.pin_d6=CAM_D6; c.pin_d7=CAM_D7;
  c.pin_vsync=CAM_VSYNC; c.pin_href=CAM_HREF; c.pin_pclk=CAM_PCLK;
  c.xclk_freq_hz=INDOORCAM_XCLK_HZ; c.ledc_timer=LEDC_TIMER_0; c.ledc_channel=LEDC_CHANNEL_0;
  c.pixel_format=PIXFORMAT_JPEG; c.frame_size=FRAMESIZE_VGA; c.jpeg_quality=INDOORCAM_JPEG_QUALITY;
  // On this board continuous two-buffer capture greatly reduced Wi-Fi throughput.
  c.fb_count=1; c.fb_location=CAMERA_FB_IN_DRAM; c.grab_mode=CAMERA_GRAB_WHEN_EMPTY;
  esp_err_t err = esp_camera_init(&c);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed 0x%x; check pinout, cable direction and power\n", err);
    while (true) delay(1000);
  }
  sensor_t* sensor = esp_camera_sensor_get();
  // Apply before JPEG encoding so snapshots, RTSP, inference and HA all agree.
  // Preserve driver defaults, then rotate and optionally undo horizontal mirroring.
  int mirror = sensor->status.hmirror ^ (INDOORCAM_ROTATION == 180) ^ INDOORCAM_MIRROR_CORRECTION;
  int flip = sensor->status.vflip ^ (INDOORCAM_ROTATION == 180);
  if (sensor->set_hmirror(sensor, mirror) != 0 || sensor->set_vflip(sensor, flip) != 0) {
    Serial.println("STOP: camera orientation failed");
    while (true) delay(1000);
  }
  // Drop the frame that may have been captured before the register changes.
  camera_fb_t* initial = esp_camera_fb_get();
  if (initial) esp_camera_fb_return(initial);
  Serial.printf("Camera rotation: %u degrees, mirror correction: %u, hmirror: %d, vflip: %d\n",
                INDOORCAM_ROTATION, INDOORCAM_MIRROR_CORRECTION, mirror, flip);
  Serial.printf("Sensor PID: 0x%04x, PSRAM: %u bytes\n", sensor->id.PID, ESP.getPsramSize());
  Serial.printf("Internal heap after camera init: %u bytes\n",heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
  indoorcam::connectWiFi(WIFI_SSID, WIFI_PASSWORD);
  ssrc = esp_random();
  rtsp.begin();
  http.on("/health", [] {
    http.send(200, "application/json", "{\"status\":\"ok\",\"rssi\":" + String(WiFi.RSSI()) +
      ",\"psram\":" + String(ESP.getPsramSize()) + ",\"sensor_pid\":" + String(esp_camera_sensor_get()->id.PID) +
      ",\"rotation\":" + String(INDOORCAM_ROTATION) +
      ",\"mirror_correction\":" + String(INDOORCAM_MIRROR_CORRECTION) +
      ",\"hmirror\":" + String(esp_camera_sensor_get()->status.hmirror) +
      ",\"vflip\":" + String(esp_camera_sensor_get()->status.vflip) + "}");
  });
  http.on("/snapshot.jpg", [] {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) { http.send(503, "text/plain", "Camera unavailable"); return; }
    http.sendHeader("Cache-Control", "no-store");
    http.setContentLength(fb->len);
    http.send(200, "image/jpeg", "");
    WiFiClient client = http.client();
    client.write(fb->buf, fb->len);
    esp_camera_fb_return(fb);
  });
  http.begin();
  Serial.printf("RTSP TCP: rtsp://%s:8554/camera\n", WiFi.localIP().toString().c_str());
}

void loop() {
  http.handleClient();
  if (!peer.connected()) {
    playing = false;
    peer = rtsp.available();
    if (peer) {
      peer.setNoDelay(false);
      peer.setTimeout(1);
      requests = "";
      lastActivity = millis();
    }
  }
  while (peer.connected() && peer.available()) {
    // Consume interleaved RTCP separately; never interpret it as RTSP headers.
    if (requests.isEmpty() && peer.peek() == '$') {
      uint8_t head[4];
      if (peer.readBytes(head, 4) != 4) { peer.stop(); break; }
      size_t remaining = (size_t(head[2]) << 8) | head[3];
      uint8_t discard[128];
      while (remaining) {
        size_t n = peer.readBytes(discard, min(remaining, sizeof(discard)));
        if (!n) { peer.stop(); break; }
        remaining -= n;
      }
      continue;
    }
    requests += char(peer.read());
    lastActivity = millis();
    if (requests.length() > 4096) { peer.stop(); break; }
    if (requests.endsWith("\r\n\r\n")) {
      // No RTSP methods supported here carry a request body.
      if (header(requests, "Content-Length").toInt() > 0) { reply(header(requests,"CSeq"),"","",400); peer.stop(); }
      else handleRequest(requests);
      requests = "";
    }
  }
  if (peer.connected() && millis()-lastActivity > 65000) peer.stop();
  uint64_t now = esp_timer_get_time();
  if (playing && peer.connected() && now-lastFrame >= 100000) {
    lastFrame = now;
    sendFrame();
  }
  delay(1);
}
