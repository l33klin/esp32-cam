#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include "wifi_setup.h"
#if __has_include("secrets.h")
#include "secrets.h"
#else
#include "secrets.example.h"
#endif

// Isolate Wi-Fi/USB power/AP performance from camera clock and DMA activity.
WebServer diagnostic(80);
static uint8_t block[4096];

void setup() {
  Serial.begin(115200);
  delay(1500);
  indoorcam::connectWiFi(WIFI_SSID, WIFI_PASSWORD);
  for (size_t i=0; i<sizeof(block); ++i) block[i] = uint8_t(i);
  diagnostic.on("/health", [] { diagnostic.send(200,"application/json","{\"mode\":\"wifi_diagnostic_no_camera\"}"); });
  diagnostic.on("/benchmark.bin", [] {
    constexpr size_t total = sizeof(block)*64;
    diagnostic.sendHeader("Cache-Control", "no-store");
    diagnostic.setContentLength(total);
    diagnostic.send(200, "application/octet-stream", "");
    WiFiClient client = diagnostic.client();
    size_t sent = 0;
    uint32_t start = millis();
    while (sent < total && client.connected() && millis()-start < 30000) {
      size_t n = client.write(block, min(sizeof(block),total-sent));
      sent += n;
      if (!n) delay(1);
    }
    Serial.printf("Wi-Fi benchmark: bytes=%u elapsed_ms=%u rssi=%d\n", sent, millis()-start, WiFi.RSSI());
    if (sent != total) client.stop();
  });
  diagnostic.begin();
  Serial.printf("WIFI_DIAGNOSTIC_URL=http://%s/benchmark.bin\n",WiFi.localIP().toString().c_str());
}

void loop() { diagnostic.handleClient(); delay(1); }
