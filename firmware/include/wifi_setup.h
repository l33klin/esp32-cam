#pragma once
#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <Preferences.h>
#include "esp_wifi.h"

// Override to 0 for access point compatibility checks without 802.11n.
#ifndef INDOORCAM_WIFI_HT
#define INDOORCAM_WIFI_HT 1
#endif

namespace indoorcam {

inline bool validWifiPassword(const String& value) {
  if (value.isEmpty() || (value.length() >= 8 && value.length() <= 63)) return true;
  if (value.length() != 64) return false;
  for (size_t i = 0; i < value.length(); ++i) if (!isxdigit(value[i])) return false;
  return true;
}

inline void connectWiFi(const char* fallbackSsid, const char* fallbackPassword) {
  Preferences prefs;
  prefs.begin("indoorcam", false);
  String ssid = prefs.getString("ssid", "");
  String password = prefs.getString("password", "");
  if (ssid.isEmpty() && fallbackSsid && strlen(fallbackSsid) && !String(fallbackSsid).startsWith("YOUR_")) {
    ssid = fallbackSsid;
    password = fallbackPassword;
  }
  WiFi.mode(WIFI_STA);
  uint8_t protocols = WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G;
#if INDOORCAM_WIFI_HT
  protocols |= WIFI_PROTOCOL_11N;
#endif
  // Set both directions explicitly: the SDK can retain this choice in NVS.
  esp_err_t protocolError = esp_wifi_set_protocol(WIFI_IF_STA, protocols);
  if (protocolError != ESP_OK) Serial.printf("Wi-Fi protocol configuration failed: 0x%x\n", protocolError);
  WiFi.setHostname("indoorcam-s3");
  WiFi.setAutoReconnect(true);
  WiFi.setSleep(false);
  if (!ssid.isEmpty()) {
    Serial.println("Connecting to saved Wi-Fi (credentials are not logged)...");
    WiFi.begin(ssid.c_str(), password.c_str());
    uint32_t start = millis();
    while (WiFi.status() != WL_CONNECTED && millis()-start < 20000) delay(100);
  }
  if (WiFi.status() != WL_CONNECTED) {
    WiFi.mode(WIFI_AP_STA);
    String suffix = WiFi.macAddress(); suffix.replace(":", ""); suffix = suffix.substring(6);
    String apName = "IndoorCam-" + suffix;
    char randomToken[33];
    snprintf(randomToken, sizeof(randomToken), "%08lx%08lx%08lx%08lx", (unsigned long)esp_random(),
             (unsigned long)esp_random(), (unsigned long)esp_random(), (unsigned long)esp_random());
    String token(randomToken), apPassword = token.substring(0, 12);
    if (!WiFi.softAP(apName.c_str(), apPassword.c_str())) {
      Serial.println("Wi-Fi setup AP failed; restarting"); delay(2000); ESP.restart();
    }
    Serial.printf("WIFI_SETUP_SSID=%s\nWIFI_SETUP_PASSWORD=%s\nWIFI_SETUP_URL=http://192.168.4.1/\n",
                  apName.c_str(), apPassword.c_str());
    WebServer portal(80);
    bool requested = false, connecting = false, acknowledged = false;
    uint32_t attemptStart = 0;
    String status = "waiting";
    portal.on("/", HTTP_GET, [&] {
      String html = R"HTML(<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>室内摄像头配网</title>
<style>body{font:17px system-ui;max-width:28rem;margin:8vh auto;padding:24px;background:#f5f5f3;color:#202522}label{display:block;margin:20px 0 6px}input,button{box-sizing:border-box;width:100%;font:inherit;padding:12px;border:1px solid #999;border-radius:6px}button{margin-top:24px;background:#244b3c;color:white;border:0}#status{line-height:1.6}</style>
<h1>室内摄像头配网</h1><p>请输入可连接到推理服务器的 2.4 GHz Wi-Fi。</p>
<form id="wifi"><input type="hidden" name="token" value="__TOKEN__">
<label for="ssid">Wi-Fi 名称</label><input id="ssid" name="ssid" required maxlength="32" autocomplete="off">
<label for="password">Wi-Fi 密码</label><input id="password" name="password" type="password" maxlength="64" autocomplete="new-password">
<button id="submit">连接 Wi-Fi</button></form><p id="status"></p>
<script>const status=document.querySelector('#status'),button=document.querySelector('#submit');
document.querySelector('form').onsubmit=async e=>{e.preventDefault();button.disabled=true;try{const r=await fetch('/wifi',{method:'POST',body:new URLSearchParams(new FormData(e.target))});status.textContent=await r.text();if(!r.ok)button.disabled=false;}catch{status.textContent='连接中，请稍候…';}};
setInterval(async()=>{try{const r=await fetch('/status',{cache:'no-store'}),s=await r.json();if(s.connected){status.textContent='连接成功，设备地址：'+s.ip+'。热点即将关闭，请切换回家庭 Wi-Fi。';button.disabled=true;}else if(s.status==='failed'){status.textContent='连接失败，请检查 Wi-Fi 名称、密码和 2.4 GHz 网络后重试。';button.disabled=false;}}catch{}},1000);</script></html>)HTML";
      html.replace("__TOKEN__", token);
      portal.sendHeader("Cache-Control", "no-store");
      portal.send(200, "text/html; charset=utf-8", html);
    });
    portal.on("/wifi", HTTP_POST, [&] {
      if (portal.arg("token") != token) { portal.send(403, "text/plain; charset=utf-8", "请重新打开配网页面。"); return; }
      String nextSsid = portal.arg("ssid"), nextPassword = portal.arg("password");
      if (nextSsid.isEmpty() || nextSsid.length() > 32 || !validWifiPassword(nextPassword)) {
        portal.send(400, "text/plain; charset=utf-8", "Wi-Fi 名称需为 1–32 字节；密码需为 8–63 字符，开放网络可留空。"); return;
      }
      ssid = nextSsid; password = nextPassword; requested = true;
      portal.sendHeader("Cache-Control", "no-store");
      portal.send(202, "text/plain; charset=utf-8", "正在连接，请等待约 20 秒…");
    });
    portal.on("/status", HTTP_GET, [&] {
      bool connected = WiFi.status() == WL_CONNECTED;
      portal.sendHeader("Cache-Control", "no-store");
      portal.send(200, "application/json", "{\"connected\":" + String(connected ? "true" : "false") +
                  ",\"status\":\"" + status + "\",\"ip\":\"" + WiFi.localIP().toString() + "\"}");
      if (connected) acknowledged = true;
    });
    portal.begin();
    while (WiFi.status() != WL_CONNECTED) {
      portal.handleClient();
      if (requested) {
        requested = false; connecting = true; status = "connecting"; attemptStart = millis();
        WiFi.begin(ssid.c_str(), password.c_str()); WiFi.setSleep(false);
      }
      if (connecting && millis()-attemptStart > 20000) { status = "failed"; connecting = false; }
      delay(2);
    }
    // Let the onboarding page read the LAN IP before the setup AP closes.
    uint32_t joined = millis();
    while (!acknowledged && millis()-joined < 10000) { portal.handleClient(); delay(2); }
    delay(500); // Allow the successful status response to reach the setup client.
    portal.stop(); WiFi.softAPdisconnect(true); WiFi.mode(WIFI_STA); WiFi.setSleep(false);
  }
  if (prefs.getString("ssid", "") != ssid || prefs.getString("password", "") != password) {
    prefs.putString("ssid", ssid); prefs.putString("password", password);
  }
  prefs.end();
  uint8_t protocol = 0;
  wifi_ps_type_t powerSave = WIFI_PS_NONE;
  esp_wifi_get_protocol(WIFI_IF_STA, &protocol);
  esp_wifi_get_ps(&powerSave);
  Serial.printf("Wi-Fi channel=%d protocol=%u power_save=%u tx_power=%d rssi=%d\n",
                WiFi.channel(), protocol, unsigned(powerSave), int(WiFi.getTxPower()), WiFi.RSSI());
  Serial.printf("WIFI_CONNECTED_IP=%s\n", WiFi.localIP().toString().c_str());
}

} // namespace indoorcam
