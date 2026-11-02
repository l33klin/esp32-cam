#pragma once
// Confirmed against the user's 鹿小班 ESP32-S3-CAM connector diagram, 2026-10-31.
// The diagram explicitly specifies CAMERA_MODEL_ESP32S3_EYE.
// PWDN is pulled to ground and camera reset is tied to ESP_RST (no GPIO).
#ifndef INDOORCAM_PINS_CONFIRMED
#define INDOORCAM_PINS_CONFIRMED 1
#endif
constexpr bool CAMERA_PINS_CONFIRMED = INDOORCAM_PINS_CONFIRMED;
constexpr int CAM_PWDN = -1;
constexpr int CAM_RESET = -1;
constexpr int CAM_XCLK = 15;
constexpr int CAM_SIOD = 4;
constexpr int CAM_SIOC = 5;
constexpr int CAM_D0 = 11; // sensor Y2, not Y0
constexpr int CAM_D1 = 9;
constexpr int CAM_D2 = 8;
constexpr int CAM_D3 = 10;
constexpr int CAM_D4 = 12;
constexpr int CAM_D5 = 18;
constexpr int CAM_D6 = 17;
constexpr int CAM_D7 = 16; // sensor Y9
constexpr int CAM_VSYNC = 6;
constexpr int CAM_HREF = 7;
constexpr int CAM_PCLK = 13;
