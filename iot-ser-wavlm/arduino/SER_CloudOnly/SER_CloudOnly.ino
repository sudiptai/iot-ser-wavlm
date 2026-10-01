/*
  SER_CloudOnly - Arduino UNO R4 WiFi
  For the Google Colab pipeline: the model runs in Colab, which publishes
  `emotion` and `confidence` to Arduino Cloud through the REST API.
  This sketch only listens to those cloud variables and shows them on the LCD.
  No USB link to a PC is needed - just WiFi and power.

  Sketch folder: SER_CloudOnly.ino, thingProperties.h, arduino_secrets.h
  Libraries: ArduinoIoTCloud, Arduino_ConnectionHandler, Adafruit GFX, Adafruit ST7735
*/
#include "thingProperties.h"
#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7735.h>

// ---- display pins: EDIT to match your wiring (software SPI, any pins work) ----
#define TFT_CS   10
#define TFT_DC    9
#define TFT_RST   8
#define TFT_BL    7
#define TFT_MOSI 11      // DIN
#define TFT_SCLK 13      // CLK

Adafruit_ST7735 tft(TFT_CS, TFT_DC, TFT_MOSI, TFT_SCLK, TFT_RST);

uint16_t colourFor(const String &l) {
  if (l == "HAPPY")     return ST77XX_YELLOW;
  if (l == "ANGRY")     return ST77XX_RED;
  if (l == "SAD")       return ST77XX_CYAN;
  if (l == "FEARFUL")   return ST77XX_MAGENTA;
  if (l == "DISGUST")   return 0x8410;
  if (l == "SURPRISED") return ST77XX_ORANGE;
  if (l == "CALM")      return 0x07FF;
  return ST77XX_GREEN;
}

void header() {
  tft.fillScreen(ST77XX_BLACK);
  tft.fillRect(0, 0, 160, 14, ST77XX_BLUE);
  tft.setTextSize(1); tft.setTextColor(ST77XX_WHITE);
  tft.setCursor(8, 3); tft.print("wav2vec2 SER  (Colab)");
  tft.fillCircle(152, 7, 4, ArduinoCloud.connected() ? ST77XX_GREEN : ST77XX_RED);
}

void showLabel() {
  header();
  uint8_t sz = emotion.length() > 7 ? 2 : 3;
  tft.setTextSize(sz); tft.setTextColor(colourFor(emotion));
  tft.setCursor((160 - emotion.length() * 6 * sz) / 2, 40);
  tft.print(emotion);
  if (confidence > 0) {
    tft.setTextSize(1); tft.setTextColor(ST77XX_WHITE);
    tft.setCursor(44, 84); tft.print("conf: "); tft.print(confidence, 1); tft.print(" %");
  }
}

// Called by the cloud library whenever Colab writes a new value
void onEmotionChange()    { detections = detections + 1; showLabel(); }
void onConfidenceChange() { showLabel(); }

void setup() {
  Serial.begin(115200);
  pinMode(TFT_BL, OUTPUT); digitalWrite(TFT_BL, HIGH);
  tft.initR(INITR_BLACKTAB);          // try INITR_GREENTAB if colours/offset look wrong
  tft.setRotation(1); tft.setTextWrap(false);

  header();
  tft.setTextColor(ST77XX_YELLOW); tft.setCursor(10, 40); tft.print("Connecting WiFi...");

  initProperties();
  ArduinoCloud.begin(ArduinoIoTPreferredConnection);
  setDebugMessageLevel(2);

  uint32_t t0 = millis();
  while (!ArduinoCloud.connected() && millis() - t0 < 30000) ArduinoCloud.update();

  header();
  tft.setTextColor(ST77XX_YELLOW); tft.setCursor(10, 40);
  tft.print(ArduinoCloud.connected() ? "Waiting for Colab..." : "Cloud not connected");
}

void loop() {
  ArduinoCloud.update();
  static bool lastConn = false;
  bool c = ArduinoCloud.connected();
  if (c != lastConn) { lastConn = c; tft.fillCircle(152, 7, 4, c ? ST77XX_GREEN : ST77XX_RED); }
}
