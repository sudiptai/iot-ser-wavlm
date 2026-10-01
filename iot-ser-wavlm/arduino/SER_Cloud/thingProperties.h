// Generated-style thingProperties.h for the "Speech Emotion" Thing.
// If you create the Thing in the Arduino Cloud web editor, it generates this
// file for you - use THAT one and just make sure the three variables below exist
// with the same names and types.
#include <ArduinoIoTCloud.h>
#include <Arduino_ConnectionHandler.h>
#include "arduino_secrets.h"

const char DEVICE_LOGIN_NAME[] = SECRET_DEVICE_ID;   // from Cloud > Devices
const char SSID[]              = SECRET_SSID;
const char PASS[]              = SECRET_OPTIONAL_PASS;
const char DEVICE_KEY[]        = SECRET_DEVICE_KEY;  // shown once when the device is added

// ---- Cloud variables (all READ-ONLY from the app's point of view) ----
String emotion;      // "HAPPY", "ANGRY", ...
float  confidence;   // 0-100 %
int    detections;   // counter

void initProperties() {
  ArduinoCloud.setBoardId(DEVICE_LOGIN_NAME);
  ArduinoCloud.setSecretDeviceKey(DEVICE_KEY);
  ArduinoCloud.addProperty(emotion,    READ, ON_CHANGE, NULL);
  ArduinoCloud.addProperty(confidence, READ, ON_CHANGE, NULL);
  ArduinoCloud.addProperty(detections, READ, ON_CHANGE, NULL);
}

WiFiConnectionHandler ArduinoIoTPreferredConnection(SSID, PASS);
