/*
 * Mecanum motor command interpreter
 * Board: Keyestudio MAX (Arduino Uno-compatible, ATmega328P).
 * 4x DC motors via 2x L298N.  Serial @ 115200 baud over USB (CP2102).
 *
 * Commands (each newline-terminated):
 *   V vx vy w        velocities, each -1.0..1.0  (vy=forward, vx=strafe-right, w=turn-right)
 *   M fl fr rl rr    direct per-wheel -1.0..1.0  (for calibration/testing)
 *   S                stop
 *
 * Safety: motors auto-stop if no valid command arrives within WATCHDOG_MS.
 *
 * This board's onboard demo peripherals (buttons/LEDs/buzzer on D2,D3,D4,D8,D9,
 * D10,D13) must be switched OFF via the board's onboard DIP switch before
 * wiring these pins to the L298Ns (confirmed silent via a pin-scan test).
 *
 * CALIBRATION (do this once, after wiring):
 *   1) Send `M 0.4 0 0 0`, `M 0 0.4 0 0`, ... to spin each wheel; flip the matching
 *      INVERT[] entry until a POSITIVE value drives that wheel "forward".
 *   2) Send `V 0 0.4 0` (fwd), `V 0.4 0 0` (strafe right), `V 0 0 0.4` (turn right);
 *      if strafe or turn go the wrong way, adjust the vx / w signs in applyVelocity().
 */

// ---------- CONFIG: match to your wiring ----------
const unsigned long WATCHDOG_MS = 300;

// motor index: 0 = front-left, 1 = front-right, 2 = rear-left, 3 = rear-right
// each motor uses: EN (PWM speed) + INA/INB (direction) on its L298N channel
// Compact rewire (2026-07-20), wheels identified via ./testwheel and reordered to
// FL/FR/RL/RR. Enables on PWM pins (Uno PWM = 3,5,6,9,10,11). FL+FR were wired with
// reversed polarity (spun backward at +command) -> INVERT flips them.
//        index:       0(FL)   1(FR)   2(RL)   3(RR)
const uint8_t EN[4]  = {  9,    11,     6,     3  };  // ENA2, ENB2, ENB1, ENA1
const uint8_t INA[4] = {  8,    13,     5,     2  };
const uint8_t INB[4] = { 10,    12,     7,     4  };
bool INVERT[4] = { true, true, false, false };  // FL,FR reversed-polarity -> flip
// --------------------------------------------------

unsigned long lastCmd = 0;
char buf[48];
uint8_t blen = 0;

void setMotor(uint8_t i, float v) {
  if (INVERT[i]) v = -v;
  if (v >  1) v =  1;
  if (v < -1) v = -1;
  bool fwd = (v >= 0);
  digitalWrite(INA[i], fwd ? HIGH : LOW);
  digitalWrite(INB[i], fwd ? LOW  : HIGH);
  analogWrite(EN[i], (int)(fabs(v) * 255.0 + 0.5));
}

void stopAll() {
  for (uint8_t i = 0; i < 4; i++) {
    analogWrite(EN[i], 0);
    digitalWrite(INA[i], LOW);
    digitalWrite(INB[i], LOW);
  }
}

void applyVelocity(float vx, float vy, float w) {
  // mecanum inverse kinematics (X roller layout)
  float m[4];
  m[0] = vy + vx + w;   // front-left
  m[1] = vy - vx - w;   // front-right
  m[2] = vy - vx + w;   // rear-left
  m[3] = vy + vx - w;   // rear-right
  // scale down together if any wheel would exceed 1.0 (keeps the heading true)
  float mx = 1.0;
  for (uint8_t i = 0; i < 4; i++) { float a = fabs(m[i]); if (a > mx) mx = a; }
  Serial.print("MIX "); Serial.print(m[0]/mx,2); Serial.print(" "); Serial.print(m[1]/mx,2); Serial.print(" "); Serial.print(m[2]/mx,2); Serial.print(" "); Serial.println(m[3]/mx,2);
  for (uint8_t i = 0; i < 4; i++) setMotor(i, m[i] / mx);
}

void handle(char* s) {
  if (s[0] == 'S' || s[0] == 's') { stopAll(); lastCmd = millis(); return; }
  if (s[0] == 'V' || s[0] == 'v') {
    char* a = strtok(s + 1, " ");
    char* b = strtok(NULL, " ");
    char* c = strtok(NULL, " ");
    if (a && b && c) { applyVelocity(atof(a), atof(b), atof(c)); lastCmd = millis(); }
    return;
  }
  if (s[0] == 'M' || s[0] == 'm') {
    float v[4]; uint8_t n = 0;
    for (char* p = strtok(s + 1, " "); p && n < 4; p = strtok(NULL, " ")) v[n++] = atof(p);
    if (n == 4) { for (uint8_t i = 0; i < 4; i++) setMotor(i, v[i]); lastCmd = millis(); }
    return;
  }
}

void setup() {
  for (uint8_t i = 0; i < 4; i++) {
    pinMode(EN[i], OUTPUT);
    pinMode(INA[i], OUTPUT);
    pinMode(INB[i], OUTPUT);
  }
  stopAll();
  Serial.begin(115200);
  Serial.println("BOOT mecanum-uno ready");
  lastCmd = millis();
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (blen) { buf[blen] = 0; handle(buf); blen = 0; }
    } else if (blen < sizeof(buf) - 1) {
      buf[blen++] = c;
    }
  }
  if (millis() - lastCmd > WATCHDOG_MS) stopAll();   // failsafe: stop if commands stop
}
