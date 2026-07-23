#!/usr/bin/env python3
"""debug_web.py — a tiny zero-dependency LAN debug dashboard for the robot.

Serves a live view (open http://<jetson-ip>:8099 in a browser):
  - the camera frame (frame.jpg),
  - live nav_state (per-column nearness bars, clearest, loom, motion, target),
  - armed/disarmed + nav mode,
  - streaming tails of the voice / nav / motor / perception logs,
  - a text chat to the bot (for when talking to it isn't practical).
Controls it exposes: the ARM/DISARM kill switch (POST /disarm creates the `.disarmed`
file, POST /arm removes it; the motor daemon enforces it every loop), and POST /chat,
which forwards a typed message to the voice sidecar's live session over a unix socket.
It still never drives the motors directly. DISARM is one-tap (safe direction) and Esc
is an emergency stop; ARM asks for confirmation first. Stdlib only (http.server)."""
import os
import json
import time
import socket
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HOME = os.path.expanduser("~")
ROBOT = os.path.join(HOME, "robot")
WORKSPACE = os.path.join(ROBOT, "workspace")
NAV_STATE = os.path.join(WORKSPACE, "nav_state.json")
FRAME = os.path.join(WORKSPACE, "frame.jpg")
DISARM = os.path.join(WORKSPACE, ".disarmed")
ULTRA = os.path.join(WORKSPACE, "ultrasonic.json")   # forward distance, published by motor daemon
MODE = os.path.join(WORKSPACE, "nav_mode")
LOGS = {"voice": os.path.join(ROBOT, "realtime.log"),
        "nav": os.path.join(ROBOT, "nav.log"),
        "motor": os.path.join(ROBOT, "motor.log"),
        "perception": os.path.join(ROBOT, "perception.log")}
PORT = int(os.environ.get("DEBUG_WEB_PORT", "8099"))
CHAT_SOCK = os.path.join(WORKSPACE, "chat.sock")   # typed messages -> voice sidecar's live session
_chatsock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)

# voice runs as a systemd --user service; the dashboard starts/stops it on request.
VOICE_UNIT = "robocar-voice"
_UID = os.getuid()
SYSTEMD_ENV = {**os.environ,
               "XDG_RUNTIME_DIR": "/run/user/%d" % _UID,
               "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/%d/bus" % _UID}
_voice_cache = {"t": 0.0, "on": False}             # cache is-active; don't spawn systemctl every poll


def voice_on():
    """Whether the voice service is active. Cached ~1.5s so the 300ms /state poll stays cheap."""
    now = time.time()
    if now - _voice_cache["t"] < 1.5:
        return _voice_cache["on"]
    on = False
    try:
        r = subprocess.run(["systemctl", "--user", "is-active", VOICE_UNIT],
                           env=SYSTEMD_ENV, capture_output=True, text=True, timeout=4)
        on = (r.stdout.strip() == "active")
    except Exception:
        on = False
    _voice_cache.update(t=now, on=on)
    return on

PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>robocar debug</title>
<style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:#0d1117;color:#c9d1d9;font:13px/1.4 ui-monospace,Menlo,Consolas,monospace}
header{padding:8px 12px;background:#161b22;border-bottom:1px solid #30363d;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
h1{font-size:14px;margin:0;font-weight:600}
.pill{padding:2px 8px;border-radius:10px;font-weight:600}
.armed{background:#5a1e1e;color:#ff9a9a}.disarmed{background:#1e3a2a;color:#7ee2a8}
.mode{background:#1f2a44;color:#9ab7ff}
.voiceon{background:#2d2148;color:#c9a3ff}.voiceoff{background:#21262d;color:#8b949e}
.sonar{font-size:30px;font-weight:700;line-height:1.1}
.btn{padding:3px 14px;border-radius:8px;border:1px solid #30363d;background:#21262d;color:#c9d1d9;cursor:pointer;font:inherit;font-weight:700}
.btn.stop{background:#f85149;border-color:#f85149;color:#fff}
.btn.go{background:#238636;border-color:#238636;color:#fff}
.hint{color:#6e7681;font-size:11px}
.wrap{display:flex;gap:12px;padding:12px;flex-wrap:wrap}
.col{flex:1;min-width:320px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px;margin-bottom:12px}
.card h2{font-size:12px;margin:0 0 8px;color:#8b949e;text-transform:uppercase;letter-spacing:.05em}
img{width:100%;border-radius:6px;display:block;background:#000}
.bars{display:flex;gap:8px;align-items:flex-end;height:120px}
.bar{flex:1;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;gap:4px}
.bar .fill{width:100%;border-radius:4px 4px 0 0;transition:height .15s,background .15s}
.bar .lab{color:#8b949e}
.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 12px}
.kv b{color:#8b949e;font-weight:500}
.tabs{display:flex;gap:4px;margin-bottom:6px;flex-wrap:wrap}
.tab{padding:3px 10px;border-radius:6px;background:#21262d;cursor:pointer;border:1px solid #30363d}
.tab.on{background:#1f6feb;border-color:#1f6feb;color:#fff}
pre{background:#010409;border:1px solid #30363d;border-radius:6px;padding:8px;height:340px;overflow:auto;margin:0;white-space:pre-wrap;word-break:break-word;font-size:12px}
.blk{color:#ff9a9a}.fwd{color:#7ee2a8}
.chatlog{height:220px;overflow:auto;display:flex;flex-direction:column;gap:6px;margin-bottom:8px;padding:8px;background:#010409;border:1px solid #30363d;border-radius:6px}
.msg{max-width:85%;padding:5px 9px;border-radius:10px;white-space:pre-wrap;word-break:break-word}
.msg.you{align-self:flex-end;background:#1f6feb;color:#fff;border-bottom-right-radius:3px}
.msg.car{align-self:flex-start;background:#21262d;color:#c9d1d9;border-bottom-left-radius:3px}
.msg.sys{align-self:center;background:transparent;color:#6e7681;font-size:11px}
#chatform{display:flex;gap:6px}
#chatinput{flex:1;background:#0d1117;border:1px solid #30363d;border-radius:6px;color:#c9d1d9;padding:7px 9px;font:inherit}
#chatform button{background:#238636;border:none;border-radius:6px;color:#fff;padding:7px 16px;cursor:pointer;font:inherit;font-weight:700}
</style></head><body>
<header>
  <h1>robocar debug</h1>
  <span id=arm class=pill>…</span>
  <button id=armBtn class=btn>…</button>
  <span class=hint>Esc = e-stop</span>
  <span id=voice class=pill>…</span>
  <button id=voiceBtn class=btn>…</button>
  <span id=mode class="pill mode">…</span>
  <span id=fps style=color:#8b949e>–</span>
  <span id=conn style=color:#8b949e>connecting…</span>
</header>
<div class=wrap>
  <div class=col>
    <div class=card><h2>camera</h2><img id=cam src="/frame.jpg" alt="camera"></div>
  </div>
  <div class=col>
    <div class=card><h2>nearness (higher = closer; red = blocked)</h2>
      <div class=bars>
        <div class=bar><div class=fill id=bL></div><div class=lab>L <span id=nL>–</span></div></div>
        <div class=bar><div class=fill id=bC></div><div class=lab>C <span id=nC>–</span></div></div>
        <div class=bar><div class=fill id=bR></div><div class=lab>R <span id=nR>–</span></div></div>
      </div>
    </div>
    <div class=card><h2>state</h2>
      <div class=kv>
        <b>clearest</b><span id=clearest>–</span>
        <b>loom</b><span id=loom>–</span>
        <b>motion</b><span id=motion>–</span>
        <b>target</b><span id=target>–</span>
        <b>stop_near</b><span>680</span>
      </div>
    </div>
    <div class=card><h2>distance ahead — ultrasonic (front)</h2>
      <div class=sonar id=sonar>–</div>
      <div class=hint id=sonarsub>waiting…</div>
    </div>
    <div class=card><h2>chat (type to the bot)</h2>
      <div id=chatlog class=chatlog></div>
      <form id=chatform>
        <input id=chatinput autocomplete=off placeholder="type a message to the bot…">
        <button type=submit>send</button>
      </form>
    </div>
  </div>
</div>
<div style=padding:0-12px><div class=card style=margin:0-12px-12px>
  <div class=tabs>
    <div class=tab data-f=voice>voice</div><div class=tab data-f=nav>nav</div>
    <div class=tab data-f=motor>motor</div><div class=tab data-f=perception>perception</div>
  </div>
  <pre id=log></pre>
</div></div>
<script>
var cur="voice", pos={voice:0,nav:0,motor:0,perception:0}, atBottom=true, disarmed=true, voiceOn=false;
var logEl=document.getElementById("log");
function setArm(which){
  fetch("/"+which,{method:"POST"}).then(r=>r.json()).then(function(d){disarmed=!!d.disarmed;poll();}).catch(function(){});
}
document.getElementById("armBtn").onclick=function(){
  if(disarmed){ if(!confirm("ARM the robot? It will be able to move.")) return; setArm("arm"); }
  else { setArm("disarm"); }
};
function setVoice(on){
  fetch("/voice",{method:"POST",headers:{"Content-Type":"text/plain"},body:on?"on":"off"})
    .then(r=>r.json()).then(function(d){voiceOn=(d.voice==="on");poll();}).catch(function(){});
}
document.getElementById("voiceBtn").onclick=function(){
  if(voiceOn){ setVoice(false); }
  else { if(!confirm("Turn the voice on? It opens the mic and the paid realtime API.")) return; setVoice(true); }
};
// Esc = emergency stop (instant disarm), from anywhere on the page
document.addEventListener("keydown",function(e){ if(e.key==="Escape"){ e.preventDefault(); setArm("disarm"); } });
logEl.addEventListener("scroll",function(){atBottom=logEl.scrollHeight-logEl.scrollTop-logEl.clientHeight<40;});
document.querySelectorAll(".tab").forEach(function(t){t.onclick=function(){
  cur=t.dataset.f; document.querySelectorAll(".tab").forEach(x=>x.classList.remove("on"));
  t.classList.add("on"); logEl.textContent=""; pos[cur]=0; atBottom=true; tail();};});
document.querySelector('.tab[data-f=voice]').classList.add("on");
function color(line){var c=line.includes("block:")||line.includes("STUCK")||line.includes("DISARMED")?"blk":(line.includes("-> forward")||line.includes("advancing")||line.includes("Car:")?"fwd":"");return c;}
function tail(){
  fetch("/tail?f="+cur+"&pos="+pos[cur]).then(r=>r.json()).then(function(d){
    pos[cur]=d.pos;
    if(d.data){d.data.split("\\n").forEach(function(ln){if(!ln)return;var s=document.createElement("span");var c=color(ln);if(c)s.className=c;s.textContent=ln+"\\n";logEl.appendChild(s);});
      if(atBottom)logEl.scrollTop=logEl.scrollHeight;}
  }).catch(function(){});
}
function bar(id,v){var mx=1400,h=Math.max(2,Math.min(100,v/mx*100));var e=document.getElementById(id);
  e.style.height=h+"%";e.style.background=v>=680?"#f85149":(v>=560?"#d29922":"#238636");}
function poll(){
  fetch("/state").then(r=>r.json()).then(function(s){
    document.getElementById("conn").textContent="● live";document.getElementById("conn").style.color="#7ee2a8";
    disarmed=!!s.disarmed;
    var a=document.getElementById("arm"),b=document.getElementById("armBtn");
    if(disarmed){a.textContent="DISARMED";a.className="pill disarmed";b.textContent="▶ ARM";b.className="btn go";}
    else{a.textContent="ARMED";a.className="pill armed";b.textContent="◼ DISARM";b.className="btn stop";}
    voiceOn=(s.voice==="on");
    var v=document.getElementById("voice"),vb=document.getElementById("voiceBtn");
    if(voiceOn){v.textContent="VOICE ON";v.className="pill voiceon";vb.textContent="◼ voice off";vb.className="btn stop";}
    else{v.textContent="VOICE OFF";v.className="pill voiceoff";vb.textContent="▶ voice on";vb.className="btn go";}
    // forward ultrasonic distance
    var son=s.sonar,se=document.getElementById("sonar"),ss=document.getElementById("sonarsub");
    if(son){
      var age=(Date.now()/1000)-(son.ts||0);
      if(age>2){se.textContent="—";se.style.color="#6e7681";ss.textContent="stale ("+age.toFixed(0)+"s) — motor daemon feeding?";}
      else if(son.valid){var cm=son.cm;se.textContent=Math.round(cm)+" cm";se.style.color=(cm<25?"#f85149":(cm<50?"#d29922":"#7ee2a8"));ss.textContent="clear ahead: "+(cm/100).toFixed(2)+" m";}
      else{se.textContent="clear";se.style.color="#7ee2a8";ss.textContent="nothing within ~3.4 m";}
    }else{se.textContent="—";se.style.color="#6e7681";ss.textContent="no reading";}
    document.getElementById("mode").textContent="mode: "+(s.mode||"idle");
    var n=s.near;
    if(n){document.getElementById("nL").textContent=Math.round(n.l);document.getElementById("nC").textContent=Math.round(n.c);document.getElementById("nR").textContent=Math.round(n.r);
      bar("bL",n.l);bar("bC",n.c);bar("bR",n.r);
      document.getElementById("clearest").textContent=s.clearest||"–";
      document.getElementById("loom").textContent=Math.round(s.loom||0);
      document.getElementById("motion").textContent=(s.motion||0).toFixed(1);
      document.getElementById("fps").textContent=(s.fps||0).toFixed(0)+" fps";
      var t=s.target||{};document.getElementById("target").textContent=t.active?(t.lost?"lost":("bearing "+(t.bearing||0).toFixed(2))):"none";
    }
  }).catch(function(){document.getElementById("conn").textContent="● no data";document.getElementById("conn").style.color="#f85149";});
}
setInterval(poll,300);setInterval(tail,500);
setInterval(function(){document.getElementById("cam").src="/frame.jpg?t="+Date.now();},300);
// ---- chat: type to the bot; replies (its spoken transcript) show here too ----
var chatLogEl=document.getElementById("chatlog"), chatBottom=true;
chatLogEl.addEventListener("scroll",function(){chatBottom=chatLogEl.scrollHeight-chatLogEl.scrollTop-chatLogEl.clientHeight<40;});
function renderChat(turns){
  chatLogEl.innerHTML="";
  turns.forEach(function(t){var d=document.createElement("div");d.className="msg "+(t.who==="you"?"you":"car");d.textContent=t.text;chatLogEl.appendChild(d);});
  if(chatBottom)chatLogEl.scrollTop=chatLogEl.scrollHeight;
}
function pollChat(){fetch("/chatlog").then(r=>r.json()).then(function(d){if(d.turns)renderChat(d.turns);}).catch(function(){});}
document.getElementById("chatform").addEventListener("submit",function(e){
  e.preventDefault();
  var inp=document.getElementById("chatinput"),text=inp.value.trim();if(!text)return;inp.value="";
  fetch("/chat",{method:"POST",headers:{"Content-Type":"text/plain"},body:text}).then(r=>r.json()).then(function(d){
    if(!d.ok){var s=document.createElement("div");s.className="msg sys";s.textContent="couldn't reach the voice service — is it running?";chatLogEl.appendChild(s);chatLogEl.scrollTop=chatLogEl.scrollHeight;}
    setTimeout(pollChat,300);
  }).catch(function(){});
});
setInterval(pollChat,1200);
poll();tail();pollChat();
</script></body></html>"""


def parse_chat(path, limit=40, tailbytes=16384):
    """Pull the recent You/Car turns out of the voice log for the chat pane. Reads only the
    tail (the log can be large) and dedups adjacent identical lines (the logger writes each
    line twice: once to the file, once to stdout which systemd also appends here)."""
    turns = []
    try:
        size = os.path.getsize(path)
        with open(path, "r", errors="replace") as f:
            if size > tailbytes:
                f.seek(size - tailbytes)
                f.readline()                        # drop the partial first line
            lines = f.readlines()
    except Exception:
        return turns
    for ln in lines:
        for tag, who in (("You (chat): ", "you"), ("You: ", "you"), ("Car: ", "car")):
            i = ln.find(tag)
            if i != -1:
                text = ln[i + len(tag):].strip()
                if text:
                    turns.append({"who": who, "text": text})
                break
    out = []
    for t in turns:
        if not out or out[-1] != t:                 # collapse the double-logged duplicates
            out.append(t)
    return out[-limit:]


class H(BaseHTTPRequestHandler):
    def _send(self, code, ctype, body, nocache=True):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        if nocache:
            self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path == "/" or u.path == "/index.html":
                self._send(200, "text/html; charset=utf-8", PAGE.encode())
            elif u.path == "/state":
                st = {}
                try:
                    with open(NAV_STATE) as f:
                        st = json.load(f)
                except Exception:
                    st = {}
                st["disarmed"] = os.path.exists(DISARM)
                try:
                    with open(MODE) as f:
                        st["mode"] = f.read().strip() or "idle"
                except Exception:
                    st["mode"] = "idle"
                try:
                    with open(ULTRA) as f:
                        st["sonar"] = json.load(f)      # {ts, cm, valid}
                except Exception:
                    st["sonar"] = None
                st["voice"] = "on" if voice_on() else "off"
                self._send(200, "application/json", json.dumps(st).encode())
            elif u.path == "/frame.jpg":
                try:
                    with open(FRAME, "rb") as f:
                        self._send(200, "image/jpeg", f.read())
                except Exception:
                    self._send(404, "text/plain", b"no frame")
            elif u.path == "/tail":
                q = parse_qs(u.query)
                name = (q.get("f", ["nav"])[0])
                pos = int(q.get("pos", ["0"])[0])
                path = LOGS.get(name)
                data = ""
                if path and os.path.exists(path):
                    size = os.path.getsize(path)
                    if pos > size:            # file truncated/rotated -> restart
                        pos = 0
                    with open(path, "r", errors="replace") as f:
                        f.seek(pos)
                        data = f.read(65536)
                        pos = f.tell()
                self._send(200, "application/json", json.dumps({"pos": pos, "data": data}).encode())
            elif u.path == "/chatlog":
                self._send(200, "application/json",
                           json.dumps({"turns": parse_chat(LOGS["voice"])}).encode())
            else:
                self._send(404, "text/plain", b"not found")
        except Exception as e:
            try:
                self._send(500, "text/plain", str(e).encode())
            except Exception:
                pass

    def do_POST(self):
        u = urlparse(self.path)
        try:
            ln = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(ln) if ln else b""   # read body (also drains it for keep-alive)
            if u.path == "/disarm":
                open(DISARM, "w").close()           # create kill switch -> daemon stops within a loop
                self._send(200, "application/json", json.dumps({"disarmed": True}).encode())
            elif u.path == "/arm":
                try:
                    os.remove(DISARM)               # clear kill switch -> daemon may drive again
                except FileNotFoundError:
                    pass
                self._send(200, "application/json", json.dumps({"disarmed": os.path.exists(DISARM)}).encode())
            elif u.path == "/voice":
                want = body.decode("utf-8", "ignore").strip().lower()
                action = "start" if want == "on" else "stop"
                ok = False
                try:
                    r = subprocess.run(["systemctl", "--user", action, VOICE_UNIT],
                                       env=SYSTEMD_ENV, capture_output=True, text=True, timeout=15)
                    ok = (r.returncode == 0)
                except Exception:
                    ok = False
                _voice_cache["t"] = 0.0             # force a fresh is-active read below
                self._send(200, "application/json",
                           json.dumps({"ok": ok, "voice": "on" if voice_on() else "off"}).encode())
            elif u.path == "/chat":
                text = body.decode("utf-8", "ignore").strip()
                ok = False
                if text:
                    try:
                        _chatsock.sendto(text.encode("utf-8"), CHAT_SOCK)   # -> voice sidecar's session
                        ok = True
                    except Exception:
                        ok = False                  # sidecar not running / socket absent
                self._send(200, "application/json", json.dumps({"ok": ok}).encode())
            else:
                self._send(404, "text/plain", b"not found")
        except Exception as e:
            try:
                self._send(500, "text/plain", str(e).encode())
            except Exception:
                pass

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), H)
    print("debug_web: http://0.0.0.0:%d" % PORT, flush=True)
    srv.serve_forever()
