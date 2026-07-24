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
import glob
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
FAST_FRAME = os.path.join(WORKSPACE, "frame_fast.jpg")   # small high-rate frame -> smooth MJPEG video feed
STREAM_FPS = float(os.environ.get("DEBUG_STREAM_FPS", "15"))
DISARM = os.path.join(WORKSPACE, ".disarmed")
ULTRA = os.path.join(WORKSPACE, "ultrasonic.json")   # forward distance, published by motor daemon
LOCATE = os.path.join(WORKSPACE, "locate.json")      # find_it's last result (the box Claude reported)
MODE = os.path.join(WORKSPACE, "nav_mode")
LOGS = {"voice": os.path.join(ROBOT, "realtime.log"),
        "nav": os.path.join(ROBOT, "nav.log"),
        "motor": os.path.join(ROBOT, "motor.log"),
        "perception": os.path.join(ROBOT, "perception.log")}
PORT = int(os.environ.get("DEBUG_WEB_PORT", "8099"))
CHAT_SOCK = os.path.join(WORKSPACE, "chat.sock")   # typed messages -> voice sidecar's live session
_chatsock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
MOTOR_SOCK = os.path.join(WORKSPACE, "motor.sock") # manual d-pad nudges -> motor daemon (it enforces DISARM)
_motorsock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
BATTERY = os.path.join(WORKSPACE, "battery.json")  # optional pack-voltage telemetry (shown only if present)
NUDGE_SPEED = float(os.environ.get("DEBUG_NUDGE_SPEED", "1.0"))   # manual-drive nudge speed (full: strafing on
                                                                 # mecanum wheels needs more push than driving straight)
NUDGE_SECS = float(os.environ.get("DEBUG_NUDGE_SECS", "0.4"))     # ...and how long each press drives for
# d-pad label -> motor-daemon direction. left/right = strafe (mecanum), cw/ccw = rotate.
DRIVE_DIRS = {"forward": "forward", "back": "back", "cw": "cw", "ccw": "ccw",
              "strafe_left": "left", "strafe_right": "right"}

# The robot's pieces each run as a systemd --user service; the dashboard reports their health and
# can start/stop the voice one. Keys are the short names shown in the health strip.
UNITS = {"motor": "robocar-motor", "perception": "robocar-perception",
         "nav": "robocar-nav", "voice": "robocar-voice", "debug": "robocar-debug"}
VOICE_UNIT = UNITS["voice"]
_UID = os.getuid()
SYSTEMD_ENV = {**os.environ,
               "XDG_RUNTIME_DIR": "/run/user/%d" % _UID,
               "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/%d/bus" % _UID}
_units_cache = {"t": 0.0, "st": {}}                # cache is-active for all units; don't spawn systemctl per poll


def units_active():
    """{short_name: bool} liveness for every service, in ONE systemctl call. Cached ~2s so the
    frequent polls stay cheap."""
    now = time.time()
    if now - _units_cache["t"] < 2.0:
        return _units_cache["st"]
    st = {}
    try:
        r = subprocess.run(["systemctl", "--user", "is-active", *UNITS.values()],
                           env=SYSTEMD_ENV, capture_output=True, text=True, timeout=4)
        lines = r.stdout.strip().split("\n")
        for (name, _unit), line in zip(UNITS.items(), lines):
            st[name] = (line.strip() == "active")
    except Exception:
        st = {}
    _units_cache.update(t=now, st=st)
    return st


def voice_on():
    """Whether the voice service is active (from the shared units cache)."""
    return bool(units_active().get("voice", False))


def _ts_age(path):
    """Seconds since the `ts` INSIDE a json state file (nav_state / ultrasonic carry their own)."""
    try:
        with open(path) as f:
            return round(time.time() - json.load(f).get("ts", 0), 1)
    except Exception:
        return None


def _mtime_age(path):
    """Seconds since a file was last written (used for frame.jpg, which has no internal ts)."""
    try:
        return round(time.time() - os.path.getmtime(path), 1)
    except Exception:
        return None


def read_temp_c():
    """Hottest on-board thermal zone in °C (Jetson exposes several), or None. Flags throttling."""
    hi = None
    for z in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            v = int(open(z).read().strip()) / 1000.0
            if 0 < v < 200 and (hi is None or v > hi):
                hi = v
        except Exception:
            pass
    return round(hi, 1) if hi is not None else None


def read_battery():
    """Optional pack telemetry, e.g. {"v": 11.8}. None unless something is writing battery.json
    (a future firmware voltage divider); the strip just hides the chip when it's absent."""
    try:
        with open(BATTERY) as f:
            return json.load(f)
    except Exception:
        return None


def health():
    """One glance at whether everything's alive: services, sensor freshness, temp, load, battery."""
    try:
        load = round(os.getloadavg()[0], 2)
    except Exception:
        load = None
    return {"units": units_active(),
            "frame_age": _mtime_age(FRAME),
            "nav_age": _ts_age(NAV_STATE),
            "sonar_age": _ts_age(ULTRA),
            "temp_c": read_temp_c(),
            "load": load,
            "battery": read_battery()}

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
.camwrap{position:relative;line-height:0}
.camwrap canvas{position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none}
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
.health{display:flex;gap:6px;padding:6px 12px;background:#0f141b;border-bottom:1px solid #30363d;flex-wrap:wrap;align-items:center}
.chip{padding:2px 8px;border-radius:6px;background:#21262d;color:#8b949e;border:1px solid #30363d;font-size:11px;white-space:nowrap}
.chip.up{color:#7ee2a8;border-color:#1e3a2a}.chip.down{color:#ff9a9a;border-color:#5a1e1e}.chip.warn{color:#e3b341;border-color:#5a4a1e}
.pad{display:flex;flex-direction:column;align-items:center;gap:6px}
.padrow{display:flex;gap:6px}
.dbtn{width:54px;height:40px;font-size:16px;border-radius:8px;border:1px solid #30363d;background:#21262d;color:#c9d1d9;cursor:pointer;font:inherit;-webkit-user-select:none;user-select:none;touch-action:manipulation}
.dbtn:active{background:#1f6feb;color:#fff}
.dbtn.stopb{background:#3a1e1e;border-color:#5a1e1e;color:#ff9a9a}.dbtn.stopb:active{background:#f85149;color:#fff}
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
<div id=health class=health><span class=chip>health…</span></div>
<div class=wrap>
  <div class=col>
    <div class=card><h2>camera <span id=camnote class=hint></span>
      <button id=vidBtn class=btn style="float:right;padding:1px 10px;font-size:11px">⚡ smooth video</button></h2>
      <div class=camwrap><img id=cam src="/frame.jpg" alt="camera"><canvas id=ov></canvas></div>
    </div>
    <div class=card><h2>manual drive (armed only · each press = a short nudge)</h2>
      <div class=pad>
        <button class=dbtn data-d=forward title="forward (↑)">▲</button>
        <div class=padrow>
          <button class=dbtn data-d=strafe_left title="strafe left (←)">◀</button>
          <button class="dbtn stopb" data-d=stop title="stop (space)">■</button>
          <button class=dbtn data-d=strafe_right title="strafe right (→)">▶</button>
        </div>
        <button class=dbtn data-d=back title="back (↓)">▼</button>
        <div class=padrow>
          <button class=dbtn data-d=ccw title="rotate left (A)">↺</button>
          <button class=dbtn data-d=cw title="rotate right (D)">↻</button>
        </div>
      </div>
      <div class=hint>arrows = move · A / D = rotate · space = stop · Esc = e-stop</div>
    </div>
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
var cur="voice", pos={voice:0,nav:0,motor:0,perception:0}, atBottom=true, disarmed=true, voiceOn=false, streaming=false;
var logEl=document.getElementById("log");
function setArm(which){
  fetch("/"+which,{method:"POST"}).then(r=>r.json()).then(function(d){disarmed=!!d.disarmed;poll();}).catch(function(){});
}
document.getElementById("armBtn").onclick=function(){ setArm(disarmed?"arm":"disarm"); };
function setVoice(on){
  fetch("/voice",{method:"POST",headers:{"Content-Type":"text/plain"},body:on?"on":"off"})
    .then(r=>r.json()).then(function(d){voiceOn=(d.voice==="on");poll();}).catch(function(){});
}
document.getElementById("voiceBtn").onclick=function(){ setVoice(!voiceOn); };
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
    drawOverlay(s);
  }).catch(function(){document.getElementById("conn").textContent="● no data";document.getElementById("conn").style.color="#f85149";});
}
// ---- bounding-box overlay on the camera: green = live tracker lock, cyan = what find_it/Claude just saw ----
var LOCATE_SHOW_SECS=8;   // the Claude box is only valid for its capture frame; hide once it's stale
function boxOn(ctx,cw,ch,b,color,label){
  var x=b[0]*cw,y=b[1]*ch,w=b[2]*cw,h=b[3]*ch;
  ctx.lineWidth=2;ctx.strokeStyle=color;ctx.strokeRect(x,y,w,h);
  if(label){ctx.font="12px ui-monospace,monospace";var tw=ctx.measureText(label).width+8;
    var ly=y>16?y-16:y+2;ctx.fillStyle=color;ctx.fillRect(x,ly,tw,15);
    ctx.fillStyle="#0d1117";ctx.fillText(label,x+4,ly+11);}
}
function drawOverlay(s){
  var cam=document.getElementById("cam"),ov=document.getElementById("ov"),note=document.getElementById("camnote");
  var cw=cam.clientWidth,ch=cam.clientHeight;if(!cw||!ch)return;
  if(ov.width!==cw)ov.width=cw;if(ov.height!==ch)ov.height=ch;
  var ctx=ov.getContext("2d");ctx.clearRect(0,0,cw,ch);
  var msg="";
  // live tracker lock (green) — follows the object frame to frame
  var t=s.target||{};
  if(t.active&&!t.lost&&t.box){boxOn(ctx,cw,ch,t.box,"#3fb950","lock");}
  // last find_it result (cyan) — the box Claude reported, only fresh for a few seconds
  var lc=s.locate;
  if(lc){var age=(Date.now()/1000)-(lc.ts||0);
    if(age<LOCATE_SHOW_SECS){
      if(lc.found&&lc.box){boxOn(ctx,cw,ch,lc.box,"#39d0d8",(lc.thing||"target")+(lc.locked?" ✓lock":" seen"));}
      else if(!lc.found){msg="looked for “"+(lc.thing||"?")+"” — not seen";}
    }
  }
  note.textContent=msg;note.style.color="#e3b341";
}
setInterval(poll,300);setInterval(tail,500);
setInterval(function(){if(!streaming)document.getElementById("cam").src="/frame.jpg?t="+Date.now();},300);
// smooth video toggle: swap the still-frame poll for the MJPEG stream (for easier remote driving)
document.getElementById("vidBtn").onclick=function(){
  streaming=!streaming;var cam=document.getElementById("cam");
  if(streaming){cam.src="/stream.mjpg";this.textContent="◼ stop video";this.classList.add("stop");}
  else{this.textContent="⚡ smooth video";this.classList.remove("stop");cam.src="/frame.jpg?t="+Date.now();}
};
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
// ---- manual drive: d-pad buttons + arrow/WASD keys -> short nudges (daemon auto-stops + enforces arm) ----
function drive(d){fetch("/drive",{method:"POST",headers:{"Content-Type":"text/plain"},body:d}).catch(function(){});}
document.querySelectorAll(".dbtn").forEach(function(b){b.onclick=function(){drive(b.dataset.d);};});
document.addEventListener("keydown",function(e){
  if(e.target&&(e.target.tagName==="INPUT"||e.target.tagName==="TEXTAREA"))return;   // don't hijack the chat box
  if(e.key===" "){e.preventDefault();drive("stop");return;}
  var m={ArrowUp:"forward",ArrowDown:"back",ArrowLeft:"strafe_left",ArrowRight:"strafe_right",
         a:"ccw",A:"ccw",d:"cw",D:"cw"};
  if(m[e.key]){e.preventDefault();drive(m[e.key]);}
});
// ---- health strip: services up/down, sensor freshness, temp, load, (optional) battery ----
function chip(host,label,cls){var s=document.createElement("span");s.className="chip "+(cls||"");s.textContent=label;host.appendChild(s);}
function ageChip(host,name,age,warn,bad){
  if(age===null||age===undefined){chip(host,name+" —");return;}
  chip(host,name+" "+age+"s",age>bad?"down":(age>warn?"warn":"up"));}
function pollHealth(){fetch("/health").then(r=>r.json()).then(function(h){
  var el=document.getElementById("health");el.innerHTML="";
  var u=h.units||{};
  ["motor","perception","nav","voice","debug"].forEach(function(k){
    if(u[k]===undefined){chip(el,k+" ?");}else{chip(el,k+(u[k]?" ✓":" ✗"),u[k]?"up":"down");}});
  ageChip(el,"cam",h.frame_age,1,3);ageChip(el,"depth",h.nav_age,1,3);ageChip(el,"sonar",h.sonar_age,2,5);
  if(h.temp_c!==null&&h.temp_c!==undefined)chip(el,h.temp_c+"°C",h.temp_c>80?"down":(h.temp_c>70?"warn":"up"));
  if(h.load!==null&&h.load!==undefined)chip(el,"load "+h.load);
  if(h.battery&&h.battery.v!==undefined&&h.battery.v!==null)
    chip(el,(+h.battery.v).toFixed(1)+"V",h.battery.v<10.5?"down":(h.battery.v<11.2?"warn":"up"));
}).catch(function(){});}
setInterval(pollHealth,2000);
poll();tail();pollChat();pollHealth();
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
                try:
                    with open(LOCATE) as f:
                        st["locate"] = json.load(f)     # {ts, thing, found, box, locked}
                except Exception:
                    st["locate"] = None
                st["voice"] = "on" if voice_on() else "off"
                self._send(200, "application/json", json.dumps(st).encode())
            elif u.path == "/health":
                self._send(200, "application/json", json.dumps(health()).encode())
            elif u.path == "/frame.jpg":
                try:
                    with open(FRAME, "rb") as f:
                        self._send(200, "image/jpeg", f.read())
                except Exception:
                    self._send(404, "text/plain", b"no frame")
            elif u.path == "/stream.mjpg":
                # smooth ~20fps MJPEG feed for teleop: loop-serve the small frame_fast.jpg as a
                # multipart stream. ThreadingHTTPServer runs this in its own thread, so the long-lived
                # connection doesn't block other endpoints. Ends when the browser closes the tab.
                self.send_response(200)
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Connection", "close")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                period = 1.0 / STREAM_FPS if STREAM_FPS > 0 else 0.05
                last = None
                try:
                    while True:
                        try:
                            with open(FAST_FRAME, "rb") as f:
                                data = f.read()
                        except Exception:
                            data = None
                        if data and data != last:            # only push changed frames
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                            self.wfile.write(("Content-Length: %d\r\n\r\n" % len(data)).encode())
                            self.wfile.write(data)
                            self.wfile.write(b"\r\n")
                            last = data
                        time.sleep(period)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass                                     # viewer closed the stream
                return
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
                _units_cache["t"] = 0.0            # force a fresh is-active read below
                self._send(200, "application/json",
                           json.dumps({"ok": ok, "voice": "on" if voice_on() else "off"}).encode())
            elif u.path == "/drive":
                # manual nudge from the dashboard d-pad -> the motor daemon as a self-expiring timed
                # move (auto-stops after NUDGE_SECS via the daemon's dead-man). The daemon still
                # enforces DISARM, so this can't move the car when the kill switch is on.
                d = body.decode("utf-8", "ignore").strip().lower()
                ok = False
                if d == "stop":
                    msg = "stop"
                elif d in DRIVE_DIRS:
                    msg = "%s %.2f %.2f" % (DRIVE_DIRS[d], NUDGE_SPEED, NUDGE_SECS)
                else:
                    msg = None
                if msg is not None:
                    try:
                        _motorsock.sendto(msg.encode(), MOTOR_SOCK)
                        ok = True
                    except Exception:
                        ok = False                  # motor daemon not running / socket absent
                self._send(200, "application/json", json.dumps({"ok": ok}).encode())
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
