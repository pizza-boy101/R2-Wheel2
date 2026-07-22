#!/usr/bin/env python3
"""debug_web.py — a tiny zero-dependency LAN debug dashboard for the robot.

Serves a live view (open http://<jetson-ip>:8099 in a browser):
  - the camera frame (frame.jpg),
  - live nav_state (per-column nearness bars, clearest, loom, motion, target),
  - armed/disarmed + nav mode,
  - streaming tails of the voice / nav / motor / perception logs.
The one control it exposes is the ARM/DISARM kill switch: POST /disarm creates the
`.disarmed` file, POST /arm removes it, and the motor daemon enforces it every loop.
It still never drives the motors directly. DISARM is one-tap (safe direction) and Esc
is an emergency stop; ARM asks for confirmation first. Stdlib only (http.server)."""
import os
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HOME = os.path.expanduser("~")
ROBOT = os.path.join(HOME, "robot")
WORKSPACE = os.path.join(ROBOT, "workspace")
NAV_STATE = os.path.join(WORKSPACE, "nav_state.json")
FRAME = os.path.join(WORKSPACE, "frame.jpg")
DISARM = os.path.join(WORKSPACE, ".disarmed")
MODE = os.path.join(WORKSPACE, "nav_mode")
LOGS = {"voice": os.path.join(ROBOT, "realtime.log"),
        "nav": os.path.join(ROBOT, "nav.log"),
        "motor": os.path.join(ROBOT, "motor.log"),
        "perception": os.path.join(ROBOT, "perception.log")}
PORT = int(os.environ.get("DEBUG_WEB_PORT", "8099"))

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
</style></head><body>
<header>
  <h1>robocar debug</h1>
  <span id=arm class=pill>…</span>
  <button id=armBtn class=btn>…</button>
  <span class=hint>Esc = e-stop</span>
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
var cur="voice", pos={voice:0,nav:0,motor:0,perception:0}, atBottom=true, disarmed=true;
var logEl=document.getElementById("log");
function setArm(which){
  fetch("/"+which,{method:"POST"}).then(r=>r.json()).then(function(d){disarmed=!!d.disarmed;poll();}).catch(function(){});
}
document.getElementById("armBtn").onclick=function(){
  if(disarmed){ if(!confirm("ARM the robot? It will be able to move.")) return; setArm("arm"); }
  else { setArm("disarm"); }
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
poll();tail();
</script></body></html>"""


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
            if ln:
                self.rfile.read(ln)                 # drain body so keep-alive stays sane
            if u.path == "/disarm":
                open(DISARM, "w").close()           # create kill switch -> daemon stops within a loop
                self._send(200, "application/json", json.dumps({"disarmed": True}).encode())
            elif u.path == "/arm":
                try:
                    os.remove(DISARM)               # clear kill switch -> daemon may drive again
                except FileNotFoundError:
                    pass
                self._send(200, "application/json", json.dumps({"disarmed": os.path.exists(DISARM)}).encode())
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
