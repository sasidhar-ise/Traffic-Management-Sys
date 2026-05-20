"""
Traffic Violation Detector — LIVE STREAMING VERSION
Shows the annotated video live in the browser as it processes.
Run: python app.py  →  Open: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_file, render_template_string, Response
import os, threading, uuid, json, queue
import numpy as np
import cv2
from pathlib import Path
from detect import detect_violations_live

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

app = Flask(__name__)
app.json_encoder = NpEncoder

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

jobs         = {}
frame_queues = {}

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Traffic Violation Detector</title>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Exo+2:wght@300;400;600&family=Share+Tech+Mono&display=swap" rel="stylesheet"/>
<style>
  :root{--bg:#060d18;--panel:#0b1829;--border:#1a3a5c;--accent:#00d4ff;--red:#ff2244;--green:#00ff88;--text:#c8dff0;--muted:#4a6a8a}
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:var(--bg);color:var(--text);font-family:'Exo 2',sans-serif;min-height:100vh;
    background-image:radial-gradient(ellipse at 15% 0%,rgba(0,80,160,.15) 0%,transparent 55%),
    radial-gradient(ellipse at 85% 100%,rgba(255,107,0,.07) 0%,transparent 55%)}
  body::before{content:'';position:fixed;inset:0;
    background-image:linear-gradient(rgba(0,212,255,.025) 1px,transparent 1px),
    linear-gradient(90deg,rgba(0,212,255,.025) 1px,transparent 1px);
    background-size:44px 44px;pointer-events:none;z-index:0}
  .wrap{max-width:980px;margin:0 auto;padding:28px 20px;position:relative;z-index:1}
  header{display:flex;align-items:center;gap:14px;margin-bottom:24px;padding-bottom:20px;border-bottom:1px solid var(--border)}
  .logo{width:48px;height:48px;background:linear-gradient(135deg,var(--accent),#0055ff);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:22px;box-shadow:0 0 20px rgba(0,212,255,.3);flex-shrink:0}
  header h1{font-family:'Rajdhani',sans-serif;font-size:24px;color:white;letter-spacing:1px}
  header p{font-size:12px;color:var(--muted);margin-top:2px}
  .badge{margin-left:auto;background:rgba(0,255,136,.1);border:1px solid var(--green);color:var(--green);padding:5px 14px;border-radius:50px;font-size:11px;font-family:'Share Tech Mono',monospace;display:flex;align-items:center;gap:6px;flex-shrink:0}
  .dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:blink 1.5s infinite}
  @keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
  .panel{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:22px;margin-bottom:16px;position:relative;overflow:hidden}
  .panel::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--accent),transparent);opacity:.4}
  .ptitle{font-family:'Rajdhani',sans-serif;font-size:12px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:var(--accent);margin-bottom:16px}
  .drop-zone{border:2px dashed var(--border);border-radius:10px;padding:36px 20px;text-align:center;cursor:pointer;transition:all .2s;background:rgba(0,20,40,.4)}
  .drop-zone:hover,.drop-zone.drag{border-color:var(--accent);background:rgba(0,100,200,.08)}
  .drop-zone .icon{font-size:36px;display:block;margin-bottom:10px}
  .drop-zone h3{font-size:16px;color:white;margin-bottom:5px}
  .drop-zone p{font-size:12px;color:var(--muted)}
  input[type=file]{display:none}
  .file-chosen{margin-top:10px;font-family:'Share Tech Mono',monospace;font-size:12px;color:var(--accent)}
  .btn{display:block;width:100%;background:linear-gradient(135deg,#0055ff,var(--accent));border:none;color:white;font-family:'Rajdhani',sans-serif;font-size:17px;font-weight:700;letter-spacing:2px;padding:14px;border-radius:10px;cursor:pointer;text-transform:uppercase;transition:all .25s;box-shadow:0 0 18px rgba(0,100,255,.25);margin-top:14px}
  .btn:hover{transform:translateY(-2px);box-shadow:0 8px 26px rgba(0,100,255,.4)}
  .btn:disabled{opacity:.35;cursor:not-allowed;transform:none}
  #liveSection{display:none}
  .live-layout{display:grid;grid-template-columns:1fr 280px;gap:16px;align-items:start}
  .video-wrap{position:relative;background:#000;border-radius:10px;overflow:hidden;border:1px solid var(--border)}
  .video-wrap img{width:100%;display:block;min-height:200px}
  .live-pill{position:absolute;top:10px;left:10px;background:rgba(220,20,40,.9);color:white;font-family:'Share Tech Mono',monospace;font-size:11px;padding:3px 10px;border-radius:50px;display:flex;align-items:center;gap:5px}
  .live-dot{width:6px;height:6px;border-radius:50%;background:white;animation:blink .8s infinite}
  .stat-card{background:rgba(0,15,35,.7);border:1px solid var(--border);border-radius:10px;padding:14px 16px;margin-bottom:10px;display:flex;align-items:center;gap:12px}
  .stat-icon{font-size:22px}
  .stat-label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:1px}
  .stat-num{font-family:'Share Tech Mono',monospace;font-size:28px;font-weight:700;line-height:1.1}
  .speeding .stat-num{color:#3399ff}
  .redlight .stat-num{color:var(--red)}
  .phone .stat-num{color:#ff44cc}
  .prog-wrap{margin-top:12px}
  .prog-bg{height:5px;background:rgba(255,255,255,.06);border-radius:3px;overflow:hidden;margin-bottom:6px}
  .prog-fill{height:100%;background:linear-gradient(90deg,#0055ff,var(--accent));border-radius:3px;width:0%;transition:width .4s}
  .prog-label{font-family:'Share Tech Mono',monospace;font-size:11px;color:var(--muted)}
  .event-list{max-height:200px;overflow-y:auto}
  .event-item{background:rgba(0,15,30,.5);border:1px solid var(--border);border-radius:7px;padding:7px 12px;margin-bottom:5px;display:flex;align-items:center;gap:10px;font-size:12px;animation:fadeIn .3s ease}
  @keyframes fadeIn{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:translateY(0)}}
  .ev-badge{padding:2px 8px;border-radius:50px;font-size:10px;font-family:'Share Tech Mono',monospace;flex-shrink:0}
  .badge-speeding{background:rgba(51,153,255,.15);color:#3399ff;border:1px solid #3399ff}
  .badge-risky{background:rgba(255,34,68,.15);color:var(--red);border:1px solid var(--red)}
  .badge-phone{background:rgba(255,68,204,.15);color:#ff44cc;border:1px solid #ff44cc}
  .ev-detail{color:var(--muted)}
  .dl-btn{display:none;background:rgba(0,212,255,.1);border:1px solid var(--accent);color:var(--accent);font-family:'Rajdhani',sans-serif;font-size:14px;font-weight:600;letter-spacing:1px;padding:9px 20px;border-radius:8px;cursor:pointer;transition:all .2s;text-decoration:none;margin-top:10px}
  .dl-btn:hover{background:rgba(0,212,255,.2)}
  ::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
  @media(max-width:680px){.live-layout{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="logo">🚦</div>
    <div>
      <h1>TRAFFIC VIOLATION DETECTOR</h1>
      <p>YOLOv8 · Live MJPEG Stream · Real-Time Detection</p>
    </div>
    <div class="badge"><div class="dot"></div><span id="statusText">READY</span></div>
  </header>

  <div class="panel" id="uploadPanel">
    <div class="ptitle">📹 Upload Traffic Video</div>
    <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
      <input type="file" id="fileInput" accept="video/*" onchange="fileSelected(this)"/>
      <span class="icon">📁</span>
      <h3>Drop your video here or click to browse</h3>
      <p>Supports MP4, AVI, MOV, MKV</p>
      <div class="file-chosen" id="fileChosen"></div>
    </div>
    <button class="btn" id="analyzeBtn" onclick="startAnalysis()" disabled>⚡ START LIVE DETECTION</button>
  </div>

  <div id="liveSection">
    <div class="panel">
      <div class="ptitle">🔴 Live Detection Feed</div>
      <div class="live-layout">
        <div>
          <div class="video-wrap">
            <img id="liveStream" src="" alt="Live Detection Feed"/>
            <div class="live-pill"><div class="live-dot"></div>LIVE</div>
          </div>
          <div class="prog-wrap">
            <div class="prog-bg"><div class="prog-fill" id="progFill"></div></div>
            <div class="prog-label" id="progLabel">Initializing...</div>
          </div>
        </div>
        <div>
          <div class="stat-card speeding"><div class="stat-icon">🚗</div><div><div class="stat-label">Speeding</div><div class="stat-num" id="cntSpeed">0</div></div></div>
          <div class="stat-card redlight"><div class="stat-icon">⚡</div><div><div class="stat-label">Risky Maneuvers</div><div class="stat-num" id="cntRisky">0</div></div></div>
          <div class="stat-card phone"><div class="stat-icon">📱</div><div><div class="stat-label">Phone Use</div><div class="stat-num" id="cntPhone">0</div></div></div>
          <a class="dl-btn" id="dlBtn" href="#" download>⬇ Download Video</a>
        </div>
      </div>
    </div>
    <div class="panel">
      <div class="ptitle">📋 Live Violation Log</div>
      <div class="event-list" id="eventList">
        <div style="color:var(--muted);font-size:12px;text-align:center;padding:14px">Violations will appear here in real-time...</div>
      </div>
    </div>
  </div>

</div>
<script>
  let selectedFile=null, currentJob=null, pollTimer=null, lastEventCount=0;

  const dz=document.getElementById('dropZone');
  dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('drag')});
  dz.addEventListener('dragleave',()=>dz.classList.remove('drag'));
  dz.addEventListener('drop',e=>{
    e.preventDefault();dz.classList.remove('drag');
    const f=e.dataTransfer.files[0];
    if(f&&f.type.startsWith('video/'))setFile(f);
  });
  function fileSelected(i){if(i.files[0])setFile(i.files[0]);}
  function setFile(f){
    selectedFile=f;
    document.getElementById('fileChosen').textContent='✓ '+f.name+' ('+( f.size/1024/1024).toFixed(1)+' MB)';
    document.getElementById('analyzeBtn').disabled=false;
  }

  async function startAnalysis(){
    if(!selectedFile)return;
    document.getElementById('analyzeBtn').disabled=true;
    document.getElementById('statusText').textContent='PROCESSING';
    const fd=new FormData();
    fd.append('video',selectedFile);
    const res=await fetch('/upload',{method:'POST',body:fd});
    const data=await res.json();
    currentJob=data.job_id;
    document.getElementById('liveSection').style.display='block';
    document.getElementById('uploadPanel').style.display='none';
    document.getElementById('liveStream').src='/stream/'+currentJob;
    pollTimer=setInterval(()=>pollStatus(currentJob),800);
  }

  async function pollStatus(jobId){
    const res=await fetch('/status/'+jobId);
    const data=await res.json();
    const s=data.summary||{};
    document.getElementById('cntSpeed').textContent=s.speeding||0;
    document.getElementById('cntRisky').textContent=s.risky||0;
    document.getElementById('cntPhone').textContent=s.phone||0;
    document.getElementById('progFill').style.width=(data.progress||0)+'%';
    document.getElementById('progLabel').textContent=data.message||'';

    const events=data.events||[];
    if(events.length>lastEventCount){
      const list=document.getElementById('eventList');
      if(lastEventCount===0)list.innerHTML='';
      for(let i=lastEventCount;i<Math.min(events.length,lastEventCount+5);i++){
        const e=events[i];
        const d=document.createElement('div');
        d.className='event-item';
        d.innerHTML=`<span class="ev-badge badge-${e.violation}">${e.violation.replace('_',' ').toUpperCase()}</span>
          <span class="ev-detail">ID:<strong style="color:white">${e.track_id}</strong> &nbsp;·&nbsp; Frame ${e.frame} &nbsp;·&nbsp; (${e.position[0]},${e.position[1]})</span>`;
        list.prepend(d);
      }
      lastEventCount=events.length;
    }

    if(data.status==='done'){
      clearInterval(pollTimer);
      document.getElementById('statusText').textContent='DONE';
      document.getElementById('progLabel').textContent='✅ Analysis complete!';
      const dl=document.getElementById('dlBtn');
      dl.href='/download/'+jobId;
      dl.style.display='inline-block';
    } else if(data.status==='error'){
      clearInterval(pollTimer);
      document.getElementById('statusText').textContent='ERROR';
      document.getElementById('progLabel').textContent='❌ '+data.message;
    }
  }
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/upload', methods=['POST'])
def upload():
    f       = request.files['video']
    job_id  = str(uuid.uuid4())[:8]
    in_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{f.filename}")
    out_path= os.path.join(OUTPUT_FOLDER, f"{job_id}_output.mp4")
    f.save(in_path)

    fq = queue.Queue(maxsize=64)
    frame_queues[job_id] = fq
    jobs[job_id] = {"status":"running","progress":0,"message":"Loading model...","summary":{},"events":[]}

    threading.Thread(target=run_job, args=(job_id, in_path, out_path, fq), daemon=True).start()
    return jsonify({"job_id": job_id})

def run_job(job_id, in_path, out_path, fq):
    try:
        detect_violations_live(in_path, out_path, fq, jobs[job_id])
        jobs[job_id]["status"]   = "done"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["message"]  = "Complete!"
    except Exception as e:
        jobs[job_id]["status"]  = "error"
        jobs[job_id]["message"] = str(e)
        import traceback; traceback.print_exc()
    finally:
        try: fq.put(None, timeout=1)
        except: pass

@app.route('/stream/<job_id>')
def stream(job_id):
    fq = frame_queues.get(job_id)
    def generate():
        blank = np.zeros((360,640,3), dtype=np.uint8)
        cv2.putText(blank,"Loading YOLOv8 model...",(140,175),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,200,255),2)
        cv2.putText(blank,"Please wait...",(220,215),cv2.FONT_HERSHEY_SIMPLEX,0.6,(100,150,200),1)
        _,buf=cv2.imencode('.jpg',blank,[cv2.IMWRITE_JPEG_QUALITY,80])
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+buf.tobytes()+b'\r\n'
        if fq is None: return
        while True:
            try:
                frame = fq.get(timeout=5)
            except queue.Empty:
                if jobs.get(job_id,{}).get("status") in ("done","error"): break
                continue
            if frame is None: break
            _,buf=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,75])
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+buf.tobytes()+b'\r\n'
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status/<job_id>')
def status(job_id):
    job  = jobs.get(job_id, {"status":"unknown"})
    safe = json.loads(json.dumps(job, cls=NpEncoder))
    return jsonify(safe)

@app.route('/download/<job_id>')
def download(job_id):
    path = os.path.join(OUTPUT_FOLDER, f"{job_id}_output.mp4")
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    print("\n🚦 Traffic Violation Detector — LIVE STREAM MODE")
    print("   Open: http://localhost:5000\n")
    app.run(debug=False, port=5000, threaded=True)
