"""
집킴이 (Zipkimi) v5 - CCTV 미행 가능성 선별 시스템

핵심 개선
  - YOLO 결과 영상을 H.264로 변환해 브라우저에서 실제로 재생되게 함
  - 리스크 리본: 영상 아래 시간대별 위험도 띠
  - 레몬 & 민트 디자인 시스템

설치
    pip install streamlit ultralytics opencv-python imageio-ffmpeg
실행
    streamlit run app_v5.py
"""

import math, os, shutil, subprocess, tempfile, uuid
from datetime import datetime
import streamlit as st

DEMO_MODE = False   # True로 바꾸면 분석 없이 예시 결과로 화면만 확인
FPS = 30

MINT="#21A179"; MINT_PALE="#E8F6F0"; MINT_DEEP="#136B4E"
LEMON="#F5C518"; LEMON_PALE="#FEF6DC"; LEMON_DEEP="#9A7400"
CORAL="#E05C4B"; CORAL_PALE="#FDEAE6"; CORAL_DEEP="#A3301F"
INK="#16241F"; INK_SOFT="#5E7269"; INK_FAINT="#93A69D"
PAPER="#FAFCFA"; CARD="#FFFFFF"; LINE="#DFEBE5"

CSS = f"""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');

html, body, [class*="css"] {{ font-family:'Pretendard',system-ui,sans-serif; }}
.stApp {{ background:{PAPER}; }}

/* 상단 툴바가 제목을 가리지 않도록 충분한 여백 확보 */
.block-container {{ padding-top:5.2rem !important; padding-bottom:4rem; max-width:1180px; }}
header[data-testid="stHeader"] {{
  background:{PAPER}; height:3.2rem; z-index:999;
  border-bottom:1px solid {LINE};
}}
[data-testid="stToolbar"] {{ right:1rem; }}
[data-testid="stDecoration"] {{ display:none; }}
#MainMenu {{ visibility:hidden; }}
footer {{ visibility:hidden; }}
h1,h2,h3,h4 {{ color:{INK}!important; letter-spacing:-.4px; }}

section[data-testid="stSidebar"] {{ background:{INK}; }}
section[data-testid="stSidebar"] > div {{ padding-top:2.6rem; }}
section[data-testid="stSidebar"] * {{ color:#DDEAE4; }}

.zk-head {{ display:flex; align-items:flex-end; justify-content:space-between;
  padding:0 0 16px; border-bottom:2px solid {INK}; margin-bottom:22px; }}
.zk-head .badge-wrap {{ display:flex; align-items:center; gap:8px; }}
.zk-live {{ font-family:'JetBrains Mono',monospace; font-size:10.5px; font-weight:700;
  color:{MINT_DEEP}; background:{MINT_PALE}; padding:5px 10px; border-radius:4px;
  display:flex; align-items:center; gap:6px; }}
.zk-live i {{ width:6px; height:6px; border-radius:50%; background:{MINT};
  display:inline-block; animation:pulse 2s ease-in-out infinite; }}
@keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.35}} }}
@media (prefers-reduced-motion: reduce) {{ .zk-live i {{ animation:none; }} }}
.zk-head .t {{ font-size:34px; font-weight:800; color:{INK}; letter-spacing:-1.2px; line-height:1; }}
.zk-head .s {{ font-size:13px; color:{INK_SOFT}; margin-top:7px; }}
.zk-head .badge {{ font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:.5px;
  background:{LEMON}; color:{INK}; padding:5px 11px; border-radius:4px; font-weight:700; }}

.zk-lab {{ font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:700;
  letter-spacing:1.4px; text-transform:uppercase; color:{INK_SOFT};
  display:flex; align-items:center; gap:9px; margin:0 0 11px; }}
.zk-lab::after {{ content:''; flex:1; height:1px; background:{LINE}; }}

.zk-card {{ background:{CARD}; border:1px solid {LINE}; border-radius:14px; padding:16px 18px; }}

.zk-kpi {{ background:{CARD}; border:1px solid {LINE}; border-radius:14px;
  padding:15px 16px; position:relative; overflow:hidden; }}
.zk-kpi .v {{ font-size:30px; font-weight:800; line-height:1.05; letter-spacing:-1px;
  font-family:'JetBrains Mono',monospace; }}
.zk-kpi .l {{ font-size:11.5px; color:{INK_SOFT}; margin-top:4px; }}

.zk-seg {{ display:flex; align-items:center; gap:9px; padding:11px 13px;
  border-radius:11px; margin-bottom:7px; }}
.zk-seg .id {{ font-family:'JetBrains Mono',monospace; font-size:12.5px; font-weight:700; }}
.zk-seg .sc {{ margin-left:auto; font-size:18px; font-weight:800;
  font-family:'JetBrains Mono',monospace; }}

.zk-chip {{ display:inline-block; padding:3px 9px; border-radius:20px;
  font-size:10.5px; font-weight:700; letter-spacing:.2px; }}

table.zk-t {{ width:100%; border-collapse:collapse; }}
table.zk-t th {{ font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:1px;
  text-transform:uppercase; color:{INK_FAINT}; font-weight:700;
  padding:0 8px 8px; text-align:left; border-bottom:1.5px solid {LINE}; }}
table.zk-t td {{ padding:9px 8px; border-bottom:1px solid {LINE}; font-size:13.5px; color:{INK}; }}
table.zk-t td.m {{ font-family:'JetBrains Mono',monospace; color:{INK_SOFT}; font-size:12.5px; }}
table.zk-t td.s {{ text-align:right; font-weight:700; font-family:'JetBrains Mono',monospace; }}
table.zk-t tr.tot td {{ border-bottom:none; padding-top:13px; font-weight:800; font-size:15px; }}

.zk-tile {{ border-radius:12px; padding:12px 10px; text-align:center; }}
.zk-tile .n {{ font-size:10.5px; font-weight:700; opacity:.9; line-height:1.3; }}
.zk-tile .v {{ font-family:'JetBrains Mono',monospace; font-size:19px; font-weight:700; margin:5px 0 2px; }}
.zk-tile .m {{ font-family:'JetBrains Mono',monospace; font-size:10.5px; opacity:.85; }}

div.stButton > button {{ border-radius:9px; border:1px solid {LINE}; background:{CARD};
  color:{INK}; font-size:12.5px; font-weight:600; }}
div.stButton > button:hover {{ border-color:{MINT}; color:{MINT_DEEP}; background:{MINT_PALE}; }}
div.stButton > button[kind="primary"] {{ background:{MINT}; border-color:{MINT}; color:#fff; }}
div.stButton > button[kind="primary"]:hover {{ background:{MINT_DEEP}; color:#fff; }}
div.stDownloadButton > button {{ border-radius:9px; border:1px solid {LINE};
  font-size:12.5px; font-weight:600; }}
[data-testid="stFileUploader"] {{ background:{CARD}; border-radius:12px;
  padding:6px 10px; border:1px solid {LINE}; }}
</style>
"""

def s_rear(r):
    p=r*100
    return (15 if p>=90 else 12 if p>=75 else 9 if p>=60 else 5 if p>=40 else 0), f"{p:.0f}%"
def s_dur(s):
    return (15 if s>=20 else 12 if s>=15 else 9 if s>=10 else 5 if s>=5 else 0), f"{s:.0f}s"
def s_head(x):
    return (10 if x>=.90 else 8 if x>=.75 else 5 if x>=.60 else 0), f"{x:.2f}"
def s_path(r):
    p=r*100
    return (20 if p>=75 else 15 if p>=60 else 10 if p>=45 else 5 if p>=30 else 0), f"{p:.0f}%"
def s_stop(c): return (20,f"{c}회") if c>=2 else ((12,"1회") if c==1 else (0,"0회"))
def s_turn(c): return (20,f"{c}회") if c>=2 else ((12,"1회") if c==1 else (0,"0회"))

MAXES={"후방 근접 유지":15,"지속 시간":15,"방향 유사도":10,
       "지연 경로 유사도":20,"정지 후 추종":20,"방향 전환 추종":20}

def level(s):
    if s>=70: return "우선 확인",CORAL,CORAL_PALE,CORAL_DEEP
    if s>=40: return "확인 필요",LEMON,LEMON_PALE,LEMON_DEEP
    return "일반 이동",MINT,MINT_PALE,MINT_DEEP

def tone(v,mx):
    r=v/mx if mx else 0
    if r>=.75: return CORAL,CORAL_PALE,CORAL_DEEP
    if r>=.40: return LEMON,LEMON_PALE,LEMON_DEEP
    return MINT,MINT_PALE,MINT_DEEP

def sec_of(t):
    a=t.split(":"); return int(a[0])*60+int(a[1])


# ─── 영상: H.264 변환 (브라우저 재생 문제 해결) ───
def to_h264(src):
    """OpenCV mp4v는 브라우저에서 재생 불가. ffmpeg로 H.264 변환해야 st.video에 보인다."""
    ff = shutil.which("ffmpeg")
    if not ff:
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return src, False
    dst = tempfile.NamedTemporaryFile(delete=False, suffix="_h264.mp4").name
    try:
        subprocess.run([ff,"-y","-i",src,"-c:v","libx264","-preset","veryfast",
                        "-pix_fmt","yuv420p","-movflags","+faststart","-an",dst],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(dst) and os.path.getsize(dst)>1000:
            return dst, True
    except Exception:
        pass
    return src, False


def analyze_video(path, prog=None):
    import cv2
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    tracks, per_frame = {}, {}
    res = model.track(path, classes=[0], tracker="bytetrack.yaml",
                      persist=True, stream=True, verbose=False)
    n=0
    for r in res:
        n+=1; rec=[]
        b=r.boxes
        if b is not None and b.id is not None:
            for pid,xyxy,xywh in zip(b.id.tolist(), b.xyxy.tolist(), b.xywh.tolist()):
                pid=int(pid)
                tracks.setdefault(pid,[]).append((n,xywh[0],xywh[1]+xywh[3]/2,xywh[3]))
                rec.append((pid,[int(v) for v in xyxy]))
        per_frame[n]=rec
        if prog and n%20==0: prog.progress(min(n/1500,.5), "사람을 추적하는 중")

    events = compute(tracks, FPS)
    risky = {e["follower"] for e in events if e["score"]>=40}

    cap=cv2.VideoCapture(path)
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=cap.get(cv2.CAP_PROP_FPS) or FPS
    raw=tempfile.NamedTemporaryFile(delete=False,suffix=".mp4").name
    w=cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W,H))
    i=0
    while True:
        ok,fr=cap.read()
        if not ok: break
        i+=1
        for pid,(x1,y1,x2,y2) in per_frame.get(i,[]):
            bad = pid in risky
            col = (75,92,224) if bad else (121,161,33)
            cv2.rectangle(fr,(x1,y1),(x2,y2),col,3 if bad else 2)
            tag = f"id{pid} FOLLOW" if bad else f"id{pid}"
            tw = 9*len(tag)+10
            cv2.rectangle(fr,(x1,max(0,y1-24)),(x1+tw,y1),col,-1)
            cv2.putText(fr,tag,(x1+5,max(14,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
        w.write(fr)
        if prog and i%20==0: prog.progress(min(.5+i/1500*.4,.9), "결과 영상을 그리는 중")
    cap.release(); w.release()
    if prog: prog.progress(.93, "브라우저 재생용으로 변환하는 중")
    out, ok = to_h264(raw)
    if prog: prog.progress(1.0, "완료")
    return events, out, ok


def _spd(p, q):
    return math.hypot(q[0]-p[0], q[1]-p[1])

def _ang(v1, v2):
    s1 = math.hypot(*v1); s2 = math.hypot(*v2)
    if s1 < 1e-6 or s2 < 1e-6: return None
    c = (v1[0]*v2[0]+v1[1]*v2[1])/(s1*s2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))

def _events_of(seq, fps, stop_thr):
    """궤적에서 실제 '정지 시작' / '방향 전환' 프레임을 찾는다"""
    stops, turns = [], []
    win = max(2, fps//6)
    prev_stopped = False
    for i in range(win, len(seq)):
        v = (seq[i][0]-seq[i-1][0], seq[i][1]-seq[i-1][1])
        sp = math.hypot(*v)
        st = sp < stop_thr
        if st and not prev_stopped:
            if not stops or i-stops[-1] > fps//2: stops.append(i)
        prev_stopped = st
        if sp > stop_thr:
            vb = (seq[i-win][0]-seq[i-win-1][0], seq[i-win][1]-seq[i-win-1][1]) \
                 if i-win-1 >= 0 else None
            if vb:
                a = _ang(vb, v)
                if a is not None and a > 35:
                    if not turns or i-turns[-1] > fps//2: turns.append(i)
    return stops, turns

def _lag_hits(ev_a, ev_b, fps):
    """A의 이벤트에 B가 1.5~4초 시간차로 반응한 횟수 / 0~0.5초 동시 반응 횟수"""
    late = sim = 0
    for a in ev_a:
        gaps = [(b-a)/fps for b in ev_b if 0 <= b-a <= fps*5]
        if not gaps: continue
        g = min(gaps)
        if g <= 0.5: sim += 1
        elif 1.5 <= g <= 4.0: late += 1
    return late, sim


def compute(tracks, fps):
    ids = list(tracks)
    out = []
    for i in range(len(ids)):
        for j in range(len(ids)):
            if i == j: continue
            L = {f:(x,y,h) for f,x,y,h in tracks[ids[i]]}
            F = {f:(x,y,h) for f,x,y,h in tracks[ids[j]]}
            com = sorted(set(L) & set(F))
            if len(com) < fps*3:            # 최소 3초 공존
                continue

            lseq = [(L[f][0], L[f][1]) for f in com]
            fseq = [(F[f][0], F[f][1]) for f in com]
            hs = [(L[f][2]+F[f][2])/2 for f in com]
            scale = sum(hs)/len(hs) or 1
            stop_thr = scale*0.004          # 키 대비 속도로 정지 판정 (초당 0.12키 이하)

            # ── 1. 후방 근접 유지 : '뒤에 있는지'까지 판정 ──
            rear = 0
            for k in range(1, len(com)):
                lv = (lseq[k][0]-lseq[k-1][0], lseq[k][1]-lseq[k-1][1])
                if math.hypot(*lv) < stop_thr: continue
                ux, uy = lv[0]/math.hypot(*lv), lv[1]/math.hypot(*lv)
                dx = fseq[k][0]-lseq[k][0]; dy = fseq[k][1]-lseq[k][1]
                along = dx*ux + dy*uy           # 음수 = 뒤에 있음
                d = math.hypot(dx, dy)/scale
                if along < 0 and 0.3 <= d <= 3.0:
                    rear += 1
            rr = rear/max(1, len(com)-1)

            # ── 2. 지속 시간 (실제 초) ──
            dur = rear/fps

            # ── 3. 방향 유사도 (실측) ──
            cos_sum = n_dir = 0
            for k in range(1, len(com)):
                lv = (lseq[k][0]-lseq[k-1][0], lseq[k][1]-lseq[k-1][1])
                fv = (fseq[k][0]-fseq[k-1][0], fseq[k][1]-fseq[k-1][1])
                a = _ang(lv, fv)
                if a is not None:
                    cos_sum += math.cos(math.radians(a)); n_dir += 1
            head = max(0.0, cos_sum/n_dir) if n_dir else 0.0

            # ── 4. 지연 경로 유사도 (후행자가 선행자의 과거 위치를 지나는가) ──
            hit = tot = 0
            for lag in (int(fps*1.0), int(fps*2.0), int(fps*3.0)):
                for k in range(lag, len(com)):
                    tot += 1
                    if _spd(lseq[k-lag], fseq[k])/scale < 0.8: hit += 1
            path = hit/tot if tot else 0.0

            # ── 5·6. 실제 정지/회전 이벤트의 반응 시차 ──
            ls, lt = _events_of(lseq, fps, stop_thr)
            fs, ft = _events_of(fseq, fps, stop_thr)
            stop_late, stop_sim = _lag_hits(ls, fs, fps)
            turn_late, turn_sim = _lag_hits(lt, ft, fps)

            rows = []
            for nm, (v, m) in [("후방 근접 유지",   s_rear(rr)),
                               ("지속 시간",        s_dur(dur)),
                               ("방향 유사도",      s_head(head)),
                               ("지연 경로 유사도", s_path(path)),
                               ("정지 후 추종",     s_stop(stop_late)),
                               ("방향 전환 추종",   s_turn(turn_late))]:
                rows.append((nm, m, v))      # (항목명, 측정값, 점수)
            susp = sum(r[2] for r in rows)

            # ── 게이트 1: 반응성이 없으면 미행으로 보지 않는다 ──
            #  거리·방향·경로만으로는 '같은 길을 가는 사람'과 구별할 수 없다.
            #  선행자의 정지·전환에 후행자가 시간차로 반응했는지가 핵심 근거다.
            reactions = stop_late + turn_late
            opportunities = len(ls) + len(lt)      # 선행자가 만든 관찰 기회
            if reactions == 0:
                continue          # 반응 없음 → 판단 근거 부족, 보고하지 않음
            if opportunities >= 3 and reactions / opportunities < 0.4:
                continue          # 기회는 많았는데 대부분 반응 안 함 → 무관

            # ── 게이트 2: 독립 신호 2개 이상 ──
            signals = sum([rr >= .6, dur >= 5, head >= .75,
                           path >= .45, stop_late >= 1, turn_late >= 1])
            if signals < 2:
                continue

            # ── 동행 감점 ──
            side = 0
            switches = 0; prev = None
            for k in range(1, len(com)):
                lv = (lseq[k][0]-lseq[k-1][0], lseq[k][1]-lseq[k-1][1])
                if math.hypot(*lv) < stop_thr: continue
                ux, uy = lv[0]/math.hypot(*lv), lv[1]/math.hypot(*lv)
                dx = fseq[k][0]-lseq[k][0]; dy = fseq[k][1]-lseq[k][1]
                along = dx*ux+dy*uy; lat = abs(-dx*uy+dy*ux)
                if abs(along) > 1 and lat/max(abs(along),1) > .8: side += 1
                cur = "F" if along > 0 else "L"
                if prev and cur != prev: switches += 1
                prev = cur
            ded = 0
            if side > (len(com)-1)*.35: ded += 15
            if stop_sim + turn_sim >= 2: ded += 10
            if switches >= 3: ded += 10
            if _spd(lseq[0], fseq[0])/scale < 2.0: ded += 5
            ded = min(ded, 40)

            fin = max(0, min(100, susp-ded))
            if fin < 40: continue

            # ── 실제 타임라인 (측정된 이벤트 기반) ──
            tl = []
            for k in range(1, len(com)):
                dx = fseq[k][0]-lseq[k][0]; dy = fseq[k][1]-lseq[k][1]
                if math.hypot(dx,dy)/scale <= 3.0:
                    tl.append((_ts(k, fps), "후방 근접 구간 진입", "근접")); break
            for a in ls[:3]:
                gaps = [(b-a)/fps for b in fs if 0 <= b-a <= fps*5]
                if gaps and 1.5 <= min(gaps) <= 4.0:
                    tl.append((_ts(a, fps),
                               f"선행자 정지 → {min(gaps):.1f}초 뒤 후행자 정지", "정지 추종"))
            for a in lt[:3]:
                gaps = [(b-a)/fps for b in ft if 0 <= b-a <= fps*5]
                if gaps and 1.5 <= min(gaps) <= 4.0:
                    tl.append((_ts(a, fps),
                               f"선행자 방향 전환 → {min(gaps):.1f}초 뒤 후행자 전환", "방향 추종"))
            tl.sort(key=lambda x: x[0])
            if not tl:
                tl = [(_ts(1, fps), "근접 관계 시작", "근접")]

            # ── 실제 위험도 곡선 (구간별 누적) ──
            cur = _curve_real(com, lseq, fseq, fps, scale, stop_thr, fin)

            out.append(ev(ids[i], ids[j], fin, susp, ded, rows, tl, cur))

    best = {}
    for e in out:
        k = tuple(sorted([e["leader"], e["follower"]]))
        if k not in best or e["score"] > best[k]["score"]: best[k] = e
    return sorted(best.values(), key=lambda e: -e["score"])


def _ts(frame_idx, fps):
    s = int(frame_idx/fps)
    return f"{s//60:02d}:{s%60:02d}"


def _curve_real(com, lseq, fseq, fps, scale, stop_thr, final):
    """실제 영상 길이에 맞춰 구간별 위험도를 계산"""
    total = len(com)
    step = max(1, total//10)
    pts = [(0, 0)]
    acc = 0
    for k in range(step, total, step):
        seg = 0
        for t in range(max(1, k-step), k):
            lv = (lseq[t][0]-lseq[t-1][0], lseq[t][1]-lseq[t-1][1])
            if math.hypot(*lv) < stop_thr: continue
            ux, uy = lv[0]/math.hypot(*lv), lv[1]/math.hypot(*lv)
            dx = fseq[t][0]-lseq[t][0]; dy = fseq[t][1]-lseq[t][1]
            if (dx*ux+dy*uy) < 0 and math.hypot(dx,dy)/scale <= 3.0: seg += 1
        acc = min(final, acc + seg/max(1,step)*final*0.35)
        pts.append((int(k/fps), int(acc)))
    if pts[-1][1] < final: pts.append((int(total/fps), final))
    return pts


def ev(a,b,sc,su,de,rows,tl,cv_):
    return {"leader":a,"follower":b,"score":sc,"suspicion":su,
            "deduction":de,"rows":rows,"timeline":tl,"curve":cv_}

def tl_of(sc,tc):
    t=[("00:03","후방 3m 이내 진입","근접")]
    if sc>=1: t.append(("00:11","선행자 정지 → 1.8초 뒤 후행자 정지","정지 추종"))
    if sc>=2: t.append(("00:24","선행자 정지 → 1.5초 뒤 후행자 정지","정지 추종"))
    if tc>=1: t.append(("00:31","선행자 좌회전 → 2.0초 뒤 후행자 좌회전","방향 추종"))
    return t

def curve(f):
    return [(0,5),(3,20),(8,29),(11,47),(16,51),(20,57),(24,69),(28,73),(31,f),(38,f)]

def demo():
    return [
      ev(3,7,82,87,5,
        [("후방 근접 유지","87%",15),("지속 시간","18s",12),("방향 유사도","0.91",10),
         ("지연 경로 유사도","76%",20),("정지 후 추종","2회",20),("방향 전환 추종","1회",12)],
        [("00:03","id7이 id3 후방 3m 이내 진입","근접"),
         ("00:11","id3 정지 → 1.8초 뒤 id7 정지","정지 추종"),
         ("00:24","id3 정지 → 1.5초 뒤 id7 정지","정지 추종"),
         ("00:31","id3 좌회전 → 2.0초 뒤 id7 좌회전","방향 추종")], curve(82)),
      ev(3,12,45,80,35,
        [("후방 근접 유지","72%",9),("지속 시간","16s",12),("방향 유사도","0.88",8),
         ("지연 경로 유사도","40%",5),("정지 후 추종","2회",20),("방향 전환 추종","1회",12)],
        [("00:05","id12 나란히 이동 시작","동행"),
         ("00:14","id3·id12 거의 동시 정지 (0.3초)","동행 신호")],
        [(0,4),(5,26),(10,39),(14,52),(20,47),(26,45),(34,45)]),
    ]


# ─── UI 조각 ───
def head(t,s,badge="",live=""):
    parts=""
    if live: parts+=f"<div class='zk-live'><i></i>{live}</div>"
    if badge: parts+=f"<div class='badge'>{badge}</div>"
    b=f"<div class='badge-wrap'>{parts}</div>" if parts else ""
    st.markdown(f"<div class='zk-head'><div><div class='t'>{t}</div>"
                f"<div class='s'>{s}</div></div>{b}</div>", unsafe_allow_html=True)

def lab(t):
    st.markdown(f"<div class='zk-lab'>{t}</div>", unsafe_allow_html=True)

def kpi(items):
    for c,(l,v,col) in zip(st.columns(len(items)), items):
        c.markdown(f"<div class='zk-kpi'>"
                   f"<div style='position:absolute;left:0;top:0;bottom:0;width:4px;background:{col}'></div>"
                   f"<div class='v' style='color:{col}'>{v}</div>"
                   f"<div class='l'>{l}</div></div>", unsafe_allow_html=True)

def ribbon(cur, total):
    segs=""
    total=max(total,1)
    for k in range(len(cur)-1):
        x0,y0=cur[k]; x1,_=cur[k+1]
        _,c,_,_=level(y0)
        segs+=(f"<div style='position:absolute;left:{x0/total*100:.2f}%;"
               f"width:{(x1-x0)/total*100:.2f}%;top:0;bottom:0;background:{c};"
               f"opacity:{.28+y0/170:.2f}'></div>")
    return (f"<div style='position:relative;height:26px;border-radius:7px;overflow:hidden;"
            f"background:{MINT_PALE};border:1px solid {LINE};margin-top:2px'>{segs}"
            f"<div style='position:absolute;inset:0;display:flex;align-items:center;"
            f"justify-content:space-between;padding:0 9px;font-family:JetBrains Mono,monospace;"
            f"font-size:9.5px;color:{INK};font-weight:700'>"
            f"<span>00:00</span><span style='letter-spacing:1px'>위험도 추이</span>"
            f"<span>{total//60:02d}:{total%60:02d}</span></div></div>")

def graph(cur, fin):
    W,H=660,178; pl,pb,pt=40,26,14
    xm=max(p[0] for p in cur) or 1
    X=lambda x: pl+(x/xm)*(W-pl-16); Y=lambda y: pt+(1-y/100)*(H-pt-pb)
    pts=" ".join(f"{X(x):.1f},{Y(y):.1f}" for x,y in cur)
    area=f"{X(0):.1f},{Y(0):.1f} {pts} {X(xm):.1f},{Y(0):.1f}"
    _,c,pale,deep=level(fin)
    g=""
    for gy,lb,gc in [(70,"우선 확인",CORAL),(40,"확인 필요",LEMON)]:
        g+=(f"<line x1='{pl}' y1='{Y(gy):.1f}' x2='{W-16}' y2='{Y(gy):.1f}' stroke='{gc}' "
            f"stroke-width='1' stroke-dasharray='3 4' opacity='.75'/>"
            f"<text x='{W-18}' y='{Y(gy)-5:.1f}' font-size='9.5' fill='{gc}' "
            f"text-anchor='end' font-family='JetBrains Mono'>{lb} {gy}</text>")
    tk=""
    for x in range(0,xm+1,max(1,xm//6)):
        tk+=(f"<text x='{X(x):.1f}' y='{H-8}' font-size='9.5' fill='{INK_FAINT}' "
             f"text-anchor='middle' font-family='JetBrains Mono'>{x//60:02d}:{x%60:02d}</text>")
    for v in (0,50,100):
        tk+=(f"<text x='{pl-7}' y='{Y(v)+3:.1f}' font-size='9.5' fill='{INK_FAINT}' "
             f"text-anchor='end' font-family='JetBrains Mono'>{v}</text>")
    dots="".join(f"<circle cx='{X(x):.1f}' cy='{Y(y):.1f}' r='2.8' fill='{c}'/>" for x,y in cur)
    lx,ly=cur[-1]
    dots+=(f"<circle cx='{X(lx):.1f}' cy='{Y(ly):.1f}' r='9' fill='{c}' opacity='.22'/>"
           f"<circle cx='{X(lx):.1f}' cy='{Y(ly):.1f}' r='5' fill='{c}'/>")
    return (f"<svg viewBox='0 0 {W} {H}' width='100%' style='display:block'>"
            f"<rect width='{W}' height='{H}' rx='13' fill='{CARD}' stroke='{LINE}'/>{g}"
            f"<polygon points='{area}' fill='{pale}'/>"
            f"<polyline points='{pts}' fill='none' stroke='{c}' stroke-width='2.6' "
            f"stroke-linejoin='round' stroke-linecap='round'/>{dots}{tk}</svg>")


def report(rec):
    L=["집킴이 분석 리포트","="*46,f"파일 : {rec['name']}",f"분석 : {rec['at']}",""]
    for e in rec["events"]:
        lv,_,_,_=level(e["score"])
        L+=[f"[{lv}] id{e['leader']} <- id{e['follower']}  ...  {e['score']}점",
            f"  의심 +{e['suspicion']} / 동행 감점 -{e['deduction']}"]
        for n,m,s in e["rows"]:
            L.append(f"    {n:<16}{m:>8}   +{s:>2} / {MAXES[n]}")
        L.append("  타임라인")
        for t,d,tag in e["timeline"]:
            L.append(f"    {t}  {d}  ({tag})")
        L.append("")
    L+=["-"*46,"본 결과는 미행을 확정하지 않으며, 확인이 필요한 구간의",
        "우선순위를 제시합니다. 최종 판단은 담당자가 수행합니다."]
    return "\n".join(L)


# ─── 화면 1 · 영상 분석 ───
def page_analyze():
    head("집킴이","CCTV 영상 속 두 사람의 상대적 이동 관계를 분석해 확인이 필요한 구간을 선별합니다",
         "FOLLOW RISK", "SYSTEM READY")
    up = st.file_uploader("분석할 CCTV 영상", type=["mp4","avi","mov"],
                          label_visibility="collapsed")
    act = st.session_state.get("active")

    if up is not None:
        tf=tempfile.NamedTemporaryFile(delete=False,suffix=".mp4")
        tf.write(up.read()); vp=tf.name
        c1,_=st.columns([1,3])
        with c1:
            go=st.button("분석 시작", type="primary", use_container_width=True)
        if go:
            bar=st.progress(0,"준비 중")
            if DEMO_MODE:
                events,video,conv=demo(),vp,True; bar.progress(1.0,"완료")
            else:
                events,video,conv=analyze_video(vp,bar)
            bar.empty()
            rec={"id":str(uuid.uuid4())[:8],"name":up.name,"video":video,"events":events,
                 "at":datetime.now().strftime("%m/%d %H:%M"),"converted":conv}
            st.session_state.setdefault("history",[]).insert(0,rec)
            st.session_state.update(active=rec,sel=0,seek=0)
            st.rerun()
        elif act is None:
            st.video(vp); return

    if act is None:
        st.markdown(f"<div class='zk-card' style='text-align:center;padding:44px 20px'>"
                    f"<div style='font-size:15px;color:{INK};font-weight:700'>분석할 영상을 올려 주세요</div>"
                    f"<div style='font-size:12.5px;color:{INK_SOFT};margin-top:6px'>"
                    f"mp4 · avi · mov 형식을 지원합니다</div></div>", unsafe_allow_html=True)
        return

    events=act["events"]
    if not events:
        st.success("확인이 필요한 구간이 감지되지 않았습니다."); return

    i=min(st.session_state.get("sel",0),len(events)-1)
    top=events[i]
    lv,c,pale,deep=level(top["score"])

    if not act.get("converted",True):
        st.warning("영상 변환 도구가 없어 결과 영상이 재생되지 않을 수 있습니다. "
                   "터미널에서 pip install imageio-ffmpeg 후 다시 분석해 주세요.")

    hits=[n for n,m,s in top["rows"] if s>=MAXES[n]*.75]
    why=" · ".join(hits[:3]) if hits else "복합 신호"
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:14px;background:{pale};"
        f"border:1px solid {c};border-left:5px solid {c};border-radius:13px;"
        f"padding:14px 18px;margin:-4px 0 20px'>"
        f"<div style='font-family:JetBrains Mono,monospace;font-size:30px;font-weight:800;"
        f"color:{deep};line-height:1'>{top['score']}</div>"
        f"<div style='flex:1'>"
        f"<div style='font-size:14.5px;font-weight:700;color:{deep}'>"
        f"{lv} · id{top['leader']} 뒤를 id{top['follower']}가 따라간 정황</div>"
        f"<div style='font-size:12.5px;color:{INK_SOFT};margin-top:3px'>"
        f"주요 근거 {why}</div></div>"
        f"<div style='text-align:right;font-family:JetBrains Mono,monospace;"
        f"font-size:11px;color:{INK_SOFT};line-height:1.7'>"
        f"{act['name']}<br>{act['at']}</div></div>", unsafe_allow_html=True)

    L,R=st.columns([1.5,1])
    with L:
        lab("분석 영상")
        st.video(act["video"], start_time=st.session_state.get("seek",0))
        st.markdown(ribbon(top["curve"], max(p[0] for p in top["curve"])),
                    unsafe_allow_html=True)
        st.markdown(f"<div style='margin-top:9px'>"
                    f"<span class='zk-chip' style='background:{CORAL_PALE};color:{CORAL_DEEP}'>■ 미행 의심</span> "
                    f"<span class='zk-chip' style='background:{MINT_PALE};color:{MINT_DEEP}'>■ 일반 보행자</span>"
                    f"</div>", unsafe_allow_html=True)

    with R:
        lab("확인 필요 구간")
        for k,e in enumerate(events):
            l2,c2,p2,d2=level(e["score"]); on=(k==i)
            st.markdown(f"<div class='zk-seg' style='background:{p2};"
                        f"border:{'2px' if on else '1px'} solid {c2 if on else LINE}'>"
                        f"<span class='id' style='color:{d2}'>id{e['leader']} ← id{e['follower']}</span>"
                        f"<span class='zk-chip' style='background:{c2};color:#fff'>{l2}</span>"
                        f"<span class='sc' style='color:{d2}'>{e['score']}</span></div>",
                        unsafe_allow_html=True)
            if len(events)>1 and not on:
                if st.button("이 구간 보기", key=f"g{k}", use_container_width=True):
                    st.session_state.update(sel=k,seek=0); st.rerun()

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        lab("이벤트 타임라인")
        st.markdown(f"<div style='font-size:11.5px;color:{INK_FAINT};margin:-4px 0 8px'>"
                    f"시간을 누르면 해당 장면부터 재생됩니다</div>", unsafe_allow_html=True)
        for k,(t,desc,tag) in enumerate(top["timeline"]):
            a,b=st.columns([1,3.2])
            with a:
                if st.button(t, key=f"t{i}_{k}", use_container_width=True):
                    st.session_state.seek=sec_of(t); st.rerun()
            with b:
                st.markdown(f"<div style='padding-top:5px;font-size:12.5px;color:{INK};"
                            f"line-height:1.45'>{desc}<br>"
                            f"<span class='zk-chip' style='background:{pale};color:{deep};"
                            f"margin-top:3px'>{tag}</span></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
    lab("시간에 따른 위험도 변화")
    st.markdown(graph(top["curve"],top["score"]), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    kpi([("최고 위험도",top["score"],c),("의심 가점",f"+{top['suspicion']}",CORAL),
         ("동행 감점",f"−{top['deduction']}",MINT),("감지 구간",f"{len(events)}",INK)])

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
    A,B=st.columns([1.15,1])
    with A:
        lab("판정 근거")
        rows=""
        for n,m,s in top["rows"]:
            _,_,dd=tone(s,MAXES[n])
            rows+=f"<tr><td>{n}</td><td class='m'>{m}</td><td class='s' style='color:{dd}'>+{s}</td></tr>"
        rows+=(f"<tr><td>동행 감점</td><td class='m'>나란히 · 동시행동</td>"
               f"<td class='s' style='color:{MINT_DEEP}'>−{top['deduction']}</td></tr>"
               f"<tr class='tot'><td>최종 위험도</td><td></td>"
               f"<td class='s' style='color:{deep};font-size:20px'>{top['score']}</td></tr>")
        st.markdown(f"<div class='zk-card'><table class='zk-t'>"
                    f"<tr><th>항목</th><th>측정값</th><th style='text-align:right'>점수</th></tr>"
                    f"{rows}</table></div>", unsafe_allow_html=True)
    with B:
        lab("항목별 점수")
        tiles=""
        for n,m,s in top["rows"]:
            cc,pp,dd=tone(s,MAXES[n])
            tiles+=(f"<div class='zk-tile' style='background:{pp};border:1.5px solid {cc}'>"
                    f"<div class='n' style='color:{dd}'>{n}</div>"
                    f"<div class='v' style='color:{dd}'>{s}"
                    f"<span style='font-size:10px;opacity:.6'>/{MAXES[n]}</span></div>"
                    f"<div class='m' style='color:{dd}'>{m}</div></div>")
        st.markdown(f"<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:8px'>"
                    f"{tiles}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='margin-top:10px'>"
                    f"<span class='zk-chip' style='background:{CORAL_PALE};color:{CORAL_DEEP}'>높음</span> "
                    f"<span class='zk-chip' style='background:{LEMON_PALE};color:{LEMON_DEEP}'>중간</span> "
                    f"<span class='zk-chip' style='background:{MINT_PALE};color:{MINT_DEEP}'>낮음</span>"
                    f"</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    a,b=st.columns(2)
    with a:
        if st.button("관제 알림으로 보내기", type="primary", use_container_width=True):
            al=st.session_state.setdefault("alerts",[])
            now=datetime.now().strftime("%H:%M:%S")
            for e in events:
                l3,_,_,_=level(e["score"])
                al.append({"time":now,"video":act["name"],"aid":act["id"],
                           "pair":f"id{e['leader']} ← id{e['follower']}",
                           "score":e["score"],"level":l3})
            st.success(f"{len(events)}건을 관제 알림으로 보냈습니다.")
    with b:
        st.download_button("리포트 내려받기", report(act),
                           file_name=f"zipkimi_{act['name']}.txt", use_container_width=True)

    st.markdown(
        f"<div style='margin-top:26px;padding:14px 16px;background:{CARD};"
        f"border:1px solid {LINE};border-radius:12px;font-size:11.5px;color:{INK_SOFT};"
        f"line-height:1.7'>"
        f"<b style='color:{INK}'>결과 해석 안내</b><br>"
        f"본 시스템은 미행 여부를 확정하지 않습니다. 확인이 필요한 구간의 우선순위를 제시하며, "
        f"최종 판단은 담당자가 수행합니다. 얼굴·신원 식별은 수행하지 않고 이동 궤적만 분석합니다. "
        f"점수 기준은 실험적 초기값으로, 현장 데이터에 따라 보정이 필요합니다."
        f"</div>", unsafe_allow_html=True)


# ─── 화면 2 · 분석 기록 ───
def page_history():
    n_h=len(st.session_state.get("history",[]))
    head("분석 기록","이전에 분석한 영상을 다시 열어볼 수 있습니다","", f"보관 {n_h}건")
    hist=st.session_state.get("history",[])
    if not hist:
        st.markdown(f"<div class='zk-card' style='text-align:center;padding:40px'>"
                    f"<div style='font-size:14px;color:{INK_SOFT}'>아직 분석한 영상이 없습니다</div>"
                    f"</div>", unsafe_allow_html=True)
        return
    for r in hist:
        sc=max((e["score"] for e in r["events"]), default=0)
        lv,c,p,d=level(sc)
        A,B=st.columns([4.2,1])
        with A:
            st.markdown(f"<div class='zk-card' style='border-left:5px solid {c};margin-bottom:8px;"
                        f"display:flex;justify-content:space-between;align-items:center'>"
                        f"<div><div style='font-size:14.5px;font-weight:700;color:{INK}'>{r['name']}</div>"
                        f"<div style='font-family:JetBrains Mono,monospace;font-size:11px;"
                        f"color:{INK_SOFT};margin-top:4px'>{r['at']} · 감지 {len(r['events'])}건</div></div>"
                        f"<div style='text-align:right'>"
                        f"<div style='font-family:JetBrains Mono,monospace;font-size:24px;"
                        f"font-weight:800;color:{d}'>{sc}</div>"
                        f"<span class='zk-chip' style='background:{p};color:{d}'>{lv}</span>"
                        f"</div></div>", unsafe_allow_html=True)
        with B:
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            if st.button("열기", key=f"h{r['id']}", use_container_width=True):
                st.session_state.update(active=r,sel=0,seek=0,page="영상 분석"); st.rerun()


# ─── 화면 3 · 관제 알림 ───
def page_alerts():
    n_al=len(st.session_state.get("alerts",[]))
    head("관제 알림","실제 운영 시에는 CCTV가 자동 분석되어 위험 구간만 이곳에 쌓입니다",
         "", f"수신 {n_al}건")
    al=st.session_state.get("alerts",[])
    if not al:
        st.markdown(f"<div class='zk-card' style='text-align:center;padding:36px'>"
                    f"<div style='font-size:14px;color:{INK};font-weight:700'>알림이 없습니다</div>"
                    f"<div style='font-size:12.5px;color:{INK_SOFT};margin-top:6px'>"
                    f"영상 분석 화면에서 관제 알림으로 보내기를 누르면 이곳에 쌓입니다</div></div>",
                    unsafe_allow_html=True)
        if st.button("예시 알림 채우기"):
            st.session_state.alerts=[
              {"time":"14:22:15","video":"cctv_역출구_0731.mp4","aid":"-",
               "pair":"id4 ← id8","score":74,"level":"우선 확인"},
              {"time":"14:27:41","video":"cctv_역출구_0731.mp4","aid":"-",
               "pair":"id2 ← id9","score":61,"level":"확인 필요"},
              {"time":"14:32:07","video":"cctv_공원_0731.mp4","aid":"-",
               "pair":"id3 ← id7","score":82,"level":"우선 확인"}]
            st.rerun()
        return

    hi=sum(1 for a in al if a["level"]=="우선 확인")
    kpi([("총 알림",len(al),INK),("우선 확인",hi,CORAL),("최근 수신",al[-1]["time"],MINT)])
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    grp={}
    for a in al: grp.setdefault(a["video"],[]).append(a)
    hist={r["id"]:r for r in st.session_state.get("history",[])}

    for vname,items in grp.items():
        items=sorted(items,key=lambda x:x["time"],reverse=True)
        mx=max(i["score"] for i in items)
        _,c,p,d=level(mx)
        st.markdown(f"<div style='display:flex;align-items:center;gap:10px;margin:4px 0 9px'>"
                    f"<span style='font-size:14.5px;font-weight:700;color:{INK}'>{vname}</span>"
                    f"<span class='zk-chip' style='background:{p};color:{d}'>{len(items)}건</span>"
                    f"</div>", unsafe_allow_html=True)
        for a in items:
            _,c2,p2,d2=level(a["score"])
            A,B=st.columns([4.2,1])
            with A:
                st.markdown(f"<div class='zk-seg' style='background:{p2};border-left:4px solid {c2}'>"
                            f"<span class='id' style='color:{d2}'>{a['time']}</span>"
                            f"<span style='font-size:13px;color:{INK};margin-left:6px'>{a['pair']}</span>"
                            f"<span class='zk-chip' style='background:{c2};color:#fff;margin-left:auto'>"
                            f"{a['level']}</span>"
                            f"<span class='sc' style='color:{d2};margin-left:10px'>{a['score']}</span>"
                            f"</div>", unsafe_allow_html=True)
            with B:
                r=hist.get(a["aid"])
                if r and st.button("영상 열기", key=f"a{a['aid']}{a['time']}{a['pair']}",
                                   use_container_width=True):
                    st.session_state.update(active=r,sel=0,seek=0,page="영상 분석"); st.rerun()

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    if st.button("알림 비우기"):
        st.session_state.alerts=[]; st.rerun()


# ─── 메인 ───
st.set_page_config(page_title="집킴이", layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

PAGES=["영상 분석","분석 기록","관제 알림"]
st.session_state.setdefault("page","영상 분석")

st.sidebar.markdown(
    f"<div style='padding:10px 0 4px'>"
    f"<div style='font-size:23px;font-weight:800;color:#fff;letter-spacing:-.7px'>집킴이</div>"
    f"<div style='font-family:JetBrains Mono,monospace;font-size:9.5px;letter-spacing:1.6px;"
    f"color:{LEMON};margin-top:5px'>FOLLOW RISK SCREENING</div></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='height:1px;background:#33473F;margin:14px 0'></div>",
                    unsafe_allow_html=True)

ch=st.sidebar.radio("화면",PAGES,index=PAGES.index(st.session_state.page),
                    label_visibility="collapsed")
if ch!=st.session_state.page:
    st.session_state.page=ch; st.rerun()

st.sidebar.markdown("<div style='height:1px;background:#33473F;margin:16px 0'></div>",
                    unsafe_allow_html=True)
st.sidebar.markdown(
    f"<div style='font-family:JetBrains Mono,monospace;font-size:11px;line-height:2;color:#9FB5AC'>"
    f"분석 기록 <span style='color:{LEMON};font-weight:700'>"
    f"{len(st.session_state.get('history',[]))}</span><br>"
    f"관제 알림 <span style='color:{LEMON};font-weight:700'>"
    f"{len(st.session_state.get('alerts',[]))}</span></div>", unsafe_allow_html=True)

if DEMO_MODE:
    st.sidebar.markdown(
        f"<div style='margin-top:22px;padding:9px 11px;border-radius:8px;background:#22352E;"
        f"font-size:11px;color:#9FB5AC;line-height:1.6'>"
        f"<b style='color:{LEMON}'>데모 모드</b><br>실제 분석은 코드 상단<br>DEMO_MODE = False</div>",
        unsafe_allow_html=True)

{"영상 분석":page_analyze,"분석 기록":page_history,"관제 알림":page_alerts}[st.session_state.page]()
