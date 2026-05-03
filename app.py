from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import cv2
import numpy as np
import base64
import threading
import time
import json
import os
import secrets
from datetime import datetime
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
 
from query_parser import parse_query
from detector import detect_all
from attribute_analyser import analyse_person
from matcher import match_person
from alerter import draw_alert_box, encode_frame
from nlp_engine import nlp
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
 
app = Flask(__name__, template_folder="templates", static_folder="templates")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB upload limit
start_time = time.time()
 
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
 
# ── Simple in-memory user store (replace with DB for production) ──────────────
USERS = {
    "admin": generate_password_hash("admin")
}
sessions = {}  # token -> username
 
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
        elif request.args.get("token"):
            token = request.args.get("token")
        if not token or token not in sessions:
            return jsonify({"error": "Unauthorized"}), 401
        request.username = sessions[token]
        return f(*args, **kwargs)
    return decorated
 
# ── Per-user state ─────────────────────────────────────────────────────────────
user_states = {}
 
def get_state(username):
    if username not in user_states:
        user_states[username] = {
            "running": False,
            "mode": None,
            "current_frame": None,
            "frame_count": 0,
            "match_count": 0,
            "live_persons": 0,
            "events": [],
            "query_attrs": {},
            "fps": 30,
            "session_name": "",
            "query_info": {},
            "thread": None,
        }
    return user_states[username]
 
# ── Timestamp helper ───────────────────────────────────────────────────────────
def _ts(frame_idx, fps):
    secs = frame_idx / fps if fps else 0
    m, s = divmod(int(secs), 60)
    h, m2 = divmod(m, 60)
    return f"{h:02d}:{m2:02d}:{s:02d}"
 
# ── Query builder ──────────────────────────────────────────────────────────────
def build_text_query(query_text):
    interpretation = nlp.interpret_query(query_text)
    query_attrs = interpretation["structured_attrs"].copy()
    if interpretation.get("spatial"):
        query_attrs["_spatial"] = interpretation["spatial"]
    return interpretation, query_attrs
 
# ── Processing loop (video only) ───────────────────────────────────────────────
def process_loop(video_path, query_attrs, threshold, skip_frames, state, rules=None):
    if rules is None:
        rules = []
 
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    state["fps"] = fps
    state["frame_count"] = 0
    state["match_count"] = 0
    state["events"] = []
 
    track_state = {}   # track_id -> {positions, last_update}
    last_alert_per_track = {}
 
    while state["running"] and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Video ended — loop back or stop
            state["running"] = False
            break
 
        state["frame_count"] += 1
 
        # Skip frames for performance
        if state["frame_count"] % skip_frames != 0:
            state["current_frame"] = encode_frame(frame)
            time.sleep(0.01)
            continue
 
        persons, vehicles, others = detect_all(frame)
        state["live_persons"] = len(persons)
        objects = persons + vehicles + others
 
        fh, fw = frame.shape[:2]
        crowd_count = len(persons)
        crowd_alert = ("crowd" in rules) and crowd_count >= 5
 
        match_scores = []
 
        for p in objects:
            tid = p.get("id")
            x1, y1, x2, y2 = p["bbox"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
 
            p.setdefault("behavior", [])
 
            # Track position history for behavior detection
            if tid is not None:
                if tid not in track_state:
                    track_state[tid] = {"positions": [], "last_update": state["frame_count"]}
                track = track_state[tid]
                track["positions"].append((cx, cy, time.time()))
                track["last_update"] = state["frame_count"]
                if len(track["positions"]) > 90:
                    track["positions"] = track["positions"][-90:]
 
                # Behavior detection
                if len(track["positions"]) > 10:
                    p1 = track["positions"][0]
                    p2 = track["positions"][-1]
                    dist = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
                    time_diff = p2[2] - p1[2]
                    velocity = dist / time_diff if time_diff > 0 else 0
                    if velocity > 300:
                        p["behavior"].append("running")
                    if len(track["positions"]) >= 90 and dist < 100:
                        p["behavior"].append("loitering")
 
            # Spatial filter
            spatial_rule_match = True
            q_spatial = query_attrs.get("_spatial")
            if q_spatial == "top" and cy > fh * 0.33: spatial_rule_match = False
            elif q_spatial == "bottom" and cy < fh * 0.66: spatial_rule_match = False
            elif q_spatial == "left" and cx > fw * 0.33: spatial_rule_match = False
            elif q_spatial == "right" and cx < fw * 0.66: spatial_rule_match = False
            elif q_spatial == "center" and (cx < fw * 0.25 or cx > fw * 0.75): spatial_rule_match = False
 
            # Behavior rule trigger
            trigger_rule = False
            if crowd_alert and p.get("type") == "person": trigger_rule = True
            if "running" in rules and "running" in p["behavior"]: trigger_rule = True
            if "loitering" in rules and "loitering" in p["behavior"]: trigger_rule = True
 
            # Match
            matched, score, details = False, 0.0, {}
            attrs = {"type": p["type"]}
 
            if p["type"] == "person":
                p_attrs = analyse_person(frame, p["bbox"])
                attrs.update(p_attrs)
                matched, score, details = match_person(p_attrs, query_attrs, threshold)
            else:
                # Keyword match on YOLO class name
                qtext = query_attrs.get("query", "").lower()
                if p["type"] in qtext:
                    matched, score = True, 0.8
                    details = {"Class Match": {"expected": p["type"], "detected": p["type"], "match": True}}
 
            if not spatial_rule_match:
                matched = False
 
            # NLP behavior context
            qtext = query_attrs.get("query", "").lower()
            if qtext:
                if "running" in qtext and "running" not in p["behavior"]: matched = False
                if "loitering" in qtext and "loitering" not in p["behavior"]: matched = False
 
            if trigger_rule and spatial_rule_match:
                matched = True
                score = 0.99
                details["Smart Rule"] = {
                    "expected": "Rule Triggered",
                    "detected": ", ".join(p["behavior"]) or "Crowd",
                    "match": True
                }
 
            match_scores.append({
                "person": p, "tid": tid,
                "matched": matched, "score": score,
                "details": details, "attrs": attrs
            })
 
        # Best match per frame
        best_match = None
        for m in match_scores:
            if m["matched"]:
                if best_match is None or m["score"] > best_match["score"]:
                    best_match = m
 
        if best_match:
            p = best_match["person"]
            tid = best_match["tid"]
            score = best_match["score"]
            details = best_match["details"]
            attrs = best_match["attrs"]
 
            frame = draw_alert_box(frame, p["bbox"], score, details)
 
            now = time.time()
            last_t = last_alert_per_track.get(tid, now - 10)
            if now - last_t > 5:
                ts = _ts(state["frame_count"], fps)
                shot_filename = f"match_{datetime.now():%Y%m%d_%H%M%S}_{state['frame_count']}.jpg"
                cv2.imwrite(os.path.join(REPORTS_DIR, shot_filename), frame)
                state["events"].append({
                    "timestamp": ts,
                    "frame": state["frame_count"],
                    "confidence": round(score * 100),
                    "bbox": list(p["bbox"]),
                    "details": details,
                    "attributes": attrs,
                    "behavior": p.get("behavior", []),
                    "shot": shot_filename,
                    "query": state.get("query_info", {}).get("content", "")
                })
                state["match_count"] += 1
                if tid is not None:
                    last_alert_per_track[tid] = now
 
        # Draw gray boxes for non-matched
        for m in match_scores:
            if not m["matched"]:
                x1, y1, x2, y2 = m["person"]["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 60), 1)
 
        state["current_frame"] = encode_frame(frame)
        time.sleep(0.01)
 
    cap.release()
    state["running"] = False
 
 
# ── Routes: Static pages ───────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("templates", "login.html")
 
@app.route("/login")
def login_page():
    return send_from_directory("templates", "login.html")
 
@app.route("/search")
def search_page():
    return send_from_directory("templates", "frontend.html")
 
@app.route("/dashboard")
def dashboard_page():
    return send_from_directory("templates", "dashboard.html")
 
@app.route("/reports-page")
def reports_page():
    return send_from_directory("templates", "reports.html")
 
@app.route("/monitor")
def monitor_page():
    return send_from_directory("templates", "dashboard.html")  # fallback to dashboard if no monitor.html
 
# ── Auth ───────────────────────────────────────────────────────────────────────
@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    data = request.json or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")
 
    if username not in USERS or not check_password_hash(USERS[username], password):
        return jsonify({"error": "Invalid credentials"}), 401
 
    token = secrets.token_hex(32)
    sessions[token] = username
    return jsonify({"token": token, "username": username, "role": "operator"})
 
@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    data = request.json or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400
    if username in USERS:
        return jsonify({"error": "User already exists"}), 409
    USERS[username] = generate_password_hash(password)
    token = secrets.token_hex(32)
    sessions[token] = username
    return jsonify({"token": token, "username": username, "role": "operator"})
 
@app.route("/api/auth/logout", methods=["POST"])
@require_auth
def auth_logout():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    sessions.pop(token, None)
    return jsonify({"status": "logged out"})
 
# ── Start / Stop / Status ──────────────────────────────────────────────────────
@app.route("/api/start", methods=["POST"])
@require_auth
def start():
    state = get_state(request.username)
    if state["running"]:
        state["running"] = False
        time.sleep(0.5)
 
    rules = []
 
    if request.content_type and "multipart/form-data" in request.content_type:
        data = request.form
        video_file = request.files.get("video_file")
    else:
        data = request.json or {}
        video_file = None
 
    source_type = data.get("source_type", "video")
 
    # Only allow video source
    if source_type == "cam":
        return jsonify({"error": "Camera input is not supported. Please upload a video file."}), 400
 
    query_text = data.get("query", "")
    threshold = float(data.get("threshold", 0.5))
    skip = int(data.get("skip_frames", 2))
    session_name = data.get("session_name", "Unnamed Session").strip()
 
    interpretation, query_attrs = build_text_query(query_text)
    query_info = {"type": "text", "content": interpretation["target_description"]}
 
    for rule in interpretation["rules"]:
        if rule not in rules:
            rules.append(rule)
 
    for rule in (data.get("rules") or []):
        if rule and rule not in rules:
            rules.append(rule)
 
    # Save uploaded video
    if video_file:
        video_path = os.path.join(UPLOADS_DIR, "temp_video.mp4")
        video_file.save(video_path)
        source = video_path
    else:
        # Allow passing a filename for a pre-existing upload
        filename = data.get("filename", "")
        source = os.path.join(UPLOADS_DIR, filename) if filename else None
        if not source or not os.path.exists(source):
            return jsonify({"error": "No video file provided. Upload a video to start."}), 400
 
    state["query_attrs"] = query_attrs
    state["mode"] = "video"
    state["session_name"] = session_name
    state["query_info"] = query_info
    state["running"] = True
 
    t = threading.Thread(
        target=process_loop,
        args=(source, query_attrs, threshold, skip, state, rules),
        daemon=True
    )
    state["thread"] = t
    t.start()
 
    clean_query = {k: v for k, v in query_attrs.items() if not k.startswith("_")}
    return jsonify({"status": "started", "query": clean_query})
 
@app.route("/api/stop", methods=["POST"])
@require_auth
def stop():
    state = get_state(request.username)
    state["running"] = False
    return jsonify({"status": "stopped", "match_count": state.get("match_count", 0)})
 
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum upload size is 500MB."}), 413
 
@app.route("/api/status", methods=["GET"])
@require_auth
def status():
    state = get_state(request.username)
    return jsonify({
        "running": state["running"],
        "mode": state["mode"],
        # frontend expects these exact keys:
        "frame": state["frame_count"],
        "frame_count": state["frame_count"],
        "matches": state["match_count"],
        "match_count": state["match_count"],
        "fps": state["fps"],
        "events": state.get("events", []),
        "session_name": state["session_name"],
        "query_info": state["query_info"],
        "query_label": state.get("query_info", {}).get("content", ""),
        "uptime": round(time.time() - start_time),
        "persons": state.get("live_persons", 0),
    })
 
# ── Frame / video feed ─────────────────────────────────────────────────────────
@app.route("/api/frame", methods=["GET"])
@require_auth
def frame():
    state = get_state(request.username)
    return jsonify({"frame": state.get("current_frame")})
 
@app.route("/api/video_feed")
def video_feed():
    token = request.args.get("token")
    if not token or token not in sessions:
        return jsonify({"error": "Unauthorized"}), 401
    username = sessions[token]
    state = get_state(username)
 
    def generate():
        while True:
            f = state.get("current_frame")
            if f:
                frame_bytes = base64.b64decode(f)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(0.05)
 
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")
 
# ── Events / Reports ───────────────────────────────────────────────────────────
@app.route("/api/events", methods=["GET"])
@require_auth
def get_events():
    state = get_state(request.username)
    return jsonify({"events": state.get("events", [])})
 
@app.route("/api/reports", methods=["GET"])
@require_auth
def reports():
    state = get_state(request.username)
    events = state.get("events", [])
    # Return a single "session" entry so dashboard table works
    sessions_list = []
    if events:
        sessions_list.append({
            "id": state.get("session_name", "session"),
            "session_name": state.get("session_name", "Session"),
            "match_count": state.get("match_count", 0),
            "created_at": events[0].get("timestamp", "") if events else "",
            "events": events,
        })
    return jsonify({
        "reports": sessions_list,
        "total": len(events),
        "session": state.get("session_name", ""),
        # also include flat events for sidebar
        "events": events,
    })
 
@app.route("/api/reports/clear", methods=["POST"])
@require_auth
def clear_reports():
    state = get_state(request.username)
    state["events"] = []
    state["match_count"] = 0
    return jsonify({"status": "cleared"})
 
@app.route("/reports/<filename>")
def report_image(filename):
    return send_from_directory(REPORTS_DIR, filename)
 
# ── Dashboard stats ────────────────────────────────────────────────────────────
@app.route("/api/dashboard/stats", methods=["GET"])
@app.route("/api/dashboard_stats", methods=["GET"])   # alias the dashboard uses
@require_auth
def dashboard_stats():
    state = get_state(request.username)
    total_events = sum(len(s.get("events", [])) for s in user_states.values())
    return jsonify({
        # keys the dashboard.html expects
        "detections_today": total_events,
        "total_reports": state.get("match_count", 0),
        "total_matches": state.get("match_count", 0),
        "active_users": len(sessions),
        "uptime_seconds": round(time.time() - start_time),
        "running": state["running"],
    })
 
# ── Upload video ───────────────────────────────────────────────────────────────
@app.route("/api/upload_video", methods=["POST"])
@require_auth
def upload_video():
    video_file = request.files.get("video")
    if not video_file:
        return jsonify({"error": "No video file"}), 400
    filename = f"video_{datetime.now():%Y%m%d_%H%M%S}.mp4"
    path = os.path.join(UPLOADS_DIR, filename)
    video_file.save(path)
    return jsonify({"filename": filename, "path": path})
 
# ── Vitals ─────────────────────────────────────────────────────────────────────
@app.route("/api/vitals", methods=["GET"])
@require_auth
def get_vitals():
    state = get_state(request.username)
    try:
        import psutil
        cpu = round(psutil.cpu_percent(interval=0.1))
        mem = round(psutil.virtual_memory().percent)
    except Exception:
        cpu, mem = 0, 0
    return jsonify({
        "uptime": round(time.time() - start_time),
        "running": state["running"],
        "frame_count": state.get("frame_count", 0),
        "match_count": state.get("match_count", 0),
        "fps": state.get("fps", 0),
        "cpu": cpu,
        "memory": mem,
        "latency": "—",
    })
 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
