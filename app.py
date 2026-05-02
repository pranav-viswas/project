from flask import Flask, request, jsonify, send_from_directory, Response, render_template
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

app = Flask(__name__)
CORS(app)
start_time = time.time()

os.makedirs("reports", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

USERS = {
    "admin": generate_password_hash("admin")
}
sessions = {}

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

user_states = {}

def get_state(username):
    if username not in user_states:
        user_states[username] = {
            "running": False,
            "mode": None,
            "current_frame": None,
            "frame_count": 0,
            "match_count": 0,
            "events": [],
            "query_attrs": {},
            "fps": 30,
            "session_name": "",
            "query_info": {},
            "thread": None,
        }
    return user_states[username]

def _ts(frame_idx, fps):
    secs = frame_idx / fps if fps else 0
    m, s = divmod(int(secs), 60)
    h, m2 = divmod(m, 60)
    return f"{h:02d}:{m2:02d}:{s:02d}"

def build_text_query(query_text):
    interpretation = nlp.interpret_query(query_text)
    query_attrs = interpretation["structured_attrs"].copy()
    if interpretation.get("spatial"):
        query_attrs["_spatial"] = interpretation["spatial"]
    return interpretation, query_attrs


# ================= ROUTES FIXED =================

@app.route("/")
def index():
    return render_template("login.html")

@app.route("/search")
def search_page():
    return render_template("frontend.html")

@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")

@app.route("/reports-page")
def reports_page():
    return render_template("reports.html")


# ================= AUTH =================

@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    data = request.json or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if username not in USERS or not check_password_hash(USERS[username], password):
        return jsonify({"error": "Invalid credentials"}), 401

    token = secrets.token_hex(32)
    sessions[token] = username
    return jsonify({"token": token, "username": username})


@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    data = request.json or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if username in USERS:
        return jsonify({"error": "User exists"}), 409

    USERS[username] = generate_password_hash(password)
    token = secrets.token_hex(32)
    sessions[token] = username
    return jsonify({"token": token, "username": username})


# ================= MAIN RUN =================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
