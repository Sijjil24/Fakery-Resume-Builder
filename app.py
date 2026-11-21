import os
import io
import csv
import json
import secrets
import sqlite3
from datetime import datetime, timedelta
from functools import wraps

import joblib
import numpy as np
import pandas as pd
from flask import (Flask, Response, flash, g, jsonify, redirect, render_template,
                   request, send_from_directory, session, url_for)
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import check_password_hash, generate_password_hash
from PIL import Image, ImageOps

BASE = os.path.dirname(os.path.abspath(__file__))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE, ".env"))
except ImportError:
    pass
try:
    from authlib.integrations.flask_client import OAuth
except ImportError:
    OAuth = None

import imgfeat

FEATURES = ["variance", "skewness", "curtosis", "entropy"]
HINTS = {
    "variance": "Spread of the wavelet-transformed image values",
    "skewness": "Asymmetry of the image value distribution",
    "curtosis": "Peakedness of the image value distribution",
    "entropy": "Amount of texture and randomness in the image",
}

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")
app.config.update(SESSION_COOKIE_SAMESITE="Lax", MAX_CONTENT_LENGTH=8 * 1024 * 1024,
                  PERMANENT_SESSION_LIFETIME=timedelta(days=14))
DB_PATH = os.environ.get("DB_PATH", os.path.join(BASE, "Fakery.db"))

# ---------------- models + reference data ----------------
model = joblib.load(os.path.join(BASE, "models", "banknote_model.joblib"))
ref = pd.read_csv(os.path.join(BASE, "models", "reference_data.csv"))
with open(os.path.join(BASE, "models", "samples.json")) as f:
    samples = json.load(f)
with open(os.path.join(BASE, "outputs", "summary.json")) as f:
    summary = json.load(f)

ref_rows = ref[FEATURES + ["class"]].round(3).values.tolist()
ref_mean = ref[FEATURES].mean().values
ref_std = ref[FEATURES].std().values
ref_scaled = (ref[FEATURES].values - ref_mean) / ref_std
ref_class = ref["class"].values

IMG_MODEL = os.path.join(BASE, "models", "image_model.joblib")
IMG_INFO = os.path.join(BASE, "models", "image_summary.json")
_img = {"mtime": None, "model": None, "info": None}


def img_model():
    """Photo model is optional. It reloads automatically if you retrain it."""
    if not os.path.exists(IMG_MODEL):
        _img.update(mtime=None, model=None, info=None)
        return None, None
    mt = os.path.getmtime(IMG_MODEL)
    if _img["mtime"] != mt:
        _img["model"] = joblib.load(IMG_MODEL)
        info = {}
        if os.path.exists(IMG_INFO):
            with open(IMG_INFO) as fh:
                info = json.load(fh)
        _img["info"] = info
        _img["mtime"] = mt
    return _img["model"], _img["info"]


# ---------------- Google sign-in (optional) ----------------
GOOGLE_ID = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
GOOGLE_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
GOOGLE_ON = bool(GOOGLE_ID and GOOGLE_SECRET and OAuth)
oauth = None
if GOOGLE_ON:
    oauth = OAuth(app)
    oauth.register(
        name="google", client_id=GOOGLE_ID, client_secret=GOOGLE_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"})

print("Photo model:", "LOADED" if os.path.exists(IMG_MODEL) else "not trained yet (run train_image.py)")


# ---------------- Google sign-in via Firebase ----------------
FIREBASE_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID", "").strip()
FIREBASE_CONFIG = {
    "apiKey": os.environ.get("FIREBASE_API_KEY", "").strip(),
    "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN", "").strip() or (FIREBASE_PROJECT_ID + ".firebaseapp.com"),
    "projectId": FIREBASE_PROJECT_ID,
    "appId": os.environ.get("FIREBASE_APP_ID", "").strip(),
}
if not (FIREBASE_CONFIG["apiKey"] and FIREBASE_PROJECT_ID):
    FIREBASE_CONFIG = {}
print("Google login (Firebase):", "ENABLED" if FIREBASE_CONFIG else "DISABLED (add FIREBASE_* values to .env)")

# ---------------- database ----------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_e):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    with sqlite3.connect(DB_PATH) as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL,
            password_hash TEXT, provider TEXT NOT NULL DEFAULT 'email', created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS scans(
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
            variance REAL, skewness REAL, curtosis REAL, entropy REAL,
            p_fake REAL, verdict TEXT, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT, subject TEXT, body TEXT, created_at TEXT);
        """)
        cols = [r[1] for r in c.execute("PRAGMA table_info(scans)")]
        if "kind" not in cols:
            c.execute("ALTER TABLE scans ADD COLUMN kind TEXT DEFAULT 'features'")
        if "label" not in cols:
            c.execute("ALTER TABLE scans ADD COLUMN label TEXT")


init_db()


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def current_user():
    if "user" not in g:
        uid = session.get("uid")
        g.user = get_db().execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone() if uid else None
    return g.user


def csrf_token():
    if "csrf" not in session:
        session["csrf"] = secrets.token_hex(16)
    return session["csrf"]


app.jinja_env.globals["csrf_token"] = csrf_token


@app.before_request
def csrf_protect():
    if request.method == "POST":
        tok = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
        if not tok or tok != session.get("csrf"):
            if request.path.startswith("/api/"):
                return jsonify(error="Your session expired. Refresh the page and try again."), 400
            flash("Your session expired. Please try again.", "error")
            return redirect(request.referrer or url_for("index"))


@app.context_processor
def inject():
    _m, info = img_model()
    return {"user": current_user(), "summary": summary, "img_info": info if info else None, "firebase_config": FIREBASE_CONFIG}


def login_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if current_user() is None:
            session.pop("uid", None)
            if request.path.startswith("/api/"):
                return jsonify(error="Please log in first."), 401
            flash("Please log in to continue.", "info")
            return redirect(url_for("login"))
        return fn(*a, **kw)
    return wrapper


def log_in(uid, remember=True):
    session["uid"] = uid
    session.permanent = bool(remember)


# ---------------- prediction ----------------
def predict_one(vals):
    row = pd.DataFrame([vals], columns=FEATURES)
    p = float(model.predict_proba(row)[0][1])
    x = (np.array([vals[f] for f in FEATURES]) - ref_mean) / ref_std
    dist = np.sqrt(((ref_scaled - x) ** 2).sum(axis=1))
    k = 15
    idx = np.argsort(dist)[:k]
    verdict = "Fake" if p >= 0.5 else "Genuine"
    conf = p if p >= 0.5 else 1 - p
    if 0.2 < p < 0.8:
        risk = "Uncertain: verify manually"
    else:
        risk = "High risk" if verdict == "Fake" else "Low risk"
    return {"p_fake": p, "verdict": verdict, "confidence": conf, "risk": risk,
            "similar_fake": int(ref_class[idx].sum()), "k": k}


def save_scan(uid, vals, p, verdict, kind="features", label=None):
    v = vals or {}
    db = get_db()
    db.execute("INSERT INTO scans(user_id,variance,skewness,curtosis,entropy,p_fake,verdict,created_at,kind,label) VALUES (?,?,?,?,?,?,?,?,?,?)",
               (uid, v.get("variance"), v.get("skewness"), v.get("curtosis"), v.get("entropy"), p, verdict, now(), kind, label))
    db.commit()


# ---------------- public pages ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/learn")
def learn():
    return render_template("learn.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name", "").strip()[:80]
        email = request.form.get("email", "").strip()[:120]
        subject = request.form.get("subject", "").strip()[:120]
        body = request.form.get("message", "").strip()[:3000]
        if not (name and "@" in email and subject and body):
            flash("Please fill in every field with a valid email.", "error")
        else:
            db = get_db()
            db.execute("INSERT INTO messages(name,email,subject,body,created_at) VALUES (?,?,?,?,?)",
                       (name, email, subject, body, now()))
            db.commit()
            flash("Thanks! Your message has been received. We will get back to you soon.", "success")
            return redirect(url_for("contact"))
    return render_template("contact.html")


@app.route("/health")
def health():
    return "ok"


# ---------------- auth ----------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user():
        return redirect(url_for("scan"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()[:80]
        email = request.form.get("email", "").strip().lower()
        pw = request.form.get("password", "")
        if not name or "@" not in email:
            flash("Please enter your name and a valid email.", "error")
        elif len(pw) < 6:
            flash("Password must be at least 6 characters.", "error")
        elif pw != request.form.get("confirm", ""):
            flash("Passwords do not match.", "error")
        else:
            try:
                db = get_db()
                cur = db.execute("INSERT INTO users(name,email,password_hash,provider,created_at) VALUES (?,?,?,?,?)",
                                 (name, email, generate_password_hash(pw), "email", now()))
                db.commit()
                log_in(cur.lastrowid)
                flash("Welcome to Fakery, " + name.split(" ")[0] + "!", "success")
                return redirect(url_for("scan"))
            except sqlite3.IntegrityError:
                flash("An account with that email already exists. Try logging in.", "error")
    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user():
        return redirect(url_for("scan"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        pw = request.form.get("password", "")
        u = get_db().execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if u is None or not u["password_hash"] or not check_password_hash(u["password_hash"], pw):
            flash("Incorrect email or password.", "error")
        else:
            log_in(u["id"], remember=bool(request.form.get("remember")))
            return redirect(url_for("scan"))
    return render_template("login.html")


@app.route("/guest")
def guest():
    db = get_db()
    tag = secrets.token_hex(3)
    cur = db.execute("INSERT INTO users(name,email,password_hash,provider,created_at) VALUES (?,?,?,?,?)",
                     ("Guest " + tag, "guest-" + tag + "@demo.local", None, "guest", now()))
    db.commit()
    log_in(cur.lastrowid, remember=False)
    flash("You are using the guest demo.", "info")
    return redirect(url_for("scan"))


@app.route("/auth/google")
def google_login():
    if not GOOGLE_ON:
        flash("Google sign-in is not set up on this server yet. Use email sign up or the guest demo.", "info")
        return redirect(url_for("login"))
    return oauth.google.authorize_redirect(url_for("google_callback", _external=True))


@app.route("/auth/google/callback")
def google_callback():
    if not GOOGLE_ON:
        return redirect(url_for("login"))
    try:
        token = oauth.google.authorize_access_token()
        info = token.get("userinfo") or {}
        email = (info.get("email") or "").lower()
        name = info.get("name") or email.split("@")[0]
        if not email or info.get("email_verified") is False:
            raise ValueError("no verified email")
    except Exception as e:
        print("Google sign-in error:", type(e).__name__, e)
        flash("Google sign-in failed. Please try again.", "error")
        return redirect(url_for("login"))
    db = get_db()
    u = db.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if u is None:
        cur = db.execute("INSERT INTO users(name,email,password_hash,provider,created_at) VALUES (?,?,?,?,?)",
                         (name, email, None, "google", now()))
        db.commit()
        uid = cur.lastrowid
    else:
        uid = u["id"]
    log_in(uid)
    flash("Signed in with Google. Welcome, " + name.split(" ")[0] + "!", "success")
    return redirect(url_for("scan"))


@app.route("/auth/firebase", methods=["POST"])
def firebase_login():
    if not FIREBASE_CONFIG:
        return jsonify(error="Google sign-in is not set up on this server yet."), 503
    token = (request.get_json(silent=True) or {}).get("idToken", "")
    try:
        from google.oauth2 import id_token as gid
        from google.auth.transport import requests as greq
        claims = gid.verify_firebase_token(token, greq.Request(), audience=FIREBASE_PROJECT_ID, clock_skew_in_seconds=10)
    except Exception as e:
        print("Firebase token error:", type(e).__name__, e)
        return jsonify(error="Could not verify your Google sign-in. Please try again."), 401
    if not claims:
        return jsonify(error="Could not verify your Google sign-in. Please try again."), 401
    email = (claims.get("email") or "").lower()
    if not email or claims.get("email_verified") is False:
        return jsonify(error="Your Google account has no verified email."), 400
    name = claims.get("name") or email.split("@")[0]
    db = get_db()
    u = db.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if u is None:
        cur = db.execute("INSERT INTO users(name,email,password_hash,provider,created_at) VALUES (?,?,?,?,?)",
                         (name, email, None, "google", now()))
        db.commit()
        uid = cur.lastrowid
    else:
        uid = u["id"]
    log_in(uid)
    flash("Signed in with Google. Welcome, " + name.split(" ")[0] + "!", "success")
    return jsonify(next=url_for("scan"))


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))


@app.route("/account", methods=["GET", "POST"])
@login_required
def account():
    u = current_user()
    db = get_db()
    if request.method == "POST":
        action = request.form.get("action")
        if action == "profile":
            name = request.form.get("name", "").strip()[:80]
            if name:
                db.execute("UPDATE users SET name=? WHERE id=?", (name, u["id"]))
                db.commit()
                flash("Profile updated.", "success")
        elif action == "password" and u["provider"] == "email":
            cur_pw, new_pw = request.form.get("current", ""), request.form.get("new", "")
            if not check_password_hash(u["password_hash"], cur_pw):
                flash("Current password is incorrect.", "error")
            elif len(new_pw) < 6:
                flash("New password must be at least 6 characters.", "error")
            else:
                db.execute("UPDATE users SET password_hash=? WHERE id=?", (generate_password_hash(new_pw), u["id"]))
                db.commit()
                flash("Password updated.", "success")
        return redirect(url_for("account"))
    n = db.execute("SELECT COUNT(*) c FROM scans WHERE user_id=?", (u["id"],)).fetchone()["c"]
    return render_template("account.html", nscans=n)


@app.route("/account/delete", methods=["POST"])
@login_required
def account_delete():
    uid = current_user()["id"]
    db = get_db()
    db.execute("DELETE FROM scans WHERE user_id=?", (uid,))
    db.execute("DELETE FROM users WHERE id=?", (uid,))
    db.commit()
    session.clear()
    flash("Your account was deleted.", "info")
    return redirect(url_for("index"))


# ---------------- scanner ----------------
@app.route("/scan")
@login_required
def scan():
    r = samples["ranges"]
    fields = [{"name": f, "label": f.capitalize(), "hint": HINTS[f],
               "min": round(r[f][0] - 1, 2), "max": round(r[f][1] + 1, 2), "val": samples["genuine"][f]}
              for f in FEATURES]
    return render_template("scan.html", fields=fields, ref=ref_rows, samples=samples)


@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    data = request.get_json(silent=True) or {}
    try:
        vals = {f: float(data[f]) for f in FEATURES}
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Please provide 4 numeric values."), 400
    if not all(np.isfinite(v) for v in vals.values()):
        return jsonify(error="Values must be finite numbers."), 400
    res = predict_one(vals)
    save_scan(session["uid"], vals, res["p_fake"], res["verdict"])
    return jsonify(res)


@app.route("/api/predict-image", methods=["POST"])
@login_required
def api_predict_image():
    f = request.files.get("image")
    if not f or not f.filename:
        return jsonify(error="Please choose an image."), 400
    try:
        img = Image.open(io.BytesIO(f.read()))
        img.load()
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        return jsonify(error="That file is not a readable image. Use JPG, PNG or WEBP."), 400
    vec, report = imgfeat.extract(img)
    m, info = img_model()
    out = {"report": report, "model_ready": m is not None}
    if m is not None:
        proba = m.predict_proba(vec.reshape(1, -1))[0]
        p = float(proba[list(m.classes_).index(1)])
        verdict = "Fake" if p >= 0.5 else "Genuine"
        if 0.35 < p < 0.65:
            risk = "Uncertain: verify manually"
        else:
            risk = "High risk" if verdict == "Fake" else "Low risk"
        nw = len(report["warnings"])
        out.update(p_fake=p, verdict=verdict, confidence=max(p, 1 - p), risk=risk,
                   reliability="Good" if nw == 0 else ("Fair" if nw == 1 else "Low"))
        save_scan(session["uid"], None, p, verdict, kind="photo", label=os.path.basename(f.filename)[:60])
    return jsonify(out)


@app.errorhandler(413)
def too_big(_e):
    if request.path.startswith("/api/"):
        return jsonify(error="File is too large (max 8 MB)."), 413
    flash("File is too large (max 8 MB).", "error")
    return redirect(url_for("index"))


@app.errorhandler(404)
def not_found(_e):
    return render_template("404.html"), 404


# ---------------- batch / history / insights ----------------
@app.route("/batch", methods=["GET", "POST"])
@login_required
def batch():
    rows, csv_text, counts = None, "", None
    if request.method == "POST":
        up = request.files.get("file")
        if not up or not up.filename:
            flash("Choose a CSV file first.", "error")
        else:
            try:
                df = pd.read_csv(up)
            except Exception:
                df = None
                flash("Could not read that file. Make sure it is a valid CSV.", "error")
            if df is not None:
                df.columns = [str(c).strip().lower() for c in df.columns]
                missing = [c for c in FEATURES if c not in df.columns]
                if missing:
                    flash("Missing columns: " + ", ".join(missing), "error")
                elif len(df) == 0 or len(df) > 5000:
                    flash("The CSV must have between 1 and 5000 rows.", "error")
                else:
                    data = df[FEATURES].apply(pd.to_numeric, errors="coerce")
                    if data.isnull().any().any():
                        flash("Some values are not valid numbers. Please check your file.", "error")
                    else:
                        p = model.predict_proba(data)[:, 1]
                        out = data.copy()
                        out["fake_probability"] = p.round(4)
                        out["verdict"] = np.where(p >= 0.5, "Fake", "Genuine")
                        csv_text = out.to_csv(index=False)
                        rows = out.head(200).to_dict("records")
                        nf = int((p >= 0.5).sum())
                        counts = {"total": len(out), "fake": nf, "genuine": len(out) - nf}
    return render_template("batch.html", rows=rows, csv_text=csv_text, counts=counts)


@app.route("/sample.csv")
@login_required
def sample_csv():
    s = pd.concat([ref[ref["class"] == 0].head(5), ref[ref["class"] == 1].head(5)])[FEATURES]
    return Response(s.to_csv(index=False), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=sample_notes.csv"})


@app.route("/history")
@login_required
def history():
    db = get_db()
    uid = session["uid"]
    scans = db.execute("SELECT * FROM scans WHERE user_id=? ORDER BY id DESC LIMIT 200", (uid,)).fetchall()
    total = db.execute("SELECT COUNT(*) c FROM scans WHERE user_id=?", (uid,)).fetchone()["c"]
    fake = db.execute("SELECT COUNT(*) c FROM scans WHERE user_id=? AND verdict='Fake'", (uid,)).fetchone()["c"]
    return render_template("history.html", scans=scans, stats={"total": total, "fake": fake, "genuine": total - fake})


@app.route("/history.csv")
@login_required
def history_csv():
    rows = get_db().execute("SELECT created_at,kind,label,variance,skewness,curtosis,entropy,p_fake,verdict FROM scans WHERE user_id=? ORDER BY id DESC",
                            (session["uid"],)).fetchall()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["when", "type", "label", "variance", "skewness", "curtosis", "entropy", "p_fake", "verdict"])
    for r in rows:
        w.writerow(list(r))
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=Fakery_history.csv"})


@app.route("/history/clear", methods=["POST"])
@login_required
def history_clear():
    db = get_db()
    db.execute("DELETE FROM scans WHERE user_id=?", (session["uid"],))
    db.commit()
    flash("Scan history cleared.", "success")
    return redirect(url_for("history"))


@app.route("/insights")
def insights():
    comparison = pd.read_csv(os.path.join(BASE, "outputs", "model_comparison.csv")).to_dict("records")
    robust = pd.read_csv(os.path.join(BASE, "outputs", "robustness.csv")).to_dict("records")
    images = [
        ("05_confusion_matrix.png", "Confusion matrix (test set)"),
        ("06_feature_importance.png", "Which feature matters most"),
        ("07_robustness.png", "Performance as input gets noisier"),
        ("04_model_comparison.png", "All models compared"),
        ("02_correlation_heatmap.png", "Feature correlation"),
        ("01_class_distribution.png", "Class distribution"),
        ("08_photo_confusion_matrix.png", "Photo model confusion matrix"),
    ]
    images = [(f, c) for f, c in images if os.path.exists(os.path.join(BASE, "outputs", f))]
    return render_template("insights.html", comparison=comparison, robust=robust, images=images)


@app.route("/outputs/<path:name>")
def output_file(name):
    return send_from_directory(os.path.join(BASE, "outputs"), name)


if __name__ == "__main__":
    app.run(debug=True)


