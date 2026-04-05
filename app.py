from flask import Flask, request, jsonify, session, redirect, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from predict import load_alzheimer_model, predict_alzheimer, get_ai_suggestions
from config import Config
from groq import Groq
import os
from datetime import datetime
import json
import uuid
import requests
# INITIALIZE
# ============================================================
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = "alzdetect_super_secure_key"

db = SQLAlchemy(app)

os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/assets", exist_ok=True)

# ============================================================
# DATABASE MODELS
# ============================================================

class Doctor(db.Model):
    __tablename__ = "doctors"

    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    license = db.Column(db.String(80), unique=True, nullable=False)
    specialization = db.Column(db.String(120), nullable=False)
    hospital = db.Column(db.String(150), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, p):
        self.password = generate_password_hash(p)

    def check_password(self, p):
        return check_password_hash(self.password, p)

    def to_dict(self):
        return {
            "id": self.id,
            "fullname": self.fullname,
            "email": self.email,
            "phone": self.phone,
            "specialization": self.specialization,
            "hospital": self.hospital
        }


class Patient(db.Model):
    __tablename__ = "patients"

    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    medical_history = db.Column(db.Text, default="")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, p):
        self.password = generate_password_hash(p)

    def check_password(self, p):
        return check_password_hash(self.password, p)

    def to_dict(self):
        return {
            "id": self.id,
            "fullname": self.fullname,
            "email": self.email,
            "phone": self.phone,
            "age": self.age,
            "gender": self.gender
        }


class MRIScan(db.Model):
    __tablename__ = "mri_scans"

    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, nullable=False)
    patient_id = db.Column(db.Integer, nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(300), nullable=False)
    prediction = db.Column(db.String(120))
    confidence = db.Column(db.Float)
    ai_suggestions = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "ai_suggestions": self.ai_suggestions,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M")
        }


class CognitiveTest(db.Model):
    __tablename__ = "cognitive_tests"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, nullable=False)
    total_score = db.Column(db.Integer)
    severity = db.Column(db.String(80))
    answers_json = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "total_score": self.total_score,
            "severity": self.severity,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M")
        }


class Prescription(db.Model):
    __tablename__ = "prescriptions"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, nullable=False)
    doctor_id = db.Column(db.Integer, nullable=False)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ============================================================
# HTML ROUTES
# ============================================================

@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")

@app.route("/login.html")
def serve_login():
    return send_from_directory(".", "login.html")

@app.route("/patient_register.html")
def serve_patient_register():
    return send_from_directory(".", "patient_register.html")

@app.route("/doctor_register.html")
def serve_doctor_register():
    return send_from_directory(".", "doctor_register.html")

@app.route("/patient_dashboard.html")
def serve_patient_dashboard():
    if "patient_id" not in session:
        return redirect("/login.html")
    return send_from_directory(".", "patient_dashboard.html")

@app.route("/doctor_dashboard.html")
def serve_doctor_dashboard():
    if "doctor_id" not in session:
        return redirect("/login.html")
    return send_from_directory(".", "doctor_dashboard.html")

@app.route("/patients_list.html")
def serve_patients_list():
    if "doctor_id" not in session:
        return redirect("/login.html")
    return send_from_directory(".", "patients_list.html")

@app.route("/patient_details.html")
def serve_patient_details():
    if "doctor_id" not in session:
        return redirect("/login.html")
    return send_from_directory(".", "patient_details.html")

@app.route("/moca.html")
def serve_moca():
    if "patient_id" not in session:
        return redirect("/login.html")
    return send_from_directory(".", "moca.html")

@app.route("/static/uploads/<path:filename>")
def serve_mri_file(filename):
    return send_from_directory("static/uploads", filename)

# ============================================================
# AUTH
# ============================================================

@app.route("/api/doctor-register", methods=["POST"])
def doctor_register():
    data = request.get_json()
    if Doctor.query.filter_by(email=data["email"]).first():
        return jsonify({"success": False}), 400

    d = Doctor(**data)
    d.set_password(data["password"])
    db.session.add(d)
    db.session.commit()
    return jsonify({"success": True})

@app.route("/api/patient-register", methods=["POST"])
def patient_register():
    data = request.get_json()
    if Patient.query.filter_by(email=data["email"]).first():
        return jsonify({"success": False}), 400

    p = Patient(
        fullname=data["fullname"],
        email=data["email"],
        phone=data["phone"],
        age=int(data["age"]),
        gender=data["gender"]
    )
    p.set_password(data["password"])
    db.session.add(p)
    db.session.commit()
    return jsonify({"success": True})

@app.route("/api/doctor-login", methods=["POST"])
def doctor_login():
    data = request.get_json()
    d = Doctor.query.filter_by(email=data["email"]).first()
    if not d or not d.check_password(data["password"]):
        return jsonify({"success": False}), 400
    session["doctor_id"] = d.id
    return jsonify({"success": True})

@app.route("/api/patient-login", methods=["POST"])
def patient_login():
    data = request.get_json()
    p = Patient.query.filter_by(email=data["email"]).first()
    if not p or not p.check_password(data["password"]):
        return jsonify({"success": False}), 400
    session["patient_id"] = p.id
    return jsonify({"success": True})

@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"success": True})

# ============================================================
# GROQ CLIENT INIT
# ============================================================

import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message")

    if not message:
        return jsonify({"success": False, "message": "Empty message"}), 400

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": "You are a medical AI assistant."},
                    {"role": "user", "content": message}
                ]
            }
        )

        result = response.json()

        if "choices" not in result:
            return jsonify({"success": False, "message": result})

        reply = result["choices"][0]["message"]["content"]

        return jsonify({"success": True, "reply": reply})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)})



# ============================================================
# DOCTOR → PATIENT LIST
# ============================================================

@app.route("/api/doctor/patients")
def get_patients():
    if "doctor_id" not in session:
        return jsonify({"success": False}), 401

    patients = Patient.query.order_by(Patient.created_at.desc()).all()

    return jsonify({
        "success": True,
        "patients": [p.to_dict() for p in patients]
    })

# ============================================================
# DOCTOR DASHBOARD STATS
# ============================================================

@app.route("/api/doctor/stats")
def doctor_stats():
    if "doctor_id" not in session:
        return jsonify({"success": False}), 401

    total_patients = db.session.query(Patient).count()
    total_mri = db.session.query(MRIScan).count()
    total_moca = db.session.query(CognitiveTest).count()

    return jsonify({
        "success": True,
        "total_patients": total_patients,
        "total_mri_reports": total_mri,
        "total_cognitive_tests": total_moca
    })

# ============================================================
# DOCTOR → PATIENT DETAILS
# ============================================================

@app.route("/api/doctor/patient/<int:pid>")
def doctor_view_patient(pid):
    if "doctor_id" not in session:
        return jsonify({"success": False}), 401

    patient = Patient.query.get(pid)

    mri_list = MRIScan.query.filter_by(patient_id=pid)\
        .order_by(MRIScan.created_at.desc()).all()

    latest_moca = CognitiveTest.query.filter_by(patient_id=pid)\
        .order_by(CognitiveTest.created_at.desc()).first()

    pres = Prescription.query.filter_by(patient_id=pid)\
        .order_by(Prescription.created_at.desc()).first()

    return jsonify({
        "success": True,
        "patient": patient.to_dict(),
        "mri_list": [m.to_dict() for m in mri_list],
        "latest_moca": latest_moca.to_dict() if latest_moca else None,
        "prescription": {"notes": pres.notes} if pres else None
    })

# ============================================================
# MRI UPLOAD
# ============================================================

@app.route("/api/doctor/upload-mri/<int:pid>", methods=["POST"])
def upload_mri(pid):
    if "doctor_id" not in session:
        return jsonify({"success": False}), 401

    file = request.files["mri"]
    unique_name = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    save_path = os.path.join("static/uploads", unique_name)
    file.save(save_path)

    result = predict_alzheimer(save_path)

    # 🔒 SAFE CHECK
    if not result or not result.get("success"):
        # Remove the file that failed the MRI test
        if os.path.exists(save_path):
            os.remove(save_path)
            
        return jsonify({
            "success": False,
            "message": result.get("message", "Invalid MRI image. Please upload a proper brain MRI.") if result else "Invalid MRI image. Please upload a proper brain MRI."
        }), 400

    suggestions = get_ai_suggestions(result["prediction"])

    scan = MRIScan(
        doctor_id=session["doctor_id"],
        patient_id=pid,
        filename=unique_name,
        filepath=save_path,
        prediction=result["prediction"],
        confidence=result["confidence"],
        ai_suggestions=suggestions
    )

    db.session.add(scan)
    db.session.commit()

    return jsonify({
        "success": True,
        "prediction": result["prediction"],
        "confidence": result["confidence"]
    })

# ============================================================
# MRI DELETE
# ============================================================

@app.route("/api/doctor/delete-mri/<int:mri_id>", methods=["DELETE"])
def delete_mri(mri_id):
    if "doctor_id" not in session:
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    
    scan = MRIScan.query.get(mri_id)
    if not scan:
        return jsonify({"success": False, "message": "MRI scan not found"}), 404
        
    try:
        # Check if the doctor owns this scan's patient, or handle permissions
        # (Assuming the doctor that deletes it is logged in properly)
        
        # Optional: Delete file from filesystem
        if os.path.exists(scan.filepath):
            os.remove(scan.filepath)
            
        db.session.delete(scan)
        db.session.commit()
        return jsonify({"success": True, "message": "MRI scan deleted successfully"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)}), 500
# ============================================================
# MoCA SUBMIT
# ============================================================

@app.route("/api/cognitive/submit", methods=["POST"])
def submit_moca():
    if "patient_id" not in session:
        return jsonify({"success": False}), 401

    data = request.get_json()
    score = data["total_score"]

    if score >= 26:
        severity = "Normal"
    elif score >= 18:
        severity = "Mild Cognitive Impairment"
    elif score >= 10:
        severity = "Moderate Dementia"
    else:
        severity = "Severe Dementia"

    test = CognitiveTest(
        patient_id=session["patient_id"],
        total_score=score,
        severity=severity,
        answers_json=json.dumps(data["answers"])
    )

    db.session.add(test)
    db.session.commit()

    return jsonify({"success": True})

# ============================================================
# PATIENT DASHBOARD DATA
# ============================================================

@app.route("/api/patient-dashboard")
def patient_dashboard():
    if "patient_id" not in session:
        return jsonify({"success": False}), 401

    p = Patient.query.get(session["patient_id"])

    return jsonify({
        "success": True,
        "patient": p.to_dict()
    })

@app.route("/api/patient/reports")
def patient_reports():
    if "patient_id" not in session:
        return jsonify({"success": False}), 401

    pid = session["patient_id"]

    cog = CognitiveTest.query.filter_by(patient_id=pid)\
        .order_by(CognitiveTest.created_at.desc()).first()

    mri = MRIScan.query.filter_by(patient_id=pid)\
        .order_by(MRIScan.created_at.desc()).first()

    pres = Prescription.query.filter_by(patient_id=pid)\
        .order_by(Prescription.created_at.desc()).first()

    return jsonify({
        "success": True,
        "cognitive": cog.to_dict() if cog else None,
        "mri": mri.to_dict() if mri else None,
        "prescription": pres.notes if pres else None
    })

# ============================================================
# INIT
# ============================================================

def init_db():
    with app.app_context():
        db.create_all()
        print("DATABASE READY ✔")

# Initialize database
init_db()

if __name__ == "__main__":
    # Preload the model so the first prediction is fast
    load_alzheimer_model()
    
    app.run(debug=True, host="0.0.0.0", port=5000)
