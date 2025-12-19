
from flask import Flask, render_template, request, jsonify
import cv2
import time
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import platform
import base64
import csv
try:
    import winsound
except Exception:
    winsound = None
from id_detector import IDDetector
from tuckin_detector import TuckInDetector
from beard_detector import BeardDetector
from uniform_detector import UniformDetector
from labcoat_detector import LabcoatDetector

app = Flask(__name__)

# Initialize detectors
id_detector = IDDetector()
tuckin_detector = TuckInDetector()
beard_detector = BeardDetector()
uniform_detector = UniformDetector()
labcoat_detector = LabcoatDetector()

def detect_from_image(frame, mode, gender):
    id_present = id_detector.detect(frame)
    id_text = "✅ ID Detected" if id_present else "❌ No ID Detected"

    uniform_required = mode == "Uniform Day"
    uniform_text = ""
    uniform_present = None
    if uniform_required:
        uniform_present = uniform_detector.detect(frame)
        uniform_text = "✅ Uniform Detected" if uniform_present else "❌ No Uniform Detected"

    tuckin_required = (
        (mode in ["Normal Day", "Uniform Day"]) and gender == "Boy"
    )
    tuckin_text = ""
    tuckin_status = None
    if tuckin_required:
        tuckin_status = tuckin_detector.detect(frame)
        tuckin_text = (
            "✅ Shirt Tucked In" if tuckin_status else "❌ Shirt Not Tucked In"
        )

    beard_text = ""
    beard_present = None
    if gender == "Boy":
        beard_present = beard_detector.detect(frame)
        beard_text = "❌ Beard Detected" if beard_present else "✅ No Beard"

    labcoat_required = mode.lower().strip().replace(" ", "") == "labday" and gender.lower().strip() == "girl"
    labcoat_text = ""
    labcoat_present = None
    if labcoat_required:
        labcoat_present = labcoat_detector.detect(frame)
        labcoat_text = "✅ Labcoat Detected" if labcoat_present else "❌ No Labcoat Detected"

    all_pass = True
    if not id_present:
        all_pass = False
    if uniform_required and uniform_text and ("❌" in uniform_text):
        all_pass = False
    if tuckin_required and tuckin_text and ("❌" in tuckin_text):
        all_pass = False
    if gender == "Boy" and beard_text and ("❌ Beard Detected" in beard_text):
        all_pass = False
    if labcoat_required and labcoat_text and ("❌" in labcoat_text):
        all_pass = False

    annotated = frame.copy()
    h, w = annotated.shape[:2]
    border_color = (0, 200, 0) if all_pass else (0, 0, 200)
    thickness = 10
    cv2.rectangle(annotated, (0, 0), (w-1, h-1), border_color, thickness)
    info_w = int(0.38 * w)
    x0 = w - info_w
    overlay = annotated.copy()
    cv2.rectangle(overlay, (x0, 0), (w, h), (0, 0, 0), -1)
    annotated = cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0)
    y = 60
    line_h = 45
    cv2.putText(annotated, ("🪪 " + id_text), (x0 + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if id_present else (0, 0, 255), 2)
    if uniform_text:
        y += line_h
        cv2.putText(annotated, ("👔 " + uniform_text), (x0 + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if "✅" in uniform_text else (0, 0, 255), 2)
    if tuckin_text:
        y += line_h
        cv2.putText(annotated, ("👕 " + tuckin_text), (x0 + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if "✅" in tuckin_text else (0, 0, 255), 2)
    if beard_text:
        y += line_h
        cv2.putText(annotated, ("🧔 " + beard_text), (x0 + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if "✅" in beard_text else (0, 0, 255), 2)
    if labcoat_text:
        y += line_h
        cv2.putText(annotated, ("🧥 " + labcoat_text), (x0 + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if "✅" in labcoat_text else (0, 0, 255), 2)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(annotated, ts, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(annotated, f"Result | {mode}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(annotated, "Smart Dress Code by Roshini", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    status_text = "✅ All rules passed. You're good to go." if all_pass else "🚫 Issue detected — please correct before proceeding."

    if winsound and platform.system().lower() == 'windows':
        try:
            if all_pass:
                winsound.Beep(1200, 200)
                winsound.Beep(1600, 200)
            else:
                winsound.Beep(400, 500)
                winsound.Beep(300, 500)
        except Exception:
            pass

    date_dir = datetime.now().strftime("%Y-%m-%d")
    out_dir = Path("records") / date_dir
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{mode.replace(' ', '')}_{gender}_{'PASS' if all_pass else 'FAIL'}.jpg"
    out_path = out_dir / fname
    try:
        cv2.imwrite(str(out_path), annotated)
        image_saved = str(out_path)
    except Exception:
        image_saved = None

    try:
        date_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H:%M:%S")
        csv_path = Path("records") / f"{date_str}_results.csv"
        file_exists = csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "Date",
                    "Time",
                    "Gender",
                    "Mode",
                    "Beard",
                    "ID",
                    "Uniform",
                    "Tucked",
                    "Labcoat",
                    "Result",
                    "Image Path",
                ])
            beard_col = "N/A" if gender != "Boy" else ("Yes" if beard_present else "No")
            id_col = "Yes" if id_present else "No"
            uniform_col = "Yes" if (uniform_present is True) else ("No" if (uniform_present is False) else "N/A")
            shirt_col = "Yes" if (tuckin_status is True) else ("No" if (tuckin_status is False) else "N/A")
            final_col = "PASS" if all_pass else "FAIL"
            labcoat_col = "Yes" if (labcoat_present is True) else ("No" if (labcoat_present is False) else "N/A")
            writer.writerow([
                date_str,
                time_str,
                gender,
                mode,
                beard_col,
                id_col,
                uniform_col,
                shirt_col,
                labcoat_col,
                final_col,
                str(out_path) if image_saved else "",
            ])
    except Exception:
        pass

    return {
        "id_text": id_text,
        "uniform_text": uniform_text,
        "tuckin_text": tuckin_text,
        "beard_text": beard_text,
        "labcoat_text": labcoat_text,
        "status_text": status_text,
        "all_pass": all_pass,
        "image_path": image_saved
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    image_data = data['image']
    mode = data['mode']
    gender = data['gender']

    # Decode base64 image
    header, encoded = image_data.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = detect_from_image(frame, mode, gender)
    return jsonify(results)

@app.route('/records')
def get_records():
    date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    records = []
    csv_path = Path("records") / f"{date}_results.csv"
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = list(reader)
    return jsonify(records)

if __name__ == "__main__":
    app.run(debug=True)
