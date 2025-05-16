import cv2
import time
import smtplib
import threading
from flask import Flask, render_template, Response, jsonify
from email.mime.text import MIMEText
from ultralytics import YOLO

# Load Fire Detection Model
model = YOLO(r"C:\Users\Bhoomika\Downloads\yolov8s.pt")  # Use the fire-trained model

# Initialize Flask App
app = Flask(__name__)

# Initialize Webcam
cap = cv2.VideoCapture(0)

# Fire Detection Parameters
fire_detected_time = None
alert_sent = False
fire_threshold = 2  # Seconds

# Email Configuration
EMAIL_ADDRESS = "bhoomikargowda2004@gmail.com"
EMAIL_PASSWORD = "Adishetty@2611"  # Replace with your App Password
TO_EMAIL = "sohafarzeen@gmail.com"

def send_email_alert():
    """Sends an email alert when fire is detected."""
    global alert_sent
    if alert_sent:
        return

    subject = "🔥 Fire Detected Alert 🔥"
    body = "Fire has been detected for more than 2 seconds. Take action immediately!"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = TO_EMAIL

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, TO_EMAIL, msg.as_string())
        server.quit()
        print("🔥 Fire alert email sent!")
        alert_sent = True  # Prevent multiple emails
    except Exception as e:
        print("❌ Failed to send email:", e)

def detect_fire():
    """Detects fire and updates dashboard."""
    global fire_detected_time, alert_sent
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Detect fire using YOLO
        results = model(frame)
        detected = any(box[4] > 0.7 for box in results[0].boxes.data)  # Confidence > 70%

        for box in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = box
            if conf > 0.7:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, "Fire", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Handle fire alert logic
        if detected:
            if fire_detected_time is None:
                fire_detected_time = time.time()
            elif time.time() - fire_detected_time >= fire_threshold:
                send_email_alert()
        else:
            fire_detected_time = None  # Reset if no fire detected
            alert_sent = False  # Reset email alert flag

def generate_frames():
    """Generates video frames for Flask dashboard."""
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Detect fire using YOLO
        results = model(frame)
        for box in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = box
            if conf > 0.7:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, "Fire", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.route("/")
def index():
    """Renders the dashboard."""
    return render_template("dashboard.html")

@app.route("/video_feed")
def video_feed():
    """Provides live video feed."""
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/fire_status")
def fire_status():
    """Returns fire detection status."""
    return jsonify({"fire_detected": alert_sent})

if __name__ == "__main__":
    # Run fire detection in a separate thread
    threading.Thread(target=detect_fire, daemon=True).start()

    # Run Flask app
    app.run(debug=True)
