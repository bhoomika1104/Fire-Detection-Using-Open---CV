from ultralytics import YOLO
import cv2
import time
import os
from datetime import datetime
import smtplib
from email.message import EmailMessage

# Suppress Ultralytics' console logs
os.environ['YOLO_VERBOSE'] = 'False'

# Email credentials
SENDER_EMAIL = "bhoomikargowda2004@gmail.com"
RECEIVER_EMAIL = "hema.bhoomika@gmail.com"
APP_PASSWORD = "ziyudicyzrwazcyp"  # Use app-specific password

# Logging function
def log_to_file(message):
    with open("logs.txt", "a", encoding="utf-8") as f:
        f.write(f"{message}\n")

# Email alert
def send_email_alert(timestamp):
    try:
        msg = EmailMessage()
        msg.set_content(f"🔥 Fire or Smoke Detected!\nTimestamp: {timestamp}")
        msg["Subject"] = "ALERT: Fire Detection Notification"
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SENDER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)
            print(f"[{timestamp}] 📧 Email sent successfully.")
            log_to_file(f"[{timestamp}] 📧 Email sent successfully.")
    except Exception as e:
        print(f"[{timestamp}] ❌ Failed to send email:", e)
        log_to_file(f"[{timestamp}] ❌ Failed to send email: {e}")

# Load model
model = YOLO(r"C:\Users\Bhoomika\Downloads\yolov8n (2).pt")

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_log_time = 0
last_email_time = 0
email_cooldown = 15  # seconds
log_cooldown = 5     # seconds

while True:
    ret, frame = cap.read()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if not ret:
        msg = f"[{timestamp}] ❌ Failed to grab frame."
        print(msg)
        log_to_file(msg)
        break

    # Inference
    results = model(frame, verbose=False)[0]
    annotated_frame = results.plot()

    fire_detected = False
    human_detected = False

    # Analyze detections
    for box in results.boxes:
        cls = int(box.cls)
        conf = float(box.conf)

        if conf < 0.6:  # skip low-confidence detections
            continue

        if cls in [0, 1]:  # fire or smoke
            fire_detected = True
        if cls == 2:  # human or person class (may vary with model)
            human_detected = True

    if time.time() - last_log_time > log_cooldown:
        if fire_detected and not human_detected:
            msg = f"[{timestamp}] 🔥 Fire or Smoke Detected!"
            print(msg)
            log_to_file(msg)

            if time.time() - last_email_time > email_cooldown:
                send_email_alert(timestamp)
                last_email_time = time.time()
        elif fire_detected and human_detected:
            msg = f"[{timestamp}] ⚠️ Fire-like detection but human present — skipped."
            print(msg)
            log_to_file(msg)
        else:
            msg = f"[{timestamp}] ✅ No fire detected."
            print(msg)
            log_to_file(msg)

        last_log_time = time.time()

    # Show the annotated video
    cv2.imshow("🔥 Fire Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
