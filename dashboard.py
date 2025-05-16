from flask import Flask, render_template
import threading
import time

app = Flask(__name__)

# Global variables to store alert data
alerts = []
total_alerts = 0

@app.route('/')
def index():
    return render_template('dashboard.html', alerts=alerts, total_alerts=total_alerts, fire_detected=alert_sent)

def run_dashboard():
    app.run(debug=True, use_reloader=False)

# Start the dashboard in a separate thread
threading.Thread(target=run_dashboard).start()

def log_alert(timestamp):
    global total_alerts
    alerts.append(timestamp)
    total_alerts += 1

# Removed the unnecessary while loop simulating alert logging
