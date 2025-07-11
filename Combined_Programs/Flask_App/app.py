from flask import Flask, render_template, request, send_file
import sqlite3
import pandas as pd
from io import BytesIO

app = Flask(__name__)

# Connect to DB
DB_PATH = "../activity_log.db"  # Adjust if needed

def get_logs():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM ActivityLog ORDER BY timestamp DESC", conn)
    conn.close()
    return df

@app.route("/")
def dashboard():
    logs = get_logs().head(50)  # Show recent 50
    return render_template("dashboard.html", logs=logs)

@app.route("/worker/<int:worker_id>")
def worker_logs(worker_id):
    df = get_logs()
    filtered = df[df["worker_id"] == worker_id]
    return render_template("worker_logs.html", logs=filtered, worker_id=worker_id)

@app.route("/export")
def export_logs():
    df = get_logs()
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(output, mimetype='text/csv', download_name="activity_log.csv", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
