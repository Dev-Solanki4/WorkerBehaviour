from flask import Flask, render_template, send_file
import sqlite3
import pandas as pd
from io import BytesIO
import time
import os

app = Flask(__name__)

# Adjust this depending on where you run the Flask app from
DB_PATH = "../activity_log.db"  # or "activity_log.db" if in same folder

def get_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM WorkerStats ORDER BY worker_id")
    rows = cursor.fetchall()
    columns = [description[0] for description in cursor.description]
    conn.close()

    def format_time(seconds):
        return time.strftime('%H:%M:%S', time.gmtime(seconds))

    stats = [
        {col: format_time(val) if 'time' in col else val for col, val in zip(columns, row)}
        for row in rows
    ]
    return stats

@app.route("/")
def dashboard():
    stats = get_stats()
    return render_template("dashboard.html", stats=stats)

@app.route("/export")
def export_logs():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM WorkerStats", conn)
    conn.close()
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(output, mimetype='text/csv', download_name="worker_stats.csv", as_attachment=True)

if __name__ == "__main__":
    print("Running from:", os.getcwd())
    app.run(debug=True)
