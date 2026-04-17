import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np

DB_PATH = "aviation_logs.db"

# -----------------------------
# Database initialization
# -----------------------------
def init_database():
    """Create tables if they don't exist"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            risk_level TEXT NOT NULL,
            error REAL NOT NULL,
            uncertainty REAL NOT NULL,
            explanation TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create retraining_events table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS retraining_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Check if predictions table is empty and populate with sample data
    cursor.execute("SELECT COUNT(*) FROM predictions")
    if cursor.fetchone()[0] == 0:
        # Insert sample data for demo purposes
        sample_data = [
            ("NORMAL", 0.0052, 0.12, '{"top_contributors": ["sensor_2", "sensor_3"]}'),
            ("NORMAL", 0.0048, 0.11, '{"top_contributors": ["sensor_4"]}'),
            ("WARNING", 0.0156, 0.25, '{"top_contributors": ["sensor_7", "sensor_11"]}'),
            ("NORMAL", 0.0061, 0.13, '{"top_contributors": ["sensor_2"]}'),
            ("WARNING", 0.0198, 0.34, '{"top_contributors": ["sensor_3", "sensor_4", "sensor_7"]}'),
        ]
        
        for risk_level, error, uncertainty, explanation in sample_data:
            cursor.execute("""
                INSERT INTO predictions (risk_level, error, uncertainty, explanation)
                VALUES (?, ?, ?, ?)
            """, (risk_level, error, uncertainty, explanation))
    
    conn.commit()
    conn.close()

# Initialize database on app start
init_database()

# -----------------------------
# Load data from database
# -----------------------------
def load_predictions():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM predictions", conn)
    conn.close()
    return df

def load_retraining():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM retraining_events", conn)
    conn.close()
    return df

# -----------------------------
# Live data simulation
# -----------------------------
def add_simulated_prediction():
    """Add a simulated prediction to demonstrate live streaming"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Simulate realistic aviation data
    risk_levels = ["NORMAL", "WARNING", "CRITICAL"]
    weights = [0.7, 0.25, 0.05]  # Most predictions should be normal

    risk_level = np.random.choice(risk_levels, p=weights)

    if risk_level == "NORMAL":
        error = np.random.uniform(0.004, 0.008)
        uncertainty = np.random.uniform(0.1, 0.2)
    elif risk_level == "WARNING":
        error = np.random.uniform(0.012, 0.025)
        uncertainty = np.random.uniform(0.2, 0.4)
    else:  # CRITICAL
        error = np.random.uniform(0.03, 0.05)
        uncertainty = np.random.uniform(0.4, 0.6)

    # Generate explanation based on risk level
    if risk_level == "NORMAL":
        explanation = '{"top_contributors": ["sensor_' + str(np.random.randint(1, 5)) + '"]}'
    elif risk_level == "WARNING":
        sensors = np.random.choice([2, 3, 4, 7, 11], size=np.random.randint(1, 3), replace=False)
        explanation = '{"top_contributors": ["sensor_' + '", "sensor_'.join(map(str, sensors)) + '"]}'
    else:
        sensors = np.random.choice([1, 2, 3, 4, 7, 8, 11, 12], size=np.random.randint(2, 4), replace=False)
        explanation = '{"top_contributors": ["sensor_' + '", "sensor_'.join(map(str, sensors)) + '"]}'

    cursor.execute("""
        INSERT INTO predictions (risk_level, error, uncertainty, explanation)
        VALUES (?, ?, ?, ?)
    """, (risk_level, error, uncertainty, explanation))

    conn.commit()
    conn.close()

# -----------------------------
# Dashboard layout
# -----------------------------
st.set_page_config(layout="wide")
st.title("✈ Aviation Engine Live Monitoring")

# Live streaming controls
col_control1, col_control2, col_control3 = st.columns([2, 2, 1])

with col_control1:
    live_mode = st.toggle("🔴 Live Streaming", value=True, help="Enable/disable automatic dashboard updates")

with col_control2:
    refresh_rate = st.selectbox(
        "Refresh Rate",
        options=[1, 2, 3, 5, 10],
        index=2,  # Default to 3 seconds
        format_func=lambda x: f"{x} seconds",
        help="How often to refresh the dashboard"
    )

with col_control3:
    if st.button("🔄 Refresh Now", help="Manually refresh the dashboard"):
        st.rerun()

with col_control1:
    if st.button("🗑️ Clear Data", help="Clear all prediction data and start fresh"):
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM predictions")
        conn.execute("DELETE FROM retraining_events")
        conn.commit()
        conn.close()
        st.success("Data cleared! Refreshing...")
        time.sleep(1)
        st.rerun()

st.divider()

placeholder = st.empty()

# Auto-refresh when live mode is on (safe default, no extra package needed).
with placeholder.container():

    # Live status indicator
    if live_mode:
        st.success(f"🔴 **LIVE** - Auto-refreshing every {refresh_rate} seconds | Last updated: {time.strftime('%H:%M:%S')}")
    else:
        st.info("⏸️ **PAUSED** - Dashboard updates disabled")

    predictions = load_predictions()
    retrain = load_retraining()

    # Add simulated live data for demonstration each refresh while live.
    if live_mode:
        add_simulated_prediction()
        predictions = load_predictions()  # Reload after adding new data

    if len(predictions) == 0:
        st.warning("No data yet. Start pipeline.")
        st.stop()

    # ====================================
    # CURRENT STATUS
    # ====================================
    latest = predictions.iloc[-1]

    col1, col2, col3 = st.columns(3)

    col1.metric("Latest Risk Level", latest["risk_level"])
    col2.metric("Reconstruction Error", round(latest["error"], 5))
    col3.metric("Uncertainty", round(latest["uncertainty"], 5))

    st.divider()

    # ====================================
    # LIVE ACTIVITY FEED
    # ====================================
    st.subheader("📡 Live Activity Feed")

    recent_predictions = predictions.tail(5).copy()
    recent_predictions["timestamp"] = pd.to_datetime(recent_predictions["timestamp"]).dt.strftime("%H:%M:%S")

    activity_feed = []
    for _, row in recent_predictions.iterrows():
        status_icon = "🟢" if row["risk_level"] == "NORMAL" else "🟡" if row["risk_level"] == "WARNING" else "🔴"
        activity_feed.append(f"{status_icon} {row['timestamp']} - {row['risk_level']} risk detected (Error: {row['error']:.4f})")

    for activity in reversed(activity_feed):
        st.text(activity)

    # ====================================
    # GRAPHS SECTION
    # ====================================
    st.subheader("📊 Analytics Dashboard")

    col_graph1, col_graph2, col_graph3 = st.columns(3)

    with col_graph1:
        st.markdown("**Risk Distribution**")
        risk_counts = predictions["risk_level"].value_counts()

        fig1, ax1 = plt.subplots(figsize=(4, 3))
        colors = ['#4CAF50', '#FF9800', '#F44336']
        risk_counts.plot(kind="bar", ax=ax1, color=colors[:len(risk_counts)])
        ax1.set_ylabel("Count", fontsize=10)
        ax1.set_xlabel("Risk Level", fontsize=10)
        ax1.tick_params(axis='both', which='major', labelsize=9)
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig1)

    with col_graph2:
        st.markdown("**Error Trend**")
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.plot(predictions.index, predictions["error"], color='#2196F3', linewidth=2)
        ax2.set_ylabel("Error", fontsize=10)
        ax2.set_xlabel("Time", fontsize=10)
        ax2.tick_params(axis='both', which='major', labelsize=9)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)

    with col_graph3:
        st.markdown("**Uncertainty Trend**")
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        ax3.plot(predictions.index, predictions["uncertainty"], color='#9C27B0', linewidth=2)
        ax3.set_ylabel("Uncertainty", fontsize=10)
        ax3.set_xlabel("Time", fontsize=10)
        ax3.tick_params(axis='both', which='major', labelsize=9)
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3)

    st.subheader("Latest Critical Explanation")
    critical = predictions[predictions["risk_level"] == "CRITICAL"]
    if len(critical) > 0:
        last_critical = critical.iloc[-1]
        st.json(last_critical["explanation"])
    else:
        st.info("No critical events recorded")

    st.divider()

    st.subheader("Model Retraining History")
    if len(retrain) > 0:
        st.dataframe(retrain)
    else:
        st.info("No retraining events yet")

if live_mode:
    time.sleep(refresh_rate)
    st.rerun()
