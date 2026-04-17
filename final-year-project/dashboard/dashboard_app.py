import streamlit as st
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import time
from datetime import datetime
from reportlab.pdfgen import canvas

DB_PATH = "aviation_logs.db"

# ======================================
# PAGE CONFIG (DARK AVIATION THEME)
# ======================================
st.set_page_config(layout="wide", page_title="Aviation Control Center")

st.markdown("""
<style>
body {background-color:#0e1117;color:white;}
</style>
""", unsafe_allow_html=True)

st.title("✈ Aviation Predictive Risk Control Center")

# ======================================
# DATABASE LOAD
# ======================================
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

# ======================================
# ALERT SOUND
# ======================================
def play_alert():
    st.audio(
        "https://www.soundjay.com/buttons/sounds/beep-07.mp3",
        autoplay=True
    )

# ======================================
# HEALTH GAUGE
# ======================================
def health_gauge(value):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text":"Engine Health %"},
        gauge={
            "axis":{"range":[0,100]},
            "bar":{"color":"white"},
            "steps":[
                {"range":[0,40],"color":"red"},
                {"range":[40,70],"color":"orange"},
                {"range":[70,100],"color":"green"},
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

# ======================================
# PDF REPORT GENERATOR
# ======================================
def generate_pdf(latest):

    file_name = "engine_report.pdf"
    c = canvas.Canvas(file_name)

    c.drawString(100,750,"Aircraft Engine Risk Report")
    c.drawString(100,720,f"Time: {latest['timestamp']}")
    c.drawString(100,700,f"Risk Level: {latest['risk_level']}")
    c.drawString(100,680,f"Error: {latest['error']}")
    c.drawString(100,660,f"Uncertainty: {latest['uncertainty']}")

    c.save()

    with open(file_name,"rb") as f:
        st.download_button("Download Report", f, file_name)

# ======================================
# MAIN LOOP
# ======================================
placeholder = st.empty()

while True:

    with placeholder.container():

        df = load_predictions()
        retrain = load_retraining()

        if len(df) == 0:
            st.warning("No data available")
            time.sleep(3)
            st.rerun()

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # ======================================
        # TIME FILTER
        # ======================================
        st.sidebar.header("Time Filter")
        start = st.sidebar.date_input("Start", df["timestamp"].min())
        end = st.sidebar.date_input("End", df["timestamp"].max())

        mask = (df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)
        df = df[mask]

        latest = df.iloc[-1]

        # ======================================
        # LIVE COLOR ALERT PANEL
        # ======================================
        if latest["risk_level"] == "SAFE":
            st.success("SYSTEM STATUS: SAFE")

        elif latest["risk_level"] == "WARNING":
            st.warning("SYSTEM STATUS: WARNING")

        else:
            st.error("SYSTEM STATUS: CRITICAL")
            play_alert()

        # ======================================
        # METRICS
        # ======================================
        c1,c2,c3 = st.columns(3)
        c1.metric("Reconstruction Error", round(latest["error"],5))
        c2.metric("Uncertainty", round(latest["uncertainty"],5))
        c3.metric("Risk Level", latest["risk_level"])

        # ======================================
        # ENGINE HEALTH GAUGE
        # ======================================
        health = max(0,100 - latest["error"]*3000)
        health_gauge(health)

        st.divider()

        # ======================================
        # MULTI ENGINE FLEET VIEW
        # ======================================
        st.subheader("Fleet Risk Overview")

        fleet = df.groupby("risk_level").size()
        st.bar_chart(fleet)

        st.divider()

        # ======================================
        # ERROR TREND
        # ======================================
        st.subheader("Error Trend")
        st.line_chart(df["error"])

        # ======================================
        # UNCERTAINTY TREND
        # ======================================
        st.subheader("Uncertainty Trend")
        st.line_chart(df["uncertainty"])

        st.divider()

        # ======================================
        # EXPLANATION
        # ======================================
        st.subheader("Latest Critical Explanation")

        critical = df[df["risk_level"]=="CRITICAL"]
        if len(critical)>0:
            st.json(critical.iloc[-1]["explanation"])
        else:
            st.info("No critical events")

        st.divider()

        # ======================================
        # RETRAIN HISTORY
        # ======================================
        st.subheader("Model Retraining History")
        st.dataframe(retrain)

        # ======================================
        # DOWNLOAD REPORT
        # ======================================
        st.subheader("Generate System Report")
        generate_pdf(latest)

    time.sleep(3)
    st.rerun()