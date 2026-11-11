"""Streamlit dashboard.  Run from project root:  streamlit run dashboard/app.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st

st.set_page_config(page_title="Pressure Performance Analytics", layout="wide")
st.title("Pressure Performance Analytics")
st.markdown("Pressure = **leverage** of a delivery (expected win-probability swing). Ratings = runs above an average player in the same "
            "situation, partially pooled toward overall skill. Use the sidebar pages for rankings, player drill-downs, matchups and the pressure engine.")
st.info("Pages: Batter Rankings - Bowler Rankings - Player Analysis - Matchup Analysis - Pressure Analysis")
