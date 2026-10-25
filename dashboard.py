
from pathlib import Path

import pandas as pd
import streamlit as st

# Find the outputs folder in this project
BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"

# Page setup
st.set_page_config(
    page_title="Fair Hire Audit",
    page_icon="⚖️",
    layout="wide",
)

# Dashboard heading
st.title("Fair Hire Audit")
st.caption("Auditing identity-related bias in resume screening")

st.info(
    "This project uses synthetic data to demonstrate how identity cues "
    "can influence resume rankings. It is an experiment, not a real hiring system."
)

# Load the audit results
overall_file = OUTPUTS_DIR / "overall.csv"
dimension_file = OUTPUTS_DIR / "per_dimension.csv"
weights_file = OUTPUTS_DIR / "identity_token_weights.csv"
report_file = OUTPUTS_DIR / "report.md"

if not overall_file.exists():
    st.error("Audit results were not found in the outputs folder.")
    st.write("Run the audit first using the command below.")
    st.code("python run_audit.py --bullet-level names+gender+college")
    st.stop()

overall = pd.read_csv(overall_file)

if overall.empty:
    st.warning("The audit results file is empty. Run the audit again.")
    st.stop()

# Overview metrics
st.header("Audit Overview")

latest = overall.iloc[-1]

col1, col2, col3 = st.columns(3)

col1.metric(
    "Resume pairs per level",
    "3,000",
    help="Number of paired comparisons in each audit level.",
)

if "rank_change_pct" in overall.columns:
    col2.metric(
        "Rank changes after full redaction",
        f"{latest['rank_change_pct']:.3f}%",
    )

if "worst_selection_rate_ratio" in overall.columns:
    col3.metric(
        "Final selection-rate ratio",
        f"{latest['worst_selection_rate_ratio']:.3f}",
    )

st.divider()

# Charts
st.header("Impact of Removing Identity Cues")
st.write(
    "Compare the results as names, gender cues, and college information "
    "are progressively removed before resume scoring."
)

if {"level", "rank_change_pct"}.issubset(overall.columns):
    st.subheader("Percentage of rankings that changed")
    st.bar_chart(
        overall.set_index("level")[["rank_change_pct"]]
    )

if {"level", "shortlist_flip_pct"}.issubset(overall.columns):
    st.subheader("Percentage of shortlist decisions that flipped")
    st.bar_chart(
        overall.set_index("level")[["shortlist_flip_pct"]]
    )

if {"level", "worst_selection_rate_ratio"}.issubset(overall.columns):
    st.subheader("Worst-case selection-rate ratio")
    st.bar_chart(
        overall.set_index("level")[["worst_selection_rate_ratio"]]
    )
    st.caption(
        "A ratio of 0.8 is a commonly used screening heuristic, "
        "not proof that a hiring system is fair or unfair."
    )

# Full results table
st.header("Complete Audit Results")
st.dataframe(overall, use_container_width=True, hide_index=True)

# Extra evidence
st.header("Explore the Evidence")

if dimension_file.exists():
    with st.expander("Results by identity dimension"):
        dimensions = pd.read_csv(dimension_file)
        st.dataframe(
            dimensions,
            use_container_width=True,
            hide_index=True,
        )

if weights_file.exists():
    with st.expander("Identity-token weights"):
        weights = pd.read_csv(weights_file)
        st.write(
            "These weights show associations learned by the text model. "
            "They do not, by themselves, prove discriminatory intent."
        )
        st.dataframe(
            weights,
            use_container_width=True,
            hide_index=True,
        )

if report_file.exists():
    with st.expander("Read the full audit report"):
        st.markdown(report_file.read_text(encoding="utf-8"))

st.divider()
st.caption(
    "Fair Hire Audit | Synthetic data | "
    "Results depend on the assumptions of the experiment."
)
