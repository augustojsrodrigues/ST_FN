# -*- coding: utf-8 -*-
"""
Streamlit app for the paper:
Monte Carlo simulation for opportunistic inspection planning under misclassification errors

This app evaluates a user-defined policy (S, T) by Monte Carlo simulation.
It does not optimize S and T and does not run differential evolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Monte Carlo opportunistic inspection",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2.5rem;
}

.hero {
    padding: 1.6rem 1.8rem;
    border-radius: 24px;
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 48%, #0f766e 100%);
    color: white;
    box-shadow: 0 18px 45px rgba(15, 23, 42, 0.20);
    margin-bottom: 1.2rem;
}

.hero h1 {
    font-size: 2.1rem;
    line-height: 1.15;
    margin-bottom: 0.35rem;
    font-weight: 800;
}

.hero p {
    font-size: 1.02rem;
    opacity: 0.93;
    margin-bottom: 0;
}

.card {
    border-radius: 18px;
    padding: 1.05rem 1.1rem;
    background: #ffffff;
    border: 1px solid rgba(15, 23, 42, 0.08);
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.07);
    margin-bottom: 1rem;
}

.dark-card {
    border-radius: 18px;
    padding: 1.1rem 1.2rem;
    background: #0f172a;
    color: white;
    box-shadow: 0 14px 34px rgba(15, 23, 42, 0.18);
    margin-bottom: 1rem;
}

.badge {
    display: inline-block;
    padding: 0.25rem 0.55rem;
    border-radius: 999px;
    background: #e0f2fe;
    color: #075985;
    font-size: 0.78rem;
    font-weight: 700;
    margin-right: 0.35rem;
}

.warning-box {
    border-left: 5px solid #f59e0b;
    background: #fffbeb;
    color: #78350f;
    padding: 0.9rem 1rem;
    border-radius: 12px;
    margin: 0.75rem 0;
}

.success-box {
    border-left: 5px solid #10b981;
    background: #ecfdf5;
    color: #064e3b;
    padding: 0.9rem 1rem;
    border-radius: 12px;
    margin: 0.75rem 0;
}

.info-box {
    border-left: 5px solid #0284c7;
    background: #eff6ff;
    color: #0c4a6e;
    padding: 0.9rem 1rem;
    border-radius: 12px;
    margin: 0.75rem 0;
}

.error-soft {
    border-left: 5px solid #ef4444;
    background: #fef2f2;
    color: #7f1d1d;
    padding: 0.9rem 1rem;
    border-radius: 12px;
    margin: 0.75rem 0;
}

.motion-box {
    height: 140px;
    border-radius: 22px;
    background: linear-gradient(135deg, #0f172a, #1e3a8a, #0f766e);
    position: relative;
    overflow: hidden;
    box-shadow: 0 18px 45px rgba(15, 23, 42, 0.18);
    margin-bottom: 1rem;
}

.motion-line {
    position: absolute;
    left: 9%;
    right: 9%;
    top: 72px;
    height: 4px;
    background: rgba(226, 232, 240, 0.75);
    border-radius: 999px;
}

.motion-dot {
    position: absolute;
    top: 58px;
    left: 9%;
    width: 28px;
    height: 28px;
    background: #fbbf24;
    border: 3px solid white;
    border-radius: 50%;
    animation: moveDot 3.2s linear infinite;
}

.motion-marker-s {
    position: absolute;
    top: 45px;
    left: 34%;
    width: 4px;
    height: 58px;
    background: #38bdf8;
    border-radius: 999px;
}

.motion-marker-t {
    position: absolute;
    top: 45px;
    left: 78%;
    width: 4px;
    height: 58px;
    background: #22c55e;
    border-radius: 999px;
}

.motion-label-s {
    position: absolute;
    top: 106px;
    left: 33%;
    color: #e0f2fe;
    font-weight: 800;
}

.motion-label-t {
    position: absolute;
    top: 106px;
    left: 77%;
    color: #dcfce7;
    font-weight: 800;
}

@keyframes moveDot {
    0% { left: 9%; }
    100% { left: 86%; }
}

hr {
    margin-top: 1rem;
    margin-bottom: 1rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class PolicyInputs:
    S: float
    T: float
    mu_x: float
    mu_h: float
    mu_z: float
    c_i: float
    c_o: float
    c_p: float
    c_f: float
    beta_s: float
    beta_o: float
    n_cycles: int
    seed: int


# ---------------------------------------------------------------------
# Simulation model
# ---------------------------------------------------------------------
def run_policy_simulation(inputs: PolicyInputs) -> Dict[str, float]:
    """
    Evaluate the modified-opportunistic inspection policy for fixed S and T.

    Outputs:
    - Cost rate;
    - MTBOF;
    - PFRBO;
    - LOM.
    """
    rng = np.random.default_rng(inputs.seed)

    S = inputs.S
    T = inputs.T
    mu_x = inputs.mu_x
    mu_h = inputs.mu_h
    mu_z = max(inputs.mu_z, 1e-12)
    CI = inputs.c_i
    CO = inputs.c_o
    CP = inputs.c_p
    CF = inputs.c_f
    beta = inputs.beta_s
    beta2 = inputs.beta_o
    n = inputs.n_cycles

    decision = 1
    cost = 0.0
    life = 0.0
    counters = np.zeros(20, dtype=np.int64)

    x = 0.0
    h = 0.0

    for _ in range(n):
        if decision == 1:
            x = rng.exponential(scale=mu_x)
            h = rng.exponential(scale=mu_h)

        z = rng.exponential(scale=mu_z) + S
        p = rng.uniform(0.0, 1.0)
        p2 = rng.uniform(0.0, 1.0)

        if x + h < S:
            cost += CF
            life += x + h
            counters[0] += 1
            decision = 1

        elif x < S <= x + h < T and z > x + h:
            cost += CF
            life += x + h
            counters[1] += 1
            decision = 1

        elif (x < S <= z <= x + h < T and x + h >= S and p > beta2) or (
            x < S <= z < T and x + h >= T and p > beta2
        ):
            cost += CP + CO
            life += z
            counters[2] += 1
            decision = 1

        elif x < S <= z <= x + h < T and p <= beta2:
            cost += CF + CO
            life += x + h
            counters[3] += 1
            decision = 1

        elif x < S and x + h >= T and z >= T:
            if p2 > beta:
                cost += CP + CI
                life += T
                counters[4] += 1
                decision = 1
            else:
                x = 0.0
                h = h - T
                cost += CI
                life += T
                counters[5] += 1
                decision = 0

        elif x < S <= z < T and x + h >= T and p <= beta2:
            if p2 > beta:
                cost += CO + CP + CI
                life += T
                counters[6] += 1
                decision = 1
            else:
                x = 0.0
                h = h - T
                cost += CI + CO
                life += T
                counters[7] += 1
                decision = 0

        elif x >= S and x + h < T and z > x + h:
            cost += CF
            life += x + h
            counters[8] += 1
            decision = 1

        elif S <= x < T and x + h >= T and z >= T:
            if p2 > beta:
                cost += CP + CI
                life += T
                counters[9] += 1
                decision = 1
            else:
                x = 0.0
                h = h - (T - x)
                cost += CI
                life += T
                counters[10] += 1
                decision = 0

        elif (S <= x <= z <= x + h < T and p > beta2) or (
            S <= x <= z < T and x + h >= T and p > beta2
        ):
            cost += CO + CP
            life += z
            counters[11] += 1
            decision = 1

        elif S <= x <= z <= x + h < T and p <= beta2:
            cost += CF + CO
            life += x + h
            counters[12] += 1
            decision = 1

        elif S <= z <= x and x + h < T:
            cost += CO + CF
            life += x + h
            counters[13] += 1
            decision = 1

        elif S <= z <= x < T and x + h >= T and p2 > beta:
            cost += CO + CP + CI
            life += T
            counters[14] += 1
            decision = 1

        elif S <= z <= x < T and x + h >= T and p2 <= beta:
            x = 0.0
            h = h - T
            cost += CO + CI
            life += T
            counters[15] += 1
            decision = 0

        elif S <= z < T and x >= T:
            x = x - T
            cost += CO + CI
            life += T
            counters[16] += 1
            decision = 0

        elif S <= x <= z < T and x + h >= T and p <= beta2:
            if p2 > beta:
                cost += CO + CP + CI
                life += T
                counters[17] += 1
                decision = 1
            else:
                x = 0.0
                h = h - (T - x)
                cost += CI + CO
                life += T
                counters[18] += 1
                decision = 0

        elif x >= T and z >= T:
            x = x - T
            cost += CI
            life += T
            counters[19] += 1
            decision = 0

        else:
            cost += CF
            life += max(1e-12, min(T, x + h))
            decision = 1

    failure_count = counters[0] + counters[1] + counters[3] + counters[8] + counters[12] + counters[13]
    cost_rate = cost / life if life > 0 else np.nan
    mtbof = life / failure_count if failure_count > 0 else np.inf
    lom = (counters[3] + counters[6] + counters[7] + counters[12] + counters[17] + counters[18]) / n
    pfrbo = (counters[2] + counters[11]) / n

    return {
        "Cost rate": cost_rate,
        "MTBOF": mtbof,
        "PFRBO": pfrbo,
        "LOM": lom,
    }


# ---------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------
st.sidebar.markdown("## Policy inputs")
st.sidebar.caption("Insert policy values and system parameters. This version evaluates the policy without optimization.")

preset = st.sidebar.selectbox(
    "Base scenario",
    [
        "Article base case",
        "Original code example",
        "High opportunistic false negative",
        "High scheduled false negative",
        "Custom",
    ],
)

defaults = {
    "Article base case": dict(S=0.6076, T=2.1599, mu_x=2.0, mu_h=1.0, mu_z=1.0, c_i=0.5, c_o=0.2, c_p=1.0, c_f=5.0, beta_s=0.0, beta_o=0.0, n_cycles=100_000),
    "Original code example": dict(S=0.287, T=1.141, mu_x=2.0, mu_h=1.0, mu_z=0.5, c_i=0.6, c_o=0.3, c_p=1.0, c_f=10.0, beta_s=0.0, beta_o=0.30, n_cycles=100_000),
    "High opportunistic false negative": dict(S=0.50, T=2.00, mu_x=2.0, mu_h=1.0, mu_z=1.0, c_i=0.5, c_o=0.2, c_p=1.0, c_f=5.0, beta_s=0.0, beta_o=0.35, n_cycles=100_000),
    "High scheduled false negative": dict(S=0.70, T=2.50, mu_x=2.0, mu_h=1.0, mu_z=1.0, c_i=0.5, c_o=0.2, c_p=1.0, c_f=5.0, beta_s=0.25, beta_o=0.0, n_cycles=100_000),
    "Custom": dict(S=0.6076, T=2.1599, mu_x=2.0, mu_h=1.0, mu_z=1.0, c_i=0.5, c_o=0.2, c_p=1.0, c_f=5.0, beta_s=0.0, beta_o=0.0, n_cycles=100_000),
}
d = defaults[preset]

with st.sidebar.expander("Decision variables", expanded=True):
    S = st.number_input("S  Opportunity acceptance threshold", min_value=0.0, value=float(d["S"]), step=0.05, format="%.4f")
    T = st.number_input("T  Scheduled inspection interval", min_value=0.0001, value=float(d["T"]), step=0.05, format="%.4f")

with st.sidebar.expander("Reliability parameters", expanded=True):
    mu_x = st.number_input("μX  Mean time to defect arrival", min_value=0.0001, value=float(d["mu_x"]), step=0.10, format="%.4f")
    mu_h = st.number_input("μH  Mean delay time from defect to failure", min_value=0.0001, value=float(d["mu_h"]), step=0.10, format="%.4f")
    mu_z = st.number_input("μZ  Mean time between opportunities", min_value=0.000001, value=float(d["mu_z"]), step=0.10, format="%.6f")

with st.sidebar.expander("Cost parameters", expanded=True):
    c_f = st.number_input("CF  Corrective replacement cost", min_value=0.0, value=float(d["c_f"]), step=0.50, format="%.4f")
    c_p = st.number_input("CP  Preventive replacement cost", min_value=0.0, value=float(d["c_p"]), step=0.10, format="%.4f")
    c_i = st.number_input("CI  Scheduled inspection cost", min_value=0.0, value=float(d["c_i"]), step=0.10, format="%.4f")
    c_o = st.number_input("CO  Opportunistic inspection cost", min_value=0.0, value=float(d["c_o"]), step=0.10, format="%.4f")

with st.sidebar.expander("Inspection quality", expanded=True):
    beta_s = st.slider("βs  False negative probability in scheduled inspections", min_value=0.0, max_value=1.0, value=float(d["beta_s"]), step=0.01)
    beta_o = st.slider("βo  False negative probability in opportunistic inspections", min_value=0.0, max_value=1.0, value=float(d["beta_o"]), step=0.01)

with st.sidebar.expander("Simulation settings", expanded=False):
    n_cycles = st.number_input("Number of simulated decision steps", min_value=1_000, max_value=2_000_000, value=int(d["n_cycles"]), step=10_000)
    seed = st.number_input("Random seed", min_value=0, max_value=999_999, value=42, step=1)

run_button = st.sidebar.button("Run policy evaluation", type="primary", use_container_width=True)


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <span class="badge">Monte Carlo simulation</span>
        <span class="badge">Opportunistic inspection</span>
        <span class="badge">Misclassification errors</span>
        <h1>Monte Carlo simulation for opportunistic inspection planning under misclassification errors</h1>
        <p>Analytical app for evaluating a fixed pair <b>(S, T)</b> in a scheduled and opportunistic inspection policy with false negative errors.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="motion-box">
        <div class="motion-line"></div>
        <div class="motion-marker-s"></div>
        <div class="motion-marker-t"></div>
        <div class="motion-label-s">S</div>
        <div class="motion-label-t">T</div>
        <div class="motion-dot"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_run, tab_policy, tab_metrics, tab_authors = st.tabs(
    ["Run model", "Policy description", "Metrics and interpretation", "Authors and optimizer"]
)


# ---------------------------------------------------------------------
# Tab 1: Run model
# ---------------------------------------------------------------------
with tab_run:
    st.markdown(
        """
        <div class="card">
        <b>Policy structure.</b> The user defines the threshold <b>S</b>, from which opportunities may be accepted, and the scheduled inspection interval <b>T</b>. The app estimates only the four main outputs: cost rate, MTBOF, PFRBO, and LOM.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if S >= T:
        st.markdown(
            """
            <div class="warning-box">
            <b>Invalid policy.</b> The opportunity threshold <b>S</b> must be smaller than the scheduled inspection interval <b>T</b>. Please set <b>S &lt; T</b> before running the model.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        if c_f < c_p:
            st.warning("CF is smaller than CP. This is allowed for testing, but corrective replacement is usually more expensive than preventive replacement.")

        if not run_button:
            st.info("Adjust the parameters in the sidebar and click **Run policy evaluation**.")
        else:
            inputs = PolicyInputs(
                S=S,
                T=T,
                mu_x=mu_x,
                mu_h=mu_h,
                mu_z=mu_z,
                c_i=c_i,
                c_o=c_o,
                c_p=c_p,
                c_f=c_f,
                beta_s=beta_s,
                beta_o=beta_o,
                n_cycles=int(n_cycles),
                seed=int(seed),
            )

            with st.spinner("Running Monte Carlo simulation..."):
                results = run_policy_simulation(inputs)

            st.markdown(
                '<div class="success-box"><b>Simulation completed.</b> Results below evaluate the selected policy only. No optimization was performed.</div>',
                unsafe_allow_html=True,
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Cost rate", f"{results['Cost rate']:.4f}")
            m2.metric("MTBOF", "∞" if np.isinf(results["MTBOF"]) else f"{results['MTBOF']:.4f}")
            m3.metric("PFRBO", f"{results['PFRBO']:.4f}")
            m4.metric("LOM", f"{results['LOM']:.4f}")

            st.caption("Cost rate is the long-run cost per unit of simulated operating time. MTBOF is the mean time between operational failures. PFRBO measures successful failure prevention by opportunities. LOM measures opportunity loss caused by false negative effects.")

            st.markdown("### Inputs used in this run")
            input_df = pd.DataFrame(
                [
                    ["S", "Opportunity acceptance threshold", S],
                    ["T", "Scheduled inspection interval", T],
                    ["μX", "Mean time to defect arrival", mu_x],
                    ["μH", "Mean delay time", mu_h],
                    ["μZ", "Mean time between opportunities", mu_z],
                    ["CF", "Corrective replacement cost", c_f],
                    ["CP", "Preventive replacement cost", c_p],
                    ["CI", "Scheduled inspection cost", c_i],
                    ["CO", "Opportunistic inspection cost", c_o],
                    ["βs", "False negative probability in scheduled inspections", beta_s],
                    ["βo", "False negative probability in opportunistic inspections", beta_o],
                    ["N", "Number of simulated decision steps", int(n_cycles)],
                ],
                columns=["Symbol", "Factor", "Value"],
            )
            st.dataframe(input_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------
# Tab 2: Policy description
# ---------------------------------------------------------------------
with tab_policy:
    st.markdown("## What is this policy?")

    st.markdown(
        """
        <div class="card">
        The policy combines two inspection modes for a critical system. The first mode is an opportunistic inspection, which may be performed when an external operational event creates a favorable inspection moment. The second mode is a scheduled inspection, which occurs at a planned time limit.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_s, col_t = st.columns(2)

    with col_s:
        st.markdown(
            """
            <div class="dark-card">
            <h3>S</h3>
            <p><b>Opportunity acceptance threshold.</b></p>
            <p>Opportunities that occur before S are ignored. Opportunities that occur after S may be used for inspection.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_t:
        st.markdown(
            """
            <div class="dark-card">
            <h3>T</h3>
            <p><b>Scheduled inspection interval.</b></p>
            <p>If no renewal occurs before T, a scheduled inspection is performed at T.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        ### System behavior

        The system starts in a good state. A defect may appear after a random time X. After the defect appears, the system remains operational for a delay time H. If the defect is not detected before the end of this delay time, functional failure occurs. If an inspection detects the defect before failure, preventive replacement is performed.

        ### Inspection quality

        The app allows two false negative probabilities. The parameter βs represents the probability of missing a defect during a scheduled inspection. The parameter βo represents the probability of missing a defect during an opportunistic inspection.
        """
    )

    st.markdown(
        """
        <div class="info-box">
        <b>Important.</b> This software evaluates the policy for the values of S and T selected by the user. It does not search for the optimal values of S and T.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Tab 3: Metrics and interpretation
# ---------------------------------------------------------------------
with tab_metrics:
    st.markdown("## Metrics and interpretation")

    col_1, col_2 = st.columns(2)

    with col_1:
        st.markdown(
            """
            <div class="card">
            <b>Cost rate</b><br>
            Long-run cost per unit of simulated operating time. Lower values indicate a cheaper policy for the selected parameters.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="card">
            <b>MTBOF</b><br>
            Mean time between operational failures. Higher values indicate longer expected operation between failures.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_2:
        st.markdown(
            """
            <div class="card">
            <b>PFRBO</b><br>
            Potential Failure Reduction by Opportunities. It measures the proportion of simulated cycles in which an opportunity successfully detects a defect and prevents a potential failure.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="card">
            <b>LOM</b><br>
            Lost Opportunity by Misclassification. It measures the proportion of simulated cycles in which an opportunity is present, but its value is lost due to false negative effects.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info("A good policy is not defined by one metric alone. The decision maker should compare cost rate, MTBOF, PFRBO, and LOM together.")


# ---------------------------------------------------------------------
# Tab 4: Authors and optimizer
# ---------------------------------------------------------------------
with tab_authors:
    st.markdown("## Authors")

    authors_df = pd.DataFrame(
        [
            ["Augusto José da Silva Rodrigues", "Universidade Federal de Pernambuco", "RANDOM"],
            ["Rodrigo Sampaio Lopes", "Universidade Federal de São João del-Rei", "RANDOM"],
            ["Yan Ribeiro de Melo", "Universidade Federal de Pernambuco", "RANDOM"],
            ["Cristiano Alexandre Virginio Cavalcante", "Universidade Federal de Pernambuco", "RANDOM"],
            ["Hanser Steven Jiménez González", "Université de Lorraine", "RANDOM"],
        ],
        columns=["Author", "Institution", "Research group"],
    )
    st.dataframe(authors_df, use_container_width=True, hide_index=True)

    st.markdown("## About optimization")

    st.markdown(
        """
        <div class="error-soft">
        <b>This public app does not optimize S and T.</b> It only evaluates the policy for values chosen by the user. It does not run differential evolution and it does not provide the optimal policy.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        The optimizer used in the research version searches for the pair of decision variables that minimizes the long-run cost rate. To obtain the optimization version, including the differential evolution procedure and additional analyses, please contact the authors.
        """
    )

    st.markdown("### Contact")
    st.write("Corresponding author: Cristiano Alexandre Virginio Cavalcante")
    st.write("E-mail: cristiano.avcavalcante@ufpe.br")


st.markdown("---")
st.caption("RANDOM, Research Group on Risk and Decision Analysis in Operations and Maintenance.")
