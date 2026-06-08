# -*- coding: utf-8 -*-
"""
Streamlit app for the paper:
Monte Carlo simulation for opportunistic inspection planning under misclassification errors

This app evaluates a user-defined policy (S, T) by Monte Carlo simulation.
It does not optimize 𝑆 and 𝑇 and does not run differential evolution.
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
    initial_sidebar_state="collapsed",
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

.stApp {
    background:
        radial-gradient(circle at 7% 0%, rgba(59, 130, 246, 0.13), transparent 25%),
        radial-gradient(circle at 92% 4%, rgba(20, 184, 166, 0.12), transparent 26%),
        linear-gradient(180deg, #f8fafc 0%, #ffffff 45%, #f8fafc 100%);
}

.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2.1rem;
    max-width: 1220px;
}

[data-testid="stSidebar"] {
    display: none;
}

.hero {
    padding: 1.25rem 1.25rem 1.0rem 1.25rem;
    border-radius: 22px;
    background:
        radial-gradient(circle at 0% 0%, rgba(34, 211, 238, 0.28), transparent 32%),
        linear-gradient(135deg, #020617 0%, #0f172a 45%, #1e3a8a 100%);
    color: white;
    border: 1px solid rgba(226, 232, 240, 0.12);
    box-shadow: 0 18px 42px rgba(15, 23, 42, 0.20);
    margin-bottom: 0.70rem;
}

.hero h1 {
    font-size: clamp(1.12rem, 1.55vw, 1.50rem);
    line-height: 1.16;
    margin-bottom: 0.28rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    max-width: 860px;
}

.hero p {
    font-size: 0.84rem;
    color: rgba(255, 255, 255, 0.88);
    margin-bottom: 0;
    max-width: 850px;
}

.badge {
    display: inline-block;
    padding: 0.17rem 0.46rem;
    border-radius: 999px;
    background: rgba(14, 165, 233, 0.16);
    color: #ffffff;
    border: 1px solid rgba(125, 211, 252, 0.22);
    font-size: 0.65rem;
    font-weight: 800;
    margin-right: 0.22rem;
    margin-bottom: 0.42rem;
}

.motion-box {
    height: 66px;
    border-radius: 22px;
    background:
        radial-gradient(circle at 10% 20%, rgba(34, 211, 238, 0.25), transparent 32%),
        linear-gradient(135deg, #020617 0%, #0f172a 48%, #0f766e 100%);
    border: 1px solid rgba(226, 232, 240, 0.12);
    position: relative;
    overflow: hidden;
    box-shadow: 0 18px 42px rgba(15, 23, 42, 0.18);
    margin-bottom: 0.55rem;
}

.motion-line {
    position: absolute;
    left: 12%;
    right: 12%;
    top: 34px;
    height: 3px;
    background: rgba(226, 232, 240, 0.62);
    border-radius: 999px;
}

.motion-dot {
    position: absolute;
    top: 22px;
    left: 12%;
    width: 24px;
    height: 24px;
    background: #22d3ee;
    border: 3px solid white;
    border-radius: 50%;
    animation: moveDot 3.0s linear infinite;
}

.motion-marker-s {
    position: absolute;
    top: 16px;
    left: 34%;
    width: 4px;
    height: 38px;
    background: #38bdf8;
    border-radius: 999px;
}

.motion-marker-t {
    position: absolute;
    top: 16px;
    left: 78%;
    width: 4px;
    height: 38px;
    background: #34d399;
    border-radius: 999px;
}

.motion-label-s {
    position: absolute;
    top: 42px;
    left: 33%;
    color: #ffffff;
    font-weight: 800;
}

.motion-label-t {
    position: absolute;
    top: 42px;
    left: 77%;
    color: #ffffff;
    font-weight: 800;
}

@keyframes moveDot {
    0% { left: 12%; }
    100% { left: 86%; }
}

/* Compact animated maintenance visual */
.maintenance-gif {
    height: 108px;
    border-radius: 22px;
    background:
        radial-gradient(circle at 25% 40%, rgba(34, 211, 238, 0.18), transparent 32%),
        radial-gradient(circle at 82% 30%, rgba(52, 211, 153, 0.14), transparent 30%),
        linear-gradient(135deg, #020617 0%, #111827 50%, #0f766e 100%);
    position: relative;
    overflow: hidden;
    box-shadow: 0 16px 38px rgba(15, 23, 42, 0.16);
    margin: 0.1rem 0 0.85rem 0;
}

.gear-a, .gear-b {
    position: absolute;
    border-radius: 50%;
    border: 7px solid #67e8f9;
    animation: spinGear 4s linear infinite;
}

.gear-a {
    width: 58px;
    height: 58px;
    top: 23px;
    left: 42px;
}

.gear-b {
    width: 38px;
    height: 38px;
    top: 48px;
    left: 100px;
    border-color: #34d399;
    animation-direction: reverse;
    animation-duration: 3s;
}

.gear-a::before, .gear-b::before {
    content: "";
    position: absolute;
    border-radius: 50%;
    background: #0f172a;
    border: 4px solid #e0f2fe;
}

.gear-a::before {
    width: 18px;
    height: 18px;
    top: 13px;
    left: 13px;
}

.gear-b::before {
    width: 10px;
    height: 10px;
    top: 7px;
    left: 7px;
}

.signal-line {
    position: absolute;
    left: 180px;
    right: 80px;
    top: 54px;
    height: 4px;
    background: linear-gradient(90deg, rgba(226,232,240,0.2), rgba(103,232,249,0.85), rgba(52,211,153,0.8), rgba(226,232,240,0.2));
    border-radius: 999px;
}

.signal-dot {
    position: absolute;
    top: 44px;
    left: 180px;
    width: 24px;
    height: 24px;
    background: #22d3ee;
    border: 3px solid white;
    border-radius: 50%;
    animation: moveSignal 2.4s ease-in-out infinite;
}

.node {
    position: absolute;
    width: 18px;
    height: 18px;
    background: #22c55e;
    border: 3px solid white;
    border-radius: 50%;
    top: 46px;
    right: 44px;
    box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.55);
    animation: pulseNode 1.7s infinite;
}

.mini-maintenance-gif {
    height: 76px;
    border-radius: 20px;
    background:
        radial-gradient(circle at 20% 35%, rgba(34, 211, 238, 0.22), transparent 30%),
        linear-gradient(135deg, #020617 0%, #0f172a 55%, #1e3a8a 100%);
    position: relative;
    overflow: hidden;
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.14);
    margin: 0.2rem 0 0.8rem 0;
}

.mini-gear-a, .mini-gear-b {
    position: absolute;
    border-radius: 50%;
    border: 6px solid #67e8f9;
    animation: spinGear 4s linear infinite;
}

.mini-gear-a {
    width: 44px;
    height: 44px;
    top: 16px;
    left: 34px;
}

.mini-gear-b {
    width: 30px;
    height: 30px;
    top: 32px;
    left: 80px;
    border-color: #34d399;
    animation-direction: reverse;
    animation-duration: 3s;
}

.mini-gear-a::before, .mini-gear-b::before {
    content: "";
    position: absolute;
    border-radius: 50%;
    background: #0f172a;
    border: 3px solid #e0f2fe;
}

.mini-gear-a::before {
    width: 14px;
    height: 14px;
    top: 9px;
    left: 9px;
}

.mini-gear-b::before {
    width: 8px;
    height: 8px;
    top: 5px;
    left: 5px;
}

.mini-pulse {
    position: absolute;
    width: 16px;
    height: 16px;
    background: #22c55e;
    border: 3px solid white;
    border-radius: 50%;
    top: 29px;
    right: 48px;
    box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.55);
    animation: pulseNode 1.7s infinite;
}

@keyframes spinGear {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

@keyframes moveSignal {
    0% { left: 180px; opacity: 0.35; }
    50% { opacity: 1; }
    100% { left: calc(100% - 105px); opacity: 0.35; }
}

@keyframes pulseNode {
    0% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.55); }
    70% { box-shadow: 0 0 0 16px rgba(34, 197, 94, 0.00); }
    100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.00); }
}

.info-box {
    border-left: 5px solid #0284c7;
    background: #eff6ff;
    color: #0c4a6e;
    padding: 0.78rem 0.95rem;
    border-radius: 14px;
    margin: 0.65rem 0;
    font-size: 0.92rem;
}

.warning-box {
    border-left: 5px solid #f59e0b;
    background: #fffbeb;
    color: #78350f;
    padding: 0.82rem 0.95rem;
    border-radius: 14px;
    margin: 0.70rem 0;
}

.success-box {
    border-left: 5px solid #10b981;
    background: #ecfdf5;
    color: #064e3b;
    padding: 0.82rem 0.95rem;
    border-radius: 14px;
    margin: 0.70rem 0;
}

.error-soft {
    border-left: 5px solid #ef4444;
    background: #fef2f2;
    color: #7f1d1d;
    padding: 0.82rem 0.95rem;
    border-radius: 14px;
    margin: 0.70rem 0;
}

.dark-card {
    border-radius: 20px;
    padding: 1.05rem 1.1rem;
    background:
        radial-gradient(circle at top right, rgba(14, 165, 233, 0.18), transparent 38%),
        linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: white;
    box-shadow: 0 16px 36px rgba(15, 23, 42, 0.18);
    margin-bottom: 1rem;
    min-height: 140px;
}

.dark-card h3 {
    color: white;
    font-size: 1.8rem;
    margin-bottom: 0.10rem;
}

.dark-card p {
    color: rgba(255, 255, 255, 0.86);
}

.compact-panel {
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid rgba(15, 23, 42, 0.08);
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.07);
    padding: 0.9rem 1rem 1rem 1rem;
    margin-bottom: 0.80rem;
}

div[data-testid="stExpander"] {
    border-radius: 18px !important;
    background: rgba(255,255,255,0.96) !important;
    border: 1px solid rgba(15, 23, 42, 0.08) !important;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
}

.stButton > button {
    border-radius: 14px !important;
    height: 3rem;
    font-weight: 800 !important;
    font-size: 1.0rem !important;
    background: linear-gradient(135deg, #1d4ed8, #0f766e) !important;
    border: 0 !important;
    color: white !important;
    box-shadow: 0 10px 24px rgba(29, 78, 216, 0.22);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1e40af, #0f766e) !important;
    color: white !important;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid rgba(15, 23, 42, 0.08);
    padding: 1rem;
    border-radius: 18px;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.07);
}

hr {
    margin-top: 0.6rem;
    margin-bottom: 0.6rem;
}

.presentation-grid {
    display: grid;
    grid-template-columns: 1.1fr 0.9fr;
    gap: 1rem;
    margin-top: 0.9rem;
    margin-bottom: 1rem;
}

.presentation-main {
    border-radius: 24px;
    padding: 1.2rem 1.25rem;
    background:
        radial-gradient(circle at 0% 0%, rgba(34, 211, 238, 0.18), transparent 35%),
        linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid rgba(15, 23, 42, 0.08);
    box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
}

.presentation-main h2 {
    font-size: 1.35rem;
    line-height: 1.22;
    margin-bottom: 0.55rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.02em;
}

.presentation-main p {
    color: #334155;
    font-size: 0.96rem;
    line-height: 1.58;
    margin-bottom: 0;
}

.presentation-side {
    border-radius: 24px;
    padding: 1.05rem 1.15rem;
    background:
        radial-gradient(circle at 100% 0%, rgba(20, 184, 166, 0.20), transparent 35%),
        linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
    color: white;
    box-shadow: 0 16px 36px rgba(15, 23, 42, 0.18);
}

.presentation-side h3 {
    font-size: 1.0rem;
    margin-bottom: 0.6rem;
    color: white;
}

.presentation-side ul {
    margin: 0;
    padding-left: 1.1rem;
}

.presentation-side li {
    margin-bottom: 0.45rem;
    color: rgba(255, 255, 255, 0.90);
    font-size: 0.92rem;
}

.feature-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin: 0.8rem 0 1.0rem 0;
}

.feature-card {
    border-radius: 18px;
    padding: 0.95rem 0.95rem;
    background: rgba(255, 255, 255, 0.96);
    border: 1px solid rgba(15, 23, 42, 0.08);
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
    min-height: 118px;
}

.feature-card h3 {
    font-size: 0.98rem;
    margin: 0 0 0.35rem 0;
    color: #0f172a;
    font-weight: 800;
}

.feature-card p {
    font-size: 0.86rem;
    margin: 0;
    color: #475569;
    line-height: 1.45;
}

.policy-mini {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.85rem;
    margin-top: 0.7rem;
}

.policy-mini-card {
    border-radius: 20px;
    padding: 1rem;
    background:
        radial-gradient(circle at top right, rgba(34, 211, 238, 0.18), transparent 36%),
        linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: white;
    min-height: 130px;
    box-shadow: 0 14px 32px rgba(15, 23, 42, 0.14);
}

.policy-mini-card h3 {
    font-size: 1.6rem;
    color: white;
    margin: 0 0 0.25rem 0;
}

.policy-mini-card p {
    font-size: 0.90rem;
    color: rgba(255, 255, 255, 0.88);
    margin: 0;
    line-height: 1.45;
}

@media (max-width: 900px) {
    .presentation-grid {
        grid-template-columns: 1fr;
    }
    .feature-row {
        grid-template-columns: 1fr 1fr;
    }
    .policy-mini {
        grid-template-columns: 1fr;
    }
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
# Header
# ---------------------------------------------------------------------
top_left, top_right = st.columns([0.72, 0.28], vertical_alignment="center")

with top_left:
    st.markdown(
        """
        <div class="hero">
            <span class="badge">Monte Carlo simulation</span>
            <span class="badge">Opportunistic inspection</span>
            <span class="badge">Misclassification errors</span>
            <h1>Monte Carlo simulation for opportunistic inspection planning under misclassification errors</h1>
            <p>Evaluate a fixed pair <b>(S, T)</b> in a scheduled and opportunistic inspection policy with false negative errors.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_right:
    st.markdown(
        """
        <div class="motion-box">
            <div class="motion-line"></div>
            <div class="motion-marker-s"></div>
            <div class="motion-marker-t"></div>
            <div class="motion-label-s">𝑆</div>
            <div class="motion-label-t">𝑇</div>
            <div class="motion-dot"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


tab_home, tab_run, tab_policy, tab_metrics, tab_authors = st.tabs(
    ["Presentation", "Run model", "Policy description", "Metrics and interpretation", "Authors and optimizer"]
)


# ---------------------------------------------------------------------
# Tab 1: Presentation
# ---------------------------------------------------------------------
with tab_home:
    st.markdown(
        """
        <div class="presentation-grid">
            <div class="presentation-main">
                <h2>Analytical interface for the inspection policy</h2>
                <p>
                This application evaluates the maintenance policy proposed in the article
                <b>Monte Carlo simulation for opportunistic inspection planning under misclassification errors</b>.
                The public version is intended for analytical evaluation. The user defines <i>S</i> and <i>T</i>,
                inserts the model parameters, and obtains the four main performance outputs.
                </p>
            </div>
            <div class="presentation-side">
                <h3>What this version provides</h3>
                <ul>
                    <li>Monte Carlo evaluation for a fixed policy;</li>
                    <li>false negative errors in scheduled and opportunistic inspections;</li>
                    <li>cost and reliability performance indicators;</li>
                    <li>no optimization of <i>S</i> and <i>T</i>.</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="feature-row">
            <div class="feature-card">
                <h3>Policy evaluation</h3>
                <p>The app evaluates one pair of decision variables selected by the user.</p>
            </div>
            <div class="feature-card">
                <h3>Monte Carlo simulation</h3>
                <p>The stochastic process is simulated to estimate long-run performance.</p>
            </div>
            <div class="feature-card">
                <h3>Inspection quality</h3>
                <p>False negative probabilities can be assigned to both inspection modes.</p>
            </div>
            <div class="feature-card">
                <h3>Research version</h3>
                <p>The optimizer is not public. Contact the authors for the optimization version.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Policy idea")

    st.markdown(
        """
        <div class="policy-mini">
            <div class="policy-mini-card">
                <h3><i>S</i></h3>
                <p><b>Opportunity acceptance threshold.</b><br>
                Opportunities before <i>S</i> are ignored. Opportunities after <i>S</i> may be used for inspection.</p>
            </div>
            <div class="policy-mini-card">
                <h3><i>T</i></h3>
                <p><b>Scheduled inspection interval.</b><br>
                If no renewal occurs before <i>T</i>, a scheduled inspection is performed at <i>T</i>.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Tab 2: Run model
# ---------------------------------------------------------------------
with tab_run:
    st.markdown(
        """
        <div class="info-box">
        Insert the policy values and model parameters below. This tab runs the analytical evaluation of the selected policy and does not optimize <b><i>S</i></b> and <b><i>T</i></b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("policy_form"):
        with st.container(border=True):
            st.markdown("#### Decision variables")
            d1, d2 = st.columns(2)
            with d1:
                S = st.number_input(
                    "𝑆  Opportunity acceptance threshold",
                    min_value=0.0,
                    value=0.6076,
                    step=0.05,
                    format="%.4f",
                    help="Opportunities before <i>S</i> are ignored. Opportunities after <i>S</i> may be used for inspection.",
                )
            with d2:
                T = st.number_input(
                    "𝑇  Scheduled inspection interval",
                    min_value=0.0001,
                    value=2.1599,
                    step=0.05,
                    format="%.4f",
                    help="Scheduled inspection is performed at <i>T</i> if no renewal occurs before that time.",
                )

        with st.expander("Reliability parameters", expanded=True):
            r1, r2, r3 = st.columns(3)
            with r1:
                mu_x = st.number_input(
                    "μX  Mean time to defect arrival",
                    min_value=0.0001,
                    value=2.0,
                    step=0.10,
                    format="%.4f",
                )
            with r2:
                mu_h = st.number_input(
                    "μH  Mean delay time from defect to failure",
                    min_value=0.0001,
                    value=1.0,
                    step=0.10,
                    format="%.4f",
                )
            with r3:
                mu_z = st.number_input(
                    "μZ  Mean time between opportunities",
                    min_value=0.000001,
                    value=1.0,
                    step=0.10,
                    format="%.6f",
                )

        with st.expander("Cost parameters", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                c_f = st.number_input(
                    "CF  Corrective replacement cost",
                    min_value=0.0,
                    value=5.0,
                    step=0.50,
                    format="%.4f",
                )
            with c2:
                c_p = st.number_input(
                    "CP  Preventive replacement cost",
                    min_value=0.0,
                    value=1.0,
                    step=0.10,
                    format="%.4f",
                )
            with c3:
                c_i = st.number_input(
                    "CI  Scheduled inspection cost",
                    min_value=0.0,
                    value=0.5,
                    step=0.10,
                    format="%.4f",
                )
            with c4:
                c_o = st.number_input(
                    "CO  Opportunistic inspection cost",
                    min_value=0.0,
                    value=0.2,
                    step=0.10,
                    format="%.4f",
                )

        with st.expander("Inspection quality and simulation", expanded=True):
            q1, q2, q3, q4 = st.columns([1.2, 1.2, 1.0, 0.85])
            with q1:
                beta_s = st.slider(
                    "βs  Scheduled inspection false negative",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.01,
                )
            with q2:
                beta_o = st.slider(
                    "βo  Opportunistic inspection false negative",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.01,
                )
            with q3:
                n_cycles = st.number_input(
                    "Number of simulated decision steps",
                    min_value=1_000,
                    max_value=2_000_000,
                    value=100_000,
                    step=10_000,
                )
            with q4:
                seed = st.number_input(
                    "Random seed",
                    min_value=0,
                    max_value=999_999,
                    value=42,
                    step=1,
                )

        submitted = st.form_submit_button("Run policy evaluation", type="primary", use_container_width=True)

    if S >= T:
        st.markdown(
            """
            <div class="warning-box">
            <b>Invalid policy.</b> The opportunity threshold <b><i>S</i></b> must be smaller than the scheduled inspection interval <b><i>T</i></b>. Please set <b><i>S</i> &lt; <i>T</i></b> before running the model.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif c_f < c_p:
        st.warning("CF is smaller than CP. This is allowed for testing, but corrective replacement is usually more expensive than preventive replacement.")

    if submitted and S < T:
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

        st.markdown("## Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cost rate", f"{results['Cost rate']:.4f}")
        m2.metric("MTBOF", "∞" if np.isinf(results["MTBOF"]) else f"{results['MTBOF']:.4f}")
        m3.metric("PFRBO", f"{results['PFRBO']:.4f}")
        m4.metric("LOM", f"{results['LOM']:.4f}")

        st.caption("Cost rate is the long-run cost per unit of simulated operating time. MTBOF is the mean time between operational failures. PFRBO measures successful failure prevention by opportunities. LOM measures opportunity loss caused by false negative effects.")
    elif not submitted:
        st.markdown(
            """
            <div class="info-box">
            After inserting the parameters, click <b>Run policy evaluation</b>. The app will show only the four main outputs used in the paper.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------
# Tab 3: Policy description
# ---------------------------------------------------------------------
with tab_policy:
    st.markdown("## What is this policy?")

    st.markdown(
        """
        <div class="info-box">
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
            <h3><i>S</i></h3>
            <p><b>Opportunity acceptance threshold.</b></p>
            <p>Opportunities that occur before <i>S</i> are ignored. Opportunities that occur after <i>S</i> may be used for inspection.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_t:
        st.markdown(
            """
            <div class="dark-card">
            <h3><i>T</i></h3>
            <p><b>Scheduled inspection interval.</b></p>
            <p>If no renewal occurs before <i>T</i>, a scheduled inspection is performed at <i>T</i>.</p>
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
        <b>Important.</b> This software evaluates the policy for the values of 𝑆 and 𝑇 selected by the user. It does not search for the optimal values of 𝑆 and 𝑇.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Tab 4: Metrics and interpretation
# ---------------------------------------------------------------------
with tab_metrics:
    st.markdown("## Metrics and interpretation")

    col_1, col_2 = st.columns(2)

    with col_1:
        with st.container(border=True):
            st.markdown("### Cost rate")
            st.write("Long-run cost per unit of simulated operating time. Lower values indicate a cheaper policy for the selected parameters.")

        with st.container(border=True):
            st.markdown("### MTBOF")
            st.write("Mean time between operational failures. Higher values indicate longer expected operation between failures.")

    with col_2:
        with st.container(border=True):
            st.markdown("### PFRBO")
            st.write("Potential Failure Reduction by Opportunities. It measures the proportion of simulated cycles in which an opportunity successfully detects a defect and prevents a potential failure.")

        with st.container(border=True):
            st.markdown("### LOM")
            st.write("Lost Opportunity by Misclassification. It measures the proportion of simulated cycles in which an opportunity is present, but its value is lost due to false negative effects.")

    st.info("A good policy is not defined by one metric alone. The decision maker should compare cost rate, MTBOF, PFRBO, and LOM together.")


# ---------------------------------------------------------------------
# Tab 5: Authors and optimizer
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
        <b>This public app does not optimize 𝑆 and 𝑇.</b> It only evaluates the policy for values chosen by the user. It does not run differential evolution and it does not provide the optimal policy.
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
