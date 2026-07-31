"""
MovieIQ — Film Intelligence Platform
====================================
An interactive analytics platform that studies a movie dataset and predicts a
film's commercial outcome with a DUAL engine:

    1. Success probability  -> will revenue beat budget? (classification)
    2. Revenue & ROI estimate -> roughly how much will it earn? (regression)

A movie is "successful" when revenue > budget.

Run it (Streamlit needs a plain .py file, not a notebook):
    streamlit run MovieIQ.py

Structure (sidebar navigation acts like pages of a platform):
    Overview        - market KPIs and the budget/revenue landscape
    Genre Explorer  - deep-dive into a single genre
    Insights        - EDA + statistical tests, told as a story
    Model Lab       - train & compare three models, inspect importance
    Predictor       - the dual engine: probability gauge + revenue/ROI + why
"""

import ast
import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, precision_score, r2_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------
# Brand palette & page config
# ----------------------------------------------------------------------
GOLD = "#E8B923"
GREEN = "#2FBF71"
RED = "#E4572E"
BLUE = "#4C9BE8"
INK = "#0B0E1A"
CARD = "#151A2E"
MUTED = "#8A93A6"

NUMERIC_COLS = ["budget", "revenue", "popularity", "runtime", "vote_average"]
# Features fed to the models. We EXCLUDE `revenue` (it defines the success
# target -> data leakage) and `title` (a unique label, not a predictor).
FEATURES = ["budget", "popularity", "runtime", "vote_average"]

st.set_page_config(page_title="MovieIQ — Film Intelligence",
                   page_icon="🎬", layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------------------------------------------------------
# Custom styling (this is what makes it not look like a default app)
# ----------------------------------------------------------------------
st.markdown(f"""
<style>
    .stApp {{ background: radial-gradient(1200px 600px at 20% -10%,
              #1a2140 0%, {INK} 55%); }}
    #MainMenu, footer {{ visibility: hidden; }}
    .hero {{
        padding: 26px 30px; border-radius: 18px; margin-bottom: 8px;
        background: linear-gradient(120deg, #1c2444 0%, #10152a 100%);
        border: 1px solid rgba(232,185,35,0.25);
    }}
    .hero h1 {{ margin: 0; font-size: 2.15rem; letter-spacing: .5px;
        color: #fff; }}
    .hero h1 span {{ color: {GOLD}; }}
    .hero p {{ margin: 6px 0 0; color: {MUTED}; font-size: 1.02rem; }}
    .kpi {{
        background: {CARD}; border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px; padding: 16px 18px; height: 100%;
    }}
    .kpi .label {{ color: {MUTED}; font-size: .78rem; text-transform: uppercase;
        letter-spacing: 1px; }}
    .kpi .value {{ color: #fff; font-size: 1.7rem; font-weight: 700;
        margin-top: 4px; }}
    .kpi .value.gold {{ color: {GOLD}; }}
    .verdict {{
        border-radius: 16px; padding: 20px 24px; margin-top: 10px;
        font-size: 1.1rem; border: 1px solid rgba(255,255,255,0.08);
    }}
    .pill {{ display:inline-block; padding: 3px 12px; border-radius: 999px;
        font-size: .8rem; font-weight: 600; }}
    section[data-testid="stSidebar"] {{ background: #0c1122; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#EAECF2"),
    margin=dict(l=10, r=10, t=50, b=10),
)


# ----------------------------------------------------------------------
# Data layer  (Stage 1)
# ----------------------------------------------------------------------
@st.cache_data
def load_data(path: str = "movies.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop rows with 0/blank budget or revenue: a 0 usually means "unknown",
    # and a 0 revenue would falsely flag a film as a failure.
    df = df.dropna(subset=NUMERIC_COLS)
    df = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()
    df["success"] = (df["revenue"] > df["budget"]).astype(int)
    df["genre"] = df["genres"].apply(_primary_genre)
    df["roi"] = df["revenue"] / df["budget"]              # return multiple
    df["profit"] = df["revenue"] - df["budget"]
    return df


def _primary_genre(raw: str) -> str:
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list) and parsed:
            return parsed[0].get("name", "Unknown")
    except (ValueError, SyntaxError, TypeError):
        pass
    return "Unknown"


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df[FEATURES + ["genre"]], columns=["genre"])


# ----------------------------------------------------------------------
# Model layer  (Stage 4) — classification + regression engines
# ----------------------------------------------------------------------
@st.cache_resource
def train_engines(df: pd.DataFrame):
    X = _encode(df)
    y = df["success"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    candidates = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)),
    }
    rows, fitted = [], {}
    for name, model in candidates.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        fitted[name] = model
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(yte, pred),
            "Precision": precision_score(yte, pred, zero_division=0),
            "Recall": recall_score(yte, pred, zero_division=0),
            "F1": f1_score(yte, pred, zero_division=0),
        })
    scores = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)

    best_name = scores.iloc[0]["Model"]
    best_clf = fitted[best_name]
    conf = confusion_matrix(yte, best_clf.predict(Xte))

    # Feature importance (tree models expose it directly).
    if hasattr(best_clf, "feature_importances_"):
        importance = pd.Series(best_clf.feature_importances_, index=X.columns)
    else:  # logistic regression -> use |coefficients|
        importance = pd.Series(np.abs(best_clf[-1].coef_[0]), index=X.columns)
    importance = importance.sort_values(ascending=False)

    # Revenue regressor (the engine that actually carries signal here).
    reg = RandomForestRegressor(n_estimators=200, random_state=42)
    Xrtr, Xrte, yrtr, yrte = train_test_split(
        X, df["revenue"], test_size=0.20, random_state=42)
    reg.fit(Xrtr, yrtr)
    reg_r2 = r2_score(yrte, reg.predict(Xrte))
    reg_mae = mean_absolute_error(yrte, reg.predict(Xrte))

    return {
        "scores": scores, "best_name": best_name, "best_clf": best_clf,
        "confusion": conf, "baseline": yte.mean(), "importance": importance,
        "columns": list(X.columns), "reg": reg, "reg_r2": reg_r2, "reg_mae": reg_mae,
    }


def score_movie(engines, movie: dict) -> dict:
    row = pd.get_dummies(pd.DataFrame([movie]), columns=["genre"])
    row = row.reindex(columns=engines["columns"], fill_value=0)
    clf = engines["best_clf"]
    prob = float(clf.predict_proba(row)[0][1])
    est_rev = float(engines["reg"].predict(row)[0])
    return {"prob": prob, "est_revenue": est_rev,
            "est_profit": est_rev - movie["budget"],
            "est_roi": est_rev / movie["budget"]}


# ----------------------------------------------------------------------
# Small UI helpers
# ----------------------------------------------------------------------
def kpi(label, value, gold=False):
    cls = "value gold" if gold else "value"
    st.markdown(
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="{cls}">{value}</div></div>', unsafe_allow_html=True)


def money(x):
    return f"${x/1e6:,.1f}M" if abs(x) < 1e9 else f"${x/1e9:,.2f}B"


# ======================================================================
# APP
# ======================================================================
df = load_data()
engines = train_engines(df)
genres = sorted(df["genre"].unique())

# ---- Sidebar: navigation + global filters ----
with st.sidebar:
    st.markdown(f"<h2 style='color:{GOLD};margin-bottom:0'>🎬 MovieIQ</h2>"
                f"<p style='color:{MUTED};margin-top:2px'>Film Intelligence Platform</p>",
                unsafe_allow_html=True)
    page = st.radio("Navigate", ["Overview", "Genre Explorer", "Insights",
                                 "Model Lab", "Predictor"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Global filters")
    f_genres = st.multiselect("Genres", genres, default=genres)
    f_vote = st.slider("Minimum vote average",
                       float(df.vote_average.min()), float(df.vote_average.max()),
                       float(df.vote_average.min()), 0.1)

mask = df["genre"].isin(f_genres) & (df["vote_average"] >= f_vote)
fdf = df[mask]

# ---- Shared hero header ----
st.markdown(
    "<div class='hero'><h1>Movie<span>IQ</span> · Film Intelligence</h1>"
    "<p>Predict a film's commercial outcome from budget, popularity, runtime, "
    "rating and genre — with a success probability <b>and</b> a revenue estimate.</p></div>",
    unsafe_allow_html=True)

if fdf.empty and page in ("Overview", "Genre Explorer", "Insights"):
    st.warning("No movies match the current filters — widen them in the sidebar.")
    st.stop()


# ======================================================================
# PAGE: OVERVIEW
# ======================================================================
if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Films in view", f"{len(fdf):,}")
    with c2: kpi("Success rate", f"{fdf.success.mean():.0%}", gold=True)
    with c3: kpi("Median budget", money(fdf.budget.median()))
    with c4: kpi("Median revenue", money(fdf.revenue.median()))

    st.markdown("### The budget–revenue landscape")
    fig = px.scatter(
        fdf, x="budget", y="revenue", color="success", size="popularity",
        color_discrete_map={0: RED, 1: GREEN}, hover_data=["genre", "vote_average"],
        labels={"success": "Success"}, size_max=18)
    mx = max(fdf.budget.max(), fdf.revenue.max())
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines",
                             line=dict(dash="dash", color=MUTED), name="break-even"))
    fig.update_layout(**PLOTLY_LAYOUT, height=430,
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")
    st.caption("Bubbles above the dashed line earned more than they cost. "
               "Bubble size = popularity. Notice budget alone doesn't decide the outcome.")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Success rate by genre")
        g = fdf.groupby("genre")["success"].mean().sort_values()
        fig = px.bar(g, orientation="h", color=g.values,
                     color_continuous_scale=["#3a2a2a", GREEN])
        fig.update_layout(**PLOTLY_LAYOUT, height=360, showlegend=False,
                          coloraxis_showscale=False,
                          xaxis_title="share succeeding", yaxis_title="")
        st.plotly_chart(fig, width="stretch")
    with colB:
        st.markdown("#### Typical ROI by genre (median return multiple)")
        r = fdf.groupby("genre")["roi"].median().sort_values()
        fig = px.bar(r, orientation="h", color=r.values,
                     color_continuous_scale=["#2a2f3a", GOLD])
        fig.update_layout(**PLOTLY_LAYOUT, height=360, showlegend=False,
                          coloraxis_showscale=False,
                          xaxis_title="revenue ÷ budget", yaxis_title="")
        st.plotly_chart(fig, width="stretch")

    # Download the filtered slice
    buf = io.StringIO(); fdf.to_csv(buf, index=False)
    st.download_button("⬇ Download this filtered dataset (CSV)",
                       buf.getvalue(), "movieiq_filtered.csv", "text/csv")


# ======================================================================
# PAGE: GENRE EXPLORER
# ======================================================================
elif page == "Genre Explorer":
    pick = st.selectbox("Choose a genre to profile", genres)
    gdf = df[df["genre"] == pick]

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Films", f"{len(gdf):,}")
    with c2: kpi("Success rate", f"{gdf.success.mean():.0%}", gold=True)
    with c3: kpi("Median ROI", f"{gdf.roi.median():.2f}×")
    with c4: kpi("Avg vote", f"{gdf.vote_average.mean():.1f}")

    st.markdown(f"### How **{pick}** compares to all films")
    metrics = ["budget", "revenue", "popularity", "runtime", "vote_average"]
    comp = pd.DataFrame({
        pick: [gdf[m].median() for m in metrics],
        "All films": [df[m].median() for m in metrics],
    }, index=metrics)
    # Normalise each row to the all-films median so bars are comparable.
    norm = comp.div(comp["All films"], axis=0)
    fig = go.Figure()
    fig.add_bar(y=metrics, x=norm[pick], orientation="h", name=pick,
                marker_color=GOLD)
    fig.add_vline(x=1, line_dash="dash", line_color=MUTED)
    fig.update_layout(**PLOTLY_LAYOUT, height=360,
                      xaxis_title=f"{pick} median ÷ all-films median (1.0 = same)")
    st.plotly_chart(fig, width="stretch")

    st.markdown(f"### Budget vs revenue for {pick}")
    fig = px.scatter(gdf, x="budget", y="revenue", color="success",
                     color_discrete_map={0: RED, 1: GREEN}, size="popularity",
                     hover_data=["vote_average"], size_max=18)
    fig.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False)
    st.plotly_chart(fig, width="stretch")


# ======================================================================
# PAGE: INSIGHTS  (Stage 2 EDA + Stage 3 stats)
# ======================================================================
elif page == "Insights":
    st.markdown("### Correlation between numeric features")
    corr = fdf[NUMERIC_COLS + ["success"]].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(**PLOTLY_LAYOUT, height=430)
    st.plotly_chart(fig, width="stretch")
    st.caption("Values near +1 or −1 mean two features move together. Here they're "
               "weak — no single feature strongly tracks success.")

    st.markdown("### Does a feature really differ for hits vs flops?")
    left, right = st.columns([1, 1.3])
    with left:
        feat = st.selectbox("Feature", ["popularity", "vote_average", "runtime", "budget"])
        if fdf["success"].nunique() == 2:
            a = fdf[fdf.success == 1][feat]
            b = fdf[fdf.success == 0][feat]
            t, p = stats.ttest_ind(a, b, equal_var=False)
            st.metric("t-statistic", f"{t:.3f}")
            st.metric("p-value", f"{p:.4f}")
            if p < 0.05:
                st.success(f"p < 0.05 → **{feat}** differs significantly between "
                           "successful and failed films.")
            else:
                st.info(f"p ≥ 0.05 → no significant difference in **{feat}**.")
        else:
            st.warning("Need both hits and flops in view.")
    with right:
        fig = px.box(fdf, x="success", y=feat, color="success",
                     color_discrete_map={0: RED, 1: GREEN})
        fig.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False,
                          xaxis=dict(tickmode="array", tickvals=[0, 1],
                                     ticktext=["Flop", "Hit"]))
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Is genre associated with success? (Chi-square)")
    if fdf["success"].nunique() == 2:
        ct = pd.crosstab(fdf["genre"], fdf["success"])
        chi2, pchi, dof, _ = stats.chi2_contingency(ct)
        c1, c2 = st.columns(2)
        c1.metric("chi² statistic", f"{chi2:.3f}")
        c2.metric("p-value", f"{pchi:.4f}")
        st.info("A **p-value** is the chance of seeing a difference this big if none "
                "truly existed. Threshold used: 0.05. "
                + ("Genre **is** associated with success here."
                   if pchi < 0.05 else
                   "No evidence genre is associated with success here."))


# ======================================================================
# PAGE: MODEL LAB  (Stage 4)
# ======================================================================
elif page == "Model Lab":
    st.markdown("### Model comparison")
    st.caption("All models trained on budget, popularity, runtime, vote average "
               "and genre. Revenue and title are excluded on purpose.")

    scores = engines["scores"].copy()
    styled = scores.style.format({c: "{:.1%}" for c in
                                  ["Accuracy", "Precision", "Recall", "F1"]})
    st.dataframe(styled, width="stretch", hide_index=True)

    b1, b2, b3 = st.columns(3)
    with b1: kpi("Best model", engines["best_name"], gold=True)
    with b2: kpi("Baseline (always 'hit')", f"{engines['baseline']:.1%}")
    with b3: kpi("Revenue model R²", f"{engines['reg_r2']:.2f}")

    st.info("Honest read: every classifier lands close to the baseline — a film's "
            "*success flag* is hard to predict from these features. But the revenue "
            f"model explains about {engines['reg_r2']:.0%} of the variance in earnings, "
            "so predicting *how much* a film makes works far better than the yes/no flag.")

    colA, colB = st.columns(2)
    with colA:
        st.markdown(f"#### Confusion matrix — {engines['best_name']}")
        cm = engines["confusion"]
        fig = px.imshow(cm, text_auto="d", color_continuous_scale="Blues",
                        x=["Pred flop", "Pred hit"], y=["Actual flop", "Actual hit"])
        fig.update_layout(**PLOTLY_LAYOUT, height=360, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
        st.caption("Most errors sit in the 'actual flop' row — because hits dominate, "
                   "the model leans toward predicting success.")
    with colB:
        st.markdown("#### What drives the prediction")
        imp = engines["importance"].head(8).sort_values()
        fig = px.bar(imp, orientation="h", color=imp.values,
                     color_continuous_scale=["#2a2f3a", GOLD])
        fig.update_layout(**PLOTLY_LAYOUT, height=360, showlegend=False,
                          coloraxis_showscale=False, xaxis_title="importance",
                          yaxis_title="")
        st.plotly_chart(fig, width="stretch")


# ======================================================================
# PAGE: PREDICTOR  (Stage 5.3) — the dual engine
# ======================================================================
elif page == "Predictor":
    st.markdown("### 🔮 Score a film")
    st.caption("Enter a concept and get a success probability, an expected-revenue "
               "estimate, an ROI outlook, and a plain-language 'why'.")

    with st.form("predict"):
        c1, c2, c3 = st.columns(3)
        with c1:
            in_budget = st.number_input("Budget ($)", 100_000, 500_000_000,
                                        50_000_000, 1_000_000)
            in_genre = st.selectbox("Genre", genres)
        with c2:
            in_pop = st.slider("Popularity", 0.0, 100.0, 50.0)
            in_runtime = st.slider("Runtime (min)", 60, 240, 120)
        with c3:
            in_vote = st.slider("Vote average", 0.0, 10.0, 6.0, 0.1)
        submitted = st.form_submit_button("Run MovieIQ", type="primary",
                                          width="stretch")

    if submitted:
        movie = {"budget": in_budget, "popularity": in_pop, "runtime": in_runtime,
                 "vote_average": in_vote, "genre": in_genre}
        r = score_movie(engines, movie)

        # ---- Row 1: gauge + headline numbers ----
        g1, g2 = st.columns([1, 1])
        with g1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=r["prob"] * 100,
                number={"suffix": "%", "font": {"color": "#fff"}},
                title={"text": "Success probability", "font": {"color": MUTED}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": MUTED},
                    "bar": {"color": GOLD},
                    "steps": [{"range": [0, 50], "color": "#2a1c1c"},
                              {"range": [50, 100], "color": "#183025"}],
                    "threshold": {"line": {"color": "#fff", "width": 3},
                                  "value": 50}}))
            gauge.update_layout(**PLOTLY_LAYOUT, height=300)
            st.plotly_chart(gauge, width="stretch")
        with g2:
            st.markdown("<br>", unsafe_allow_html=True)
            k1, k2 = st.columns(2)
            with k1: kpi("Estimated revenue", money(r["est_revenue"]), gold=True)
            with k2: kpi("Estimated ROI", f"{r['est_roi']:.2f}×")
            k3, k4 = st.columns(2)
            with k3: kpi("Estimated profit", money(r["est_profit"]))
            with k4: kpi("Budget", money(in_budget))

        # ---- Verdict banner ----
        hit = r["prob"] >= 0.5 and r["est_profit"] > 0
        colr = GREEN if hit else RED
        label = "LIKELY SUCCESS" if hit else "RISKY"
        st.markdown(
            f"<div class='verdict' style='background:rgba({'47,191,113' if hit else '228,87,46'},0.12);"
            f"border-color:{colr}'><span class='pill' style='background:{colr};color:#08110c'>"
            f"{label}</span>&nbsp;&nbsp;MovieIQ estimates this film earns "
            f"<b>{money(r['est_revenue'])}</b> against a <b>{money(in_budget)}</b> budget "
            f"— a projected <b>{money(r['est_profit'])}</b>.</div>",
            unsafe_allow_html=True)

        # ---- "Why": compare inputs to dataset medians ----
        st.markdown("#### Why this result")
        med = df[FEATURES].median()
        why_rows = []
        for f in FEATURES:
            diff = (movie[f] - med[f]) / med[f] * 100
            why_rows.append({"Feature": f, "Your film": movie[f],
                             "Typical film": round(med[f], 1),
                             "vs typical %": diff})
        why = pd.DataFrame(why_rows)
        fig = px.bar(why, x="vs typical %", y="Feature", orientation="h",
                     color="vs typical %", color_continuous_scale=["#E4572E", "#2a2f3a", GREEN],
                     color_continuous_midpoint=0)
        fig.update_layout(**PLOTLY_LAYOUT, height=300, coloraxis_showscale=False,
                          xaxis_title="how your film compares to a typical film (%)",
                          yaxis_title="")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Genre: **{in_genre}**. Bars to the right mean your film is above "
                   "the typical film on that feature. Note the success flag is only "
                   "weakly predictable — treat the probability as a guide, and lean on "
                   "the revenue/ROI estimate for the money question.")

        # ---- Downloadable report ----
        report = (f"MovieIQ prediction report\n{'='*30}\n"
                  f"Genre: {in_genre}\nBudget: {money(in_budget)}\n"
                  f"Popularity: {in_pop}  Runtime: {in_runtime}m  Vote: {in_vote}\n\n"
                  f"Success probability: {r['prob']:.0%}\n"
                  f"Estimated revenue: {money(r['est_revenue'])}\n"
                  f"Estimated profit: {money(r['est_profit'])}\n"
                  f"Estimated ROI: {r['est_roi']:.2f}x\n"
                  f"Verdict: {label}\n")
        st.download_button("⬇ Download prediction report", report,
                           "movieiq_report.txt", "text/plain")

st.markdown(f"<p style='text-align:center;color:{MUTED};margin-top:26px'>"
            "MovieIQ · built with Streamlit, scikit-learn, Plotly & SciPy</p>",
            unsafe_allow_html=True)
