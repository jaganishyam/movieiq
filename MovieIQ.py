"""
MovieIQ — Film Intelligence Platform
====================================
An interactive analytics platform that studies a movie dataset and predicts a
film's commercial outcome with a DUAL engine:

    1. Success probability   -> will revenue beat budget?  (classification)
    2. Revenue & ROI estimate -> roughly how much will it earn? (regression)

A movie is "successful" when revenue > budget.

Run it (Streamlit needs a plain .py file, not a notebook):
    streamlit run MovieIQ.py

Pages (sidebar):
    Problem Statement          - what we're solving and why
    Overview                   - KPIs, budget/revenue picture, genre charts
    Genre Explorer             - radar profile of any single genre
    Statistics                 - correlation, hit-vs-flop profile, tests
    Model Lab                  - compare 3 models, confusion matrix, importance
    Predictor                  - gauge + revenue/ROI + similar films + sensitivity
    Insights & Recommendations - plain-language findings and business advice
"""

import ast
import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, precision_score, r2_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------
# Light brand palette
# ----------------------------------------------------------------------
GOLD = "#C99700"
GREEN = "#17916B"
RED = "#D64545"
BLUE = "#2F6FED"
INK = "#14181F"
MUTED = "#6B7385"
LINE = "#E6EAF2"

NUMERIC_COLS = ["budget", "revenue", "popularity", "runtime", "vote_average"]
# Fed to models. EXCLUDE `revenue` (defines the target -> leakage) & `title`.
FEATURES = ["budget", "popularity", "runtime", "vote_average"]

st.set_page_config(page_title="MovieIQ — Film Intelligence",
                   page_icon="🎬", layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------------------------------------------------------
# Light custom styling
# ----------------------------------------------------------------------
st.markdown(f"""
<style>
    .stApp {{ background:
        linear-gradient(180deg, #FAFBFE 0%, #F3F5FA 100%); }}
    #MainMenu, footer {{ visibility: hidden; }}
    .hero {{
        padding: 24px 28px; border-radius: 18px; margin-bottom: 10px;
        background: linear-gradient(120deg, #FFFFFF 0%, #F1F4FB 100%);
        border: 1px solid {LINE}; box-shadow: 0 6px 22px rgba(20,24,31,0.05);
    }}
    .hero h1 {{ margin: 0; font-size: 2.1rem; color: {INK}; letter-spacing:.3px; }}
    .hero h1 span {{ color: {GOLD}; }}
    .hero p {{ margin: 6px 0 0; color: {MUTED}; font-size: 1.0rem; }}
    .kpi {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 14px;
        padding: 16px 18px; height: 100%; box-shadow: 0 4px 14px rgba(20,24,31,0.04); }}
    .kpi .label {{ color: {MUTED}; font-size:.74rem; text-transform: uppercase;
        letter-spacing: 1px; }}
    .kpi .value {{ color: {INK}; font-size: 1.65rem; font-weight: 700; margin-top: 4px; }}
    .kpi .value.gold {{ color: {GOLD}; }}
    .verdict {{ border-radius: 16px; padding: 18px 22px; margin-top: 8px;
        font-size: 1.05rem; border: 1px solid {LINE}; color:{INK}; }}
    .pill {{ display:inline-block; padding: 3px 12px; border-radius: 999px;
        font-size:.8rem; font-weight: 700; }}
    .card {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:14px;
        padding:18px 22px; margin-bottom:12px; box-shadow:0 4px 14px rgba(20,24,31,0.04); }}
    .card h4 {{ margin:0 0 6px; color:{INK}; }}
    .card p {{ margin:0; color:#333A48; font-size:.98rem; line-height:1.5; }}
    .rec {{ background:#F0F7F4; border:1px solid #CFE8DE; border-radius:14px;
        padding:18px 22px; color:#14332a; }}
    section[data-testid="stSidebar"] {{ background:#FFFFFF; border-right:1px solid {LINE}; }}
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                     plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK),
                     margin=dict(l=10, r=10, t=50, b=10))


# ----------------------------------------------------------------------
# Data layer  (Stage 1)
# ----------------------------------------------------------------------
@st.cache_data
def load_data(path: str = "movies.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=NUMERIC_COLS)
    df = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()
    df["success"] = (df["revenue"] > df["budget"]).astype(int)
    df["Outcome"] = np.where(df["success"] == 1, "Hit", "Flop")
    df["genre"] = df["genres"].apply(_primary_genre)
    df["roi"] = df["revenue"] / df["budget"]
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
# Model layer  (Stage 4)
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
        rows.append({"Model": name, "Accuracy": accuracy_score(yte, pred),
                     "Precision": precision_score(yte, pred, zero_division=0),
                     "Recall": recall_score(yte, pred),
                     "F1": f1_score(yte, pred, zero_division=0)})
    scores = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)

    best_name = scores.iloc[0]["Model"]
    best_clf = fitted[best_name]
    conf = confusion_matrix(yte, best_clf.predict(Xte))
    if hasattr(best_clf, "feature_importances_"):
        importance = pd.Series(best_clf.feature_importances_, index=X.columns)
    else:
        importance = pd.Series(np.abs(best_clf[-1].coef_[0]), index=X.columns)
    importance = importance.sort_values(ascending=False)

    reg = RandomForestRegressor(n_estimators=200, random_state=42)
    Xrtr, Xrte, yrtr, yrte = train_test_split(
        X, df["revenue"], test_size=0.20, random_state=42)
    reg.fit(Xrtr, yrtr)
    reg_r2 = r2_score(yrte, reg.predict(Xrte))
    reg_mae = mean_absolute_error(yrte, reg.predict(Xrte))

    scaler = StandardScaler().fit(df[FEATURES])
    nn = NearestNeighbors(n_neighbors=6).fit(scaler.transform(df[FEATURES]))

    return {"scores": scores, "best_name": best_name, "best_clf": best_clf,
            "confusion": conf, "baseline": yte.mean(), "importance": importance,
            "columns": list(X.columns), "reg": reg, "reg_r2": reg_r2,
            "reg_mae": reg_mae, "scaler": scaler, "nn": nn}


def score_movie(engines, movie: dict) -> dict:
    row = pd.get_dummies(pd.DataFrame([movie]), columns=["genre"])
    row = row.reindex(columns=engines["columns"], fill_value=0)
    prob = float(engines["best_clf"].predict_proba(row)[0][1])
    est_rev = float(engines["reg"].predict(row)[0])
    return {"prob": prob, "est_revenue": est_rev,
            "est_profit": est_rev - movie["budget"],
            "est_roi": est_rev / movie["budget"]}


def similar_films(engines, df, movie, k=5):
    q = engines["scaler"].transform(pd.DataFrame([{f: movie[f] for f in FEATURES}]))
    dist, idx = engines["nn"].kneighbors(q, n_neighbors=k)
    out = df.iloc[idx[0]].copy()
    out["similarity"] = 1 / (1 + dist[0])
    return out


# ----------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------
def kpi(label, value, gold=False):
    cls = "value gold" if gold else "value"
    st.markdown(f'<div class="kpi"><div class="label">{label}</div>'
                f'<div class="{cls}">{value}</div></div>', unsafe_allow_html=True)


def money(x):
    return f"${x/1e6:,.1f}M" if abs(x) < 1e9 else f"${x/1e9:,.2f}B"


# ======================================================================
# APP
# ======================================================================
df = load_data()
engines = train_engines(df)
genres = sorted(df["genre"].unique())

with st.sidebar:
    st.markdown(f"<h2 style='color:{GOLD};margin-bottom:0'>🎬 MovieIQ</h2>"
                f"<p style='color:{MUTED};margin-top:2px'>Film Intelligence Platform</p>",
                unsafe_allow_html=True)
    page = st.radio("Navigate",
                    ["Problem Statement", "Overview", "Genre Explorer", "Statistics",
                     "Model Lab", "Predictor", "Insights & Recommendations"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.caption("Global filters")
    f_genres = st.multiselect("Genres", genres, default=genres)
    f_vote = st.slider("Minimum vote average",
                       float(df.vote_average.min()), float(df.vote_average.max()),
                       float(df.vote_average.min()), 0.1)

mask = df["genre"].isin(f_genres) & (df["vote_average"] >= f_vote)
fdf = df[mask]

st.markdown(
    "<div class='hero'><h1>Movie<span>IQ</span> · Film Intelligence</h1>"
    "<p>Predict a film's commercial outcome from budget, popularity, runtime, "
    "rating and genre — with a success probability <b>and</b> a revenue estimate.</p></div>",
    unsafe_allow_html=True)

if fdf.empty and page in ("Overview", "Genre Explorer", "Statistics"):
    st.warning("No movies match the current filters — widen them in the sidebar.")
    st.stop()


# ======================================================================
# PROBLEM STATEMENT
# ======================================================================
if page == "Problem Statement":
    st.markdown("## The problem we're solving")
    st.markdown(
        "<div class='card'><h4>What counts as success?</h4><p>Every film is a bet: "
        "money goes in as <b>budget</b>, money comes back as <b>revenue</b>. In this "
        "project a film is a <b>success</b> when it earns back more than it cost — "
        "put simply, when <b>revenue is greater than budget</b>. Every film in the "
        "data is labelled a hit (1) or a flop (0) by that one rule.</p></div>",
        unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "<div class='card'><h4>Why it matters</h4><p>A <b>studio</b> deciding which "
            "scripts to green-light, and an <b>investor</b> deciding where to put money, "
            "both want the same thing early: a read on whether a film will make its money "
            "back. A reliable signal helps them back the right projects and size their "
            "budgets sensibly.</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(
            "<div class='card'><h4>What MovieIQ does</h4><p>It learns from 2,000 past "
            "films, then for a new film idea it answers two questions: <b>“Will it turn a "
            "profit?”</b> (a yes/no probability) and <b>“Roughly how much will it earn?”</b> "
            "(a revenue and return estimate). It's a classification task — the answer is a "
            "category, hit or flop.</p></div>", unsafe_allow_html=True)

    st.markdown("### How to use this app")
    st.markdown(
        "<div class='card'><p>Move through the pages on the left: <b>Overview</b> for the "
        "big picture, <b>Genre Explorer</b> to study one genre, <b>Statistics</b> to test "
        "which factors really matter, <b>Model Lab</b> to see how well the prediction works, "
        "the <b>Predictor</b> to score your own film idea, and <b>Insights & "
        "Recommendations</b> for the plain-English takeaways.</p></div>",
        unsafe_allow_html=True)


# ======================================================================
# OVERVIEW
# ======================================================================
elif page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Films in view", f"{len(fdf):,}")
    with c2: kpi("Success rate", f"{fdf.success.mean():.0%}", gold=True)
    with c3: kpi("Median budget", money(fdf.budget.median()))
    with c4: kpi("Median revenue", money(fdf.revenue.median()))

    st.markdown("### Where films land: budget vs what they earned")
    fig = px.density_heatmap(fdf, x="budget", y="revenue", nbinsx=30, nbinsy=30,
                             color_continuous_scale="YlOrRd")
    mx = max(fdf.budget.max(), fdf.revenue.max())
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines",
                             line=dict(dash="dash", color=MUTED), name="break-even"))
    fig.update_layout(**PLOTLY_LAYOUT, height=420)
    st.plotly_chart(fig, width="stretch")
    st.caption("Warmer areas are where many films cluster. The dashed line is break-even — "
               "films above it earned back more than they cost. Notice how most of the "
               "cloud sits above the line, and how a bigger budget doesn't neatly push a "
               "film higher.")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Which genres win, and which lose?")
        fig = px.sunburst(fdf, path=["genre", "Outcome"], color="Outcome",
                          color_discrete_map={"Hit": GREEN, "Flop": RED})
        fig.update_layout(**PLOTLY_LAYOUT, height=430)
        st.plotly_chart(fig, width="stretch")
        st.caption("Click a genre to zoom in. The green part of each genre is how many "
                   "of its films made a profit; the red part is how many lost money.")
    with colB:
        st.markdown("#### Which genres bring in the most money?")
        fig = px.treemap(fdf, path=["genre"], values="revenue",
                         color="roi", color_continuous_scale="YlGn")
        fig.update_layout(**PLOTLY_LAYOUT, height=430)
        st.plotly_chart(fig, width="stretch")
        st.caption("Bigger boxes are the genres that earned the most in total. Greener "
                   "boxes gave a better return for every dollar spent making them.")

    buf = io.StringIO(); fdf.to_csv(buf, index=False)
    st.download_button("⬇ Download this filtered dataset (CSV)",
                       buf.getvalue(), "movieiq_filtered.csv", "text/csv")


# ======================================================================
# GENRE EXPLORER
# ======================================================================
elif page == "Genre Explorer":
    pick = st.selectbox("Choose a genre to profile", genres)
    gdf = df[df["genre"] == pick]

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Films", f"{len(gdf):,}")
    with c2: kpi("Success rate", f"{gdf.success.mean():.0%}", gold=True)
    with c3: kpi("Median ROI", f"{gdf.roi.median():.2f}×")
    with c4: kpi("Avg vote", f"{gdf.vote_average.mean():.1f}")

    st.markdown(f"### How **{pick}** films compare to the typical film")
    metrics = ["budget", "revenue", "popularity", "runtime", "vote_average"]
    ranges = {m: (df[m].min(), df[m].max()) for m in metrics}
    def scale(vals):
        return [(vals[m] - ranges[m][0]) / (ranges[m][1] - ranges[m][0]) for m in metrics]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=scale(df[metrics].median()) + [scale(df[metrics].median())[0]],
                                  theta=metrics + [metrics[0]], fill="toself",
                                  name="Typical film", line_color=MUTED, opacity=.6))
    fig.add_trace(go.Scatterpolar(r=scale(gdf[metrics].median()) + [scale(gdf[metrics].median())[0]],
                                  theta=metrics + [metrics[0]], fill="toself",
                                  name=pick, line_color=GOLD))
    fig.update_layout(**PLOTLY_LAYOUT, height=430,
                      polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Each spoke is one factor. The gold shape is a typical **{pick}** film; the "
               "grey shape is a typical film overall. Where gold reaches further out, "
               f"{pick} films tend to score higher on that factor.")

    st.markdown(f"### Budget vs revenue for {pick}")
    fig = px.scatter(gdf, x="budget", y="revenue", color="Outcome",
                     color_discrete_map={"Hit": GREEN, "Flop": RED},
                     size="popularity", hover_data=["vote_average"], size_max=18)
    fig.update_layout(**PLOTLY_LAYOUT, height=380)
    st.plotly_chart(fig, width="stretch")
    st.caption("Green dots earned a profit, red dots didn't. Larger dots are more popular films.")


# ======================================================================
# STATISTICS  (correlation + hit-vs-flop profile + tests)
# ======================================================================
elif page == "Statistics":
    st.markdown("### Do the factors move together?")
    corr = fdf[NUMERIC_COLS + ["success"]].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(**PLOTLY_LAYOUT, height=420)
    st.plotly_chart(fig, width="stretch")
    st.caption("A number near +1 means two things rise together; near −1 means one rises as "
               "the other falls; near 0 means no link. Here almost every number is close to "
               "0 — no single factor strongly tracks whether a film succeeds.")

    st.markdown("### What does an average hit look like vs an average flop?")
    if fdf["success"].nunique() == 2:
        overall = fdf[FEATURES].mean()
        hit_m = fdf[fdf.success == 1][FEATURES].mean()
        flop_m = fdf[fdf.success == 0][FEATURES].mean()
        rows = []
        for f in FEATURES:
            rows.append({"Factor": f, "Group": "Average hit", "Index": hit_m[f] / overall[f] * 100})
            rows.append({"Factor": f, "Group": "Average flop", "Index": flop_m[f] / overall[f] * 100})
        prof = pd.DataFrame(rows)
        fig = px.bar(prof, x="Index", y="Factor", color="Group", barmode="group",
                     orientation="h",
                     color_discrete_map={"Average hit": GREEN, "Average flop": RED})
        fig.add_vline(x=100, line_dash="dash", line_color=MUTED)
        fig.update_layout(**PLOTLY_LAYOUT, height=360,
                          xaxis_title="compared to the typical film (100 = average)",
                          yaxis_title="", legend_title="")
        st.plotly_chart(fig, width="stretch")
        st.caption("The dashed line at 100 is the typical film. Green bars are the average "
                   "hit, red bars the average flop. When the green and red bars sit almost "
                   "on top of each other, that factor barely tells hits and flops apart.")
    else:
        st.warning("Need both hits and flops in view.")

    st.markdown("### Is the difference real, or just luck? (T-Test)")
    left, right = st.columns([1, 1.3])
    with left:
        feat = st.selectbox("Factor to test", ["popularity", "vote_average", "runtime", "budget"])
        if fdf["success"].nunique() == 2:
            t, p = stats.ttest_ind(fdf[fdf.success == 1][feat],
                                   fdf[fdf.success == 0][feat], equal_var=False)
            st.metric("p-value", f"{p:.4f}")
            if p < 0.05:
                st.success(f"Below 0.05 → the gap in **{feat}** between hits and flops is "
                           "real, not chance.")
            else:
                st.info(f"Above 0.05 → **{feat}** doesn't reliably separate hits from flops.")
        else:
            st.warning("Need both hits and flops in view.")
    with right:
        fig = px.violin(fdf, x="Outcome", y=feat, color="Outcome", box=True,
                        color_discrete_map={"Hit": GREEN, "Flop": RED})
        fig.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False)
        st.plotly_chart(fig, width="stretch")
        st.caption("Each shape shows how that factor is spread for hits vs flops. "
                   "Heavily overlapping shapes mean the factor doesn't separate them.")

    st.markdown("### Does genre affect success? (Chi-square)")
    if fdf["success"].nunique() == 2:
        ct = pd.crosstab(fdf["genre"], fdf["success"])
        chi2, pchi, dof, _ = stats.chi2_contingency(ct)
        st.metric("p-value", f"{pchi:.4f}")
        st.info("This checks whether some genres succeed more often than others. "
                + ("Below 0.05 → genre does matter for success here."
                   if pchi < 0.05 else
                   "Above 0.05 → there's no real evidence that genre changes the odds "
                   "of success here."))


# ======================================================================
# MODEL LAB
# ======================================================================
elif page == "Model Lab":
    st.markdown("### How well can we predict a hit?")
    st.caption("Each model learns from budget, popularity, runtime, vote average and genre. "
               "We leave out revenue (it would give the answer away) and title (just a name).")
    scores = engines["scores"].copy()
    st.dataframe(scores.style.format({c: "{:.1%}" for c in
                 ["Accuracy", "Precision", "Recall", "F1"]}),
                 width="stretch", hide_index=True)

    b1, b2, b3 = st.columns(3)
    with b1: kpi("Best model", engines["best_name"], gold=True)
    with b2: kpi("Baseline (always 'hit')", f"{engines['baseline']:.1%}")
    with b3: kpi("Revenue model R²", f"{engines['reg_r2']:.2f}")

    st.info("The honest picture: because most films succeed, simply guessing 'hit' every "
            f"time already scores {engines['baseline']:.0%}. Our models barely beat that — "
            "the yes/no flag is hard to predict. But the revenue model explains about "
            f"{engines['reg_r2']:.0%} of the differences in earnings, so predicting *how much* "
            "a film makes works far better than predicting *whether* it wins.")

    colA, colB = st.columns(2)
    with colA:
        st.markdown(f"#### Where the {engines['best_name']} gets it right and wrong")
        fig = px.imshow(engines["confusion"], text_auto="d",
                        color_continuous_scale="Blues",
                        x=["Predicted flop", "Predicted hit"],
                        y=["Actually flop", "Actually hit"])
        fig.update_layout(**PLOTLY_LAYOUT, height=360, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
        st.caption("The diagonal (top-left, bottom-right) is correct calls. Most mistakes "
                   "are real flops the model wrongly called hits — because hits dominate the "
                   "data, it leans optimistic.")
    with colB:
        st.markdown("#### Which factors the model leans on most")
        imp = engines["importance"].head(8).sort_values()
        fig = px.bar(imp, orientation="h", color=imp.values,
                     color_continuous_scale=["#EFE3B8", GOLD])
        fig.update_layout(**PLOTLY_LAYOUT, height=360, showlegend=False,
                          coloraxis_showscale=False, xaxis_title="how much it's used",
                          yaxis_title="")
        st.plotly_chart(fig, width="stretch")
        st.caption("Longer bars are factors the model relies on more. Genre labels barely "
                   "register, matching what the statistics showed.")


# ======================================================================
# PREDICTOR
# ======================================================================
elif page == "Predictor":
    st.markdown("### 🔮 Score a film")
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
        submitted = st.form_submit_button("Run MovieIQ", type="primary", width="stretch")

    if submitted:
        movie = {"budget": in_budget, "popularity": in_pop, "runtime": in_runtime,
                 "vote_average": in_vote, "genre": in_genre}
        r = score_movie(engines, movie)

        g1, g2 = st.columns([1, 1])
        with g1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=r["prob"] * 100, number={"suffix": "%"},
                title={"text": "Success probability", "font": {"color": MUTED}},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": GOLD},
                       "steps": [{"range": [0, 50], "color": "#FBE9E7"},
                                 {"range": [50, 100], "color": "#E4F3EC"}],
                       "threshold": {"line": {"color": INK, "width": 3}, "value": 50}}))
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

        hit = r["prob"] >= 0.5 and r["est_profit"] > 0
        colr = GREEN if hit else RED
        label = "LIKELY SUCCESS" if hit else "RISKY"
        st.markdown(
            f"<div class='verdict' style='border-color:{colr}'>"
            f"<span class='pill' style='background:{colr};color:#fff'>{label}</span>"
            f"&nbsp;&nbsp;MovieIQ estimates <b>{money(r['est_revenue'])}</b> revenue against a "
            f"<b>{money(in_budget)}</b> budget — projected <b>{money(r['est_profit'])}</b> "
            f"profit.</div>", unsafe_allow_html=True)

        st.markdown("#### 🎯 Real films most like yours")
        sim = similar_films(engines, df, movie, k=5)
        show = sim[["title", "genre", "budget", "revenue", "roi", "Outcome"]].copy()
        show["budget"] = show["budget"].apply(money)
        show["revenue"] = show["revenue"].apply(money)
        show["roi"] = show["roi"].round(2).astype(str) + "×"
        st.dataframe(show, width="stretch", hide_index=True)
        st.caption(f"These are the five real films in the data closest to your idea. "
                   f"{int(sim.success.sum())} of them were hits — a sanity check on the verdict.")

        st.markdown("#### 📈 What if you changed the budget?")
        sweep = np.linspace(max(1e6, in_budget * 0.3), in_budget * 2.0, 25)
        probs, revs = [], []
        for b in sweep:
            rr = score_movie(engines, {**movie, "budget": b})
            probs.append(rr["prob"] * 100); revs.append(rr["est_revenue"] / 1e6)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=sweep / 1e6, y=probs, name="Success chance (%)",
                                 line=dict(color=BLUE)), secondary_y=False)
        fig.add_trace(go.Scatter(x=sweep / 1e6, y=revs, name="Est. revenue ($M)",
                                 line=dict(color=GOLD)), secondary_y=True)
        fig.add_vline(x=in_budget / 1e6, line_dash="dash", line_color=MUTED)
        fig.update_xaxes(title_text="budget ($M)")
        fig.update_yaxes(title_text="success chance (%)", secondary_y=False)
        fig.update_yaxes(title_text="est. revenue ($M)", secondary_y=True)
        fig.update_layout(**PLOTLY_LAYOUT, height=360, legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, width="stretch")
        st.caption("Keeping everything else the same, this slides the budget up and down. "
                   "The dashed line is your chosen budget. Spending more tends to raise "
                   "expected revenue, but not necessarily the chance of beating that larger budget.")

        report = (f"MovieIQ prediction report\n{'='*30}\n"
                  f"Genre: {in_genre}\nBudget: {money(in_budget)}\n"
                  f"Popularity: {in_pop}  Runtime: {in_runtime}m  Vote: {in_vote}\n\n"
                  f"Success probability: {r['prob']:.0%}\n"
                  f"Estimated revenue: {money(r['est_revenue'])}\n"
                  f"Estimated profit: {money(r['est_profit'])}\n"
                  f"Estimated ROI: {r['est_roi']:.2f}x\nVerdict: {label}\n")
        st.download_button("⬇ Download prediction report", report,
                           "movieiq_report.txt", "text/plain")


# ======================================================================
# INSIGHTS & RECOMMENDATIONS
# ======================================================================
elif page == "Insights & Recommendations":
    # Live numbers so every claim is reproducible from the data.
    succ = df.success.mean()
    budget_corr = df[["budget", "success"]].corr().iloc[0, 1]
    _, pop_p = stats.ttest_ind(df[df.success == 1]["popularity"],
                               df[df.success == 0]["popularity"], equal_var=False)
    _, chi_p, _, _ = stats.chi2_contingency(pd.crosstab(df["genre"], df["success"]))
    best_acc = engines["scores"].iloc[0]["Accuracy"]

    st.markdown("## What the data is telling us")
    c1, c2, c3 = st.columns(3)
    with c1: kpi("Films that profit", f"{succ:.0%}", gold=True)
    with c2: kpi("Best model vs baseline", f"{best_acc:.0%} vs {engines['baseline']:.0%}")
    with c3: kpi("Revenue explained (R²)", f"{engines['reg_r2']:.0%}")

    findings = [
        ("Most films here make money",
         f"About {succ:.0%} of films in this dataset earned back more than they cost. "
         "That's good to know, but it also means the data is lopsided — any model has to "
         "beat that high bar to be genuinely useful."),
        ("A bigger budget does not buy success",
         f"Budget and success barely move together (correlation of just {budget_corr:+.2f}). "
         "Spending more does not reliably turn a film into a hit — several big-budget films "
         "still lost money."),
        ("Whether a film wins is hard to call in advance",
         f"The best model reaches only {best_acc:.0%}, essentially matching the "
         f"{engines['baseline']:.0%} you'd get by always guessing 'hit'. Of all the factors, "
         f"popularity is the only one with even a slight real edge (p = {pop_p:.2f})."),
        ("Genre barely changes the odds",
         f"A chi-square test gives p = {chi_p:.2f}, well above 0.05 — there's no real "
         "evidence that some genres succeed more often than others in this data."),
        ("But how much a film earns IS predictable",
         f"The revenue model explains about {engines['reg_r2']:.0%} of the differences in "
         "earnings. So the smarter question isn't 'will it win?' but 'how much will it "
         "earn, and what's the return?'"),
    ]
    for title, body in findings:
        st.markdown(f"<div class='card'><h4>{title}</h4><p>{body}</p></div>",
                    unsafe_allow_html=True)

    st.markdown("## Recommendations")
    st.markdown(
        "<div class='rec'>"
        "<b>1. Judge films by expected return, not a yes/no guess.</b> Lead with the "
        "revenue and ROI estimate on the Predictor page — that's where the model is "
        "trustworthy.<br><br>"
        "<b>2. Don't treat budget as a lever for success.</b> A larger budget raises the "
        "bar the film must clear. Watch ROI, not spend.<br><br>"
        "<b>3. Collect richer signals.</b> To predict the yes/no outcome better, the data "
        "needs more than five numbers — think cast and director, release timing, marketing "
        "spend, and whether it's a sequel or franchise.<br><br>"
        "<b>4. Use MovieIQ as a guide, not a guarantee.</b> Treat its verdict as one input "
        "for a green-light decision, alongside human judgement."
        "</div>", unsafe_allow_html=True)

st.markdown(f"<p style='text-align:center;color:{MUTED};margin-top:26px'>"
            "MovieIQ · built with Streamlit, scikit-learn, Plotly & SciPy</p>",
            unsafe_allow_html=True)
