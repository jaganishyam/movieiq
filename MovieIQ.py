"""
MovieIQ — Film Intelligence Platform (light edition)
====================================================
An interactive analytics platform that studies a movie dataset and predicts a
film's commercial outcome with a DUAL engine:

    1. Success probability   -> will revenue beat budget?  (classification)
    2. Revenue & ROI estimate -> roughly how much will it earn? (regression)

A movie is "successful" when revenue > budget.

Run it (Streamlit needs a plain .py file, not a notebook):
    streamlit run MovieIQ.py

Pages (sidebar):
    Overview       - KPIs, budget/revenue landscape, sunburst & treemap
    Genre Explorer - radar profile of any single genre
    Insights       - correlation, parallel coordinates, t-test, chi-square
    Model Lab      - compare 3 models, confusion matrix, importance
    Predictor      - gauge + revenue/ROI + "why" + similar films + sensitivity
    Global         - reference map of major film-production hubs
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
        border: 1px solid {LINE};
        box-shadow: 0 6px 22px rgba(20,24,31,0.05);
    }}
    .hero h1 {{ margin: 0; font-size: 2.1rem; color: {INK}; letter-spacing:.3px; }}
    .hero h1 span {{ color: {GOLD}; }}
    .hero p {{ margin: 6px 0 0; color: {MUTED}; font-size: 1.0rem; }}
    .kpi {{
        background: #FFFFFF; border: 1px solid {LINE}; border-radius: 14px;
        padding: 16px 18px; height: 100%;
        box-shadow: 0 4px 14px rgba(20,24,31,0.04);
    }}
    .kpi .label {{ color: {MUTED}; font-size:.74rem; text-transform: uppercase;
        letter-spacing: 1px; }}
    .kpi .value {{ color: {INK}; font-size: 1.65rem; font-weight: 700;
        margin-top: 4px; }}
    .kpi .value.gold {{ color: {GOLD}; }}
    .verdict {{ border-radius: 16px; padding: 18px 22px; margin-top: 8px;
        font-size: 1.05rem; border: 1px solid {LINE}; color:{INK}; }}
    .pill {{ display:inline-block; padding: 3px 12px; border-radius: 999px;
        font-size:.8rem; font-weight: 700; }}
    .note {{ background:#FFF9E8; border:1px solid #F2E2B0; border-radius:10px;
        padding:10px 14px; color:#6b5a1e; font-size:.9rem; }}
    section[data-testid="stSidebar"] {{ background:#FFFFFF; border-right:1px solid {LINE}; }}
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK),
    margin=dict(l=10, r=10, t=50, b=10),
)


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
# Model layer  (Stage 4) — classification + regression + neighbours
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
        rows.append({"Model": name,
                     "Accuracy": accuracy_score(yte, pred),
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

    # Nearest-neighbour index on scaled numeric features (for "similar films").
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
    out["similarity"] = (1 / (1 + dist[0]))
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
                    ["Overview", "Genre Explorer", "Insights", "Model Lab",
                     "Predictor", "Global"], label_visibility="collapsed")
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

if fdf.empty and page in ("Overview", "Genre Explorer", "Insights"):
    st.warning("No movies match the current filters — widen them in the sidebar.")
    st.stop()


# ======================================================================
# OVERVIEW
# ======================================================================
if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Films in view", f"{len(fdf):,}")
    with c2: kpi("Success rate", f"{fdf.success.mean():.0%}", gold=True)
    with c3: kpi("Median budget", money(fdf.budget.median()))
    with c4: kpi("Median revenue", money(fdf.revenue.median()))

    st.markdown("### Budget–revenue density")
    fig = px.density_heatmap(fdf, x="budget", y="revenue", nbinsx=30, nbinsy=30,
                             color_continuous_scale="YlOrRd")
    mx = max(fdf.budget.max(), fdf.revenue.max())
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines",
                             line=dict(dash="dash", color=MUTED), name="break-even"))
    fig.update_layout(**PLOTLY_LAYOUT, height=420)
    st.plotly_chart(fig, width="stretch")
    st.caption("Brighter cells = more films at that budget/revenue combination. "
               "Anything above the dashed line turned a profit.")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Genre → outcome (sunburst)")
        fig = px.sunburst(fdf, path=["genre", "Outcome"],
                          color="Outcome",
                          color_discrete_map={"Hit": GREEN, "Flop": RED,
                                              "(?)": LINE})
        fig.update_layout(**PLOTLY_LAYOUT, height=430)
        st.plotly_chart(fig, width="stretch")
    with colB:
        st.markdown("#### Total revenue by genre (treemap)")
        fig = px.treemap(fdf, path=["genre"], values="revenue",
                         color="roi", color_continuous_scale="YlGn")
        fig.update_layout(**PLOTLY_LAYOUT, height=430)
        st.plotly_chart(fig, width="stretch")
    st.caption("Sunburst: inner ring = genre, outer ring = hit/flop split. "
               "Treemap: box size = total revenue, colour = median ROI.")

    buf = io.StringIO(); fdf.to_csv(buf, index=False)
    st.download_button("⬇ Download this filtered dataset (CSV)",
                       buf.getvalue(), "movieiq_filtered.csv", "text/csv")


# ======================================================================
# GENRE EXPLORER  (radar chart)
# ======================================================================
elif page == "Genre Explorer":
    pick = st.selectbox("Choose a genre to profile", genres)
    gdf = df[df["genre"] == pick]

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Films", f"{len(gdf):,}")
    with c2: kpi("Success rate", f"{gdf.success.mean():.0%}", gold=True)
    with c3: kpi("Median ROI", f"{gdf.roi.median():.2f}×")
    with c4: kpi("Avg vote", f"{gdf.vote_average.mean():.1f}")

    st.markdown(f"### {pick} profile vs all films (radar)")
    metrics = ["budget", "revenue", "popularity", "runtime", "vote_average"]
    # Scale each metric to 0-1 across the whole dataset so axes are comparable.
    ranges = {m: (df[m].min(), df[m].max()) for m in metrics}
    def scale(vals):
        return [(vals[m] - ranges[m][0]) / (ranges[m][1] - ranges[m][0]) for m in metrics]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=scale(df[metrics].median()) + [scale(df[metrics].median())[0]],
                                  theta=metrics + [metrics[0]], fill="toself",
                                  name="All films", line_color=MUTED, opacity=.6))
    fig.add_trace(go.Scatterpolar(r=scale(gdf[metrics].median()) + [scale(gdf[metrics].median())[0]],
                                  theta=metrics + [metrics[0]], fill="toself",
                                  name=pick, line_color=GOLD))
    fig.update_layout(**PLOTLY_LAYOUT, height=430,
                      polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    st.plotly_chart(fig, width="stretch")
    st.caption("Each spoke is a metric scaled 0–1 across all films. "
               "Gold = this genre's median, grey = all-films median.")

    st.markdown(f"### Budget vs revenue for {pick}")
    fig = px.scatter(gdf, x="budget", y="revenue", color="Outcome",
                     color_discrete_map={"Hit": GREEN, "Flop": RED},
                     size="popularity", hover_data=["vote_average"], size_max=18)
    fig.update_layout(**PLOTLY_LAYOUT, height=380)
    st.plotly_chart(fig, width="stretch")


# ======================================================================
# INSIGHTS  (correlation + parallel coordinates + tests)
# ======================================================================
elif page == "Insights":
    st.markdown("### Correlation between numeric features")
    corr = fdf[NUMERIC_COLS + ["success"]].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(**PLOTLY_LAYOUT, height=420)
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Multivariate view — parallel coordinates")
    pcdf = fdf.copy()
    fig = px.parallel_coordinates(
        pcdf, dimensions=["budget", "popularity", "runtime", "vote_average", "roi"],
        color="success", color_continuous_scale=[[0, RED], [1, GREEN]],
        labels={"vote_average": "vote", "budget": "budget"})
    fig.update_layout(**PLOTLY_LAYOUT, height=420)
    st.plotly_chart(fig, width="stretch")
    st.caption("Each line is a film flowing across all axes at once. Drag along an "
               "axis to filter. Green lines are hits, red are flops — notice they "
               "weave together, showing no single axis cleanly separates them.")

    st.markdown("### Does a feature differ for hits vs flops?")
    left, right = st.columns([1, 1.3])
    with left:
        feat = st.selectbox("Feature", ["popularity", "vote_average", "runtime", "budget"])
        if fdf["success"].nunique() == 2:
            t, p = stats.ttest_ind(fdf[fdf.success == 1][feat],
                                   fdf[fdf.success == 0][feat], equal_var=False)
            st.metric("t-statistic", f"{t:.3f}")
            st.metric("p-value", f"{p:.4f}")
            if p < 0.05:
                st.success(f"p < 0.05 → **{feat}** differs significantly.")
            else:
                st.info(f"p ≥ 0.05 → no significant difference in **{feat}**.")
        else:
            st.warning("Need both hits and flops in view.")
    with right:
        fig = px.violin(fdf, x="Outcome", y=feat, color="Outcome", box=True,
                        color_discrete_map={"Hit": GREEN, "Flop": RED})
        fig.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False)
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Is genre associated with success? (Chi-square)")
    if fdf["success"].nunique() == 2:
        ct = pd.crosstab(fdf["genre"], fdf["success"])
        chi2, pchi, dof, _ = stats.chi2_contingency(ct)
        c1, c2 = st.columns(2)
        c1.metric("chi² statistic", f"{chi2:.3f}")
        c2.metric("p-value", f"{pchi:.4f}")
        st.info("A **p-value** is the chance of a difference this big if none truly "
                "existed (threshold 0.05). "
                + ("Genre **is** associated with success here." if pchi < 0.05
                   else "No evidence genre is associated with success here."))


# ======================================================================
# MODEL LAB
# ======================================================================
elif page == "Model Lab":
    st.markdown("### Model comparison")
    st.caption("Trained on budget, popularity, runtime, vote average and genre. "
               "Revenue and title excluded on purpose.")
    scores = engines["scores"].copy()
    st.dataframe(scores.style.format({c: "{:.1%}" for c in
                 ["Accuracy", "Precision", "Recall", "F1"]}),
                 width="stretch", hide_index=True)

    b1, b2, b3 = st.columns(3)
    with b1: kpi("Best model", engines["best_name"], gold=True)
    with b2: kpi("Baseline (always 'hit')", f"{engines['baseline']:.1%}")
    with b3: kpi("Revenue model R²", f"{engines['reg_r2']:.2f}")

    st.info("Honest read: every classifier lands near the baseline — the *success "
            "flag* is hard to predict here. But the revenue model explains about "
            f"{engines['reg_r2']:.0%} of the variance in earnings, so predicting "
            "*how much* a film makes works far better than the yes/no flag.")

    colA, colB = st.columns(2)
    with colA:
        st.markdown(f"#### Confusion matrix — {engines['best_name']}")
        fig = px.imshow(engines["confusion"], text_auto="d",
                        color_continuous_scale="Blues",
                        x=["Pred flop", "Pred hit"], y=["Actual flop", "Actual hit"])
        fig.update_layout(**PLOTLY_LAYOUT, height=360, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    with colB:
        st.markdown("#### What drives the prediction")
        imp = engines["importance"].head(8).sort_values()
        fig = px.bar(imp, orientation="h", color=imp.values,
                     color_continuous_scale=["#EFE3B8", GOLD])
        fig.update_layout(**PLOTLY_LAYOUT, height=360, showlegend=False,
                          coloraxis_showscale=False, xaxis_title="importance",
                          yaxis_title="")
        st.plotly_chart(fig, width="stretch")


# ======================================================================
# PREDICTOR — dual engine + similar films + sensitivity
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
        submitted = st.form_submit_button("Run MovieIQ", type="primary",
                                          width="stretch")

    if submitted:
        movie = {"budget": in_budget, "popularity": in_pop, "runtime": in_runtime,
                 "vote_average": in_vote, "genre": in_genre}
        r = score_movie(engines, movie)

        g1, g2 = st.columns([1, 1])
        with g1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=r["prob"] * 100,
                number={"suffix": "%"},
                title={"text": "Success probability", "font": {"color": MUTED}},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": GOLD},
                       "steps": [{"range": [0, 50], "color": "#FBE9E7"},
                                 {"range": [50, 100], "color": "#E4F3EC"}],
                       "threshold": {"line": {"color": INK, "width": 3},
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

        hit = r["prob"] >= 0.5 and r["est_profit"] > 0
        colr = GREEN if hit else RED
        label = "LIKELY SUCCESS" if hit else "RISKY"
        st.markdown(
            f"<div class='verdict' style='border-color:{colr}'>"
            f"<span class='pill' style='background:{colr};color:#fff'>{label}</span>"
            f"&nbsp;&nbsp;MovieIQ estimates <b>{money(r['est_revenue'])}</b> revenue "
            f"against a <b>{money(in_budget)}</b> budget — projected "
            f"<b>{money(r['est_profit'])}</b> profit.</div>", unsafe_allow_html=True)

        # ---- NEW: similar films from the real dataset ----
        st.markdown("#### 🎯 Films most similar to yours (from the dataset)")
        sim = similar_films(engines, df, movie, k=5)
        show = sim[["title", "genre", "budget", "revenue", "roi", "Outcome"]].copy()
        show["budget"] = show["budget"].apply(money)
        show["revenue"] = show["revenue"].apply(money)
        show["roi"] = show["roi"].round(2).astype(str) + "×"
        st.dataframe(show, width="stretch", hide_index=True)
        st.caption(f"Of these 5 real neighbours, {int(sim.success.sum())} were hits — "
                   "a reality check on the model's verdict.")

        # ---- NEW: sensitivity sweep on budget ----
        st.markdown("#### 📈 Sensitivity — how budget changes the outcome")
        sweep = np.linspace(max(1e6, in_budget * 0.3), in_budget * 2.0, 25)
        probs, revs = [], []
        for b in sweep:
            rr = score_movie(engines, {**movie, "budget": b})
            probs.append(rr["prob"] * 100); revs.append(rr["est_revenue"] / 1e6)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=sweep / 1e6, y=probs, name="Success prob (%)",
                                 line=dict(color=BLUE)), secondary_y=False)
        fig.add_trace(go.Scatter(x=sweep / 1e6, y=revs, name="Est. revenue ($M)",
                                 line=dict(color=GOLD)), secondary_y=True)
        fig.add_vline(x=in_budget / 1e6, line_dash="dash", line_color=MUTED)
        fig.update_xaxes(title_text="budget ($M)")
        fig.update_yaxes(title_text="success prob (%)", secondary_y=False)
        fig.update_yaxes(title_text="est. revenue ($M)", secondary_y=True)
        fig.update_layout(**PLOTLY_LAYOUT, height=360,
                          legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, width="stretch")
        st.caption("Everything else fixed, this sweeps the budget. The dashed line is "
                   "your chosen budget — see how spending more raises revenue but not "
                   "necessarily the odds of clearing that bigger budget.")

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
# GLOBAL — reference map of film-production hubs
# ======================================================================
elif page == "Global":
    st.markdown("### 🌍 Major global film-production hubs")
    st.markdown("<div class='note'>Your dataset has no country or location field, so "
                "this map is <b>industry context</b>, not derived from movies.csv. "
                "It marks well-known real production centres for reference.</div>",
                unsafe_allow_html=True)

    hubs = pd.DataFrame([
        ("Hollywood (Los Angeles)", "United States", 34.05, -118.24),
        ("Bollywood (Mumbai)", "India", 19.08, 72.88),
        ("Nollywood (Lagos)", "Nigeria", 6.52, 3.38),
        ("Chinese cinema (Beijing)", "China", 39.90, 116.40),
        ("Korean cinema (Seoul)", "South Korea", 37.57, 126.98),
        ("British cinema (London)", "United Kingdom", 51.51, -0.13),
        ("French cinema (Paris)", "France", 48.86, 2.35),
        ("Japanese cinema (Tokyo)", "Japan", 35.68, 139.69),
        ("Hong Kong cinema", "Hong Kong", 22.32, 114.17),
        ("Canadian hub (Toronto)", "Canada", 43.65, -79.38),
    ], columns=["Hub", "Country", "lat", "lon"])

    fig = px.scatter_geo(hubs, lat="lat", lon="lon", hover_name="Hub",
                         hover_data={"Country": True, "lat": False, "lon": False},
                         projection="natural earth")
    fig.update_traces(marker=dict(size=13, color=GOLD,
                                  line=dict(width=1, color=INK)))
    fig.update_geos(showland=True, landcolor="#F1F4FB", showocean=True,
                    oceancolor="#FFFFFF", showcountries=True, countrycolor=LINE,
                    coastlinecolor=LINE)
    fig.update_layout(**PLOTLY_LAYOUT, height=520)
    st.plotly_chart(fig, width="stretch")

    st.markdown("These hubs anchor most of global film output. If you later add a "
                "`country` column to your data, MovieIQ could colour this map by real "
                "success rate or revenue per market.")

st.markdown(f"<p style='text-align:center;color:{MUTED};margin-top:26px'>"
            "MovieIQ · built with Streamlit, scikit-learn, Plotly & SciPy</p>",
            unsafe_allow_html=True)
