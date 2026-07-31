"""
MovieIQ - Predictive Analytics on Film Success
================================================
A Streamlit dashboard that explores a movie dataset, runs statistical tests,
trains a Random Forest model, and predicts whether a film will be successful.

A movie is "successful" when its revenue is greater than its budget.

HOW TO RUN (this is what "pure python file" means):
    Streamlit runs a plain .py script, NOT a Jupyter notebook.
    From a terminal, in this folder, type:
        streamlit run MovieIQ.py

Maps to the project brief stages:
    Stage 1 -> load_data()            (data prep + cleaning)
    Stage 2 -> "EDA" tab              (exploratory charts)
    Stage 3 -> "Statistical Tests" tab (t-test + chi-square)
    Stage 4 -> train_model()          (Random Forest + evaluation)
    Stage 5 -> the whole app          (interactive Streamlit dashboard)
"""

import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="MovieIQ", page_icon="🎬", layout="wide")

sns.set_theme(style="whitegrid")

# The numeric columns we analyse and model on.
NUMERIC_COLS = ["budget", "revenue", "popularity", "runtime", "vote_average"]

# Features we FEED the model. Note two deliberate exclusions:
#   - "revenue" is excluded because success is DEFINED as revenue > budget.
#     Including it would let the model "cheat" (this is called data leakage).
#   - "title" is excluded because it is a unique label, not a predictor.
MODEL_NUMERIC_FEATURES = ["budget", "popularity", "runtime", "vote_average"]


# ----------------------------------------------------------------------
# STAGE 1 - Data preparation
# ----------------------------------------------------------------------
@st.cache_data
def load_data(path: str = "movies.csv") -> pd.DataFrame:
    """
    Load and clean the dataset.

    @st.cache_data means Streamlit runs this once and reuses the result,
    so the file is not re-read on every click (keeps the app fast).
    """
    df = pd.read_csv(path)

    # 2. Handle zeros/missing in budget & revenue.
    #    A budget or revenue of 0 is almost always missing/unknown data, not a
    #    real value. If revenue were 0, revenue > budget would wrongly say the
    #    film failed. So we drop those rows to keep the success label honest.
    df = df.dropna(subset=NUMERIC_COLS)
    df = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()

    # 3. Create the target: success = 1 when revenue > budget, else 0.
    df["success"] = (df["revenue"] > df["budget"]).astype(int)

    # 4. The genres column is stored in TMDB's raw format, e.g.
    #        "[{'id': 18, 'name': 'Drama'}]"
    #    We parse it into a clean primary-genre string we can filter on.
    df["genre"] = df["genres"].apply(_extract_primary_genre)

    return df


def _extract_primary_genre(raw: str) -> str:
    """Turn "[{'id': 18, 'name': 'Drama'}]" into "Drama". Falls back safely."""
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list) and parsed:
            return parsed[0].get("name", "Unknown")
    except (ValueError, SyntaxError, TypeError):
        pass
    return "Unknown"


# ----------------------------------------------------------------------
# STAGE 4 - Predictive model (Random Forest)
# ----------------------------------------------------------------------
@st.cache_resource
def train_model(df: pd.DataFrame):
    """
    Train a Random Forest on the FULL (unfiltered) dataset so predictions
    stay stable no matter what the user filters in the sidebar.

    @st.cache_resource caches the trained model object itself.

    Returns the model, its feature-column order, and an evaluation dict.
    """
    # Build the feature matrix: numeric features + one-hot encoded genre.
    X = pd.get_dummies(
        df[MODEL_NUMERIC_FEATURES + ["genre"]], columns=["genre"]
    )
    y = df["success"]

    # 2. Train/test split. We hold out 20% the model never sees during
    #    training, so accuracy is measured on unseen data (an honest test).
    #    stratify=y keeps the same success/failure ratio in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. A Random Forest builds many decision trees on random slices of the
    #    data and features, then lets them "vote". The majority vote is the
    #    prediction. Averaging many trees reduces overfitting.
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluate on the held-out test set.
    preds = model.predict(X_test)
    evaluation = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "confusion": confusion_matrix(y_test, preds),
        "baseline": y_test.mean(),  # accuracy if we always guessed "success"
        "importance": pd.Series(
            model.feature_importances_, index=X.columns
        ).sort_values(ascending=False),
    }
    return model, list(X.columns), evaluation


def predict_success(model, feature_cols, movie: dict) -> tuple:
    """Predict success for one movie described by a dict of its details."""
    row = pd.DataFrame([movie])
    row = pd.get_dummies(row, columns=["genre"])
    # Line up columns with training order; genres the movie doesn't have = 0.
    row = row.reindex(columns=feature_cols, fill_value=0)
    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0][1]  # probability of class "success"
    return int(pred), float(proba)


# ======================================================================
# APP BODY
# ======================================================================
df = load_data()
model, feature_cols, ev = train_model(df)

st.title("🎬 MovieIQ — Predictive Analytics on Film Success")
st.caption(
    "A movie is **successful** when its revenue is greater than its budget. "
    "Explore the data, test it statistically, and predict a film's outcome."
)

# ----------------------------------------------------------------------
# Sidebar filters (Stage 5.1)
# ----------------------------------------------------------------------
st.sidebar.header("Filters")

all_genres = sorted(df["genre"].unique())
chosen_genres = st.sidebar.multiselect(
    "Genre", options=all_genres, default=all_genres,
    help="Filter every chart below by one or more genres.",
)
min_vote = st.sidebar.slider(
    "Minimum vote average", float(df["vote_average"].min()),
    float(df["vote_average"].max()), float(df["vote_average"].min()), 0.1,
)

# Apply the filters. This filtered view drives the EDA + stats tabs.
mask = df["genre"].isin(chosen_genres) & (df["vote_average"] >= min_vote)
fdf = df[mask]

st.sidebar.markdown("---")
st.sidebar.metric("Movies after filter", f"{len(fdf):,}")
if len(fdf):
    st.sidebar.metric("Success rate", f"{fdf['success'].mean():.0%}")

# Top-line KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Movies (total)", f"{len(df):,}")
c2.metric("Overall success rate", f"{df['success'].mean():.0%}")
c3.metric("Median budget", f"${df['budget'].median()/1e6:.0f}M")
c4.metric("Median revenue", f"${df['revenue'].median()/1e6:.0f}M")

tab_eda, tab_stats, tab_model, tab_predict = st.tabs(
    ["📊 EDA", "🧪 Statistical Tests", "🌲 Model", "🔮 Predict"]
)

# ----------------------------------------------------------------------
# STAGE 2 - Exploratory Data Analysis
# ----------------------------------------------------------------------
with tab_eda:
    if fdf.empty:
        st.warning("No movies match the current filters. Widen them in the sidebar.")
    else:
        st.subheader("Budget vs. Revenue")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(
                data=fdf, x="budget", y="revenue", hue="success",
                palette={0: "#d62728", 1: "#2ca02c"}, alpha=0.6, ax=ax,
            )
            # Reference line where revenue == budget (break-even).
            lims = [0, max(fdf["budget"].max(), fdf["revenue"].max())]
            ax.plot(lims, lims, "--", color="gray", label="break-even")
            ax.set_title("Budget vs Revenue (green = success)")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
        with col2:
            st.markdown(
                "**Reading it:** points above the dashed break-even line made "
                "more than they cost (successes). A rising cloud would mean "
                "bigger budgets earn more — check whether that holds here."
            )

        st.subheader("Genre trends")
        col1, col2 = st.columns(2)
        with col1:
            counts = fdf["genre"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#4c72b0")
            ax.set_title("Most common genres")
            ax.set_xlabel("Number of movies")
            st.pyplot(fig)
            plt.close(fig)
        with col2:
            succ_by_genre = (
                fdf.groupby("genre")["success"].mean().sort_values(ascending=False)
            )
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=succ_by_genre.values, y=succ_by_genre.index,
                        ax=ax, color="#2ca02c")
            ax.set_title("Success rate by genre")
            ax.set_xlabel("Share of movies that succeeded")
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("How features relate to success")
        feat = st.selectbox(
            "Compare a feature across successful vs failed movies",
            ["popularity", "runtime", "vote_average", "budget"],
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(
            data=fdf, x="success", y=feat, ax=ax,
            palette={0: "#d62728", 1: "#2ca02c"},
        )
        ax.set_xticklabels(["Failure (0)", "Success (1)"])
        ax.set_title(f"{feat} vs success")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Correlation heatmap")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            fdf[NUMERIC_COLS + ["success"]].corr(),
            annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
        )
        ax.set_title("Correlation between numeric features")
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Strongly correlated feature pairs (near +1 or -1) carry "
            "overlapping information, which can make a model less stable."
        )

# ----------------------------------------------------------------------
# STAGE 3 - Statistical testing
# ----------------------------------------------------------------------
with tab_stats:
    st.subheader("Is the difference real, or just chance?")
    st.markdown(
        "A **p-value** is the probability of seeing a difference this large "
        "if there were truly no difference. We use the common threshold "
        "**0.05**: below it, we call the result *statistically significant*."
    )

    if fdf["success"].nunique() < 2:
        st.warning("Need both successful and failed movies in view to run tests.")
    else:
        # --- T-Test: does a numeric feature differ between the two groups? ---
        st.markdown("### T-Test")
        num_feat = st.selectbox(
            "Numeric feature to test", ["popularity", "vote_average", "runtime", "budget"],
        )
        group_ok = fdf.groupby("success")[num_feat]
        t_stat, p_val = stats.ttest_ind(
            fdf[fdf["success"] == 1][num_feat],
            fdf[fdf["success"] == 0][num_feat],
            equal_var=False,
        )
        st.write(
            f"**H₀ (null):** the mean *{num_feat}* is the same for successful "
            f"and unsuccessful movies."
        )
        cc1, cc2 = st.columns(2)
        cc1.metric("t-statistic", f"{t_stat:.3f}")
        cc2.metric("p-value", f"{p_val:.4f}")
        if p_val < 0.05:
            st.success(
                f"p < 0.05 → reject H₀. *{num_feat}* differs significantly "
                "between successful and failed movies."
            )
        else:
            st.info(
                f"p ≥ 0.05 → cannot reject H₀. No significant difference in "
                f"*{num_feat}* between the groups."
            )

        # --- Chi-Square: is genre associated with success? ---
        st.markdown("### Chi-Square Test (genre vs success)")
        contingency = pd.crosstab(fdf["genre"], fdf["success"])
        chi2, p_chi, dof, _ = stats.chi2_contingency(contingency)
        st.write("**H₀ (null):** genre and success are independent (unrelated).")
        cc1, cc2 = st.columns(2)
        cc1.metric("chi² statistic", f"{chi2:.3f}")
        cc2.metric("p-value", f"{p_chi:.4f}")
        if p_chi < 0.05:
            st.success("p < 0.05 → genre and success are associated.")
        else:
            st.info("p ≥ 0.05 → no evidence that genre and success are related.")

# ----------------------------------------------------------------------
# STAGE 4 - Model results
# ----------------------------------------------------------------------
with tab_model:
    st.subheader("Random Forest — how well can we predict success?")
    st.markdown(
        "The model is trained on **budget, popularity, runtime, vote average "
        "and genre**. We deliberately leave out *revenue* (it defines the "
        "target, so using it would be cheating) and *title* (just a label)."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{ev['accuracy']:.1%}")
    m2.metric("Precision", f"{ev['precision']:.1%}")
    m3.metric("Recall", f"{ev['recall']:.1%}")
    m4.metric("Baseline*", f"{ev['baseline']:.1%}",
              help="Accuracy if we blindly guessed 'success' every time.")

    st.caption(
        "*The dataset is imbalanced (most movies succeed), so compare accuracy "
        "against this baseline — beating it is what actually shows skill."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            ev["confusion"], annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Pred fail", "Pred success"],
            yticklabels=["Actual fail", "Actual success"], ax=ax,
        )
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        st.markdown("**Feature importance**")
        top = ev["importance"].head(8)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x=top.values, y=top.index, ax=ax, color="#4c72b0")
        ax.set_xlabel("Importance")
        st.pyplot(fig)
        plt.close(fig)

    st.info(
        "Where it makes mistakes: because most movies succeed, the model leans "
        "toward predicting 'success' and misses many of the rarer failures "
        "(the bottom-left cell of the matrix)."
    )

# ----------------------------------------------------------------------
# STAGE 5.3 - Prediction section
# ----------------------------------------------------------------------
with tab_predict:
    st.subheader("Will this film succeed?")
    st.markdown("Enter a movie's details and get a prediction from the model.")

    col1, col2 = st.columns(2)
    with col1:
        in_budget = st.number_input(
            "Budget ($)", min_value=100_000, max_value=500_000_000,
            value=50_000_000, step=1_000_000,
        )
        in_pop = st.slider("Popularity", 0.0, 100.0, 50.0)
        in_runtime = st.slider("Runtime (minutes)", 60, 240, 120)
    with col2:
        in_vote = st.slider("Vote average", 0.0, 10.0, 6.0, 0.1)
        in_genre = st.selectbox("Genre", all_genres)

    if st.button("Predict", type="primary"):
        movie = {
            "budget": in_budget, "popularity": in_pop, "runtime": in_runtime,
            "vote_average": in_vote, "genre": in_genre,
        }
        pred, proba = predict_success(model, feature_cols, movie)
        if pred == 1:
            st.success(f"✅ Predicted **SUCCESS** — {proba:.0%} confidence.")
        else:
            st.error(f"❌ Predicted **NOT a success** — {(1 - proba):.0%} confidence.")
        st.progress(proba)
        st.caption(
            "Confidence is the model's estimated probability that revenue will "
            "exceed budget. Treat it as a guide, not a guarantee."
        )

st.markdown("---")
st.caption("MovieIQ · Built with Streamlit, scikit-learn, seaborn & scipy.")
