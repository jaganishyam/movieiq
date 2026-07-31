# MovieIQ — Film Intelligence Platform

An interactive analytics platform (light theme) that studies a movie dataset and
predicts a film's commercial outcome with a **dual engine**:

- **Success probability** — will revenue beat budget? (classification)
- **Revenue & ROI estimate** — roughly how much will it earn? (regression)

A movie is **successful** when revenue > budget.

## Files

| File | What it is |
|------|------------|
| `MovieIQ.py` | The app — a **pure Python file** (`streamlit run MovieIQ.py`). |
| `.streamlit/config.toml` | The light theme. Upload this folder too. |
| `movies.csv` | The dataset (2,000 films). |
| `requirements.txt` | Libraries needed to run and deploy. |
| `ANSWERS.md` | Written answers to the project brief, grounded in this data. |

## Seven pages (sidebar navigation)

1. **Problem Statement** — what "success" means, why it matters, and what the app does.
2. **Overview** — KPIs, a budget/revenue density view, a genre hit/flop sunburst,
   and a revenue treemap.
3. **Genre Explorer** — a radar chart profiling any genre against the typical film.
4. **Statistics** — correlation, an "average hit vs average flop" comparison, a
   t-test (with a violin plot), and a chi-square test.
5. **Model Lab** — compares Random Forest, Gradient Boosting and Logistic
   Regression; shows the confusion matrix, feature importance, and revenue-model R².
6. **Predictor** — probability gauge + revenue/ROI estimate + a "real films most
   like yours" table + a budget sensitivity chart + a downloadable report.
7. **Insights & Recommendations** — plain-language findings (computed live from the
   data) and business recommendations.

## Charts used

Density heatmap, sunburst, treemap, radar (polar), grouped comparison bars,
violin, gauge, and a dual-axis sensitivity line — every caption is written in
plain language so a non-technical reader can follow it.

## Run locally

```bash
pip install -r requirements.txt
streamlit run MovieIQ.py
```

## Deploy on Streamlit Community Cloud (free)

1. Push these files to a **public GitHub repo** — including the **`.streamlit`
   folder** and **`movies.csv`**.
2. Go to <https://share.streamlit.io>, sign in with GitHub.
3. Click **Create app**, set Repository, Branch `main`, Main file `MovieIQ.py`.
4. **Deploy.**

To add the theme folder on GitHub's website: **Add file → Create new file**, type
`.streamlit/config.toml` as the name, paste the contents.
