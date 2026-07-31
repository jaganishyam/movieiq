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

## Six pages (sidebar navigation)

1. **Overview** — KPIs, a budget/revenue **density heatmap**, a genre→outcome
   **sunburst**, and a revenue **treemap**.
2. **Genre Explorer** — a **radar chart** profiling any genre against all films.
3. **Insights** — correlation heatmap, a **parallel-coordinates** plot, a
   **violin** t-test view, and a chi-square test.
4. **Model Lab** — compares Random Forest, Gradient Boosting and Logistic
   Regression; shows the confusion matrix, feature importance, and revenue-model R².
5. **Predictor** — probability gauge + revenue/ROI estimate + a **"similar films"**
   table (real nearest neighbours from the data) + a **budget sensitivity** sweep
   + a downloadable report.
6. **Global** — a reference **map** of major world film-production hubs.

## Uncommon charts used

Density heatmap, sunburst, treemap, radar (polar), parallel coordinates, violin,
gauge, and a geographic scatter map — chosen to stand apart from the usual
bar/scatter-only projects.

## A note on the map

`movies.csv` has no country/location column, so the map is **industry context**
(real production hubs), not derived from the dataset. This is stated in the app
itself so it's transparent to any reviewer.

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
