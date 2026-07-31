# MovieIQ — Predictive Analytics on Film Success

An interactive Streamlit dashboard that explores a movie dataset, runs
statistical tests, trains a Random Forest model, and predicts whether a film
will be **successful** (revenue greater than budget).

## Files

| File | What it is |
|------|------------|
| `MovieIQ.py` | The app — a **pure Python file** (this is what your reviewer meant). |
| `movies.csv` | The dataset (2,000 movies). |
| `requirements.txt` | Libraries needed to run and deploy. |
| `ANSWERS.md` | Written answers to the project brief, grounded in this data. |

## "Pure Python file" — what that means

Streamlit can only run a plain `.py` script, **not a Jupyter notebook**
(`.ipynb`). The command below runs the file top to bottom and turns it into a
web app. All your logic — loading data, charts, statistics, the model — lives
inside `MovieIQ.py`.

## Run it locally

```bash
# 1. (optional but recommended) create a clean environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. install libraries
pip install -r requirements.txt

# 3. launch the app
streamlit run MovieIQ.py
```

Your browser opens at `http://localhost:8501`. Use the sidebar to filter by
genre and minimum vote average; the tabs hold EDA, statistical tests, the
model results, and a live prediction form.
4. Deploy. You get a public link like the ones your fellow interns shared.

**What usually needs changing for deployment:** make sure the data path is
relative (`movies.csv`, not `C:\Users\...`), and that everything you import is
listed in `requirements.txt`. Both are already handled here.
