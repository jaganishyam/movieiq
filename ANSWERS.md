# MovieIQ — Written Answers

Answer notes for the project brief, grounded in the actual `movies.csv`
(2,000 movies). Put these in your own words before submitting.

## Stage 0 — Problem statement

1. **What makes a movie "successful."** A movie succeeds when it earns back more
   than it cost. Exact rule: `success = 1 if revenue > budget else 0`.
2. **Why it's valuable.** *Studios* can decide which scripts to green-light and
   how much to spend; *investors/financiers* can judge risk before putting money
   in. Both want to avoid funding films that lose money.
3. **Objective + steps.** Objective: predict, from a film's basic attributes,
   whether it will make a profit. Steps: (a) clean the data and build the target,
   (b) explore patterns with EDA, (c) confirm patterns with statistical tests,
   (d) train and evaluate a classifier, (e) ship it as a Streamlit app.
4. **Classification.** Classification means predicting a *category* rather than a
   number. Here the target variable is `success`, which is either 1 (success) or
   0 (failure) — a binary classification problem.

## Stage 1 — Data preparation

1. **Shape.** 2,000 rows × 7 columns (`budget, revenue, popularity, runtime,
   vote_average, title, genres`). Summary stats are printed in the app / notebook.
2. **Zeros & missing.** A budget or revenue of 0 usually means the value is
   unknown, not truly zero — and a 0 revenue would wrongly label a film a failure.
   In *this* dataset there happen to be **no zeros or missing values**, but the
   code still drops any row with a 0 or null budget/revenue so the rule stays
   honest on other data.
3. **Target balance.** Success rate ≈ **81%** (1,614 successes vs 386 failures).
   The dataset is **imbalanced** — most movies succeed. This matters: a model that
   blindly guesses "success" already scores ~81% accuracy, so accuracy alone is
   misleading and we also report precision and recall.
4. **Genres.** They're stored in TMDB's raw format, e.g.
   `[{'id': 18, 'name': 'Drama'}]`. The code parses that string with
   `ast.literal_eval` and extracts the genre name into a clean `genre` column.
   (181 rows have an empty genre list and become `Unknown`.)

## Stage 2 — EDA (what the charts show in this data)

1. **Budget vs revenue.** The scatter has no strong upward trend — bigger budgets
   do **not** reliably earn more here. Most points sit above the break-even line,
   matching the ~81% success rate.
2. **Genres.** The nine genres are evenly represented (~190–220 films each), and
   success rates are similar across them — no genre clearly out-earns the rest.
3. **Popularity / runtime / vote_average vs success.** The box plots for
   successful vs failed movies overlap heavily. Popularity shows the *slightest*
   separation; the others barely differ.
4. **Correlation heatmap.** No strong correlations between the numeric features
   (values near 0), so there's little multicollinearity to worry about — but also
   little signal linking any single feature to success.

## Stage 3 — Statistical testing

1. **T-Test (popularity vs success).** H₀: mean popularity is the same for
   successful and failed movies. p ≈ **0.04** → just below 0.05, so we *reject H₀*:
   popularity differs slightly between the groups. (vote_average, runtime and
   budget all give p > 0.05 → no significant difference.)
2. **Chi-Square (genre vs success).** H₀: genre and success are independent.
   p ≈ **0.99** → far above 0.05, so we *cannot reject H₀*: there's no evidence
   that genre is associated with success in this data.
3. **P-value in plain words.** It's the chance of seeing a difference this big if
   there were really no difference at all. Threshold used: **0.05** — the standard
   convention meaning we accept a 5% risk of a false alarm.

## Stage 4 — Modeling (Random Forest)

1. **Features & exclusions.** Fed: `budget, popularity, runtime, vote_average`
   plus one-hot `genre`. Excluded **revenue** (success is *defined* from it — using
   it is data leakage / cheating) and **title** (a unique label, not a predictor).
2. **Train/test split.** 80/20, `stratify=y` to keep the success ratio in both
   sets. A held-out test set is essential so we measure performance on data the
   model never saw — otherwise we'd just be measuring memorisation.
3. **How a random forest predicts.** It grows many decision trees, each on a random
   sample of rows and features, then takes a majority vote across the trees.
   Averaging many trees makes it more robust than a single tree.
4. **Evaluation.** Accuracy ≈ **81%**, but that *equals the ~81% baseline* of
   always guessing "success." Recall for the success class is near 100% while it
   catches almost none of the failures — you can see this in the confusion matrix
   (the "actual failure" row is mostly misclassified). So the model mostly predicts
   the majority class.
5. **Feature importance.** Popularity, budget, vote_average and runtime come out
   roughly equal and on top; genre dummies barely matter. This lines up with the
   EDA and stats: popularity was the only feature with a significant t-test, and
   genre was not associated with success (chi-square).

## Stage 5 — Streamlit

Covered by `MovieIQ.py`: sidebar filters (genre + min vote average), EDA charts,
statistical-test results, and a prediction form — all in one deployable file.
See `README.md` for local run and deployment steps.

## Reflection

If a studio asked "Will our next film succeed?", I'd be **cautiously honest**:
MovieIQ predicts, but on this dataset the chosen features carry little real signal
(the model barely beats a coin-flip-against-base-rate), so I wouldn't stake a
budget on it.

- **Limitation:** the data looks synthetic and lightly correlated with success —
  and it's imbalanced (81% winners), which pushes the model toward always saying
  "success."
- **What I'd improve with more time:** add richer, real predictors (director,
  cast, release month, marketing spend, franchise/sequel flag), address the
  imbalance (class weights or resampling), and compare models beyond Random Forest.
