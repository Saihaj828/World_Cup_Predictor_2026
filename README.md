# World Cup 2026 — Match Predictor

A deliberately small machine-learning project, in two parts:

1. **`World_Cup_Predictor.ipynb`** — the notebook where the model was explored and built.
2. **`app.py`** — a one-file **web app** that serves predictions for upcoming
   World Cup 2026 matches (group stage now; Round of 32 and beyond automatically,
   once those matchups are fixed).

Given two national teams it predicts **home win / draw / away win** using a small,
transparent model: an Elo rating, recent form, and a random forest.

---

## Quick start (run the website locally)

```bash
cd wc2026-predictor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py            # opens http://localhost:8501
```

> **No `pip` in your venv?** Some minimal systems ship Python without
> `ensurepip`. Bootstrap it once with:
> `python3 -m venv .venv --without-pip && curl -sL https://bootstrap.pypa.io/get-pip.py | .venv/bin/python`

## Deploy it (free, ~2 minutes)

The whole app is one file, so hosting is trivial:

1. Push this folder to a **GitHub** repo.
2. Go to **[share.streamlit.io](https://share.streamlit.io)**, connect the repo,
   and point it at `app.py`.
3. Streamlit Community Cloud installs `requirements.txt` and gives you a public
   URL. Done — no servers, Dockerfiles, or build steps to manage.

---

## What the website does

- **📅 Upcoming matches** — every unplayed World Cup 2026 fixture in the data,
  grouped by round, each with predicted probabilities and a pick.
  - *Group stage*: shows **Home win / Draw / Away win**.
  - *Knockouts* (Round of 32 onward): shows each side's chance to **advance**.
- **⚔️ Head-to-head** — pick any two teams and get an instant prediction.
- **ℹ️ How it works** — a short in-app explainer.

### How group stage *and* Round of 32 are handled

The app predicts whatever **real, scheduled** fixtures exist in the dataset:

- Group-stage matchups are known, so they're predicted right now.
- Knockout matchups don't exist until the group stage ends and the bracket is set.
  Because the app reads the dataset **live**, the moment the Round of 32 fixtures
  are published upstream they appear in the app and get predicted — **no code
  change or redeploy needed**. (There's a *Refresh data & model* button in the
  sidebar to pull the latest immediately.)

---

## Design decisions — and *why*

Since the goal was "proper but basic," every choice optimised for **simplicity,
transparency, and a free one-click deploy**. Here's the reasoning behind each one:

| Decision | Why |
|---|---|
| **Streamlit** (not Flask + React/HTML) | Streamlit lets the **frontend and backend live in one Python file** with no HTML/CSS/JS, no API layer, and no separate build. It's purpose-built for ML demos and is the shortest path from "model" to "website." |
| **Everything in a single `app.py`** | You asked for it all "within one downloadable file." One file is easy to read top-to-bottom, easy to hand off, and removes glue code between a frontend and a backend. |
| **Pull the dataset live** (with a bundled fallback) | This is what makes the **Round-of-32 requirement work by itself**: when new fixtures are scheduled upstream they show up automatically. The bundled `data/results.csv` is a fallback so the app still runs offline / if the download fails. |
| **Train the model on startup, then cache it** | Rather than ship a pickled `.pkl` binary, the app rebuilds the model from data each launch (cached for 6h via `@st.cache_resource`). The logic stays **visible and auditable**, there's no stale artifact to keep in sync, and the model **retrains on fresh results** as the tournament progresses. Caching means the heavy work runs once, not on every click. |
| **Predict the *real* fixtures, don't simulate the bracket** | Simulating who advances would mean encoding group standings, FIFA tiebreakers, and best-third-place ranking — genuinely advanced and error-prone. Predicting fixtures only once they're **fixed** is far simpler and is exactly what was asked. |
| **Label rounds by date** | The dataset has no "round" column, but the 2026 schedule windows are fixed. Mapping a fixture's date to its round means new rounds (R32, R16, …) **slot in automatically** with zero new code. |
| **Knockout "advances" = win% + ½·draw%** | Knockout games can't end level. Splitting the draw probability 50/50 is a simple, honest stand-in for the extra-time/penalty coin-flip — no extra model required. |
| **Lean `requirements.txt`** (streamlit, pandas, numpy, scikit-learn) | Dropping `matplotlib`/`jupyterlab` keeps the cloud deploy small and fast. They're only needed for the notebook (`pip install jupyterlab matplotlib`). |
| **Same model as the notebook** | The website is just the notebook's pipeline, productised — so what you explored is exactly what gets served. |

---

## The model (shared by notebook and app)

Three **leak-free** features (computed only from matches *before* each game):

1. **`elo_diff`** — difference in Elo rating. Teams start at 1500; winners gain
   points and losers lose them, scaled by how surprising the result was. The
   single strongest signal.
2. **`form_diff`** — difference in recent form (avg goal difference, last 5 games).
3. **`neutral`** — neutral-venue flag (most World Cup games are neutral; hosts
   USA/Canada/Mexico are not), which controls for home advantage.

A `RandomForestClassifier` then predicts the three outcomes.

**Reality check:** 3-way football prediction tops out around **~50–55%** accuracy
because draws are inherently hard to call. The notebook checks the model against
two naive baselines (always-home, higher-Elo) and confirms it **beats both** —
that's the bar for success here, not a big raw number. The model rarely predicts a
draw as the single most-likely outcome, but still assigns draws a realistic
20–30% probability.

---

## Project structure

```
wc2026-predictor/
├── app.py                      # the web app: frontend + backend + model, one file
├── World_Cup_Predictor.ipynb   # exploration / model-building notebook
├── data/results.csv            # bundled snapshot (offline fallback for the app)
├── requirements.txt            # deploy deps (streamlit, pandas, numpy, scikit-learn)
└── README.md
```

## Data
From the community-maintained
[martj42/international_results](https://github.com/martj42/international_results)
dataset (one row per international match, 1872–present, including the scheduled
2026 World Cup fixtures).

## Ideas to extend
- Add FIFA ranking as a feature.
- Swap in `LogisticRegression` for a more interpretable model.
- Simulate the full 48-team bracket to estimate each team's title odds.
- Add a results page that scores past predictions once matches are played.
