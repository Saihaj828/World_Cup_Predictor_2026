# ⚽ World Cup 2026 Match Predictor

A machine-learning web app I built to predict the outcome of World Cup 2026 matches
(win / draw / loss). It updates itself as the tournament progresses: it forecasts
every upcoming fixture, compares several models, and scores its own past predictions
against real results.

<!-- Live demo: https://your-app-name.streamlit.app -->
## What it does

I designed it as a small multi-page Streamlit app:

- **📅 Home — Upcoming matches:** predictions for every unplayed WC2026 fixture,
  grouped by round. Knockout games show each side's chance to *advance*.
- **⚔️ Head-to-head:** I can pick any two teams and any model and get an instant
  win/draw/loss prediction.
- **📊 Model comparison:** accuracy and log-loss for each model, scored
  out-of-sample on the World Cup matches, plus naive baselines to beat.
- **✅ Results:** a scoreboard of my past predictions vs. what actually happened,
  with an overall hit-rate.
- **ℹ️ How it works:** a plain-English explainer of the data, features, and models.

The data is pulled **live**, so new rounds (Round of 32, Round of 16, …) appear and
get predicted automatically as their matchups are confirmed — no code changes needed.

---

## How it works

### Data
I use the community-maintained
[martj42/international_results](https://github.com/martj42/international_results)
dataset — every international match since 1872, including the scheduled 2026 World
Cup fixtures. The app fetches it live (with a bundled snapshot as an offline
fallback) and trains on matches from 2000 onward.

### Features
For every match I engineer five **leak-free** features (computed only from matches
that happened *before* it, so there's no peeking at the result):

| Feature | Meaning |
|---|---|
| `elo_diff` | Difference in Elo rating — overall strength. I compute Elo myself in a single forward pass. |
| `att_diff` | Attack: difference in average goals **scored** over the last 5 games. |
| `def_diff` | Defense: difference in average goals **conceded** (fewer is better). |
| `winrate_diff` | Difference in share of points won over the last 5 games. |
| `neutral` | Whether the match is on neutral ground (most World Cup games are). |

I split the old single "form" feature into separate **attack** and **defense**
signals, which carry the same information but tell the model *how* a team is strong.

### Models
I train and compare three scikit-learn models:

- **Logistic Regression** (with feature scaling)
- **Random Forest**
- **Gradient Boosting** (`HistGradientBoostingClassifier`)

### Honest, out-of-sample scoring
This is the part I care about most. I train every model on all matches **except**
the 2026 World Cup, then score them only on the World Cup matches that have actually
been played. So the **Model comparison** and **Results** pages show genuine
out-of-sample performance — the models never saw those games during training. The
best-performing model is selected automatically to power the Home-page predictions.

**Reality check:** on a 3-way outcome, ~50–55% accuracy is strong because draws are
genuinely hard to call, so my real goal is to beat the naive baselines (*always pick
home*, *always pick the higher-Elo team*). A representative run scored the trained
models around **63–64% accuracy**, ahead of both baselines. (These numbers are
computed live and shift as more matches are played.)

---

## Tech stack

| Layer | Choice |
|---|---|
| Language | Python |
| ML | scikit-learn (Logistic Regression, Random Forest, Gradient Boosting) |
| Data | pandas, NumPy |
| Web app | Streamlit (multi-page) |
| Data source | martj42/international_results (fetched live) |
| Hosting | Streamlit Community Cloud (free) |

I kept the tables rendering as **Markdown** rather than interactive widgets, on
purpose — it makes the app immune to a Streamlit chunk-loading error that can hit
after a redeploy.

---

## Project structure

```
wc2026-predictor/
├── engine.py                    # shared logic: data, features, models, helpers
├── app.py                       # Home page (upcoming matches)
├── pages/
│   ├── 1_Head_to_head.py
│   ├── 2_Model_comparison.py
│   ├── 3_Results.py
│   └── 4_How_it_works.py
├── data/results.csv             # offline fallback snapshot
├── requirements.txt
└── README.md
```

All the heavy work lives in `engine.py` and runs once (cached with a 6-hour TTL),
shared across every page.

---


## Notes

- **The "best model" can change over time.** As the live data updates, whichever
  model scores best on the World Cup test set is auto-selected for the Home
  predictions, so it may flip between Logistic Regression and Random Forest — they
  run neck-and-neck.
- **Draws are hard.** The models rarely pick a draw as the single most-likely
  outcome, but they still assign it a realistic 20–30% probability.

## Ideas I might add next

- Add FIFA ranking as a feature (needs a historical ranking dataset).
- Simulate the full 48-team bracket to estimate each team's title odds.
- Track prediction accuracy over time as a chart.
- A calibration plot to check how honest the probabilities are.

---

# Built as a personal project to learn end-to-end ML: data, feature engineering,model comparison, honest evaluation, and deployment.


