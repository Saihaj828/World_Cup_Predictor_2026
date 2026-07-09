"""
Shared engine for the World Cup 2026 Predictor (multi-page Streamlit app).

Every page (app.py + pages/*.py) imports this module so the heavy work runs once
and is shared via @st.cache_resource:
  - load match data (live, with a bundled fallback)
  - engineer leak-free features (Elo, attack, defense, win-rate, venue)
  - train several models and score them out-of-sample on the 2026 World Cup
  - helpers to predict a matchup and render Markdown tables
"""
from __future__ import annotations

import io
import urllib.request
from collections import defaultdict, deque
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_URL = ("https://raw.githubusercontent.com/martj42/"
            "international_results/master/results.csv")
LOCAL_CSV = Path(__file__).parent / "data" / "results.csv"

START_YEAR  = 2000   # modern era only
FORM_WINDOW = 5      # matches used for attack / defense / win-rate form
K           = 30     # Elo update speed
HOME_ADV    = 60     # Elo bonus for a non-neutral home side
FEATURES    = ["elo_diff", "att_diff", "def_diff", "winrate_diff", "neutral"]
LABELS      = ["away_win", "draw", "home_win"]   # fixed order (log-loss / proba)

SCHEDULE = [
    (date(2026, 6, 11), date(2026, 6, 27), "Group Stage"),
    (date(2026, 6, 28), date(2026, 7,  3), "Round of 32"),
    (date(2026, 7,  4), date(2026, 7,  7), "Round of 16"),
    (date(2026, 7,  9), date(2026, 7, 11), "Quarter-finals"),
    (date(2026, 7, 14), date(2026, 7, 15), "Semi-finals"),
    (date(2026, 7, 18), date(2026, 7, 18), "Third-place play-off"),
    (date(2026, 7, 19), date(2026, 7, 19), "Final"),
]
STAGE_ORDER = [name for *_, name in SCHEDULE]


def stage_for(d: date) -> str:
    """Map a fixture date to its tournament round."""
    for start, end, name in SCHEDULE:
        if start <= d <= end:
            return name
    return "Knockout"


def make_models() -> dict:
    """The models we train and compare (all scikit-learn)."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=20,
            random_state=42, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.08, max_iter=300, random_state=42),
    }


def _align_proba(proba: np.ndarray, classes) -> np.ndarray:
    """Reorder predict_proba columns into LABELS order (defensive: sklearn sorts
    classes alphabetically, which already equals LABELS)."""
    idx = [list(classes).index(lbl) for lbl in LABELS]
    return proba[:, idx]


def _mean(d) -> float:
    return float(np.mean(d)) if d else 0.0


# --------------------------------------------------------------------------- #
# Data + features + training (cached: runs once, shared across all pages)
# --------------------------------------------------------------------------- #
def _read_results() -> tuple[pd.DataFrame, str]:
    """Return (dataframe, source-label). Prefer live data, fall back to bundle."""
    try:
        with urllib.request.urlopen(DATA_URL, timeout=20) as resp:
            text = resp.read().decode("utf-8")
        return pd.read_csv(io.StringIO(text), parse_dates=["date"]), "live (GitHub)"
    except Exception:
        return pd.read_csv(LOCAL_CSV, parse_dates=["date"]), "bundled snapshot"


@st.cache_resource(ttl=6 * 3600,
                   show_spinner="Crunching match history & training models…")
def build_engine() -> dict:
    df, source = _read_results()
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")

    played = df.dropna(subset=["home_score", "away_score"]).copy()
    played["home_score"] = played["home_score"].astype(int)
    played["away_score"] = played["away_score"].astype(int)
    played = (played[played["date"].dt.year >= START_YEAR]
              .sort_values("date").reset_index(drop=True))

    played["result"] = np.select(
        [played.home_score > played.away_score,
         played.home_score < played.away_score],
        ["home_win", "away_win"], default="draw")

    # --- one leak-free forward pass: Elo + rolling goals-for/against/result --
    ratings: dict[str, float] = {}
    gf = defaultdict(lambda: deque(maxlen=FORM_WINDOW))   # goals scored
    ga = defaultdict(lambda: deque(maxlen=FORM_WINDOW))   # goals conceded
    res = defaultdict(lambda: deque(maxlen=FORM_WINDOW))  # 1 win / .5 draw / 0 loss

    h_elo, a_elo, h_att, a_att, h_def, a_def, h_win, a_win = ([] for _ in range(8))

    for row in played.itertuples(index=False):
        ra = ratings.get(row.home_team, 1500.0)
        rb = ratings.get(row.away_team, 1500.0)
        # snapshot PRE-match state (no leakage)
        h_elo.append(ra); a_elo.append(rb)
        h_att.append(_mean(gf[row.home_team])); a_att.append(_mean(gf[row.away_team]))
        h_def.append(_mean(ga[row.home_team])); a_def.append(_mean(ga[row.away_team]))
        h_win.append(_mean(res[row.home_team])); a_win.append(_mean(res[row.away_team]))

        # Elo update
        adv = 0 if row.neutral else HOME_ADV
        exp_home = 1.0 / (1.0 + 10 ** ((rb - (ra + adv)) / 400))
        if   row.home_score > row.away_score: s = 1.0
        elif row.home_score < row.away_score: s = 0.0
        else:                                 s = 0.5
        ratings[row.home_team] = ra + K * (s - exp_home)
        ratings[row.away_team] = rb - K * (s - exp_home)

        # update rolling stats (AFTER snapshot)
        gf[row.home_team].append(row.home_score); ga[row.home_team].append(row.away_score)
        gf[row.away_team].append(row.away_score); ga[row.away_team].append(row.home_score)
        res[row.home_team].append(s); res[row.away_team].append(1 - s)

    played["elo_diff"]     = np.array(h_elo) - np.array(a_elo)
    played["att_diff"]     = np.array(h_att) - np.array(a_att)
    played["def_diff"]     = np.array(a_def) - np.array(h_def)   # home concedes fewer ⇒ +
    played["winrate_diff"] = np.array(h_win) - np.array(a_win)
    played["neutral"]      = played["neutral"].astype(float)

    latest = {
        "elo": ratings,
        "att": {t: _mean(v) for t, v in gf.items()},
        "def": {t: _mean(v) for t, v in ga.items()},
        "win": {t: _mean(v) for t, v in res.items()},
    }

    # --- hold out the 2026 World Cup so scoring is honest (out-of-sample) ----
    is_wc26 = (played.tournament == "FIFA World Cup") & (played.date.dt.year == 2026)
    train_df = played[~is_wc26]
    test_df = played[is_wc26].copy()

    X_train, y_train = train_df[FEATURES].astype(float), train_df["result"]
    models = {name: m.fit(X_train, y_train) for name, m in make_models().items()}

    # --- score every model on the played WC2026 matches ---------------------
    metrics = []
    if len(test_df):
        Xte, yte = test_df[FEATURES].astype(float), test_df["result"]
        for name, m in models.items():
            proba = _align_proba(m.predict_proba(Xte), m.classes_)
            metrics.append({"Model": name,
                            "Accuracy": accuracy_score(yte, m.predict(Xte)),
                            "Log-loss": log_loss(yte, proba, labels=LABELS)})
        metrics.append({"Model": "Baseline: always home",
                        "Accuracy": accuracy_score(yte, ["home_win"] * len(yte)),
                        "Log-loss": None})
        elo_pick = np.where(test_df["elo_diff"] >= 0, "home_win", "away_win")
        metrics.append({"Model": "Baseline: higher Elo",
                        "Accuracy": accuracy_score(yte, elo_pick), "Log-loss": None})

    metrics_df = pd.DataFrame(metrics)
    if len(metrics_df) and metrics_df["Log-loss"].notna().any():
        best_model = (metrics_df[metrics_df["Log-loss"].notna()]
                      .sort_values(["Accuracy", "Log-loss"], ascending=[False, True])
                      ["Model"].iloc[0])
    else:
        best_model = list(models)[0]

    # --- upcoming (unplayed) WC2026 fixtures --------------------------------
    fix = df[(df.tournament == "FIFA World Cup")
             & (df.home_score.isna())
             & (df.date.dt.year >= 2026)].sort_values("date").copy()
    fix["stage"] = fix["date"].dt.date.map(stage_for)

    return {
        "models": models,
        "metrics": metrics_df,
        "best_model": best_model,
        "latest": latest,
        "teams": sorted(ratings),
        "fixtures": fix,
        "test_df": test_df,
        "source": source,
        "n_played": len(played),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "last_played": played["date"].max().date(),
    }


# --------------------------------------------------------------------------- #
# Prediction + display helpers
# --------------------------------------------------------------------------- #
def feature_row(latest: dict, home: str, away: str, neutral: bool) -> pd.DataFrame:
    """Build a one-row feature frame for a matchup from the latest team snapshots."""
    L = latest
    return pd.DataFrame([{
        "elo_diff":     L["elo"].get(home, 1500.0) - L["elo"].get(away, 1500.0),
        "att_diff":     L["att"].get(home, 0.0) - L["att"].get(away, 0.0),
        "def_diff":     L["def"].get(away, 0.0) - L["def"].get(home, 0.0),
        "winrate_diff": L["win"].get(home, 0.0) - L["win"].get(away, 0.0),
        "neutral":      float(neutral),
    }], columns=FEATURES)


def predict_proba(engine: dict, model_name: str, home: str, away: str,
                  neutral: bool) -> dict:
    """{home_win, draw, away_win} probabilities for a matchup, using one model."""
    model = engine["models"][model_name]
    X = feature_row(engine["latest"], home, away, neutral)
    proba = _align_proba(model.predict_proba(X), model.classes_)[0]
    return dict(zip(LABELS, proba))


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def to_md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a Markdown table (core bundle only — immune to the
    st.dataframe 'failed to fetch dynamically imported module' error)."""
    cols = list(df.columns)
    header = "| " + " | ".join(map(str, cols)) + " |"
    divider = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join([header, divider, *body])


def fixtures_table(engine: dict, model_name: str, sub: pd.DataFrame,
                   knockout: bool) -> pd.DataFrame:
    """Predictions table for a set of fixtures. Knockouts show 'advance' chances
    (draw probability split 50/50 for the extra-time/penalty coin-flip)."""
    rows = []
    for f in sub.itertuples(index=False):
        p = predict_proba(engine, model_name, f.home_team, f.away_team, bool(f.neutral))
        if knockout:
            adv_h = p["home_win"] + 0.5 * p["draw"]
            adv_a = p["away_win"] + 0.5 * p["draw"]
            rows.append({"Date": f.date.date(), "Home": f.home_team,
                         "Home advances": pct(adv_h), "Away advances": pct(adv_a),
                         "Away": f.away_team,
                         "Favourite": f.home_team if adv_h >= adv_a else f.away_team})
        else:
            label = {"home_win": f.home_team, "draw": "Draw", "away_win": f.away_team}
            rows.append({"Date": f.date.date(), "Home": f.home_team,
                         "Home win": pct(p["home_win"]), "Draw": pct(p["draw"]),
                         "Away win": pct(p["away_win"]), "Away": f.away_team,
                         "Prediction": label[max(p, key=p.get)]})
    return pd.DataFrame(rows)


def render_sidebar(engine: dict) -> None:
    """Shared sidebar shown on every page."""
    with st.sidebar:
        st.header("⚽ WC2026 Predictor")
        st.caption(f"Trained on {engine['n_train']:,} matches · "
                   f"scored on {engine['n_test']} played WC2026 games · "
                   f"data: {engine['source']}")
        if st.button("🔄 Refresh data & models"):
            st.cache_resource.clear()
            st.rerun()
