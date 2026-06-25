"""
World Cup 2026 Match Predictor — a single-file Streamlit web app.

Frontend + backend + ML model all live in THIS one file (see README.md for the
reasoning). The same Elo + recent-form + random-forest pipeline as the notebook,
wrapped in a small web UI.

Run locally:   streamlit run app.py
Deploy free:   push to GitHub, then deploy at https://share.streamlit.io
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
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# We pull the dataset LIVE so newly-scheduled fixtures (e.g. the Round of 32,
# once the group stage ends) show up automatically with no code change. If the
# download fails we fall back to the snapshot bundled in data/.
DATA_URL = ("https://raw.githubusercontent.com/martj42/"
            "international_results/master/results.csv")
LOCAL_CSV = Path(__file__).parent / "data" / "results.csv"

START_YEAR  = 2000   # train on the modern era only
FORM_WINDOW = 5      # matches that define a team's "recent form"
K           = 30     # Elo update speed
HOME_ADV    = 60     # Elo bonus given to a non-neutral home side
FEATURES    = ["elo_diff", "form_diff", "neutral"]

# Official WC2026 schedule windows, used to label each fixture's round purely by
# its date. Knockout matchups only enter the data once the bracket is fixed, so
# labelling by date means new rounds slot in automatically as the data updates.
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


# --------------------------------------------------------------------------- #
# Data + model  (cached so we download & train once, not on every click)
# --------------------------------------------------------------------------- #
def _read_results() -> tuple[pd.DataFrame, str]:
    """Return (dataframe, source-label). Prefer live data, fall back to bundle."""
    try:
        with urllib.request.urlopen(DATA_URL, timeout=20) as resp:
            text = resp.read().decode("utf-8")
        return pd.read_csv(io.StringIO(text), parse_dates=["date"]), "live (GitHub)"
    except Exception:
        return pd.read_csv(LOCAL_CSV, parse_dates=["date"]), "bundled snapshot"


@st.cache_resource(ttl=6 * 3600, show_spinner="Crunching match history…")
def build_engine() -> dict:
    """Load data, engineer features, train the model, list upcoming fixtures.

    Cached as a resource with a 6-hour TTL: heavy work runs once, then refreshes
    periodically so freshly-scheduled rounds are picked up automatically.
    """
    df, source = _read_results()
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")

    played = df.dropna(subset=["home_score", "away_score"]).copy()
    played["home_score"] = played["home_score"].astype(int)
    played["away_score"] = played["away_score"].astype(int)
    played = (played[played["date"].dt.year >= START_YEAR]
              .sort_values("date").reset_index(drop=True))

    # label: home_win / draw / away_win
    played["result"] = np.select(
        [played.home_score > played.away_score,
         played.home_score < played.away_score],
        ["home_win", "away_win"], default="draw")

    # --- Elo ratings: one leak-free forward pass over the matches ------------
    ratings: dict[str, float] = {}
    home_elo, away_elo = [], []
    for row in played.itertuples(index=False):
        ra = ratings.get(row.home_team, 1500.0)
        rb = ratings.get(row.away_team, 1500.0)
        home_elo.append(ra)
        away_elo.append(rb)
        adv = 0 if row.neutral else HOME_ADV
        exp_home = 1.0 / (1.0 + 10 ** ((rb - (ra + adv)) / 400))
        if   row.home_score > row.away_score: s = 1.0
        elif row.home_score < row.away_score: s = 0.0
        else:                                 s = 0.5
        ratings[row.home_team] = ra + K * (s - exp_home)
        ratings[row.away_team] = rb - K * (s - exp_home)
    played["elo_diff"] = np.array(home_elo) - np.array(away_elo)

    # --- recent form: avg goal difference over last FORM_WINDOW matches ------
    recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    hf_list, af_list = [], []
    for row in played.itertuples(index=False):
        hf = np.mean(recent[row.home_team]) if recent[row.home_team] else 0.0
        af = np.mean(recent[row.away_team]) if recent[row.away_team] else 0.0
        hf_list.append(hf)
        af_list.append(af)
        gd = row.home_score - row.away_score
        recent[row.home_team].append(gd)
        recent[row.away_team].append(-gd)
    played["form_diff"] = np.array(hf_list) - np.array(af_list)
    latest_form = {t: (float(np.mean(v)) if v else 0.0) for t, v in recent.items()}

    # --- train the classifier ------------------------------------------------
    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        random_state=42, n_jobs=-1)
    model.fit(played[FEATURES].astype(float), played["result"])

    # --- upcoming (unplayed) World Cup 2026 fixtures -------------------------
    fix = df[(df.tournament == "FIFA World Cup")
             & (df.home_score.isna())
             & (df.date.dt.year >= 2026)].sort_values("date").copy()
    fix["stage"] = fix["date"].dt.date.map(stage_for)

    return {
        "model": model,
        "ratings": ratings,
        "latest_form": latest_form,
        "teams": sorted(ratings),
        "fixtures": fix,
        "source": source,
        "n_played": len(played),
        "last_played": played["date"].max().date(),
    }


def predict_proba(engine: dict, home: str, away: str, neutral: bool) -> dict:
    """Outcome probabilities {home_win, draw, away_win} for one matchup."""
    r, f = engine["ratings"], engine["latest_form"]
    elo_diff = r[home] - r[away]
    form_diff = f.get(home, 0.0) - f.get(away, 0.0)
    X = pd.DataFrame([[elo_diff, form_diff, float(neutral)]], columns=FEATURES)
    proba = engine["model"].predict_proba(X)[0]
    return dict(zip(engine["model"].classes_, proba))


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def fixtures_table(engine: dict, sub: pd.DataFrame, knockout: bool) -> pd.DataFrame:
    """Build a display table of predictions for a set of fixtures.

    Group games show W/D/L. Knockout games can't end level, so we report the
    chance each side *advances*, splitting the draw probability 50/50 (a simple
    stand-in for the extra-time / penalty coin-flip)."""
    rows = []
    for f in sub.itertuples(index=False):
        p = predict_proba(engine, f.home_team, f.away_team, bool(f.neutral))
        if knockout:
            adv_h = p["home_win"] + 0.5 * p["draw"]
            adv_a = p["away_win"] + 0.5 * p["draw"]
            rows.append({
                "Date": f.date.date(),
                "Home": f.home_team,
                "Home advances": pct(adv_h),
                "Away advances": pct(adv_a),
                "Away": f.away_team,
                "Favourite": f.home_team if adv_h >= adv_a else f.away_team,
            })
        else:
            label = {"home_win": f.home_team, "draw": "Draw", "away_win": f.away_team}
            rows.append({
                "Date": f.date.date(),
                "Home": f.home_team,
                "Home win": pct(p["home_win"]),
                "Draw": pct(p["draw"]),
                "Away win": pct(p["away_win"]),
                "Away": f.away_team,
                "Prediction": label[max(p, key=p.get)],
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="⚽",
                   layout="wide")

engine = build_engine()

st.title("⚽ World Cup 2026 Match Predictor")
st.caption(
    f"Elo + recent form → random forest · trained on {engine['n_played']:,} "
    f"international matches (through {engine['last_played']}) · "
    f"data source: {engine['source']}")

with st.sidebar:
    st.header("About")
    st.write(
        "Predicts upcoming World Cup 2026 matches from historical results. "
        "Knockout rounds (Round of 32 onward) appear automatically once their "
        "matchups are confirmed in the source data.")
    if st.button("🔄 Refresh data & model"):
        st.cache_resource.clear()
        st.rerun()
    st.caption("See README.md for how it works and why each choice was made.")

tab_fixtures, tab_h2h, tab_info = st.tabs(
    ["📅 Upcoming matches", "⚔️ Head-to-head", "ℹ️ How it works"])

# --- Tab 1: upcoming fixtures, grouped by round ---------------------------- #
with tab_fixtures:
    fix = engine["fixtures"]
    if fix.empty:
        st.info("No upcoming fixtures in the data right now. As soon as the next "
                "round is scheduled, it will show up here automatically.")
    else:
        present = [s for s in STAGE_ORDER if s in set(fix["stage"])]
        present += [s for s in fix["stage"].unique() if s not in STAGE_ORDER]
        for stage in present:
            sub = fix[fix["stage"] == stage]
            knockout = stage != "Group Stage"
            st.subheader(f"{stage} · {len(sub)} match{'es' if len(sub) != 1 else ''}")
            st.dataframe(fixtures_table(engine, sub, knockout),
                         hide_index=True, width="stretch")
        st.caption(
            "ℹ️ The Round of 32 and later rounds slot in here as soon as their "
            "matchups are confirmed upstream — no code changes needed. Use "
            "**Refresh data & model** in the sidebar to pull the latest.")

# --- Tab 2: pick any two teams --------------------------------------------- #
with tab_h2h:
    st.write("Pick any two teams to see the predicted outcome.")
    teams = engine["teams"]
    c1, c2 = st.columns(2)
    idx_a = teams.index("Brazil") if "Brazil" in teams else 0
    idx_b = teams.index("Argentina") if "Argentina" in teams else 1
    team_a = c1.selectbox("Team A (home side)", teams, index=idx_a)
    team_b = c2.selectbox("Team B (away side)", teams, index=idx_b)
    neutral = st.checkbox("Neutral venue", value=True,
                          help="Tick for a World Cup match on neutral ground; "
                               "untick if Team A is genuinely at home.")
    if team_a == team_b:
        st.warning("Pick two different teams.")
    else:
        p = predict_proba(engine, team_a, team_b, neutral)
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{team_a} win", pct(p["home_win"]))
        m2.metric("Draw", pct(p["draw"]))
        m3.metric(f"{team_b} win", pct(p["away_win"]))
        st.bar_chart(pd.DataFrame(
            {"probability": [p["home_win"], p["draw"], p["away_win"]]},
            index=[f"{team_a} win", "Draw", f"{team_b} win"]))

# --- Tab 3: explainer ------------------------------------------------------ #
with tab_info:
    st.markdown(
        """
### How it works

1. **Data** — every international match since 1872 (community dataset on GitHub),
   pulled live so new fixtures appear on their own.
2. **Features** (computed only from *past* matches, so no peeking at results):
   - **Elo rating difference** — the single strongest signal of relative strength.
   - **Recent form** — average goal difference over each team's last 5 games.
   - **Neutral venue** flag — most World Cup games are on neutral ground.
3. **Model** — a `RandomForestClassifier` predicting home win / draw / away win.
4. **Knockouts** — draws can't be final, so the chance to *advance* splits the
   draw probability 50/50 (a simple stand-in for extra time + penalties).

**Reality check:** 3-way football prediction tops out around ~50–55% accuracy
because draws are genuinely hard to call. This is a deliberately simple model —
see `README.md` for the full design rationale and ideas to extend it.
        """)
