"""Head-to-head page: predict any matchup with any of the trained models."""
import streamlit as st

import engine as E

st.set_page_config(page_title="Head-to-head", page_icon="⚔️", layout="wide")

eng = E.build_engine()
E.render_sidebar(eng)

st.title("⚔️ Head-to-head")
st.write("Pick any two teams and a model to see the predicted outcome.")

teams = eng["teams"]
model_names = list(eng["models"])

c1, c2, c3 = st.columns(3)
idx_a = teams.index("Brazil") if "Brazil" in teams else 0
idx_b = teams.index("Argentina") if "Argentina" in teams else 1
team_a = c1.selectbox("Team A (home side)", teams, index=idx_a)
team_b = c2.selectbox("Team B (away side)", teams, index=idx_b)
model_name = c3.selectbox("Model", model_names,
                          index=model_names.index(eng["best_model"]))
neutral = st.checkbox("Neutral venue", value=True,
                      help="Tick for a match on neutral ground; untick if Team A "
                           "is genuinely at home.")

if team_a == team_b:
    st.warning("Pick two different teams.")
else:
    p = E.predict_proba(eng, model_name, team_a, team_b, neutral)
    m1, m2, m3 = st.columns(3)
    m1.metric(f"{team_a} win", E.pct(p["home_win"]))
    m2.metric("Draw", E.pct(p["draw"]))
    m3.metric(f"{team_b} win", E.pct(p["away_win"]))
    for label, prob in ((f"{team_a} win", p["home_win"]),
                        ("Draw", p["draw"]),
                        (f"{team_b} win", p["away_win"])):
        filled = round(prob * 24)
        st.markdown(f"**{label}** — {E.pct(prob)}  \n"
                    f"`{'█' * filled}{'░' * (24 - filled)}`")
