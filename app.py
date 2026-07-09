"""
World Cup 2026 Predictor — Home page (upcoming match predictions).

This is now a multi-page Streamlit app. All shared logic lives in engine.py;
the other pages are in pages/. Run:  streamlit run app.py
"""
import streamlit as st

import engine as E

st.set_page_config(page_title="WC2026 Predictor", page_icon="⚽", layout="wide")

eng = E.build_engine()
E.render_sidebar(eng)

st.title("⚽ World Cup 2026 Match Predictor")
st.caption(
    f"Elo + attack/defense/form → {len(eng['models'])} models compared · "
    f"current best: **{eng['best_model']}** · data through {eng['last_played']}")

st.markdown(
    f"Predictions for every upcoming World Cup 2026 fixture, by round, using the "
    f"current best model (**{eng['best_model']}**). Explore **Head-to-head**, "
    f"**Model comparison**, and **Results** from the sidebar.")

fix = eng["fixtures"]
if fix.empty:
    st.info("No upcoming fixtures in the data right now — the tournament may be "
            "between rounds or finished. New rounds appear here automatically.")
else:
    present = [s for s in E.STAGE_ORDER if s in set(fix["stage"])]
    present += [s for s in fix["stage"].unique() if s not in E.STAGE_ORDER]
    for stage in present:
        sub = fix[fix["stage"] == stage]
        knockout = stage != "Group Stage"
        st.subheader(f"{stage} · {len(sub)} match{'es' if len(sub) != 1 else ''}")
        st.markdown(E.to_md_table(
            E.fixtures_table(eng, eng["best_model"], sub, knockout)))
    st.caption("Later rounds appear automatically once matchups are confirmed. "
               "Use **Refresh data & models** in the sidebar to pull the latest.")
