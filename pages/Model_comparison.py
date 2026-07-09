"""Model comparison page: accuracy & log-loss for each model, scored out-of-sample
on the played 2026 World Cup matches, plus naive baselines."""
import pandas as pd
import streamlit as st

import engine as E

st.set_page_config(page_title="Model comparison", page_icon="📊", layout="wide")

eng = E.build_engine()
E.render_sidebar(eng)

st.title("📊 Model comparison")

m = eng["metrics"]
if m.empty:
    st.info("No played World Cup 2026 matches to score yet — this table appears "
            "once matches have results.")
else:
    st.write(
        f"Each model trained on **{eng['n_train']:,}** matches (everything *except* "
        f"the 2026 World Cup), then scored **out-of-sample** on the **{eng['n_test']}** "
        f"WC2026 matches played so far. Higher accuracy and lower log-loss are better.")

    disp = m.copy()
    disp["Accuracy"] = disp["Accuracy"].map(lambda x: f"{x * 100:.1f}%")
    disp["Log-loss"] = disp["Log-loss"].map(lambda x: "—" if pd.isna(x) else f"{x:.3f}")
    disp["Model"] = disp["Model"].map(
        lambda n: f"⭐ {n}" if n == eng["best_model"] else n)
    st.markdown(E.to_md_table(disp))

    st.caption("⭐ = current best model (used for the Home predictions). "
               "Baselines make no probabilities, so they have no log-loss.")
    st.markdown(
        "**What this means:** the trained models should beat the naive baselines "
        "(*always home*, *higher Elo*). On a 3-way outcome, ~50–55% accuracy is "
        "strong — draws are inherently hard to call, so a small edge over the Elo "
        "baseline is a real win.")
