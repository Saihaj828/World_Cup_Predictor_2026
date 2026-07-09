"""Results page: scoreboard of predictions vs. actual outcomes for the played
2026 World Cup matches (truly out-of-sample — models never trained on them)."""
import pandas as pd
import streamlit as st

import engine as E

st.set_page_config(page_title="Results", page_icon="✅", layout="wide")

eng = E.build_engine()
E.render_sidebar(eng)

st.title("✅ Results — predictions vs. reality")

test = eng["test_df"]
if test.empty:
    st.info("No played World Cup 2026 matches yet. Once matches have results, this "
            "page scores what each model predicted against what actually happened.")
    st.stop()

model_names = list(eng["models"])
model_name = st.selectbox("Score this model", model_names,
                          index=model_names.index(eng["best_model"]))
model = eng["models"][model_name]

X = test[E.FEATURES].astype(float)
preds = model.predict(X)
proba = E._align_proba(model.predict_proba(X), model.classes_)

rows, correct = [], 0
for i, row in enumerate(test.itertuples(index=False)):
    pred = preds[i]
    conf = proba[i][E.LABELS.index(pred)]
    ok = bool(pred == row.result)
    correct += ok
    name = {"home_win": row.home_team, "away_win": row.away_team, "draw": "Draw"}
    rows.append({
        "Date": row.date.date(),
        "Stage": E.stage_for(row.date.date()),
        "Match": f"{row.home_team} {row.home_score}–{row.away_score} {row.away_team}",
        "Actual": name[row.result],
        "Predicted": name[pred],
        "Conf.": E.pct(conf),
        "Hit": "✅" if ok else "❌",
    })

acc = correct / len(test)
st.metric(f"{model_name}: correct predictions",
          f"{correct} / {len(test)}", f"{acc * 100:.0f}% accuracy")

tbl = pd.DataFrame(rows).sort_values("Date", ascending=False)
st.markdown(E.to_md_table(tbl))
st.caption("Out-of-sample: the models were never trained on any 2026 World Cup "
           "match, so these are honest predictions. Draws are scored as a miss "
           "unless the model's top pick was a draw.")
