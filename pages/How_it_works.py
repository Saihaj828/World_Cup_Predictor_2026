"""How it works page: plain-English explainer of the data, features, and models."""
import streamlit as st

import engine as E

st.set_page_config(page_title="How it works", page_icon="ℹ️", layout="wide")

eng = E.build_engine()
E.render_sidebar(eng)

st.title("ℹ️ How it works")
st.markdown(f"""
### Data
Every international match since 1872 (community dataset on GitHub), pulled **live**
so new fixtures appear automatically. The models currently train on
**{eng['n_train']:,}** matches (2000 onward), holding out the 2026 World Cup for
honest scoring.

### Features — all leak-free (computed only from *past* matches)
- **Elo rating difference** — overall strength; winners gain points, losers lose them.
- **Attack difference** — average goals *scored* over each team's last 5 games.
- **Defense difference** — average goals *conceded* (fewer is better).
- **Win-rate difference** — share of points won over the last 5 games.
- **Neutral venue** — most World Cup games are on neutral ground.

*(Attack + defense together replace the old single "form" feature — they carry the
same information but split into how a team scores vs. how it defends.)*

### Models
Three scikit-learn models are trained and compared — **Logistic Regression**,
**Random Forest**, and **Gradient Boosting**. The **Model comparison** page scores
each one out-of-sample on the played 2026 World Cup matches; the best is used for
the Home-page predictions.

### Honest scoring
Because the models never train on 2026 World Cup matches, the **Results** page
scores their predictions against real outcomes truly out-of-sample — no cheating.

### Knockouts
Draws can't be final, so knockout "advance" chances split the draw probability
50/50 (a simple stand-in for extra time + penalties).

**Reality check:** ~50–55% accuracy on 3-way outcomes is strong — draws are hard,
so beating the naive baselines is the real measure of success.
""")
