import streamlit as st
from dashboard.pages._common import players
st.header("Top batters under pressure")
d = players("bat"); d = d[d.eligible]
n = st.slider("Show top N", 5, 50, 10)
st.dataframe(d.head(n)[["name", "n_matches", "hi_n", "g", "P", "rating", "p_top10", "median_rank", "rank_p05", "rank_p95", "form"]], use_container_width=True)
