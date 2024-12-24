import streamlit as st

st.set_page_config(
    page_title="Quantum Home base",
    page_icon="⚛️",
    layout="wide"
)

st.header("Miguel's dashboard", divider="blue")

quicklinks = {
    "🌊️ Collapsed Wave": "https://collapsedwave.com",
}

columns = st.columns(len(quicklinks))

for i, text in enumerate(quicklinks.keys()):
    url = quicklinks[text]
    with columns[i]:
        st.markdown(f"[{text}]({url})")

# TODO: Random generator for different distributions. Include random angle in a n-sphere.

