import streamlit as st
from dataclasses import dataclass

st.set_page_config(
    page_title="Miguel's dashboard",
    page_icon="⚛️",
    layout="wide"
)

# TODO: Random generator for different distributions. Include random angle in a n-sphere.

st.header("Miguel's dashboard", divider="blue")
st.markdown("""
    I build my code into interactive apps, shown here.
    Feel free to explore them.
    Do reach out if you want to build something together
""")

quicklinks = {
    "🌊️ Collapsed Wave Notes": "https://collapsedwave.com",
    "〰 MAWI Project": "https://mawi-net.eu/",
    "📺 Source code": "https://github.com/migueltorrescosta/notebooks/tree/master/pages",
    "GitHub": "https://github.com/migueltorrescosta",
    "LinkedIn": "https://www.linkedin.com/in/miguel-torres-costa/"
}

st.markdown("   |   ".join([f"[{text}]({url})" for text, url in quicklinks.items()]))

st.header("Productivity", divider="orange")

@dataclass
class ProductivityTip:
    summary: str
    url: str
    description: str

productivity_tips = [
    ProductivityTip(
        summary="Don't Ask to Ask",
        url="https://dontasktoask.com/",
        description="In shared groups, it's a lot more productive to ask a specific question, 'Does anyone know how to do X with Y when trying to solve problem Z' than asking 'Does anyone know about Z?'"
    ),
    ProductivityTip(
        summary="No Hello",
        url="https://nohello.net/en/",
        description="Don't send a hello message as a way to check if someone is online."
    ),
    ProductivityTip(
        summary="Write a brag document",
        url="https://jvns.ca/blog/brag-documents/",
        description="A document describing your contributions ensures your work is focused on deliverables, rather than focusing on being/feeling busy. This avoids wasting time and effort. This page can be seen as my brag document."
    ),
    ProductivityTip(
        summary="Link everything",
        url="https://www.asianefficiency.com/productivity/linking/",
        description="There’s just something so satisfying about clicking or tapping a link and being taken right to the thing you need. No searching around, no scrolling, no navigating projects or folders. You’re just… ready to go."
    )
]

st.markdown("""
    Interactions often consume most of our time.
    I recommend the tips below to minimise the friction of online conversations.
    I hope you find them useful too.
    """)

columns = st.columns(2)
colors = ["red", "violet", "gray"]
for i, tip in enumerate(productivity_tips):
    with columns[i%len(columns)]:
        st.subheader(f"[{tip.summary}]({tip.url})")
        st.markdown(f"{tip.description}")
        st.divider()

