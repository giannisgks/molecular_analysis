import streamlit as st

from tabs import upload_tab, preprocess_tab, visualize_tab, algorithms_tab, about_tab


st.set_page_config(page_title="scRNA-seq App", layout="wide")

with open("./app/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

with open("./app/loader.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Create tab layout
tabs = st.tabs(["Upload Data", "Preprocessing", "Visualisation (before vs after)","Algorithms", "About Us"])
# --------- Tab 1: Upload Data ---------
with tabs[0]:
    upload_tab.show()

with tabs[1]:
    preprocess_tab.show()
    

with tabs[2]:
    visualize_tab.show()
    

with tabs[3]:
    algorithms_tab.show()

with tabs[4]:
    about_tab.show()
