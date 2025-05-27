import streamlit as st

def show():
    st.header("About us")

    col1, col2 = st.columns(2)

    with col1:
        st.image("./app/developers.jpg", caption="Development Team", use_container_width=True)

    with col2:
        st.markdown("""
        **Our Mission**  
        A team working on making single-cell data analysis more accessible and properly visualized.  
        This project was created for the course *Software Technology*, supervised by Aristeidis Vrahatis and co-supervised by Konstantinos Lazaros.

        **Team Members**  
        - Mohammad-Matin Marzie – Planning, Code Optimization  
        - Ioannis Giakisikloglou – Visualization, Plotting, Algorithm Implementation

        **Contact Us**  
        - inf2022001@ionio.gr – Matin  
        - inf2022034@ionio.gr – Giannis
        """, unsafe_allow_html=True)