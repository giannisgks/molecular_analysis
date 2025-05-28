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
        - Ioannis Giakisikloglou – Data visualization, Plotting και plot comparison visualization, Υλοποίηση αλγορίθμων(Aνάλυση διαφορικής γονιδιακής έκφρασης (DEG analysis, Batch correction, κ.ά.), επεξεργασία και τροποποίηση γονιδίων (gene manipulation), σχεδίαση και σύνταξη διαγραμμάτων UML,Ορισμός δυναμικής παραμετροποίησης των συναρτήσεων, pipeline integration and combinating with streamlit, Implementation of data preview before analysis, interactive streamlit tab implementation with custom theme.
        - Mohammad-Matin Marzie – Planning(μεθοδολογίες υλοποίησης), Code Optimization, Αρχιτεκτονικός καταμερισμός και δομικός διαχωρισμός του κώδικα,ολοκλήρωση της κύριας έκτασης και κατάρτηση της τεχνικής ααφοράς, Προσαρμογή και παραμετροποίηση εμφάνισης με Css για UI στοιχεία(buttons&animation like pulse, glitch, rolling loading, io-button effects and κ.ά.), Δημιουργία κοντέινερ για απομόνωση και φορητότητα με Docker(Dockerization), Εισαγωγή ηχητικής επένδυσης.

        **Contact Us**  
        - inf2022034@ionio.gr – Giannis
        - inf2022001@ionio.gr – Matin  
        """, unsafe_allow_html=True)