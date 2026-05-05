import streamlit as st


# * Function to setup the page
def setup(
    icon="Deployment/assets/icons/icons8-fire-48.png",
    title="Fire and Smoke Detection",
    sidebar="auto",
):
    # Set the page config
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout="centered",
        initial_sidebar_state=sidebar,
    )
    # Set the sidebar
    st.sidebar.title("Links:")
    st.sidebar.markdown("GitHub [Profile](https://github.com/hank921109)")
    st.sidebar.markdown(
        "GitHub [Repository](https://github.com/hank921109/114-2-Fire_Smoke_Detection)"
    )
    st.sidebar.caption("Developed by ingPrivate, 2026.")
    st.sidebar.image(
        "Deployment/assets/wildfire2.png",
        caption="AI generated image of a wildfire",
        width="stretch",
    )
