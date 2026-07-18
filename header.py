import streamlit as st

def show_header():

    col1, col2 = st.columns([1,5])

    with col1:
        st.image(
            "assets/logo.png",
            width=100
        )

    with col2:
        st.title("Restaurant Profit Optimization")
        st.caption("Predict • Analyze • Optimize • Grow")

    st.image(
        "assets/banner.png",
        use_container_width=True
    )

    st.divider()