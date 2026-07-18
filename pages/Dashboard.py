import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.header import show_header

st.set_page_config(
    page_title="Restaurant Profit Optimization",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------
# Page Config
# ------------------------------------------------


show_header()

# ------------------------------------------------
# Load Dataset
# ------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/restaurant_profit_dataset_20000_rows.csv")

df = load_data()

# ------------------------------------------------
# Feature Engineering
# ------------------------------------------------

df["TotalRevenue"] = (
    df["InStoreRevenue"] +
    df["UberEatsRevenue"] +
    df["DoorDashRevenue"] +
    df["SelfDeliveryRevenue"]
)

df["TotalNetProfit"] = (
    df["InStoreNetProfit"] +
    df["UberEatsNetProfit"] +
    df["DoorDashNetProfit"] +
    df["SelfDeliveryNetProfit"]
)

# ------------------------------------------------
# Dashboard Title
# ------------------------------------------------

st.title("Executive Dashboard")
st.write("Business Overview of SkyCity Auckland Restaurants")

st.markdown("---")

# ------------------------------------------------
# KPI Cards
# ------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Revenue",
        f"${df['TotalRevenue'].sum():,.0f}"
    )

with col2:

    st.metric(
        "Total Profit",
        f"${df['TotalNetProfit'].sum():,.0f}"
    )

with col3:

    st.metric(
        "Total Orders",
        f"{df['MonthlyOrders'].sum():,.0f}"
    )

with col4:

    st.metric(
        "Restaurants",
        df["RestaurantName"].nunique()
    )

st.markdown("---")

# ------------------------------------------------
# Revenue & Profit Charts
# ------------------------------------------------

left, right = st.columns(2)

with left:

    revenue = pd.DataFrame({

        "Channel":[

            "InStore",
            "Uber Eats",
            "DoorDash",
            "Self Delivery"

        ],

        "Revenue":[

            df["InStoreRevenue"].sum(),

            df["UberEatsRevenue"].sum(),

            df["DoorDashRevenue"].sum(),

            df["SelfDeliveryRevenue"].sum()

        ]

    })

    fig = px.bar(

        revenue,

        x="Channel",

        y="Revenue",

        color="Revenue",

        title="Revenue by Channel"

    )

    st.plotly_chart(fig,use_container_width=True)

with right:

    profit = pd.DataFrame({

        "Channel":[

            "InStore",

            "Uber Eats",

            "DoorDash",

            "Self Delivery"

        ],

        "Profit":[

            df["InStoreNetProfit"].sum(),

            df["UberEatsNetProfit"].sum(),

            df["DoorDashNetProfit"].sum(),

            df["SelfDeliveryNetProfit"].sum()

        ]

    })

    fig = px.pie(

        profit,

        names="Channel",

        values="Profit",

        hole=0.5,

        title="Profit Distribution"

    )

    st.plotly_chart(fig,use_container_width=True)

# ------------------------------------------------
# Profit by Cuisine
# ------------------------------------------------

st.markdown("---")

fig = px.bar(

    df.groupby("CuisineType")["TotalNetProfit"]

    .mean()

    .reset_index(),

    x="CuisineType",

    y="TotalNetProfit",

    color="CuisineType",

    title="Average Profit by Cuisine"

)

st.plotly_chart(fig,use_container_width=True)

# ------------------------------------------------
# Monthly Orders
# ------------------------------------------------

st.markdown("---")

fig = px.histogram(

    df,

    x="MonthlyOrders",

    nbins=25,

    title="Monthly Orders Distribution"

)

st.plotly_chart(fig,use_container_width=True)

# ------------------------------------------------
# Average Order Value
# ------------------------------------------------

col1,col2=st.columns(2)

with col1:

    fig=px.box(

        df,

        y="AOV",

        title="Average Order Value"

    )

    st.plotly_chart(fig,use_container_width=True)

with col2:

    fig=px.scatter(

        df,

        x="MonthlyOrders",

        y="TotalNetProfit",

        color="CuisineType",

        title="Orders vs Profit"

    )

    st.plotly_chart(fig,use_container_width=True)

# ------------------------------------------------
# Top Restaurants
# ------------------------------------------------

st.markdown("---")

top=df[

    ["RestaurantName","TotalNetProfit"]

].sort_values(

    by="TotalNetProfit",

    ascending=False

).head(10)

st.subheader("Top 10 Restaurants")

st.dataframe(top,use_container_width=True)

# ------------------------------------------------
# Footer
# ------------------------------------------------

st.markdown("---")

st.success(
    "Restaurant Profit Optimization Dashboard Loaded Successfully"
)


st.divider()
st.caption(
    "Restaurant Profit Optimization System | Final Project | 2026"
)