
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from utils.header import show_header


st.set_page_config(
    page_title="Restaurant Profit Optimization",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------



show_header()

# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/restaurant_profit_dataset_20000_rows.csv")

df = load_data()

# ----------------------------------------------------
# Create Required Columns
# ----------------------------------------------------

if "TotalRevenue" not in df.columns:

    df["TotalRevenue"] = (
        df["InStoreRevenue"] +
        df["UberEatsRevenue"] +
        df["DoorDashRevenue"] +
        df["SelfDeliveryRevenue"]
    )

if "TotalNetProfit" not in df.columns:

    df["TotalNetProfit"] = (
        df["InStoreNetProfit"] +
        df["UberEatsNetProfit"] +
        df["DoorDashNetProfit"] +
        df["SelfDeliveryNetProfit"]
    )

# ----------------------------------------------------
# Sidebar Filters
# ----------------------------------------------------

st.sidebar.title("Filters")

cuisine = st.sidebar.multiselect(
    "Cuisine",
    sorted(df["CuisineType"].unique()),
    default=sorted(df["CuisineType"].unique())
)

segment = st.sidebar.multiselect(
    "Segment",
    sorted(df["Segment"].unique()),
    default=sorted(df["Segment"].unique())
)

subregion = st.sidebar.multiselect(
    "Subregion",
    sorted(df["Subregion"].unique()),
    default=sorted(df["Subregion"].unique())
)

filtered_df = df[
    (df["CuisineType"].isin(cuisine)) &
    (df["Segment"].isin(segment)) &
    (df["Subregion"].isin(subregion))
]

# ----------------------------------------------------
# Page Title
# ----------------------------------------------------

st.title("Exploratory Data Analysis")

st.write(
    "Explore restaurant performance, revenue, customer behaviour, "
    "and profitability through interactive visualizations."
)

st.divider()

# ----------------------------------------------------
# Dataset Overview
# ----------------------------------------------------

st.subheader("Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Rows",
        filtered_df.shape[0]
    )

with c2:
    st.metric(
        "Columns",
        filtered_df.shape[1]
    )

with c3:
    st.metric(
        "Missing Values",
        int(filtered_df.isnull().sum().sum())
    )

with c4:
    st.metric(
        "Duplicate Rows",
        int(filtered_df.duplicated().sum())
    )

st.divider()

# ----------------------------------------------------
# Numeric Summary
# ----------------------------------------------------

st.subheader("Statistical Summary")

st.dataframe(
    filtered_df.describe().T,
    use_container_width=True
)

st.divider()

# ----------------------------------------------------
# Revenue Summary
# ----------------------------------------------------

left, right = st.columns(2)

with left:

    revenue = pd.DataFrame({

        "Channel": [
            "InStore",
            "Uber Eats",
            "DoorDash",
            "Self Delivery"
        ],

        "Revenue": [

            filtered_df["InStoreRevenue"].sum(),

            filtered_df["UberEatsRevenue"].sum(),

            filtered_df["DoorDashRevenue"].sum(),

            filtered_df["SelfDeliveryRevenue"].sum()

        ]

    })

    fig = px.bar(
        revenue,
        x="Channel",
        y="Revenue",
        color="Revenue",
        title="Revenue by Channel"
    )

    st.plotly_chart(fig, use_container_width=True)

with right:

    profit = pd.DataFrame({

        "Channel": [
            "InStore",
            "Uber Eats",
            "DoorDash",
            "Self Delivery"
        ],

        "Profit": [

            filtered_df["InStoreNetProfit"].sum(),

            filtered_df["UberEatsNetProfit"].sum(),

            filtered_df["DoorDashNetProfit"].sum(),

            filtered_df["SelfDeliveryNetProfit"].sum()

        ]

    })

    fig = px.pie(
        profit,
        names="Channel",
        values="Profit",
        hole=0.55,
        title="Profit Contribution"
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ----------------------------------------------------
# Dataset Preview
# ----------------------------------------------------

st.subheader("Dataset Preview")

st.dataframe(
    filtered_df.head(20),
    use_container_width=True
)

st.divider()



# ----------------------------------------------------
# Distribution Analysis
# ----------------------------------------------------

st.subheader("Distribution Analysis")

col1, col2 = st.columns(2)

with col1:

    fig = px.histogram(
        filtered_df,
        x="MonthlyOrders",
        nbins=30,
        color_discrete_sequence=["#4F8BF9"],
        title="Monthly Orders Distribution"
    )

    fig.update_layout(
        xaxis_title="Monthly Orders",
        yaxis_title="Count"
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:

    fig = px.histogram(
        filtered_df,
        x="AOV",
        nbins=30,
        color_discrete_sequence=["#00CC96"],
        title="Average Order Value Distribution"
    )

    fig.update_layout(
        xaxis_title="Average Order Value",
        yaxis_title="Count"
    )

    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    fig = px.histogram(
        filtered_df,
        x="TotalRevenue",
        nbins=30,
        color_discrete_sequence=["#FFA15A"],
        title="Total Revenue Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:

    fig = px.histogram(
        filtered_df,
        x="TotalNetProfit",
        nbins=30,
        color_discrete_sequence=["#EF553B"],
        title="Net Profit Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ----------------------------------------------------
# Orders by Channel
# ----------------------------------------------------

st.subheader("📦 Orders by Channel")

orders = pd.DataFrame({

    "Channel":[
        "InStore",
        "Uber Eats",
        "DoorDash",
        "Self Delivery"
    ],

    "Orders":[

        filtered_df["InStoreOrdersCount"].sum(),

        filtered_df["UberEatsOrdersCount"].sum(),

        filtered_df["DoorDashOrdersCount"].sum(),

        filtered_df["SelfDeliveryOrdersCount"].sum()

    ]

})

fig = px.bar(

    orders,

    x="Channel",

    y="Orders",

    color="Orders",

    text_auto=".2s",

    title="Orders Received Across Channels"

)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ----------------------------------------------------
# Revenue by Cuisine
# ----------------------------------------------------

st.subheader("Revenue by Cuisine")

cuisine_rev = (

    filtered_df

    .groupby("CuisineType")["TotalRevenue"]

    .sum()

    .reset_index()

    .sort_values("TotalRevenue", ascending=False)

)

fig = px.bar(

    cuisine_rev,

    x="CuisineType",

    y="TotalRevenue",

    color="CuisineType",

    text_auto=".2s"

)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------
# Profit by Cuisine
# ----------------------------------------------------

cuisine_profit = (

    filtered_df

    .groupby("CuisineType")["TotalNetProfit"]

    .mean()

    .reset_index()

)

fig = px.bar(

    cuisine_profit,

    x="CuisineType",

    y="TotalNetProfit",

    color="CuisineType",

    title="Average Profit by Cuisine"

)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ----------------------------------------------------
# Profit by Segment
# ----------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    segment_profit = (

        filtered_df

        .groupby("Segment")["TotalNetProfit"]

        .mean()

        .reset_index()

    )

    fig = px.bar(

        segment_profit,

        x="Segment",

        y="TotalNetProfit",

        color="Segment",

        title="Average Profit by Segment"

    )

    st.plotly_chart(fig, use_container_width=True)

with col2:

    region_profit = (

        filtered_df

        .groupby("Subregion")["TotalNetProfit"]

        .mean()

        .reset_index()

    )

    fig = px.bar(

        region_profit,

        x="Subregion",

        y="TotalNetProfit",

        color="Subregion",

        title="Average Profit by Subregion"

    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ----------------------------------------------------
# Top Restaurants
# ----------------------------------------------------

st.subheader("🏆 Top 10 Most Profitable Restaurants")

top10 = (

    filtered_df

    .sort_values(

        by="TotalNetProfit",

        ascending=False

    )[["RestaurantName","CuisineType","TotalNetProfit"]]

    .head(10)

)

st.dataframe(top10, use_container_width=True)




# ----------------------------------------------------
# Correlation Analysis
# ----------------------------------------------------

st.subheader("🔥 Correlation Heatmap")

numeric_df = filtered_df.select_dtypes(include=["number"])

corr = numeric_df.corr(numeric_only=True)

fig = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    aspect="auto",
    title="Correlation Matrix"
)

fig.update_layout(height=800)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ----------------------------------------------------
# Relationship Analysis
# ----------------------------------------------------

st.subheader("Relationship Analysis")

col1, col2 = st.columns(2)

with col1:

    fig = px.scatter(
        filtered_df,
        x="MonthlyOrders",
        y="TotalNetProfit",
        color="CuisineType",
        size="AOV",
        hover_name="RestaurantName",
        title="Monthly Orders vs Net Profit"
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:

    fig = px.scatter(
        filtered_df,
        x="TotalRevenue",
        y="TotalNetProfit",
        color="Segment",
        size="MonthlyOrders",
        hover_name="RestaurantName",
        title="Revenue vs Net Profit"
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ----------------------------------------------------
# Delivery Radius Impact
# ----------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    fig = px.scatter(
        filtered_df,
        x="DeliveryRadiusKM",
        y="TotalNetProfit",
        color="CuisineType",
        trendline="ols",
        hover_name="RestaurantName",
        title="Delivery Radius vs Profit"
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:

    fig = px.scatter(
        filtered_df,
        x="CommissionRate",
        y="TotalNetProfit",
        color="Segment",
        trendline="ols",
        hover_name="RestaurantName",
        title="Commission Rate vs Profit"
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ----------------------------------------------------
# Growth Analysis
# ----------------------------------------------------

st.subheader("Growth Factor Analysis")

fig = px.scatter(
    filtered_df,
    x="GrowthFactor",
    y="TotalNetProfit",
    color="CuisineType",
    size="MonthlyOrders",
    hover_name="RestaurantName",
    title="Growth Factor vs Net Profit"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ----------------------------------------------------
# Bubble Chart
# ----------------------------------------------------

st.subheader("Restaurant Performance Bubble Chart")

fig = px.scatter(
    filtered_df,
    x="TotalRevenue",
    y="TotalNetProfit",
    size="MonthlyOrders",
    color="CuisineType",
    hover_name="RestaurantName",
    size_max=60,
    title="Revenue vs Profit vs Orders"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ----------------------------------------------------
# Profit Margin Analysis
# ----------------------------------------------------

if "ProfitMargin" in filtered_df.columns:

    st.subheader("Profit Margin by Cuisine")

    margin = (
        filtered_df
        .groupby("CuisineType")["ProfitMargin"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        margin,
        x="CuisineType",
        y="ProfitMargin",
        color="CuisineType",
        text_auto=".2f"
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ----------------------------------------------------
# Revenue Composition
# ----------------------------------------------------

st.subheader("Revenue Composition")

revenue_df = pd.DataFrame({
    "Channel": [
        "InStore",
        "Uber Eats",
        "DoorDash",
        "Self Delivery"
    ],
    "Revenue": [
        filtered_df["InStoreRevenue"].sum(),
        filtered_df["UberEatsRevenue"].sum(),
        filtered_df["DoorDashRevenue"].sum(),
        filtered_df["SelfDeliveryRevenue"].sum()
    ]
})

fig = px.pie(
    revenue_df,
    names="Channel",
    values="Revenue",
    hole=0.45,
    title="Revenue Share by Channel"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ----------------------------------------------------
# Profit Composition
# ----------------------------------------------------

profit_df = pd.DataFrame({
    "Channel": [
        "InStore",
        "Uber Eats",
        "DoorDash",
        "Self Delivery"
    ],
    "Profit": [
        filtered_df["InStoreNetProfit"].sum(),
        filtered_df["UberEatsNetProfit"].sum(),
        filtered_df["DoorDashNetProfit"].sum(),
        filtered_df["SelfDeliveryNetProfit"].sum()
    ]
})

fig = px.funnel(
    profit_df,
    x="Profit",
    y="Channel",
    title="Profit Contribution Funnel"
)

st.plotly_chart(fig, use_container_width=True)



# ----------------------------------------------------
# Outlier Analysis
# ----------------------------------------------------

st.subheader("Outlier Detection")

numeric_columns = [
    "MonthlyOrders",
    "AOV",
    "TotalRevenue",
    "TotalNetProfit",
    "CommissionRate",
    "DeliveryRadiusKM"
]

selected_column = st.selectbox(
    "Select Numeric Column",
    numeric_columns
)

fig = px.box(
    filtered_df,
    y=selected_column,
    color_discrete_sequence=["#636EFA"],
    title=f"Outlier Detection - {selected_column}"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ----------------------------------------------------
# Interactive Data Explorer
# ----------------------------------------------------

st.subheader("Data Explorer")

search = st.text_input(
    "🔍 Search Restaurant"
)

if search:

    data = filtered_df[
        filtered_df["RestaurantName"]
        .str.contains(search, case=False, na=False)
    ]

else:

    data = filtered_df

st.dataframe(
    data,
    use_container_width=True,
    height=450
)

st.download_button(
    label="⬇ Download Filtered Dataset",
    data=data.to_csv(index=False),
    file_name="restaurant_filtered_data.csv",
    mime="text/csv"
)

st.divider()

# ----------------------------------------------------
# KPI Summary
# ----------------------------------------------------

st.subheader("Business Summary")

c1, c2, c3 = st.columns(3)

with c1:

    best_restaurant = filtered_df.loc[
        filtered_df["TotalNetProfit"].idxmax(),
        "RestaurantName"
    ]

    st.success(f"Highest Profit\n\n**{best_restaurant}**")

with c2:

    best_cuisine = (
        filtered_df
        .groupby("CuisineType")["TotalNetProfit"]
        .mean()
        .idxmax()
    )

    st.info(f"Best Cuisine\n\n**{best_cuisine}**")

with c3:

    best_segment = (
        filtered_df
        .groupby("Segment")["TotalNetProfit"]
        .mean()
        .idxmax()
    )

    st.warning(f"Best Segment\n\n**{best_segment}**")

st.divider()

# ----------------------------------------------------
# Key Metrics
# ----------------------------------------------------

st.subheader("Key Business Metrics")

metrics = pd.DataFrame({

    "Metric": [

        "Average Revenue",
        "Average Profit",
        "Average Orders",
        "Average AOV",
        "Average Commission Rate",
        "Average Delivery Radius"

    ],

    "Value": [

        round(filtered_df["TotalRevenue"].mean(),2),
        round(filtered_df["TotalNetProfit"].mean(),2),
        round(filtered_df["MonthlyOrders"].mean(),2),
        round(filtered_df["AOV"].mean(),2),
        round(filtered_df["CommissionRate"].mean(),2),
        round(filtered_df["DeliveryRadiusKM"].mean(),2)

    ]

})

st.dataframe(
    metrics,
    use_container_width=True,
    hide_index=True
)

st.divider()

# ----------------------------------------------------
# Executive Insights
# ----------------------------------------------------

st.subheader("Executive Insights")

highest_profit = (
    filtered_df["TotalNetProfit"].max()
)

lowest_profit = (
    filtered_df["TotalNetProfit"].min()
)

avg_profit = (
    filtered_df["TotalNetProfit"].mean()
)

st.markdown(f"""
###Business Highlights

- **Highest Net Profit:** ${highest_profit:,.2f}

- **Lowest Net Profit:** ${lowest_profit:,.2f}

- **Average Net Profit:** ${avg_profit:,.2f}

- Analyze which cuisine generates the highest revenue.

- Compare delivery radius with profitability.

- Monitor commission rates to maximize earnings.

- Increase high-performing delivery channels.

- Use the Prediction and Optimization pages for decision making.
""")

st.divider()

# ----------------------------------------------------
# Footer
# ----------------------------------------------------

st.caption(
    "Restaurant Profit Optimization System | Exploratory Data Analysis Dashboard"
)

st.divider()
st.caption(
    "Restaurant Profit Optimization System | Final Project | 2026"
)