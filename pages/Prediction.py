import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from utils.header import show_header

st.set_page_config(
    page_title="Restaurant Profit Optimization",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ============================================================
# PAGE CONFIG
# ============================================================


show_header()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title{
    font-size:38px;
    font-weight:bold;
    color:#0E7490;
}

.subtitle{
    font-size:18px;
    color:#6B7280;
    margin-bottom:15px;
}

.block{
    background:#ffffff;
    padding:18px;
    border-radius:14px;
    box-shadow:0 2px 8px rgba(0,0,0,.08);
}

</style>
""", unsafe_allow_html=True)


st.markdown(
    "<div class='main-title'>Restaurant Profit Prediction</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Predict expected monthly restaurant profit using the trained Machine Learning model.</div>",
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("models/best_profit_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv(
        "data/restaurant_profit_dataset_20000_rows.csv"
    )


model = load_model()
df = load_data()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Prediction Settings")

restaurant_name = st.sidebar.selectbox(
    "Restaurant",
    sorted(df["RestaurantName"].unique())
)

restaurant = df[
    df["RestaurantName"] == restaurant_name
].iloc[0]

st.sidebar.success("Restaurant Loaded")


# ============================================================
# BUSINESS INFORMATION
# ============================================================

st.subheader("Business Information")

col1, col2 = st.columns(2)

with col1:

    cuisine = st.selectbox(
        "Cuisine Type",
        sorted(df["CuisineType"].unique()),
        index=sorted(
            df["CuisineType"].unique()
        ).index(
            restaurant["CuisineType"]
        )
    )

    segment = st.selectbox(
        "Business Segment",
        sorted(df["Segment"].unique()),
        index=sorted(
            df["Segment"].unique()
        ).index(
            restaurant["Segment"]
        )
    )

    subregion = st.selectbox(
        "Subregion",
        sorted(df["Subregion"].unique()),
        index=sorted(
            df["Subregion"].unique()
        ).index(
            restaurant["Subregion"]
        )
    )

with col2:

    monthly_orders = st.number_input(
        "Monthly Orders",
        value=int(restaurant["MonthlyOrders"]),
        min_value=100
    )

    aov = st.number_input(
        "Average Order Value",
        value=float(restaurant["AOV"]),
        min_value=1.0
    )

    growth = st.slider(
        "Growth Factor",
        0.5,
        2.0,
        float(restaurant["GrowthFactor"]),
        0.01
    )


st.divider()


# ============================================================
# OPERATIONAL PARAMETERS
# ============================================================

st.subheader("Operational Parameters")

col1, col2 = st.columns(2)

with col1:

    commission = st.slider(
        "Commission Rate",
        0.05,
        0.40,
        float(restaurant["CommissionRate"]),
        0.01
    )

    delivery_radius = st.slider(
        "Delivery Radius (KM)",
        1.0,
        20.0,
        float(restaurant["DeliveryRadiusKM"]),
        0.5
    )

with col2:

    delivery_cost = st.slider(
        "Delivery Cost",
        1.0,
        20.0,
        float(restaurant["DeliveryCostOrder"]),
        0.5
    )

    cogs = st.slider(
        "COGS Rate",
        0.10,
        0.90,
        float(restaurant["COGSRate"]),
        0.01
    )


st.divider()


# ============================================================
# ORDER DISTRIBUTION
# ============================================================

st.subheader("Order Distribution")

c1, c2, c3, c4 = st.columns(4)

with c1:
    instore_share = st.number_input(
        "InStore %",
        0,
        100,
        int(restaurant["InStoreShare"] * 100)
    )

with c2:
    uber_share = st.number_input(
        "Uber Eats %",
        0,
        100,
        int(restaurant["UE_share"] * 100)
    )

with c3:
    doordash_share = st.number_input(
        "DoorDash %",
        0,
        100,
        int(restaurant["DD_share"] * 100)
    )

with c4:
    self_delivery_share = st.number_input(
        "Self Delivery %",
        0,
        100,
        int(restaurant["SD_share"] * 100)
    )


total_share = (
    instore_share
    + uber_share
    + doordash_share
    + self_delivery_share
)

if total_share != 100:
    st.warning(
        f"Total share is {total_share}%. "
        "The model performs best when it equals 100%."
    )


predict_button = st.button(
    "Predict Monthly Profit",
    use_container_width=True,
    type="primary"
)


# ============================================================
# PREDICTION LOGIC
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Validate Order Distribution
    # --------------------------------------------------------

    
    # --------------------------------------------------------
    # Convert Percentage to Decimal
    # --------------------------------------------------------

    instore = instore_share
    uber = uber_share
    doordash = doordash_share
    self_delivery = self_delivery_share

    total = instore + uber + doordash + self_delivery

    if total > 0:
        instore /= total
        uber /= total
        doordash /= total
        self_delivery /= total
    # --------------------------------------------------------
    # Estimated Revenue
    # --------------------------------------------------------

    estimated_revenue = monthly_orders * aov * growth

    # --------------------------------------------------------
    # Prepare Input
    # --------------------------------------------------------

    input_df = pd.DataFrame({

        "CuisineType": [cuisine],
        "Segment": [segment],
        "Subregion": [subregion],

        "MonthlyOrders": [monthly_orders],
        "AOV": [aov],

        "GrowthFactor": [growth],

        "CommissionRate": [commission],
        "DeliveryRadiusKM": [delivery_radius],
        "DeliveryCostOrder": [delivery_cost],
        "COGSRate": [cogs],

        "InStoreShare": [instore],
        "UE_share": [uber],
        "DD_share": [doordash],
        "SD_share": [self_delivery]

    })

    # --------------------------------------------------------
    # One Hot Encoding
    # --------------------------------------------------------

    input_encoded = pd.get_dummies(input_df)

    input_encoded = input_encoded.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predicted_profit = float(
        model.predict(input_encoded)[0]
    )

    # --------------------------------------------------------
    # Business Metrics
    # --------------------------------------------------------

    profit_margin = (
        predicted_profit /
        estimated_revenue
    ) * 100

    total_delivery_share = (
        uber_share +
        doordash_share +
        self_delivery_share
    )

    # --------------------------------------------------------
    # Business Health
    # --------------------------------------------------------

    if predicted_profit >= 60000:
        health = "Excellent"
        health_color = "green"

    elif predicted_profit >= 40000:
        health = "Good"
        health_color = "blue"

    elif predicted_profit >= 20000:
        health = "Average"
        health_color = "orange"

    else:
        health = "Needs Improvement"
        health_color = "red"

    # --------------------------------------------------------
    # Profit Category
    # --------------------------------------------------------

    if predicted_profit >= 60000:
        profit_level = "High Profit"

    elif predicted_profit >= 35000:
        profit_level = "Medium Profit"

    else:
        profit_level = "Low Profit"

    # --------------------------------------------------------
    # Generate Recommendations
    # --------------------------------------------------------

    recommendations = []

    if commission > 0.25:
        recommendations.append(
            "Reduce dependency on third-party delivery platforms."
        )

    if delivery_cost > 8:
        recommendations.append(
            "Optimize delivery routes to reduce operational costs."
        )

    if cogs > 0.60:
        recommendations.append(
            "Review supplier contracts and menu pricing."
        )

    if instore_share < 30:
        recommendations.append(
            "Increase dine-in promotions to improve direct sales."
        )

    if total_delivery_share > 70:
        recommendations.append(
            "Heavy dependence on delivery channels may reduce profitability."
        )

    if growth < 0.90:
        recommendations.append(
            "Business growth is slow. Focus on marketing campaigns."
        )

    if profit_margin > 35:
        recommendations.append(
            "Strong profitability. Expansion opportunities can be explored."
        )

    if len(recommendations) == 0:
        recommendations.append(
            "Business performance is balanced. Continue monitoring KPIs."
        )

    # --------------------------------------------------------
    # Channel Summary
    # --------------------------------------------------------

    channel_df = pd.DataFrame({

        "Channel": [
            "InStore",
            "Uber Eats",
            "DoorDash",
            "Self Delivery"
        ],

        "Share": [
            instore_share,
            uber_share,
            doordash_share,
            self_delivery_share
        ]

    })

    highest_channel = channel_df.loc[
        channel_df["Share"].idxmax()
    ]

    lowest_channel = channel_df.loc[
        channel_df["Share"].idxmin()
    ]

    # --------------------------------------------------------
    # Revenue Breakdown
    # --------------------------------------------------------

    revenue_df = pd.DataFrame({

        "Category": [
            "Predicted Profit",
            "Operating Cost"
        ],

        "Amount": [
            predicted_profit,
            max(
                estimated_revenue - predicted_profit,
                0
            )
        ]

    })

    # --------------------------------------------------------
    # Prediction Report
    # --------------------------------------------------------

    report_df = pd.DataFrame({

        "Metric": [

            "Restaurant",
            "Cuisine",
            "Segment",
            "Region",
            "Monthly Orders",
            "Average Order Value",
            "Estimated Revenue",
            "Predicted Profit",
            "Profit Margin",
            "Business Health",
            "Profit Category"

        ],

        "Value": [

            restaurant_name,
            cuisine,
            segment,
            subregion,
            monthly_orders,
            round(aov, 2),
            round(estimated_revenue, 2),
            round(predicted_profit, 2),
            round(profit_margin, 2),
            health,
            profit_level

        ]

    })
    
    
    
        # ============================================================
    # RESULTS
    # ============================================================

    st.divider()
    st.header("Prediction Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Predicted Profit",
            f"${predicted_profit:,.2f}"
        )

    with col2:
        st.metric(
            "Estimated Revenue",
            f"${estimated_revenue:,.2f}"
        )

    with col3:
        st.metric(
            "Profit Margin",
            f"{profit_margin:.2f}%"
        )

    with col4:
        st.metric(
            "Business Health",
            health
        )

    st.divider()

    # ============================================================
    # CHARTS
    # ============================================================

    left, right = st.columns(2)

    with left:

        fig = px.pie(
            channel_df,
            values="Share",
            names="Channel",
            title="Order Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        fig2 = px.bar(
            revenue_df,
            x="Category",
            y="Amount",
            text="Amount",
            title="Revenue Breakdown"
        )

        fig2.update_traces(
            texttemplate="$%{text:,.0f}",
            textposition="outside"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    st.divider()

    # ============================================================
    # CHANNEL INSIGHTS
    # ============================================================

    st.subheader("Business Insights")

    c1, c2 = st.columns(2)

    with c1:
        st.info(
            f"Highest Order Channel: "
            f"{highest_channel['Channel']} "
            f"({highest_channel['Share']}%)"
        )

    with c2:
        st.warning(
            f"Lowest Order Channel: "
            f"{lowest_channel['Channel']} "
            f"({lowest_channel['Share']}%)"
        )

    st.divider()

    # ============================================================
    # RECOMMENDATIONS
    # ============================================================

    st.subheader("Recommendations")

    for rec in recommendations:
        st.write(f"✅ {rec}")

    st.divider()

    # ============================================================
    # REPORT TABLE
    # ============================================================

    st.subheader("Prediction Report")

    st.dataframe(
        report_df,
        use_container_width=True
    )

    csv = report_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Prediction Report",
        csv,
        "prediction_report.csv",
        "text/csv",
        use_container_width=True
    )



st.divider()
st.caption(
    "Restaurant Profit Optimization System | Final Project | 2026"
)