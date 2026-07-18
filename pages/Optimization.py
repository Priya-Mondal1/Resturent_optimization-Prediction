import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

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

st.title("Restaurant Profit Optimization")
st.write(
    "Optimize delivery channels, commission rates, and operational parameters "
    "to maximize restaurant profit."
)

st.divider()

# ----------------------------------------------------
# Load Model
# ----------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/best_profit_model.pkl")

model = load_model()

# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/restaurant_profit_dataset_20000_rows.csv")

df = load_data()

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

st.sidebar.header("Optimization Settings")

selected_restaurant = st.sidebar.selectbox(
    "Restaurant",
    sorted(df["RestaurantName"].unique())
)

restaurant = df[df["RestaurantName"] == selected_restaurant].iloc[0]

# ----------------------------------------------------
# Input Parameters
# ----------------------------------------------------

st.subheader("Business Parameters")

col1, col2 = st.columns(2)

with col1:

    monthly_orders = st.number_input(
        "Monthly Orders",
        min_value=100,
        max_value=10000,
        value=int(restaurant["MonthlyOrders"])
    )

    aov = st.number_input(
        "Average Order Value",
        min_value=5.0,
        max_value=500.0,
        value=float(restaurant["AOV"])
    )

    growth = st.slider(
        "Growth Factor",
        0.50,
        2.00,
        float(restaurant["GrowthFactor"]),
        0.01
    )

with col2:

    commission = st.slider(
        "Commission Rate",
        0.05,
        0.40,
        float(restaurant["CommissionRate"]),
        0.01
    )

    radius = st.slider(
        "Delivery Radius (KM)",
        1.0,
        20.0,
        float(restaurant["DeliveryRadiusKM"]),
        0.5
    )

    delivery_cost = st.slider(
        "Delivery Cost / Order",
        1.0,
        20.0,
        float(restaurant["DeliveryCostOrder"]),
        0.5
    )

st.divider()

# ----------------------------------------------------
# Channel Shares
# ----------------------------------------------------

st.subheader("Order Distribution")

c1, c2 = st.columns(2)

with c1:

    instore = st.slider(
        "InStore %",
        0,
        100,
        int(restaurant["InStoreShare"] * 100)
    )

    uber = st.slider(
        "Uber Eats %",
        0,
        100,
        int(restaurant["UE_share"] * 100)
    )

with c2:

    doordash = st.slider(
        "DoorDash %",
        0,
        100,
        int(restaurant["DD_share"] * 100)
    )

    self_delivery = st.slider(
        "Self Delivery %",
        0,
        100,
        int(restaurant["SD_share"] * 100)
    )

total_share = instore + uber + doordash + self_delivery

st.info(
    f"Current Share Total : {total_share}%"
)

st.success("Channel distribution is valid.")




# ----------------------------------------------------
# Create Input Data
# ----------------------------------------------------

input_data = restaurant.copy()

input_data["MonthlyOrders"] = monthly_orders
input_data["AOV"] = aov
input_data["GrowthFactor"] = growth
input_data["CommissionRate"] = commission
input_data["DeliveryRadiusKM"] = radius
input_data["DeliveryCostOrder"] = delivery_cost

total = (
    instore +
    uber +
    doordash +
    self_delivery
)

if total == 0:
    total = 1

input_data["InStoreShare"] = instore / total
input_data["UE_share"] = uber / total
input_data["DD_share"] = doordash / total
input_data["SD_share"] = self_delivery / total
# ----------------------------------------------------
# Feature Engineering
# ----------------------------------------------------

input_data["InStoreOrdersCount"] = monthly_orders * input_data["InStoreShare"]
input_data["UberEatsOrdersCount"] = monthly_orders * input_data["UE_share"]
input_data["DoorDashOrdersCount"] = monthly_orders * input_data["DD_share"]
input_data["SelfDeliveryOrdersCount"] = monthly_orders * input_data["SD_share"]

input_data["InStoreRevenue"] = (
    input_data["InStoreOrdersCount"] * aov
)

input_data["UberEatsRevenue"] = (
    input_data["UberEatsOrdersCount"] * aov
)

input_data["DoorDashRevenue"] = (
    input_data["DoorDashOrdersCount"] * aov
)

input_data["SelfDeliveryRevenue"] = (
    input_data["SelfDeliveryOrdersCount"] * aov
)

# ----------------------------------------------------
# Prepare Model Input
# ----------------------------------------------------


# ----------------------------------------------------
# Prepare Model Input
# ----------------------------------------------------

X = pd.DataFrame([input_data])

# One Hot Encode categorical columns
X = pd.get_dummies(X)

# Add missing columns expected by the model
for col in model.feature_names_in_:
    if col not in X.columns:
        X[col] = 0

# Keep only model columns in correct order
X = X.reindex(
    columns=model.feature_names_in_,
    fill_value=0
)

# Prediction

# ----------------------------------------------------
# Current Prediction
# ----------------------------------------------------

current_profit = model.predict(X)[0]

st.subheader("Current Predicted Profit")

st.metric(
    "Predicted Monthly Profit",
    f"${current_profit:,.2f}"
)

st.divider()

# ----------------------------------------------------
# Optimization Engine
# ----------------------------------------------------

st.subheader("Running Optimization")

progress = st.progress(0)

best_profit = current_profit
best_values = None

share_options = [0.20, 0.30, 0.40, 0.50]

commission_options = [0.10, 0.15, 0.20, 0.25]

radius_options = [3,5,7,9,11]

count = 0
total_iterations = (
    len(share_options)
    * len(commission_options)
    * len(radius_options)
)

for s in share_options:

    for c in commission_options:

        for r in radius_options:

            scenario = X.copy()

            scenario["CommissionRate"] = c
            scenario["DeliveryRadiusKM"] = r

            scenario["UE_share"] = s
            scenario["DD_share"] = s

            scenario["InStoreShare"] = (1 - 2*s)/2
            scenario["SD_share"] = (1 - 2*s)/2

            pred = model.predict(scenario)[0]

            if pred > best_profit:

                best_profit = pred

                best_values = {

                    "Commission": c,

                    "Radius": r,

                    "UE Share": scenario["UE_share"].values[0],

                    "DD Share": scenario["DD_share"].values[0],

                    "InStore Share": scenario["InStoreShare"].values[0],

                    "Self Delivery Share": scenario["SD_share"].values[0]

                }

            count += 1

            progress.progress(count / total_iterations)

progress.empty()

# ----------------------------------------------------
# Results
# ----------------------------------------------------

st.success("Optimization Completed")

improvement = best_profit - current_profit

improvement_pct = (
    improvement / current_profit * 100
    if current_profit != 0
    else 0
)

col1,col2,col3 = st.columns(3)

with col1:

    st.metric(
        "Current Profit",
        f"${current_profit:,.2f}"
    )

with col2:

    st.metric(
        "Optimized Profit",
        f"${best_profit:,.2f}",
        delta=f"${improvement:,.2f}"
    )

with col3:

    st.metric(
        "Improvement %",
        f"{improvement_pct:.2f}%"
    )
    
    
    
    
    # ----------------------------------------------------
# Optimization Summary
# ----------------------------------------------------

st.divider()
st.subheader("Optimized Business Strategy")

if best_values is not None:

    recommendations = pd.DataFrame({

        "Parameter": [

            "Commission Rate",
            "Delivery Radius (KM)",
            "InStore Share",
            "Uber Eats Share",
            "DoorDash Share",
            "Self Delivery Share"

        ],

        "Recommended Value": [

            f"{best_values['Commission']*100:.1f}%",
            f"{best_values['Radius']} KM",
            f"{best_values['InStore Share']*100:.1f}%",
            f"{best_values['UE Share']*100:.1f}%",
            f"{best_values['DD Share']*100:.1f}%",
            f"{best_values['Self Delivery Share']*100:.1f}%"

        ]

    })

    st.dataframe(
        recommendations,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "Current configuration is already close to the best scenario."
    )

# ----------------------------------------------------
# Before vs After Profit
# ----------------------------------------------------

st.divider()

st.subheader("Profit Comparison")

comparison = pd.DataFrame({

    "Scenario": [

        "Current",
        "Optimized"

    ],

    "Profit": [

        current_profit,
        best_profit

    ]

})

fig = px.bar(

    comparison,

    x="Scenario",

    y="Profit",

    color="Scenario",

    text_auto=".2f",

    title="Current vs Optimized Profit"

)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------
# Channel Distribution Comparison
# ----------------------------------------------------

st.divider()

st.subheader("Optimized Channel Distribution")

col1, col2 = st.columns(2)

with col1:

    current_channel = pd.DataFrame({

        "Channel":[

            "InStore",
            "Uber Eats",
            "DoorDash",
            "Self Delivery"

        ],

        "Share":[

            instore,
            uber,
            doordash,
            self_delivery

        ]

    })

    fig = px.pie(

        current_channel,

        names="Channel",

        values="Share",

        hole=0.45,

        title="Current Distribution"

    )

    st.plotly_chart(fig, use_container_width=True)

with col2:

    if best_values is not None:

        optimized_channel = pd.DataFrame({

            "Channel":[

                "InStore",
                "Uber Eats",
                "DoorDash",
                "Self Delivery"

            ],

            "Share":[

                best_values["InStore Share"]*100,
                best_values["UE Share"]*100,
                best_values["DD Share"]*100,
                best_values["Self Delivery Share"]*100

            ]

        })

        fig = px.pie(

            optimized_channel,

            names="Channel",

            values="Share",

            hole=0.45,

            title="Optimized Distribution"

        )

        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------
# Profit Improvement
# ----------------------------------------------------

st.divider()

st.subheader("Improvement Summary")

summary = pd.DataFrame({

    "Metric":[

        "Current Profit",
        "Optimized Profit",
        "Improvement"

    ],

    "Amount":[

        current_profit,
        best_profit,
        improvement

    ]

})

fig = px.bar(

    summary,

    x="Metric",

    y="Amount",

    color="Metric",

    text_auto=".2f"

)

st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------
# Gauge Chart
# ----------------------------------------------------

st.divider()

st.subheader("Optimization Score")

score = min(100, max(0, improvement_pct))

fig = go.Figure(go.Indicator(

    mode="gauge+number",

    value=score,

    title={"text":"Optimization Gain (%)"},

    gauge={

        "axis":{"range":[0,100]},

        "bar":{"color":"green"},

        "steps":[

            {"range":[0,30],"color":"lightgray"},
            {"range":[30,60],"color":"gold"},
            {"range":[60,100],"color":"lightgreen"}

        ]

    }

))

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------
# KPI Cards
# ----------------------------------------------------

st.divider()

st.subheader("Key Performance Indicators")

k1, k2, k3 = st.columns(3)

with k1:

    st.metric(
        "Profit Increase",
        f"${improvement:,.2f}"
    )

with k2:

    st.metric(
        "Growth %",
        f"{improvement_pct:.2f}%"
    )

with k3:

    if best_values:

        st.metric(
            "Recommended Commission",
            f"{best_values['Commission']*100:.1f}%"
        )
        
        
        
        # ----------------------------------------------------
# AI Business Recommendations
# ----------------------------------------------------

st.divider()

st.subheader("AI Business Recommendations")

recommendations = []

if improvement_pct > 15:
    recommendations.append(
        " Excellent opportunity for profit improvement. Consider implementing the optimized strategy."
    )

elif improvement_pct > 5:
    recommendations.append(
        "Moderate profit improvement is possible with the suggested configuration."
    )

else:
    recommendations.append(
        "Current strategy is already performing well. Only minor optimizations are recommended."
    )

if commission > 0.25:
    recommendations.append(
        "High commission rate detected. Negotiating lower marketplace commissions could significantly improve profit."
    )

if radius > 10:
    recommendations.append(
        "Large delivery radius increases delivery costs. Consider reducing the service area."
    )

if self_delivery < 20:
    recommendations.append(
        "Increasing self-delivery orders may reduce third-party commission costs."
    )

if growth < 1:
    recommendations.append(
        "Growth factor is below average. Marketing campaigns and promotions could help increase demand."
    )

for rec in recommendations:
    st.info(rec)

# ----------------------------------------------------
# Optimization Report
# ----------------------------------------------------

st.divider()

st.subheader(" Optimization Report")

report = pd.DataFrame({

    "Metric":[
        "Current Profit",
        "Optimized Profit",
        "Profit Increase",
        "Improvement %",
        "Recommended Commission",
        "Recommended Radius"
    ],

    "Value":[
        round(current_profit,2),
        round(best_profit,2),
        round(improvement,2),
        round(improvement_pct,2),

        best_values["Commission"]
        if best_values
        else commission,

        best_values["Radius"]
        if best_values
        else radius

    ]

})

st.dataframe(
    report,
    use_container_width=True,
    hide_index=True
)

# ----------------------------------------------------
# Download Report
# ----------------------------------------------------

csv = report.to_csv(index=False)

st.download_button(

    label="Download Optimization Report",

    data=csv,

    file_name="optimization_report.csv",

    mime="text/csv"

)

# ----------------------------------------------------
# Executive Summary
# ----------------------------------------------------

st.divider()

st.subheader(" Executive Summary")

col1, col2 = st.columns(2)

with col1:

    st.success(f"""
### Optimization Complete

Current Profit

**${current_profit:,.2f}**

Optimized Profit

**${best_profit:,.2f}**
""")

with col2:

    st.info(f"""
### Expected Improvement

Increase

**${improvement:,.2f}**

Growth

**{improvement_pct:.2f}%**
""")

# ----------------------------------------------------
# Best Configuration
# ----------------------------------------------------

if best_values:

    st.divider()

    st.subheader("Best Recommended Configuration")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Commission",
            f"{best_values['Commission']*100:.1f}%"
        )

        st.metric(
            "Delivery Radius",
            f"{best_values['Radius']} KM"
        )

    with c2:

        st.metric(
            "Uber Eats",
            f"{best_values['UE Share']*100:.1f}%"
        )

        st.metric(
            "DoorDash",
            f"{best_values['DD Share']*100:.1f}%"
        )

    with c3:

        st.metric(
            "InStore",
            f"{best_values['InStore Share']*100:.1f}%"
        )

        st.metric(
            "Self Delivery",
            f"{best_values['Self Delivery Share']*100:.1f}%"
        )

# ----------------------------------------------------
# Footer
# ----------------------------------------------------

st.divider()

st.caption(
    "Restaurant Profit Optimization System | AI Optimization Dashboard"
)

st.divider()
st.caption(
    "Restaurant Profit Optimization System | Final Project | 2026"
)