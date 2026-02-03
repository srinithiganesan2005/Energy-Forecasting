import streamlit as st
import pandas as pd
import numpy as np
import joblib

from weather_api import get_temperature
from sms_notifier import send_sms

# ======================================
# LOAD TRAINED MODEL (HYBRID)
# ======================================
model = joblib.load("energy_model_hybrid.pkl")

st.set_page_config(page_title="Energy Forecaster", layout="centered")
st.title("⚡ AI-Powered Energy Consumption Forecaster")

# ======================================
# WEATHER (DISPLAY ONLY)
# ======================================
temperature = get_temperature()
st.info(f"🌡 Live Temperature: {temperature} °C")

# ======================================
# USER INPUTS (MATCH TRAINING FEATURES)
# ======================================
usage_hours = st.slider("Daily Usage Hours", 0.5, 24.0, 6.0, step=0.5)
power_watt = st.slider("Appliance Power Rating (Watt)", 40, 3000, 500)

# ======================================
# BUTTON: PREDICT + SEND SMS
# ======================================
if st.button("📩 Predict & Send SMS Alert"):

    # ----- MODEL PREDICTION -----
    input_df = pd.DataFrame(
        [[usage_hours, power_watt]],
        columns=["usage_hours", "power_watt"]
    )

    prediction = round(model.predict(input_df)[0], 2)

    # ----- USAGE LEVEL -----
    if prediction < 3:
        level = "Low"
    elif prediction < 8:
        level = "Moderate"
    else:
        level = "High"

    # ----- ENERGY SUGGESTIONS -----
    suggestions_map = {
        "Low": [
            "Maintain current energy-efficient usage",
            "Continue using star-rated appliances"
        ],
        "Moderate": [
            "Reduce usage during peak hours",
            "Turn off idle appliances"
        ],
        "High": [
            "Limit AC usage",
            "Unplug unused devices immediately"
        ]
    }

    suggestions = suggestions_map[level]

    # ----- UI OUTPUT -----
    st.subheader("🔮 Prediction Result")
    st.success(f"{prediction} kWh — {level} Usage")

    st.info("💡 Energy Saving Suggestions:")
    for tip in suggestions:
        st.write("•", tip)

    # ----- SEND SMS -----
    sms_status = send_sms(
        "+919790235865",   # RECEIVER NUMBER (must be verified)
        temperature,
        prediction,
        level,
        suggestions
    )

    if sms_status:
        st.success("📨 SMS alert sent successfully")
    else:
        st.warning("⚠️ SMS failed to send")

    # ----- STORE HISTORY FOR CHATBOT -----
    st.session_state["last_prediction"] = prediction
    st.session_state["level"] = level

# ======================================
# CHATBOT SECTION
# ======================================
st.divider()
st.subheader("🤖 Energy Assistant Chatbot")

# Initialize history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input(
    "Ask about prediction, dataset, trends, history, or energy tips"
)

# Simulated historical data (for demo & chatbot)
history_df = pd.DataFrame({
    "Energy (kWh)": np.random.normal(
        st.session_state.get("last_prediction", 5),
        1.5,
        30
    )
})

if query:
    q = query.lower()
    st.session_state.chat_history.append(("user", query))

    # -------- DATASET --------
    if "dataset" in q or "data" in q:
        response = (
            "The model is trained using real Firebase data combined with "
            "simulated data. Features used are usage hours and power rating."
        )

    # -------- PREDICTION --------
    elif "prediction" in q:
        if "last_prediction" in st.session_state:
            response = (
                f"Your latest predicted usage is "
                f"{st.session_state['last_prediction']} kWh "
                f"({st.session_state['level']})."
            )
        else:
            response = "Please click Predict first to generate a prediction."

    # -------- TRENDS --------
    elif "trend" in q or "graph" in q:
        response = "Here is the energy consumption trend."

    # -------- HISTORY --------
    elif "history" in q:
        response = "Here is a sample of historical energy consumption data."

    # -------- ENERGY TIPS --------
    elif "tip" in q or "save" in q:
        if "level" in st.session_state:
            response = "Here are energy-saving tips based on your usage level."
        else:
            response = "Generate a prediction first to get personalized tips."

    # -------- MODEL --------
    elif "model" in q:
        response = (
            "A Random Forest Regression model is used because it handles "
            "non-linear energy usage patterns effectively."
        )

    # -------- HELP --------
    elif "help" in q:
        response = (
            "You can ask me about:\n"
            "• Prediction\n"
            "• Dataset\n"
            "• Trends\n"
            "• History\n"
            "• Energy saving tips\n"
            "• Model details"
        )

    else:
        response = (
            "I didn’t understand that. Try asking about prediction, dataset, "
            "trends, history, or energy tips."
        )

    st.session_state.chat_history.append(("assistant", response))

# Display chat
for role, msg in st.session_state.chat_history:
    st.chat_message(role).write(msg)

# Extra outputs
if query:
    if "trend" in query.lower():
        st.line_chart(history_df)

    if "history" in query.lower():
        st.dataframe(history_df)
