import streamlit as st
import pandas as pd
import requests
import numpy as np
import json

st.set_page_config(page_title="Credit Risk Prediction Dashboard", layout="wide")

st.title("Credit Risk Prediction Dashboard")
st.write("Upload customer data and get default risk predictions.")

API_URL = "http://localhost:8000/predict"

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    if st.button("Predict"):
        try:
            df_clean = df.replace([np.inf, -np.inf], np.nan)
            df_clean = df_clean.astype(object).where(pd.notnull(df_clean), None)

            payload = {
                        "inputs": df_clean.to_dict(orient="records"),
                        "model_name": "xgboost"
}

            response = requests.post(
                API_URL,
                data=json.dumps(payload, allow_nan=False),
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                result = response.json()

                predictions = pd.DataFrame(result["predictions"])

                output_df = df.copy()
                output_df["risk_probability"] = predictions["probability"]
                output_df["prediction"] = predictions["prediction"]
                output_df["risk_label"] = output_df["prediction"].map({
                   0: "Low Risk",
                    1: "High Risk"
    })

                st.success("Prediction complete")

                st.subheader("Prediction Summary")
                st.write(f"Model used: {result['model_used']}")
                st.write(f"Model timestamp: {result['timestamp']}")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Customers", len(output_df))

                with col2:
                    st.metric("High Risk Customers", int(output_df["prediction"].sum()))

                with col3:
                    st.metric(
                        "Average Risk",
                        round(output_df["risk_probability"].mean(), 3)
        )

                st.subheader("Prediction Results")

                display_cols = ["risk_probability", "prediction", "risk_label"]

                if "SK_ID_CURR" in output_df.columns:
                    display_cols.insert(0, "SK_ID_CURR")

                st.dataframe(
                    output_df[display_cols],
                    use_container_width=True
    )

                st.subheader("Risk Probability Distribution")
                st.bar_chart(output_df["risk_probability"])

                st.download_button(
                    "Download Predictions CSV",
                    output_df.to_csv(index=False),
                    file_name="credit_risk_predictions.csv",
                    mime="text/csv"
    )

        except Exception as e:
            st.error("Something went wrong")
            st.exception(e)