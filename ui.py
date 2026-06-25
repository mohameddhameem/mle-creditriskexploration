import streamlit as st
import pandas as pd
import requests
import numpy as np
import json

st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="💳",
    layout="wide"
)

API_URL = "http://localhost:8000/predict"

st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: 800;
    color: #0B1F3A;
}
.sub-title {
    font-size: 18px;
    color: #5A6473;
}
.card {
    padding: 22px;
    border-radius: 16px;
    background-color: #F8FAFC;
    border: 1px solid #E5E7EB;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.risk-high {
    color: #B91C1C;
    font-weight: 700;
}
.risk-low {
    color: #047857;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">💳 Credit Risk Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Upload customer data to generate default-risk probabilities using the deployed ML model.</div>',
    unsafe_allow_html=True
)

st.divider()

with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox("Model", ["xgboost", "logistic_regression"])
    st.info("Upload a CSV file with customer application features.")
    st.caption("API endpoint: `/predict`")

uploaded_file = st.file_uploader(
    "📤 Upload customer CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Rows", df.shape[0])
    col_b.metric("Columns", df.shape[1])
    col_c.metric("Missing Values", int(df.isna().sum().sum()))

    st.dataframe(df.head(10), use_container_width=True)

    st.divider()

    if st.button("🚀 Run Prediction", use_container_width=True):
        try:
            with st.spinner("Running prediction..."):
                df_clean = df.replace([np.inf, -np.inf], np.nan)
                df_clean = df_clean.astype(object).where(pd.notnull(df_clean), None)

                payload = {
                    "inputs": df_clean.to_dict(orient="records"),
                    "model_name": model_name
                }

                response = requests.post(
                    API_URL,
                    data=json.dumps(payload, allow_nan=False),
                    headers={"Content-Type": "application/json"},
                    timeout=60
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

                st.success("Prediction completed successfully")

                st.subheader("📊 Prediction Summary")

                total_customers = len(output_df)
                high_risk = int(output_df["prediction"].sum())
                low_risk = total_customers - high_risk
                avg_risk = output_df["risk_probability"].mean()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Customers", total_customers)
                c2.metric("High Risk", high_risk)
                c3.metric("Low Risk", low_risk)
                c4.metric("Average Risk", round(avg_risk, 3))

                st.caption(f"Model used: **{result.get('model_used', model_name)}** | Timestamp: **{result.get('timestamp', 'N/A')}**")

                st.divider()

                left, right = st.columns([2, 1])

                with left:
                    st.subheader("📋 Prediction Results")

                    display_cols = ["risk_probability", "prediction", "risk_label"]

                    if "SK_ID_CURR" in output_df.columns:
                        display_cols.insert(0, "SK_ID_CURR")

                    styled_df = output_df[display_cols].copy()

                    st.dataframe(
                        styled_df,
                        use_container_width=True,
                        hide_index=True
                    )

                with right:
                    st.subheader("⚠️ Risk Split")
                    risk_counts = output_df["risk_label"].value_counts()
                    st.bar_chart(risk_counts)

                    st.subheader("📈 Risk Distribution")
                    st.line_chart(output_df["risk_probability"])

                st.divider()

                high_risk_df = output_df[output_df["prediction"] == 1]

                with st.expander("View High Risk Customers"):
                    if len(high_risk_df) > 0:
                        st.dataframe(high_risk_df, use_container_width=True, hide_index=True)
                    else:
                        st.success("No high-risk customers found.")

                st.download_button(
                    "⬇️ Download Predictions CSV",
                    output_df.to_csv(index=False),
                    file_name="credit_risk_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            else:
                st.error(f"API Error: {response.status_code}")
                st.code(response.text)

        except Exception as e:
            st.error("Something went wrong while generating predictions.")
            st.exception(e)

else:
    st.info("Upload a CSV file to begin.")