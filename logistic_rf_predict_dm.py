import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.title("糖尿病風險預測器")
st.markdown("請輸入您的基本資料")

input_data = {
    'Pregnancies': st.number_input("懷孕次數", min_value=0, max_value=17, value=0, key='Pregnancies'),
    'Glucose': st.number_input("葡萄糖濃度", min_value=44, max_value=199, value=120, key='Glucose'),
    'BloodPressure': st.number_input("血壓", min_value=24, max_value=122, value=70, key='BloodPressure'),
    'SkinThickness': st.number_input("皮膚厚度", min_value=7, max_value=99, value=20, key='SkinThickness'),
    'Insulin': st.number_input("胰島素", min_value=14, max_value=846, value=79, key='Insulin'),
    'BMI': st.number_input("BMI", min_value=18.2, max_value=67.1, value=25.6, key='BMI'),
    'DiabetesPedigreeFunction': st.number_input("糖尿病遺傳指數", min_value=0.07, max_value=2.42, value=0.5, key='DiabetesPedigreeFunction'),
    'Age': st.number_input("年齡", min_value=21, max_value=81, value=33, key='Age')
}

if st.button("預測糖尿病風險", key="predict_button"):
    interaction_model = joblib.load("interaction_model.joblib")
    scaler = joblib.load("scaler.joblib")
    top_features = joblib.load("top_features.joblib")

    input_df = pd.DataFrame([input_data])
    for feature in top_features:
        f1, f2 = feature.split("*")
        input_df[feature] = input_df[f1] * input_df[f2]

    input_selected = input_df[top_features]
    input_scaler = scaler.transform(input_selected)

    pred = interaction_model.predict(input_scaler)[0]
    prob = interaction_model.predict_proba(input_scaler)[0][1]

    st.subheader("預測結果")
    st.write(f"您有{'**高**' if pred == 1 else '**低**'}的糖尿病風險")
    st.write(f"預測機率為: {prob:.2%}")

average_values = {
    'Pregnancies': 3.8,
    'Glucose': 121.67,
    'BloodPressure': 72.39,
    'SkinThickness': 29.09,
    'Insulin': 141.75,
    'BMI': 32.43,
    'DiabetesPedigreeFunction': 0.47,
    'Age': 33.24
}

st.subheader("輸入資料 VS 平均值")
for key, user_val in input_data.items():
    avg = average_values[key]
    delta = user_val - avg
    if delta > 0:
        label = "高於平均"
    elif delta < 0:
        label = "低於平均"
    else:
        label = "與平均相同"

    st.write(f"**{key}**: {user_val} ({label}, 平均為 {avg})")

    fig, ax = plt.subplots(figsize=(5, 0.8))
    ax.barh(["使用者"], [user_val], color="orange")
    ax.axvline(avg, color='blue', linestyle='--', label='average')
    ax.set_xlim(0, max(user_val, avg) * 1.2)
    ax.set_xlabel(key)
    ax.legend()
    st.pyplot(fig)