import streamlit as st

st.title("Sleep Disorder Risk Prediction")
st.write("睡眠障礙風險預測系統")
st.write("請輸入您的生活與睡眠資訊，系統將預測睡眠障礙風險。")

st.subheader("基本資料")

age = st.number_input(
    "年齡",
    min_value=18,
    max_value=69,
    value=33,
    step=1
)

mental_health_condition = st.selectbox(
    "心理健康狀況",
    ["Healthy", "Anxiety", "Depression", "Both"]
)

shift_work_text = st.radio(
    "是否從事輪班工作？",
    ["否", "是"]
)

shift_work = 1 if shift_work_text == "是" else 0

sleep_duration_hrs = st.slider(
    "平均每晚睡眠時間（小時）",
    min_value=3.0,
    max_value=10.5,
    value=6.4,
    step=0.1
)

st.subheader("身體資料")

height_cm = st.number_input(
    "身高（cm）",
    min_value=120.0,
    max_value=220.0,
    value=160.0,
    step=0.5
)

weight_kg = st.number_input(
    "體重（kg）",
    min_value=30.0,
    max_value=200.0,
    value=60.0,
    step=0.5
)

height_m = height_cm / 100

bmi = weight_kg / (height_m ** 2)

st.write(f"BMI：{bmi:.1f}")

st.subheader("睡眠與壓力資料")

wake_episodes_per_night = st.number_input(
    "平均每晚醒來幾次？",
    min_value=0,
    max_value=8,
    value=3,
    step=1
)

stress_score = st.slider(
    "最近的壓力程度",
    min_value=1.0,
    max_value=10.0,
    value=5.8,
    step=0.1
)

sleep_latency_mins = st.number_input(
    "躺下後通常需要多久才能睡著？（分鐘）",
    min_value=1,
    max_value=58,
    value=19,
    step=1
)

sleep_quality_score = st.slider(
    "最近的睡眠品質",
    min_value=1.0,
    max_value=10.0,
    value=4.9,
    step=0.1
)

st.subheader("生活習慣資料")

alcohol_units_before_bed = st.number_input(
    "睡前飲酒量（單位）",
    min_value=0.0,
    max_value=6.0,
    value=0.0,
    step=0.5
)

caffeine_mg_before_bed = st.number_input(
    "睡前咖啡因攝取量（mg）",
    min_value=0,
    max_value=400,
    value=0,
    step=10
)

weekend_sleep_diff_hrs = st.slider(
    "假日通常比平日多睡或少睡幾小時？",
    min_value=-1.0,
    max_value=3.0,
    value=1.2,
    step=0.1
)

screen_time_before_bed_mins = st.slider(
    "睡前使用手機、平板、電腦或電視多久？（分鐘）",
    min_value=2,
    max_value=180,
    value=51,
    step=5
)
import pandas as pd

input_data = pd.DataFrame([{
    "mental_health_condition": mental_health_condition,
    "sleep_duration_hrs": sleep_duration_hrs,
    "bmi": bmi,
    "wake_episodes_per_night": wake_episodes_per_night,
    "stress_score": stress_score,
    "sleep_latency_mins": sleep_latency_mins,
    "shift_work": shift_work,
    "sleep_quality_score": sleep_quality_score,
    "alcohol_units_before_bed": alcohol_units_before_bed,
    "age": age,
    "caffeine_mg_before_bed": caffeine_mg_before_bed,
    "weekend_sleep_diff_hrs": weekend_sleep_diff_hrs,
    "screen_time_before_bed_mins": screen_time_before_bed_mins
}])

st.subheader("模型輸入資料")
st.dataframe(input_data)



from catboost import CatBoostClassifier

model = CatBoostClassifier()
model.load_model("catboost_web13.cbm")






if st.button("開始預測"):

    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    predicted_class = prediction[0][0]
    prob = probabilities[0]

    st.subheader("預測結果")

    if predicted_class == "Healthy":
        st.success("🟢 預測結果：Healthy")

    elif predicted_class == "Mild":
        st.info("🟡 預測結果：Mild")

    elif predicted_class == "Moderate":
        st.warning("🟠 預測結果：Moderate")

    elif predicted_class == "Severe":
        st.error("🔴 預測結果：Severe")

    st.write("### 各風險類別機率")

    for class_name, p in zip(model.classes_, prob):
        st.write(f"{class_name}: {p:.2%}")