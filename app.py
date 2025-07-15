import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load models
reg_model = joblib.load('regression_model.pkl')
cls_model = joblib.load('classification_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load dataset
df = pd.read_csv('student_data.csv')
df['Pass_Fail'] = df['Final_Score'].apply(lambda x: 'Pass' if x >= 50 else 'Fail')

# Streamlit page config
st.set_page_config(page_title="Student Score Predictor", layout="centered")
st.title("🎓 Student Performance Predictor")

# Sidebar user input
st.sidebar.header("📌 Input Student Details")
hours = st.sidebar.slider("Hours Studied", 0, 10, 5)
attendance = st.sidebar.slider("Attendance (%)", 50, 100, 75)
previous = st.sidebar.slider("Previous Exam Score", 0, 100, 60)

# Prediction
if st.sidebar.button("🔍 Predict"):
    input_data = np.array([[hours, attendance, previous]])
    predicted_score = reg_model.predict(input_data)[0]
    predicted_label = cls_model.predict(input_data)[0]
    predicted_status = label_encoder.inverse_transform([predicted_label])[0]

    st.subheader("📈 Predicted Results")
    st.success(f"🎯 Predicted Final Score: {round(predicted_score, 2)}")
    
    if predicted_status == "Pass":
        st.info("✅ Student is likely to Pass")
    else:
        st.warning("❌ Student is likely to Fail")

    # Store prediction history
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    st.session_state.history.append({
        'Hours': hours,
        'Attendance': attendance,
        'Previous Score': previous,
        'Predicted Score': round(predicted_score, 2),
        'Prediction': predicted_status
    })

# Prediction History
if 'history' in st.session_state:
    st.write("### 🧾 Prediction History")
    st.dataframe(pd.DataFrame(st.session_state.history))

# Visualization - Scatter Plot
st.write("### 📊 EDA - Study Hours vs Final Score")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='Hours_Studied', y='Final_Score', hue='Pass_Fail', ax=ax)
plt.title("Study Hours vs Final Score")
st.pyplot(fig)

# Correlation heatmap
st.write("### 🔗 Correlation Heatmap")
fig2, ax2 = plt.subplots()
numeric_df = df.drop(columns=['Pass_Fail'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax2)
st.pyplot(fig2)

st.caption("Built by 3rd Year Engineering Student 🧑‍🎓")
