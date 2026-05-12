import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Page configuration
st.set_page_config(page_title="Responsible AI: Gender Equity", layout="wide")

# 1. DATA LOADING & MODEL TRAINING
@st.cache_data
def get_processed_data():
    # Load data from the data folder
    df = pd.read_csv('data/WA_Fn-UseC_-HR-Employee-Attrition.csv')
    
    # ML Preprocessing for Audit
    le = LabelEncoder()
    df_ml = df.copy()
    for col in df_ml.select_dtypes(include=['object']).columns:
        df_ml[col] = le.fit_transform(df_ml[col])
    
    X = df_ml.drop("Attrition", axis=1)
    y = df_ml["Attrition"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return df, X_test, y_test, y_pred, model, X

# Initialize data
df, X_test, y_test, y_pred, model, X = get_processed_data()

# 2. SIDEBAR NAVIGATION
st.sidebar.title("📊 Project Navigation")
page = st.sidebar.radio("Go to:", [
    "Overview", 
    "Gender Analytics", 
    "Fairness Audit", 
    "Explainability", 
    "Governance Insights"
])

# --- STAGE 1: OVERVIEW ---
if page == "Overview":
    st.title("🌐 Project Overview")
    st.markdown("""
    This dashboard analyzes **Gender Equity** and **AI Fairness** within the workforce.
    Using the HR Employee Attrition dataset, we explore how data science can be applied responsibly.
    """)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Employees", len(df))
    col2.metric("Avg Monthly Income", f"${df['MonthlyIncome'].mean():,.0f}")
    col3.metric("Attrition Rate", f"{(df['Attrition']=='Yes').mean()*100:.1f}%")
    
    st.subheader("Data Preview")
    st.dataframe(df.head(10))

# --- STAGE 2: GENDER ANALYTICS ---
elif page == "Gender Analytics":
    st.title("📊 Gender Analytics")
    st.write("Exploration of salary and workforce distribution by gender.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Monthly Income Gap Check")
        fig, ax = plt.subplots()
        sns.boxplot(x='Gender', y='MonthlyIncome', data=df, palette='Set2', ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.subheader("Workforce Composition")
        fig, ax = plt.subplots()
        df['Gender'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, colors=['#66b3ff','#ff9999'])
        st.pyplot(fig)

# --- STAGE 3: FAIRNESS AUDIT ---
elif page == "Fairness Audit":
    st.title("⚖️ AI Fairness Audit")
    st.write("Auditing the AI model's predictions to detect gender bias.")
    
    fairness_df = pd.DataFrame({
        "Gender": df.loc[X_test.index, "Gender"],
        "Prediction": y_pred
    })
    fairness_stats = fairness_df.groupby("Gender")["Prediction"].mean()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Prediction Rate by Gender")
        st.bar_chart(fairness_stats)
    with col2:
        # Handling potential key issues
        female_rate = fairness_stats.get("Female", 0)
        male_rate = fairness_stats.get("Male", 0)
        diff = abs(female_rate - male_rate)
        
        st.metric("Demographic Parity Gap", f"{diff:.4f}")
        if diff < 0.05:
            st.success("✅ Model is Fair: Minimal bias detected.")
        else:
            st.warning("⚠️ Bias Detected: Significant difference in prediction rates.")

# --- STAGE 4: EXPLAINABILITY ---
elif page == "Explainability":
    st.title("🔍 Explainability (XAI)")
    st.write("Identifying the key factors that drive the AI's predictions.")
    
    importances = model.feature_importances_
    feat_importances = pd.Series(importances, index=X.columns).nlargest(10)
    
    fig, ax = plt.subplots()
    feat_importances.plot(kind='barh', ax=ax, color='teal')
    plt.title("Top 10 Drivers of Attrition")
    st.pyplot(fig)
    st.info("Insights: Factors like 'MonthlyIncome' and 'OverTime' are higher priorities for the model than 'Gender'.")

# --- STAGE 5: GOVERNANCE INSIGHTS ---
elif page == "Governance Insights":
    st.title("📜 Governance & Policy Insights")
    st.markdown("""
    ### Responsible AI Framework:
    1. **Bias Mitigation:** Since we detected a parity gap, we must ensure salary and overtime are balanced.
    2. **Transparency:** This dashboard provides stakeholders with clear insights into how AI decisions are made.
    3. **Privacy:** All employee data is processed according to data protection standards.
    
    **Project Goal:** To integrate **Data Feminism** and **Responsible AI** into corporate decision-making.
    """)