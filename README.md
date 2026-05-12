Gender Equity in Responsible AI

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Executive Summary
This project addresses the critical intersection of **Machine Learning** and **Social Justice**. In an era where HR decisions are increasingly automated, this pipeline serves as a diagnostic tool to audit Algorithmic Bias. 

By applying a **Data Feminism** lens to the IBM HR Analytics dataset, this dashboard evaluates whether predictive models for employee attrition exhibit gender-based disparities, ensuring that "efficiency" does not come at the cost of "equity."

---

## 🚀 Key Research Modules

### 1. Gender Analytics & Economic Parity
* **Income Gap Analysis:** Visualizing the distribution of monthly income across genders to identify systemic pay disparities.
* **Workforce Composition:** Evaluating the representation of genders across various departments.

### 2. Algorithmic Fairness Audit
* **Demographic Parity Check:** Auditing the model's predictions to ensure the "Positive Rate" is consistent across genders.
* **Bias Detection:** Automating the calculation of the Parity Gap to alert stakeholders of potential discriminatory outcomes.

### 3. Explainable AI (XAI)
* **Feature Attribution:** Utilizing Random Forest feature importance to ensure that the model relies on professional metrics (like Performance Rating or Years at Company) rather than protected characteristics.
* **Transparency:** Providing a clear view of the "Black Box" to facilitate human-in-the-loop governance.

---

## 🛠️ Technical Implementation
* **Backend:** Python 3.x
* **ML Core:** Scikit-learn (Random Forest Classifier for robust non-linear feature mapping)
* **Data Audit:** Custom-built fairness evaluation logic.
* **Frontend:** Streamlit for real-time interactive policy simulation.

---

## 📂 Project Architecture
```text
├── data/
│   ├── app.py                      # Core Dashboard Logic
│   └── HR-Employee-Attrition.csv   # Audited Dataset
├── .gitignore                      # Environment Protection
├── requirements.txt                # Dependency Manifest
└── README.md                       # Research Documentation
📖 Theoretical Framework
This work is heavily inspired by the Data Feminism framework (D'Ignazio & Klein), focusing on:

Challenging Power: Auditing how HR models might penalize marginalized groups.

Making Labour Visible: Highlighting internal workforce trends that are often hidden in aggregate data.

🤝 Contact & Collaboration
Developed by Fazeela Saleem Undergraduate Data Science Student | Feminist Writer | AI Ethics Researcher

Currently exploring the role of AI Governance in shaping inclusive media and policy frameworks. Open to research collaborations in Responsible AI and Data for Social Good.

“The numbers have no voice; it is our responsibility to give them a conscience.”