# 📌 Credit Default Prediction

A **Machine Learning web application** built with **Streamlit** that predicts whether a customer will default on their credit card payment using a trained ML model. This project is useful for banks and financial institutions to assess credit risk before approving loans.

---

## 🚀 Table of Contents
- [📌 Project Overview](#-project-overview)
- [📊 Dataset Details](#-dataset-details)
- [⚙️ Installation Guide](#️-installation-guide)
- [🛠️ How to Use](#️-how-to-use)
- [💡 Model Details](#-model-details)
- [📊 Features Used](#-features-used)
- [🔥 Live Demo](#-live-demo)
- [📜 License](#-license)

---

## 📌 Project Overview
This project predicts whether a customer will **default** on their **credit card payment** using a Machine Learning model. It takes various financial and demographic inputs and predicts the likelihood of default.

**🛠️ Technologies Used:**
- Python 🐍
- Machine Learning (scikit-learn, joblib)
- Streamlit (for the web UI)
- Pandas & NumPy (for data processing)

---

## 📊 Dataset Details
- **Source:** [UCI Credit Card Default Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- **Size:** 30,000 records
- **Target Variable:** `Default Payment (Yes/No)`
- **Key Features:** Credit Limit, Payment History, Bill Amounts, etc.

---

## ⚙️ Installation Guide
**Step 1: Clone the Repository**
```bash
git clone https://github.com/yourusername/credit-default-prediction.git
cd credit-default-prediction
```

**Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 3: Run the Application**
```bash
streamlit run app/app.py
```

---

## 🛠️ How to Use
1. Open the app in your browser.
2. Enter customer details (Credit Limit, Age, Repayment History, etc.).
3. Click **"Predict Credit Default"**.
4. View the prediction result and probability.

---

## 💡 Model Details
- **Preprocessing:** StandardScaler
- **Algorithm:** Machine Learning model (Random Forest, Logistic Regression, etc.)
- **Prediction Output:** "Default" or "No Default" with confidence score

---

## 📊 Features Used
- **LIMIT_BAL** - Credit limit of the customer
- **AGE** - Age of the customer
- **PAY_X** - Repayment status for previous months
- **BILL_AMTX** - Bill amount for previous months
- **PAY_AMTX** - Payment amount for previous months

---

## 🔥 Live Demo
🔗 **[Try it Here (If Hosted)](https://your-live-demo-link.com)**

---

## 📜 License
This project is licensed under the **MIT License**.
