import streamlit as st
import pandas as pd
import numpy as np
import base64
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Function to encode image to base64
def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Function to set background image
def set_background(image_file):
    base64_img = get_base64_of_image(image_file)
    background_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{base64_img}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

# Set background image
set_background("img.jpg")  # Ensure "img.jpg" is in the same directory

# Load the trained model
model = joblib.load('model.pkl')  # Ensure the model is saved with this name

# Sidebar for Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Prediction"])

if page == "Introduction":
    st.title("Welcome to Credit Default Prediction App")
    st.subheader("About the Model")
    st.write("This model predicts whether a customer will default on their credit payment.")
    
    st.subheader("Features Used")
    st.write("- **LIMIT_BAL**: Credit limit\n- **SEX**: Gender of the customer\n- **EDUCATION**: Education level\n- **MARRIAGE**: Marital status\n- **AGE**: Age of the customer\n- **PAY_X**: Repayment status for previous months\n- **BILL_AMTX**: Bill amount for previous months\n- **PAY_AMTX**: Payment amount for previous months")
    
    st.subheader("Use Cases")
    st.write("- Banks and financial institutions can use this app to assess credit risk.")
    st.write("- Helps in making informed decisions about loan approvals.")
    st.write("- Reduces the risk of defaults by identifying potential defaulters.")

else:
    st.title("Credit Default Prediction")
    st.subheader("Predict if a customer will default on their credit card payment")

    def preprocess_input(data):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        return scaled_data

    def predict_default(input_data):
        processed_data = preprocess_input(input_data)
        prediction_proba = model.predict_proba(processed_data)
        return prediction_proba

    st.sidebar.header("User Input Features")
    
    def user_input_features():
        limit_balance = st.sidebar.number_input("Limit Balance", min_value=0, max_value=1000000, value=20000)
        sex = st.sidebar.selectbox("Sex", options=[1, 2], format_func=lambda x: 'Male' if x == 1 else 'Female')
        education = st.sidebar.selectbox("Education", options=[1, 2, 3, 4], format_func=lambda x: ['Graduate School', 'University', 'High School', 'Others'][x-1])
        marriage = st.sidebar.selectbox("Marital Status", options=[1, 2, 3], format_func=lambda x: ['Married', 'Single', 'Others'][x-1])
        age = st.sidebar.slider("Age", 21, 79, 35)
        
        pay_0 = st.sidebar.slider("Repayment Status (September)", -2, 8, 0)
        pay_2 = st.sidebar.slider("Repayment Status (August)", -2, 8, 0)
        pay_3 = st.sidebar.slider("Repayment Status (July)", -2, 8, 0)
        pay_4 = st.sidebar.slider("Repayment Status (June)", -2, 8, 0)
        pay_5 = st.sidebar.slider("Repayment Status (May)", -2, 8, 0)
        pay_6 = st.sidebar.slider("Repayment Status (April)", -2, 8, 0)
        
        bill_amt1 = st.sidebar.number_input("Bill Amount (September)", min_value=-5000, max_value=500000, value=50000)
        bill_amt2 = st.sidebar.number_input("Bill Amount (August)", min_value=-5000, max_value=500000, value=45000)
        bill_amt3 = st.sidebar.number_input("Bill Amount (July)", min_value=-5000, max_value=500000, value=40000)
        bill_amt4 = st.sidebar.number_input("Bill Amount (June)", min_value=-5000, max_value=500000, value=35000)
        bill_amt5 = st.sidebar.number_input("Bill Amount (May)", min_value=-5000, max_value=500000, value=30000)
        bill_amt6 = st.sidebar.number_input("Bill Amount (April)", min_value=-5000, max_value=500000, value=25000)
        
        pay_amt1 = st.sidebar.number_input("Previous Payment (September)", min_value=0, max_value=500000, value=2000)
        pay_amt2 = st.sidebar.number_input("Previous Payment (August)", min_value=0, max_value=500000, value=2000)
        pay_amt3 = st.sidebar.number_input("Previous Payment (July)", min_value=0, max_value=500000, value=2000)
        pay_amt4 = st.sidebar.number_input("Previous Payment (June)", min_value=0, max_value=500000, value=2000)
        pay_amt5 = st.sidebar.number_input("Previous Payment (May)", min_value=0, max_value=500000, value=2000)
        pay_amt6 = st.sidebar.number_input("Previous Payment (April)", min_value=0, max_value=500000, value=2000)
        
        data = {
            'LIMIT_BAL': limit_balance,
            'SEX': sex,
            'EDUCATION': education,
            'MARRIAGE': marriage,
            'AGE': age,
            'PAY_0': pay_0,
            'PAY_2': pay_2,
            'PAY_3': pay_3,
            'PAY_4': pay_4,
            'PAY_5': pay_5,
            'PAY_6': pay_6,
            'BILL_AMT1': bill_amt1,
            'BILL_AMT2': bill_amt2,
            'BILL_AMT3': bill_amt3,
            'BILL_AMT4': bill_amt4,
            'BILL_AMT5': bill_amt5,
            'BILL_AMT6': bill_amt6,
            'PAY_AMT1': pay_amt1,
            'PAY_AMT2': pay_amt2,
            'PAY_AMT3': pay_amt3,
            'PAY_AMT4': pay_amt4,
            'PAY_AMT5': pay_amt5,
            'PAY_AMT6': pay_amt6
        }
        features = pd.DataFrame(data, index=[0])
        return features
    
    input_df = user_input_features()
    st.subheader("User Input Features")
    st.write(input_df)
    
    if st.button("Predict Credit Default"):
        prediction_proba = predict_default(input_df)
        result = "**Default**" if prediction_proba[0][1] > 0.3 else "**No Default**"
        st.subheader(f"Prediction Result: {result}")
        st.write(f"Prediction Confidence: {prediction_proba[0][1] * 100:.2f}%")
