import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Load Data set
def load_data():
    load_data = "https://raw.githubusercontent.com/shreyasnimkhedkar/loan_predition/refs/heads/master/train.csv"  # Example dataset
    data = pd.read_csv(load_data)
    
    # Drop rows with missing values
    data = data.dropna()

    # Encode categorical features
    label_enc = LabelEncoder()
    data['Gender'] = label_enc.fit_transform(data['Gender'])
    data['Married'] = label_enc.fit_transform(data['Married'])
    data['Education'] = label_enc.fit_transform(data['Education'])
    data['Self_Employed'] = label_enc.fit_transform(data['Self_Employed'])
    data['Property_Area'] = label_enc.fit_transform(data['Property_Area'])
    data['Loan_Status'] = label_enc.fit_transform(data['Loan_Status'])

    data['Dependents'] = data['Dependents'].replace('3+', 3).astype(int)

    return data

# Load data
data = load_data()

# Split dataset
x = data.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
y = data.iloc[:, 12]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=2)
assert x_train.shape[1] == x_test.shape[1], "Mismatch in feature count between train and test sets"

xgb = XGBClassifier()
xgb.fit(x_train, y_train)
y_pred = xgb.predict(x_test)
print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Streamlit UI
def main():
    st.title("SHREYAS NIMKHEDKAR")
    st.title("Loan Prediction System")
    
    if st.checkbox("Show Dataset"):
        st.write(data.head())

    st.write(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    st.header("Enter Loan Details for Prediction")

    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=2000)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=100)
    loan_amount_term = st.selectbox("Loan Amount Term", [12, 36, 60, 84, 120, 180, 240, 300, 360])
    credit_history = st.selectbox("Credit History", [0.0, 1.0])
    property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

    input_data = np.array([
        [
            1 if gender == "Male" else 0,
            1 if married == "Yes" else 0,
            int(dependents[0]) if dependents != "3+" else 3,
            1 if education == "Graduate" else 0,
            1 if self_employed == "Yes" else 0,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_amount_term,
            credit_history,
            ["Rural", "Semiurban", "Urban"].index(property_area)
        ]
    ])

    if st.button("Predict Loan Status"):
        prediction = xgb.predict(input_data)
        if prediction[0] == 1:
            st.success("Loan Approved")
        else:
            st.error("Loan Not Approved")

if __name__ == "__main__":
    main()
