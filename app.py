import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('Extra_Trees_Classifier.pkl', 'rb'))
sc = pickle.load(open('min_max_scalar.pkl', 'rb'))
st.title("Diabetes Prediction App")

st.write("""
This app predicts whether a person has diabetes based on their medical data.
""")
st.markdown("""
### Instructions
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration
- **Blood Pressure**: Diastolic blood pressure (mm Hg)
- **Skin Thickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **Diabetes Pedigree Function**: Diabetes pedigree function ( 0.08 to 2.42)
- **Age**: Age in years
""")


# Input fields for user data
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0, step=1)
glucose = st.number_input('Glucose', min_value=0, max_value=200, value=0, step=1)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=150, value=0, step=1)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=0, step=1)
insulin = st.number_input('Insulin', min_value=0, max_value=900, value=0, step=1)
bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=0.0, step=0.1)
diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.0, step=0.01)
age = st.number_input('Age', min_value=0, max_value=120, value=0, step=1)
# Create a numpy array from the input values
user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

# Apply MinMaxScaler to the input features
# scaled_data = sc.transform(user_data)


btn = st.button('Click Here to Check')
# convert them into numpy array
if btn:
    # inp = float(inp)
    inp = np.array(user_data).reshape(1, -1)

    # apply MinMaxScalar to the input features
    inp_scaled = sc.transform(inp)

    # predict model output
    prediction = model.predict_proba(inp_scaled)
    st.header('Diabetes Probability')
    st.html(f"{prediction[0][1]*100} %")



