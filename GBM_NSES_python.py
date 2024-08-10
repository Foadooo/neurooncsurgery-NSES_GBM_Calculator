import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pyreadstat
import pickle

# Load the dataset
file_path = "/Users/foadk/Downloads/Shiny app 15/NSES_GBM_Calculator/June 23rd full v3 .sav"
df, meta = pyreadstat.read_sav(file_path)

# Convert to numeric
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['NSESindex'] = pd.to_numeric(df['NSESindex'], errors='coerce')
df['MFI'] = pd.to_numeric(df['MFI'], errors='coerce')
df['KPSscore'] = pd.to_numeric(df['KPSscore'], errors='coerce')
df['SurvFromDxmo'] = pd.to_numeric(df['SurvFromDxmo'], errors='coerce')

# Convert to categorical (binary/factor)
df['Sex'] = df['Sex'].astype('category')
df['Ethnicity'] = df['Ethnicity'].astype('category')
df['Retired10'] = df['Retired10'].astype('category')
df['ERAdmit'] = df['ERAdmit'].astype('category')
df['NonHomeDispo'] = df['NonHomeDispo'].astype('category')
df['racebinary'] = df['racebinary'].astype('category')
df['MDandotherneighbors'] = df['MDandotherneighbors'].astype('category')
df['didnotinitiateStupp'] = df['didnotinitiateStupp'].astype('category')
df['LOSbinary'] = df['LOSbinary'].astype('category')
df['maritalfinalbinary'] = df['maritalfinalbinary'].astype('category')
df['insurancebinaryfinal'] = df['insurancebinaryfinal'].astype('category')

# Model 1: LOSbinary outcome with ERAdmit, KPSscore, and NSESindex predictors
model1 = sm.GLM(df['LOSbinary'].cat.codes, sm.add_constant(df[['ERAdmit', 'KPSscore', 'NSESindex']]), family=sm.families.Binomial()).fit()

# Model 2: NonHomeDispo outcome with insurancebinaryfinal, ERAdmit, MFI, and NSESindex predictors
model2 = sm.GLM(df['NonHomeDispo'].cat.codes, sm.add_constant(df[['insurancebinaryfinal', 'ERAdmit', 'MFI', 'NSESindex']]), family=sm.families.Binomial()).fit()

# Model 4: didnotinitiateStupp outcome with Age, insurancebinaryfinal, KPSscore, NSESindex, and MDandotherneighbors predictors
model4 = sm.GLM(df['didnotinitiateStupp'].cat.codes, sm.add_constant(df[['Age', 'insurancebinaryfinal', 'KPSscore', 'NSESindex', 'MDandotherneighbors']]), family=sm.families.Binomial()).fit()

# Save models for later use
with open('model1.pickle', 'wb') as f:
    pickle.dump(model1, f)
with open('model2.pickle', 'wb') as f:
    pickle.dump(model2, f)
with open('model4.pickle', 'wb') as f:
    pickle.dump(model4, f)

# Streamlit UI
st.title("Glioblastoma Resection Outcome Predictor")

st.sidebar.header("Input Parameters")

age = st.sidebar.slider("Age", 18, 90, 72)
insurance = st.sidebar.selectbox("Please select type of insurance:", ("Private", "Medicare/Medicaid/Uninsured/other"))
state_of_residence = st.sidebar.selectbox("Please select state of residence:", ("MD and neighboring states", "Other states"))
admission_source = st.sidebar.selectbox("Please select admission source:", ("Elective admission", "Non-elective admission"))
mfi_5 = st.sidebar.slider("mFI-5", 0, 5, 5)
kps = st.sidebar.slider("KPS", 20, 100, 80, step=10)
nses = st.sidebar.slider("NSES", 25, 90, 65)

if st.sidebar.button("Submit"):
    # Process inputs for predictions
    input_data1 = pd.DataFrame({
        'ERAdmit': [1 if admission_source == "Non-elective admission" else 0],
        'KPSscore': [kps],
        'NSESindex': [nses]
    })
    input_data1 = sm.add_constant(input_data1, has_constant='add')

    input_data2 = pd.DataFrame({
        'insurancebinaryfinal': [1 if insurance == "Medicare/Medicaid/Uninsured/other" else 0],
        'ERAdmit': [1 if admission_source == "Non-elective admission" else 0],
        'MFI': [mfi_5],
        'NSESindex': [nses]
    })
    input_data2 = sm.add_constant(input_data2, has_constant='add')

    input_data4 = pd.DataFrame({
        'Age': [age],
        'insurancebinaryfinal': [1 if insurance == "Medicare/Medicaid/Uninsured/other" else 0],
        'KPSscore': [kps],
        'NSESindex': [nses],
        'MDandotherneighbors': [1 if state_of_residence == "Other states" else 0]
    })
    input_data4 = sm.add_constant(input_data4, has_constant='add')

    # Make predictions
    try:
        pred1 = model1.predict(input_data1)[0] * 100
        pred2 = model2.predict(input_data2)[0] * 100
        pred4 = model4.predict(input_data4)[0] * 100

        # Display predictions
        st.header("Predicted Outcomes")
        st.subheader(f"Probability of extended Length of Stay: {pred1:.2f}%")
        st.subheader(f"Probability of non-routine discharge disposition: {pred2:.2f}%")
        st.subheader(f"Probability of non-initiation of Stupp protocol: {pred4:.2f}%")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

else:
    st.write("Server is ready for calculation.")
