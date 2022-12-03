import pickle

import numpy as np
import pandas as pd
import streamlit as st

model = pickle.load(open("storage/models/last_pipeline.pkl", "rb"))
st.title("Credit Card Approval Prediction")

CODE_GENDER = st.selectbox("Gender Code", ["M", "F"])
FLAG_OWN_CAR = st.selectbox("Car Owner", ["Y", "N"])
FLAG_OWN_REALTY = st.selectbox("Realty Owner", ["Y", "N"])
CNT_CHILDREN = st.slider("Children Count", 0, 20)
AMT_INCOME_TOTAL = st.number_input("Total Income", 0, 10_000_000)
NAME_INCOME_TYPE = st.selectbox(
    "Income Type",
    ["Working", "Commercial associate", "Pensioner", "State servant", "Student"],
)
NAME_EDUCATION_TYPE = st.selectbox(
    "Education Type",
    [
        "Higher education",
        "Secondary / secondary special",
        "Incomplete higher",
        "Lower secondary",
        "Academic degree",
    ],
)
NAME_FAMILY_STATUS = st.selectbox(
    "Family Status",
    ["Civil marriage", "Married", "Single / not married", "Separated", "Widow"],
)
NAME_HOUSING_TYPE = st.selectbox(
    "Housing Type",
    [
        "Rented apartment",
        "House / apartment",
        "Municipal apartment",
        "With parents",
        "Co-op apartment",
        "Office apartment",
    ],
)
DAYS_BIRTH = st.number_input("Days Birth", -28000, -6000)
DAYS_EMPLOYED = st.number_input("Days Birth", -20000, 365243)
FLAG_MOBIL = st.selectbox(
    "Mobile Flag",
    [0, 1],
)
FLAG_WORK_PHONE = st.selectbox(
    "Work Phone Flag",
    [0, 1],
)
FLAG_PHONE = st.selectbox(
    "Phone Flag",
    [0, 1],
)
FLAG_EMAIL = st.selectbox(
    "Email Flag",
    [0, 1],
)
OCCUPATION_TYPE = st.selectbox(
    "Occupation Type",
    [
        "unkown",
        "Security staff",
        "Sales staff",
        "Accountants",
        "Laborers",
        "Managers",
        "Drivers",
        "Core staff",
        "High skill tech staff",
        "Cleaning staff",
        "Private service staff",
        "Cooking staff",
        "Low-skill Laborers",
        "Medicine staff",
        "Secretaries",
        "Waiters/barmen staff",
        "HR staff",
        "Realty agents",
        "IT staff",
    ],
)
CNT_FAM_MEMBERS = st.slider("Family Number Count", 0, 30)


def predict():
    row = np.array(
        [
            CODE_GENDER,
            FLAG_OWN_CAR,
            FLAG_OWN_REALTY,
            CNT_CHILDREN,
            AMT_INCOME_TOTAL,
            NAME_INCOME_TYPE,
            NAME_EDUCATION_TYPE,
            NAME_FAMILY_STATUS,
            NAME_HOUSING_TYPE,
            DAYS_BIRTH,
            DAYS_EMPLOYED,
            FLAG_MOBIL,
            FLAG_WORK_PHONE,
            FLAG_PHONE,
            FLAG_EMAIL,
            OCCUPATION_TYPE,
            CNT_FAM_MEMBERS,
        ]
    )
    X = pd.DataFrame(
        [row],
        columns=[
            "code_gender",
            "flag_own_car",
            "flag_own_realty",
            "cnt_children",
            "amt_income_total",
            "name_income_type",
            "name_education_type",
            "name_family_status",
            "name_housing_type",
            "days_birth",
            "days_employed",
            "flag_mobil",
            "flag_work_phone",
            "flag_phone",
            "flag_email",
            "occupation_type",
            "cnt_fam_members",
        ],
    )
    prediction = model.predict(X)
    if prediction[0] == 1:
        st.success("Reject")
    else:
        st.error("Accept")


trigger = st.button("Predict", on_click=predict)
