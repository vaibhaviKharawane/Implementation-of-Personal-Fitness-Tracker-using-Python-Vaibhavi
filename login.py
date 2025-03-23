import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# --- INITIALIZE SESSION STATE ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False


# --- ✅ LOGIN SYSTEM (Fixes Experimental Rerun Error) ---
def login():
    st.write("## 🔐 Login to Access the Fitness Tracker")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "password123":  # Change as needed
            st.session_state["logged_in"] = True
            st.success("✅ Logged in successfully!")
            st.rerun()  # ✅ Use `st.rerun()` instead of `st.experimental_rerun()`
        else:
            st.error("❌ Invalid username or password. Please try again.")

# --- CHECK IF LOGGED IN ---
if not st.session_state["logged_in"]:
    login()
    st.stop()  # Stops execution until logged in

# --- MAIN APP STARTS AFTER LOGIN ---
st.write("## Personal Fitness Tracker")
st.write("This WebApp predicts your calories burned based on `Age`, `Gender`, `BMI`, etc.")

st.sidebar.header("User Input Parameters: ")

def user_input_features():
    age = st.sidebar.slider("Age: ", 10, 100, 30)
    bmi = st.sidebar.slider("BMI: ", 15, 40, 20)
    duration = st.sidebar.slider("Duration (min): ", 0, 35, 15)
    heart_rate = st.sidebar.slider("Heart Rate: ", 60, 130, 80)
    body_temp = st.sidebar.slider("Body Temperature (C): ", 36, 42, 38)
    gender_button = st.sidebar.radio("Gender: ", ("Male", "Female"))

    gender = 1 if gender_button == "Male" else 0

    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender  
    }

    return pd.DataFrame(data_model, index=[0])

df = user_input_features()

st.write("---")
st.header("Your Parameters: ")
st.write(df)

# --- LOAD DATA ---
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

for data in [exercise_train_data, exercise_test_data]:
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
    data["BMI"] = round(data["BMI"], 2)

exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]

exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]

random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
random_reg.fit(X_train, y_train)

df = df.reindex(columns=X_train.columns, fill_value=0)
prediction = random_reg.predict(df)

st.write("---")
st.header("Prediction: ")
st.write(f"{round(prediction[0], 2)} **kilocalories**")
