import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Function to authenticate user
def authenticate(username, password):
    return username == "admin" and password == "password123"  # Example credentials

# Initialize session state if not set
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Login page
if not st.session_state.authenticated:
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate(username, password):
            st.session_state.authenticated = True
            st.experimental_rerun()
        else:
            st.error("Invalid credentials")
else:
    # Main App Page after Login
    st.title("Personal Fitness Tracker")
    st.write("In this WebApp you will be able to observe your predicted calories burned in your body. Pass your parameters such as `Age`, `Gender`, `BMI`, etc., into this WebApp and then you will see the predicted value of kilocalories burned.")
    
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
        
        features = pd.DataFrame(data_model, index=[0])
        return features
    
    df = user_input_features()
    
    st.write("---")
    st.header("Your Parameters: ")
    bar = st.progress(0)
    for i in range(100):
        bar.progress(i + 1)
        time.sleep(0.01)
    st.write(df)
    
    # Load and preprocess data
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
    
    X_test = exercise_test_data.drop("Calories", axis=1)
    y_test = exercise_test_data["Calories"]
    
    random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
    random_reg.fit(X_train, y_train)
    
    df = df.reindex(columns=X_train.columns, fill_value=0)
    prediction = random_reg.predict(df)
    
    st.write("---")
    st.header("Prediction: ")
    bar = st.progress(0)
    for i in range(100):
        bar.progress(i + 1)
        time.sleep(0.01)
    
    st.write(f"{round(prediction[0], 2)} **kilocalories**")
    
    st.write("---")
    st.header("Similar Results: ")
    bar = st.progress(0)
    for i in range(100):
        bar.progress(i + 1)
        time.sleep(0.01)
    
    calorie_range = [prediction[0] - 10, prediction[0] + 10]
    similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]
    st.write(similar_data.sample(5))
    
    st.write("---")
    st.header("General Information: ")
    
    boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
    boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
    boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
    boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()
    
    st.write("You are older than", round(sum(boolean_age) / len(boolean_age), 2) * 100, "% of other people.")
    st.write("Your exercise duration is higher than", round(sum(boolean_duration) / len(boolean_duration), 2) * 100, "% of other people.")
    st.write("You have a higher heart rate than", round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100, "% of other people during exercise.")
    st.write("You have a higher body temperature than", round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100, "% of other people during exercise.")
    
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.experimental_rerun()
