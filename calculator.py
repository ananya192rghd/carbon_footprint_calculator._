import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import time
import psutil

# Streamlit UI Header
st.title("Carbon Footprint Calculator with Green AI")
st.write("This app predicts carbon footprint using an optimized, energy-efficient model.")

# Load a dataset with more complexity
data = {
    "transport_emissions": [120, 80, 100, 200, 150, 90, 50, 300, 250, 170, 95, 280],
    "diet_emissions": [50, 60, 55, 40, 70, 45, 50, 90, 85, 65, 60, 75],
    "waste_emissions": [30, 40, 35, 20, 45, 25, 30, 60, 50, 45, 40, 55],
    "carbon_footprint": [200, 180, 190, 260, 265, 160, 130, 450, 390, 300, 210, 400]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Feature Selection: Dropping less significant features (if necessary)
X = df.drop("carbon_footprint", axis=1)
y = df["carbon_footprint"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Using Ridge Regression (optimized, lightweight model)
model = Ridge(alpha=0.1)  # Less complex than deep learning

# Measure training energy efficiency
start_time = time.time()
cpu_before = psutil.cpu_percent()
model.fit(X_train_scaled, y_train)
cpu_after = psutil.cpu_percent()
end_time = time.time()

training_time = end_time - start_time
cpu_usage = cpu_after - cpu_before

# Streamlit Input Fields
transport = st.number_input("Transport Emissions (kg CO2)", min_value=0.0, step=1.0)
diet = st.number_input("Diet Emissions (kg CO2)", min_value=0.0, step=1.0)
waste = st.number_input("Waste Emissions (kg CO2)", min_value=0.0, step=1.0)

if st.button("Predict Carbon Footprint"):
    input_data = np.array([[transport, diet, waste]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Carbon Footprint: {prediction[0]:.2f} kg CO2")
    st.write(f"Model Training Time: {training_time:.4f} seconds")
    st.write(f"CPU Usage During Training: {cpu_usage:.2f}%")

    # Dynamic Heatmap
    st.subheader("Correlation Heatmap")
    df_dynamic = df.copy()
    df_dynamic.loc[len(df_dynamic)] = [transport, diet, waste, prediction[0]]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df_dynamic.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
    st.pyplot(fig)

    # Dynamic Pie Chart
    st.subheader("Contribution of Different Emissions")
    sizes = [transport, diet, waste]
    labels = ["Transport", "Diet", "Waste"]
    colors = ['red', 'blue', 'green']
    fig2, ax2 = plt.subplots()
    ax2.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140, shadow=True)
    st.pyplot(fig2)
