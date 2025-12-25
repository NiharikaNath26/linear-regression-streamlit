import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.title("Mobile Price Prediction using Linear Regression Models")

st.write("This app trains and evaluates Linear, Ridge, Lasso, and ElasticNet regression models.")

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Cellphone.csv")

data = load_data()

st.subheader("Dataset Preview")
st.dataframe(data.head())

# --------------------------------------------------
# Feature / Target Split
# --------------------------------------------------
X = data.drop(columns=["Price"], errors="ignore")
y = data["Price"]

# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Feature Scaling
# --------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------
# Models
# --------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "ElasticNet Regression": ElasticNet(alpha=0.01, l1_ratio=0.3)
}

# --------------------------------------------------
# Train & Evaluate
# --------------------------------------------------
results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_rmse = mean_squared_error(y_train, y_train_pred) ** 0.5
    test_rmse = mean_squared_error(y_test, y_test_pred) ** 0.5
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    results.append({
        "Model": name,
        "Train RMSE": round(train_rmse, 2),
        "Test RMSE": round(test_rmse, 2),
        "Train R2": round(train_r2, 4),
        "Test R2": round(test_r2, 4)
    })

# --------------------------------------------------
# Results Table
# --------------------------------------------------
results_df = pd.DataFrame(results)

st.subheader("Model Performance Comparison")
st.dataframe(results_df)

# --------------------------------------------------
# Conclusion
# --------------------------------------------------
st.subheader("Conclusion")
st.write(
    "ElasticNet performs best due to balanced regularization, "
    "reducing overfitting while maintaining strong predictive power."
)
st.divider()
st.header("Used Phone Price Prediction")

# ------------------------------
# User Inputs
# ------------------------------
col1, col2 = st.columns(2)

with col1:
    storage = st.number_input("Storage (GB)", min_value=0, max_value=512, value=64)
    screen_size = st.number_input("Screen Size (inches)", min_value=0.0, max_value=8.0, value=6.0)
    rear_camera = st.number_input("Rear Camera (MP)", min_value=0, max_value=200, value=48)
    battery = st.number_input("Battery Capacity (mAh)", min_value=0, max_value=7000, value=4000)

with col2:
    front_camera = st.number_input("Front Camera (MP)", min_value=0, max_value=100, value=16)
    ram = st.number_input("RAM (GB)", min_value=0, max_value=32, value=4)
    cpu_core = st.number_input("CPU Cores", min_value=1, max_value=16, value=8)
    cpu_freq = st.number_input("CPU Frequency (GHz)", min_value=0.5, max_value=4.0, value=2.0)

condition = st.selectbox("Condition", ["Like New", "Good", "Average", "Poor"])
phone_type = st.selectbox("Type", ["Android", "iOS"])
brand = st.selectbox("Brand", ["Samsung", "Apple", "Google", "Xiaomi", "Other"])

# ------------------------------
# Encode categorical inputs
# ------------------------------
condition_map = {"Like New": 3, "Good": 2, "Average": 1, "Poor": 0}
type_map = {"Android": 0, "iOS": 1}

condition_encoded = condition_map[condition]
type_encoded = type_map[phone_type]

# ------------------------------
# Predict Button
# ------------------------------
if st.button("Predict Price"):
    input_data = np.array([[ 
        storage,
        screen_size,
        rear_camera,
        front_camera,
        battery,
        ram,
        cpu_core,
        cpu_freq,
        condition_encoded,
        type_encoded
    ]])

    # Scale input using training scaler
    input_scaled = scaler.transform(input_data)

    # Use ElasticNet (best balanced model)
    prediction = models["ElasticNet"].predict(input_scaled)

    st.success(f"Estimated Price: ₹ {prediction[0]:,.2f}")

