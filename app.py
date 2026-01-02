import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------------
# Load and Preprocess Data
# --------------------------------------------------
@st.cache_data
def load_and_clean_data():
    # Load your CSV
    df = pd.read_csv("Cellphone.csv")
    
    # Map categorical text to numbers so the model can read them
    condition_map = {"Like New": 3, "Good": 2, "Average": 1, "Poor": 0}
    type_map = {"Android": 0, "iOS": 1}
    
    if 'Condition' in df.columns:
        df['Condition'] = df['Condition'].map(condition_map)
    if 'Type' in df.columns:
        df['Type'] = df['Type'].map(type_map)
        
    # One-Hot Encoding for Brand (Creates Samsung, Apple, etc. columns)
    # This prevents the model from thinking Brand 4 is "greater than" Brand 1
    df = pd.get_dummies(df, columns=['Brand'], drop_first=False)
    
    return df

# Initialize
st.title("Mobile Price Prediction")

try:
    data = load_and_clean_data()
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # --------------------------------------------------
    # Feature / Target Split
    # --------------------------------------------------
    X = data.drop(columns=["Price"], errors="ignore")
    y = data["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --------------------------------------------------
    # Feature Scaling
    # --------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------
    # Models Training
    # --------------------------------------------------
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.3)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)
        results.append({"Model": name, "Test RMSE": round(rmse, 2), "Test R2": round(r2, 4)})

    st.subheader("Model Comparison")
    st.table(pd.DataFrame(results))

    # --------------------------------------------------
    # Prediction UI
    # --------------------------------------------------
    st.divider()
    st.header("Predict a Phone Price")

    col1, col2 = st.columns(2)
    with col1:
        storage = st.number_input("Storage (GB)", 0, 512, 64)
        screen_size = st.number_input("Screen Size", 0.0, 8.0, 6.0)
        rear_cam = st.number_input("Rear Camera (MP)", 0, 200, 48)
        battery = st.number_input("Battery (mAh)", 0, 7000, 4000)
        brand = st.selectbox("Brand", ["Samsung", "Apple", "Google", "Xiaomi", "Other"])

    with col2:
        front_cam = st.number_input("Front Camera (MP)", 0, 100, 16)
        ram = st.number_input("RAM (GB)", 0, 32, 4)
        cores = st.number_input("CPU Cores", 1, 16, 8)
        freq = st.number_input("CPU Freq (GHz)", 0.5, 4.0, 2.0)
        condition = st.selectbox("Condition", ["Like New", "Good", "Average", "Poor"])

    phone_type = st.selectbox("Type", ["Android", "iOS"])

    if st.button("Predict Price"):
        # 1. Create a dictionary for input
        input_dict = {col: 0 for col in X.columns} # Initialize all columns with 0
        
        # 2. Update with user values
        input_dict.update({
            "Storage": storage,
            "Screen Size": screen_size,
            "Rear Camera": rear_cam,
            "Front Camera": front_cam,
            "Battery": battery,
            "RAM": ram,
            "CPU Cores": cores,
            "CPU Frequency": freq,
            "Condition": {"Like New": 3, "Good": 2, "Average": 1, "Poor": 0}[condition],
            "Type": {"Android": 0, "iOS": 1}[phone_type]
        })
        
        # 3. Handle the One-Hot Encoded Brand
        brand_col = f"Brand_{brand}"
        if brand_col in input_dict:
            input_dict[brand_col] = 1

        # 4. Convert to DataFrame to ensure correct column order
        input_df = pd.DataFrame([input_dict])[X.columns]
        
        # 5. Scale and Predict
        input_scaled = scaler.transform(input_df)
        prediction = models["ElasticNet"].predict(input_scaled)

        st.success(f"Estimated Price: ₹ {max(0, prediction[0]):,.2f}")

except FileNotFoundError:
    st.error("Please ensure 'Cellphone.csv' is in the same folder as this script.")
