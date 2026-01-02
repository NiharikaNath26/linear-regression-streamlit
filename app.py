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
   
    # Drop Product_id as it doesn't help with price prediction
    if 'Product_id' in df.columns:
        df = df.drop(columns=['Product_id'])
   
    return df

# Initialize
st.set_page_config(page_title="Mobile Price Prediction", layout="wide")
st.title("📱 Mobile Price Prediction")

try:
    data = load_and_clean_data()
    st.subheader("Dataset Preview")
    st.dataframe(data.head(), use_container_width=True)

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
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        preds = model.predict(X_test_scaled)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)
        results.append({"Model": name, "Test RMSE": round(rmse, 2), "Test R2": round(r2, 4)})

    st.subheader("Model Performance Comparison")
    st.table(pd.DataFrame(results))

    # --------------------------------------------------
    # Prediction UI
    # --------------------------------------------------
    st.divider()
    st.header("Predict a Phone Price")
   
    # Layout with three columns for all features in the CSV
    col1, col2, col3 = st.columns(3)
   
    with col1:
        internal_mem = st.number_input("Internal Memory (GB)", 0, 1024, 64)
        ram = st.number_input("RAM (GB)", 0.0, 64.0, 4.0)
        cpu_core = st.number_input("CPU Cores", 1, 16, 8)
        cpu_freq = st.number_input("CPU Frequency (GHz)", 0.5, 4.0, 2.0)

    with col2:
        resoloution = st.number_input("Screen Size (Inches)", 0.0, 10.0, 6.0)
        ppi = st.number_input("PPI (Pixel Density)", 100, 1000, 400)
        rear_cam = st.number_input("Rear Camera (MP)", 0.0, 200.0, 13.0)
        front_cam = st.number_input("Front Camera (MP)", 0.0, 100.0, 8.0)

    with col3:
        battery = st.number_input("Battery (mAh)", 500, 10000, 3000)
        weight = st.number_input("Weight (g)", 50.0, 500.0, 150.0)
        thickness = st.number_input("Thickness (mm)", 1.0, 20.0, 8.0)
        sale = st.number_input("Sale Count/Index", 0, 10000, 10)

    if st.button("Predict Price"):
        # 1. Create a dictionary for input exactly matching CSV columns
        input_dict = {
            "Sale": sale,
            "weight": weight,
            "resoloution": resoloution,
            "ppi": ppi,
            "cpu core": cpu_core,
            "cpu freq": cpu_freq,
            "internal mem": internal_mem,
            "ram": ram,
            "RearCam": rear_cam,
            "Front_Cam": front_cam,
            "battery": battery,
            "thickness": thickness
        }
       
        # 2. Convert to DataFrame to ensure correct column order
        input_df = pd.DataFrame([input_dict])[X.columns]
       
        # 3. Scale and Predict using the best performing model (ElasticNet or Linear)
        input_scaled = scaler.transform(input_df)
        prediction = trained_models["Linear Regression"].predict(input_scaled)

        st.success(f"### Estimated Price: ${max(0, prediction[0]):,.2f}")

except FileNotFoundError:
    st.error("Please ensure 'Cellphone.csv' is in the same folder as this script.")
except Exception as e:
    st.error(f"An error occurred: {e}")
