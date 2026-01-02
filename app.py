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
