#!/usr/bin/env python
# coding: utf-8

# ### Linear regression project
# ### Niharika Nath(Batch-12)
# ### Emial: nihunath12@gmail.com

# ### A.	Explain the main assumptions of Linear Regression in detail

# ### Assumptions of Linear Regression
# 
# Linear Regression makes several assumptions about the data. These assumptions must be checked to ensure the validity of the model’s outputs, especially when interpreting the coefficients and making inferences.
# 
# ---
# 
# ### 1. Linearity
# 
# 1. Linearity
# - The relationship between the independent variables (features) and the dependent variable (target) should be linear.
# - How to check:
#   - Use scatter plots of actual vs predicted values.
#   - Plot residuals vs predicted values and check if there is no visible pattern.
# 
# ---
# 
# ### 2. Independence of Errors
# 
# - The residuals (errors) should be **independent** of each other.
# - This is crucial for time series or sequential data.
# - **How to check:**  
#   - Plot residuals over time  
#   - Use **Durbin-Watson** statistic (value ~2 is ideal)
# 
# ---
# 
# ### 3. Homoscedasticity
# 
# - The residuals should have **constant variance** at every level of the independent variables.
# - If this is violated, the model suffers from **heteroscedasticity**.
# - **How to check:**  
#   - Residuals vs Fitted values plot  
#   - Look for consistent spread (no funnel shape)
# 
# ---
# 
# ### 4. No Multicollinearity
# 
# - Independent variables should not be **highly correlated** with each other.
# - High multicollinearity inflates the variance of coefficients and makes them unreliable.
# - **How to check:**  
#   - Correlation matrix  
#   - Variance Inflation Factor (VIF) — VIF > 5 or 10 indicates a problem
# 
# ---
# 
# ### 5. Normality of Residuals
# 
# - The residuals (errors) should be **normally distributed**.
# - Important for confidence intervals and hypothesis tests.
# - **How to check:**  
#   - Histogram of residuals  
#   - Q-Q plot  
#   - Shapiro-Wilk or Kolmogorov-Smirnov test
# 
# ---
# 
# ###  Summary Table
# 
# | Assumption           | Description                                  | How to Check                       |
# |----------------------|----------------------------------------------|------------------------------------|
# | **Linearity**        | X and y have a linear relationship           | Residual plots, scatter plots      |
# | **Independence**     | Residuals are independent                    | Durbin-Watson test                 |
# | **Homoscedasticity** | Constant error variance                      | Residuals vs fitted plot           |
# | **No Multicollinearity** | Predictors aren't highly correlated   | Correlation matrix, VIF            |
# | **Normality of Errors** | Residuals are normally distributed      | Q-Q plot, histogram, normality test|
# 
# ---
# 
# Violating these assumptions affects the **interpretability** and **reliability** of the linear regression model. While the model may still make predictions, statistical inferences (like p-values and confidence intervals) may not be valid.
# 

# ### B. Difference Between R-squared and Adjusted R-squared

# R-squared and Adjusted R-squared are both evaluation metrics used to measure how well a linear regression model fits the data. 
# However, they differ in how they handle the number of predictors.
# 
# 1. R-squared 
# - R-squared measures the proportion of the variance in the dependent variable that is explained by the independent variables.
# - Its value ranges from 0 to 1.
# - A higher R² means the model explains more of the variation in the target variable.
# - However, R² always increases or stays the same when new variables are added, even if they are not useful.
# 
# 2. Adjusted R-squared
# - Adjusted R-squared is a modified version of R² that adjusts for the number of independent variables in the model.
# - It increases only if the new variable improves the model more than would be expected by chance.
# - If a variable does not add value, Adjusted R² may decrease.
# - This helps to prevent overfitting when using multiple features.
# 
# Key Differences:
# 
# R-squared:
# - Measures how well the model explains the variation in the data.
# - Always increases or stays the same when more variables are added.
# - Does not account for the number of predictors.
# 
# Adjusted R-squared:
# - Adjusts R² based on the number of predictors.
# - May decrease when irrelevant features are added.
# - More reliable when comparing models with different numbers of predictors.
# 
# Conclusion:
# - Use R-squared to understand model fit.
# - Use Adjusted R-squared when comparing models with different numbers of features.
# 

# ### C.	What are the different types of Regularization techniques in Regression. Explain in detail with cost functions of each technique.

# Regularization is a technique used in regression to reduce overfitting by adding a penalty term to the cost function. This penalty discourages the model from fitting too closely to the training data by shrinking the model coefficients.
# 
# There are three main types of regularization techniques in regression:
# 
# 1. Ridge Regression (L2 Regularization)
# 
# - Ridge regression adds the squared magnitude of coefficients as a penalty term to the cost function.
# - It reduces the impact of less important features by shrinking their coefficients, but it does not set them exactly to zero.
# - Ridge is useful when all features contribute a little to the prediction.
# 
# Cost Function:
# J(theta) = MSE + lambda * sum(theta_i^2)
# Where:
# - MSE is the Mean Squared Error
# - lambda is the regularization parameter (also called alpha)
# - theta_i are the model coefficients
# 
# 2. Lasso Regression (L1 Regularization)
# 
# - Lasso adds the absolute value of coefficients as a penalty term to the cost function.
# - It can shrink some coefficients exactly to zero, effectively performing feature selection.
# - Useful when we want to eliminate irrelevant features.
# 
# Cost Function:
# J(theta) = MSE + lambda * sum(abs(theta_i))
# 
# 3. ElasticNet Regression (Combination of L1 and L2)
# 
# - ElasticNet combines both L1 (Lasso) and L2 (Ridge) penalties.
# - It uses two parameters: lambda (for strength of regularization) and l1_ratio (to balance L1 vs L2).
# - Useful when there are multiple correlated features.
# 
# Cost Function:
# J(theta) = MSE + lambda * (l1_ratio * sum(abs(theta_i)) + (1 - l1_ratio) * sum(theta_i^2))
# 
# Summary:
# 
# Technique     Penalty Term             Feature Selection    Best Use Case
# Ridge         sum(theta_i^2)           No                   Many small/medium effects
# Lasso         sum(abs(theta_i))        Yes                  Sparse models, feature elimination
# ElasticNet    L1 + L2 combination      Yes                  When features are correlated
# 
# Conclusion:
# 
# - Regularization helps prevent overfitting and improves generalization.
# - Ridge is good when all features matter slightly.
# - Lasso is good when only a few features are important.
# - ElasticNet is a balanced approach between Ridge and Lasso.
# 

# ### D. How Logistic Regression Works for Multiclass Classification

# 
# 
# Logistic Regression is primarily a binary classification algorithm. However, it can be extended to handle multiclass classification using strategies like One-vs-Rest (OvR) or Softmax Regression.
# 
# There are two main ways Logistic Regression handles multiclass classification:
# 
# 1. One-vs-Rest (OvR) or One-vs-All
# 
# - This approach trains one binary classifier per class.
# - For each class, it considers that class as "positive" and all others as "negative".
# - If there are K classes, the model trains K separate binary logistic regression classifiers.
# - During prediction, the classifier that gives the highest probability is selected as the predicted class.
# 
# Example:
# For 3 classes (A, B, C), three classifiers are trained:
# - Classifier 1: A vs not A
# - Classifier 2: B vs not B
# - Classifier 3: C vs not C
# 
# 2. Multinomial Logistic Regression (Softmax Regression)
# 
# - This method uses a single model with a generalized cost function called the softmax function.
# - Instead of producing binary outputs, it gives probabilities for each class.
# - The class with the highest probability is selected.
# 
# How it works:
# - For K classes, the model computes K separate linear combina
# 

# ### E. Performance Metrics of Logistic Regression

# E. Performance Metrics of Logistic Regression
# 
# Logistic Regression is used for binary and multiclass classification problems. It is evaluated using classification metrics that measure how well the predicted classes match the actual classes.
# 
# 1. Accuracy  
# Accuracy is the ratio of correctly predicted observations to the total number of observations.  
# Formula: (TP + TN) / (TP + TN + FP + FN)
# 
# 2. Precision  
# Precision is the ratio of correctly predicted positive observations to the total predicted positive observations.  
# Formula: TP / (TP + FP)
# 
# 3. Recall (Sensitivity or True Positive Rate)  
# Recall is the ratio of correctly predicted positive observations to all actual positive observations.  
# Formula: TP / (TP + FN)
# 
# 4. F1 Score  
# F1 Score is the harmonic mean of precision and recall. It balances the trade-off between precision and recall, especially useful when classes are imbalanced.  
# Formula: 2 * (Precision * Recall) / (Precision + Recall)
# 
# 5. Confusion Matrix  
# A confusion matrix is a table used to describe the performance of a classification model on a set of test data for which the true values are known.  
# It shows the number of correct and incorrect predictions made by the model.
# 
# Structure:  
# Actual 0: True Negative (TN), False Positive (FP)  
# Actual 1: False Negative (FN), True Positive (TP)
# 
# 6. ROC Curve and AUC  
# The ROC (Receiver Operating Characteristic) Curve is a graphical plot that shows the diagnostic ability of a binary classifier.  
# AUC (Area Under the Curve) measures the entire two-dimensional area underneath the entire ROC curve.  
# The closer the AUC is to 1, the better the model performance.
# 
# Summary:  
# - Use Accuracy when the classes are balanced.  
# - Use Precision when false positives are more important to avoid.  
# - Use Recall when false negatives are more important to avoid.  
# - Use F1 Score when both false positives and false negatives are important.  
# - Use ROC-AUC to evaluate model performance regardless of the threshold.
# 

# ### i.	Download the dataset from above link and load it into your Python environment.

# In[1]:


get_ipython().system('pip install xgboost')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import xgboost as xgb

from pandas.plotting import scatter_matrix
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Fix the typo in the magic command
get_ipython().run_line_magic('matplotlib', 'inline')

# Configure plot styles
sns.set(style="whitegrid")


# In[2]:


# Adjust the path as needed in your Jupyter/Colab environment
mobile_price = pd.read_excel('Cellphone.xlsx', sheet_name='Sheet1')


# ### ii.	Perform the EDA and do the visualizations.

# In[3]:


mobile_price.info()


# In[4]:


mobile_price.describe()


# In[5]:


mobile_price.isnull().sum()


# In[6]:


mobile_price.nunique()


# In[7]:


mobile_price.describe().T


# In[8]:


#Pairplot
sns.pairplot(mobile_price[['Price', 'ram', 'battery', 'ppi', 'cpu freq']], diag_kind='kde')
plt.suptitle("Pairwise Relationships", y=1.02)
plt.tight_layout()
plt.show()


# In[9]:


#Boxplot
num_cols = ['weight', 'resoloution', 'ppi', 'cpu core', 'cpu freq', 
            'internal mem', 'ram', 'RearCam', 'Front_Cam', 'battery', 'thickness']

for col in num_cols:
    plt.figure(figsize=(11, 8))
    sns.boxplot(data=mobile_price, x=col, y='Price')
    plt.title(f"Price vs {col}")
    plt.xticks(rotation=50)
    plt.show()


# In[10]:


plt.figure(figsize=(8, 4))
sns.violinplot(data=mobile_price, x='cpu core', y='Price')
plt.title("Price vs CPU Core")
plt.show()


# In[11]:


plt.figure(figsize=(12, 8))
sns.heatmap(mobile_price.drop(columns='Product_id').corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Features")
plt.show()


# ### iii.	Check the distributions/skewness in the variables and do the transformations if required.

# In[12]:


skew_vals = mobile_price.drop(columns='Product_id').skew(numeric_only=True).sort_values(ascending=False)
print("Skewness in variables:\n", skew_vals)


# ### iv.	Check/Treat the outliers and do the feature scaling if required.

# In[13]:


# Apply log1p to reduce skewness
skewed_cols = skew_vals[abs(skew_vals) > 1].index.tolist()
mobile_price[skewed_cols] = mobile_price[skewed_cols].apply(lambda x: np.log1p(x))

# Visualize after transformation
for col in skewed_cols:
    plt.figure(figsize=(6, 3))
    sns.histplot (mobile_price[col], kde=True)
    plt.title(f"Transformed Distribution of {col}")
    plt.show()


# In[14]:


#Removing ouliers
def remove_outliers_iqr(mobile_price, columns):
    mobile_price_out = mobile_price.copy()
    for col in columns:
        Q1 = mobile_price_out[col].quantile(0.25)
        Q3 = mobile_price_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        mobile_price_out = mobile_price_out[(mobile_price_out[col] >= lower) & (mobile_price_out[col] <= upper)]
    return mobile_price_out

# Drop Product_id, and clean
features = mobile_price.drop(columns='Product_id').columns.tolist()
mobile_price_cleaned = remove_outliers_iqr(mobile_price, features)


# In[15]:


#defining features and target
X = mobile_price_cleaned.drop(columns=['Product_id', 'Price'])
y = mobile_price_cleaned['Price']

#scaling the features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)  # Convert back to DataFrame with column names
y = mobile_price_cleaned['Price']

# Show shape of cleaned dataset
mobile_price_cleaned.shape

# OR show first few rows of scaled features
X.head()  # Now we can use .head() directly on X since it's a DataFrame


# ### v.	Create a ML model to predict the price of the phone based on the specifications given

# In[16]:


# Building the models

# Linear Regression model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

train_rmse, test_rmse, train_r2, test_r2


# ### vi.	Check for overfitting and use the Regularization techniques if required

# In[17]:


from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    results.append({
        "Model": name,
        "Train RMSE": round(train_rmse, 2),
        "Test RMSE": round(test_rmse, 2),
        "Train R²": round(train_r2, 4),
        "Test R²": round(test_r2, 4)
    })

# Display as a DataFrame
import pandas as pd
pd.DataFrame(results)


# ### vii.	Compare the performance metrics of training dataset and testing dataset for all the different algorithms used (Linear/Ridge/Lasso/ElasticNet)
# 
# ### Step vii – Performance Comparison of Linear, Ridge, Lasso, and ElasticNet
# 
# ## Performance Metrics:
# 
# | Model       | Train RMSE | Test RMSE | Train R2 | Test R2 |
# |-------------|------------|-----------|----------|---------|
# | Linear      | 159.45     | 226.11    | 0.9246   | 0.8115  |
# | Ridge       | 159.67     | 222.14    | 0.9244   | 0.8180  |
# | Lasso       | 159.45     | 225.87    | 0.9246   | 0.8119  |
# | ElasticNet  | 161.80     | 213.68    | 0.9224   | **0.8316**  |
# 
# ---
# 
# ## Observations:
# 
# - All models perform similarly on the **training set** (R2 = 0.92), showing no underfitting.
# - The *test R2* reveals generalization ability:
#   - Linear, Ridge, and Lasso perform well, but ElasticNet performs **best**.
# - *ElasticNet* has:
#   - The *lowest Test RMSE (213.68)*  lowest average error
#   - The *highest Test R2 (0.8316)* best variance explanation on test data
# 
# ---
# ### Conclusion: 
# ElasticNet is the best model for this dataset, offering the best trade-off between bias and variance. It generalizes better than the other models and slightly improves predictive performance.
# 

# # Final Model: ElasticNet with Polynomial Features + Tuning

# In[18]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate polynomial features (degree=2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Train-test split on new polynomial features
X_train_poly, X_test_poly, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Tune ElasticNet
param_grid = {
    'alpha': np.logspace(-3, 1, 5),      # [0.001, 0.01, 0.1, 1, 10]
    'l1_ratio': np.linspace(0.1, 0.9, 5) # [0.1, 0.3, 0.5, 0.7, 0.9]
}

grid = GridSearchCV(ElasticNet(max_iter=10000), param_grid, cv=5, scoring='r2', n_jobs=-1)
grid.fit(X_train_poly, y_train)

# Best model from grid
best_model = grid.best_estimator_

# Predict
y_train_pred = best_model.predict(X_train_poly)
y_test_pred = best_model.predict(X_test_poly)

# Evaluate
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Show results
{
    "Best Alpha": grid.best_params_['alpha'],
    "Best L1 Ratio": grid.best_params_['l1_ratio'],
    "Train RMSE": round(train_rmse, 2),
    "Test RMSE": round(test_rmse, 2),
    "Train R²": round(train_r2, 4),
    "Test R²": round(test_r2, 4)
}


# ---
# 
# ###  Final Model Improvement Summary (After Tuning)
# 
# After comparing the basic linear models (Linear, Ridge, Lasso, ElasticNet), we observed that **ElasticNet** gave the best generalization performance — but still showed slight overfitting.
# 
# To further improve it, we:
# 
# - Added **Polynomial Features (degree = 2)** to model non-linear relationships
# - Used **GridSearchCV** to tune ElasticNet's alpha and l1_ratio
# 
# ---
# 
# ### Final Optimized ElasticNet Model:
# 
# - **Train R²:** 0.9998
# - **Test R²:** 0.9227
# - **Train RMSE:** 8.51
# - **Test RMSE:** 139.19
# - **Best Hyperparameters:**
#   - alpha: 0.01
#   - l1_ratio: 0.3
# 
# ---
# 
# ###  Final Conclusion:
# 
# - The tuned **ElasticNet with polynomial features** significantly improved the model.
# - It achieved **high accuracy** with **minimal overfitting**, indicating excellent generalisation.
# - This is the final selected model for deployment.
# 
# ---
# 

# In[ ]:




