# ElasticNet Streamlit ML Project

This project aims to build and deploy an ElasticNet regression model for predicting cellphone prices using a dataset of cellphone specifications. The model is trained and evaluated using various techniques, and a user-friendly interface is provided through a Streamlit application.

## Project Structure

```
elasticnet-streamlit-ml
├── data
│   ├── raw
│   │   └── Cellphone.xlsx          # Raw dataset used for analysis and modeling
│   └── processed                    # Directory for processed data files
├── notebooks
│   ├── 01_eda.ipynb                # Exploratory Data Analysis (EDA) notebook
│   ├── 02_feature_engineering.ipynb # Feature engineering notebook
│   └── 03_modeling_elasticnet.ipynb # Modeling and evaluation of the ElasticNet model
├── src
│   ├── streamlit_app.py             # Streamlit application code
│   ├── config.py                     # Configuration settings for the project
│   ├── data
│   │   └── load_data.py             # Functions for loading the dataset
│   ├── features
│   │   └── preprocess.py             # Functions for preprocessing the data
│   ├── models
│   │   ├── train.py                  # Functions for training the ElasticNet model
│   │   ├── predict.py                # Functions for making predictions
│   │   └── elasticnet_pipeline.py    # Pipeline for training and predicting with ElasticNet
│   └── utils
│       └── metrics.py                # Functions for calculating evaluation metrics
├── models
│   └── elasticnet_v1.pkl             # Serialized version of the trained ElasticNet model
├── tests
│   └── test_model.py                 # Unit tests for model functions
├── scripts
│   └── run_local.sh                  # Shell script to run the Streamlit app locally
├── requirements.txt                  # Python dependencies for the project
├── Dockerfile                        # Instructions for building a Docker image
├── .gitignore                        # Files and directories to ignore by Git
└── README.md                         # Documentation for the project
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd elasticnet-streamlit-ml
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment. You can create one using `venv` or `conda`.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   You can run the application locally using the provided script:
   ```bash
   ./scripts/run_local.sh
   ```

4. **Access the application:**
   Open your web browser and go to `http://localhost:8501` to interact with the ElasticNet model.

## Usage Guidelines

- Use the Streamlit interface to input cellphone specifications and get price predictions.
- Explore the notebooks for detailed analysis, feature engineering, and model evaluation.
- Modify the configuration in `src/config.py` as needed for different datasets or model parameters.

## Contribution

Contributions are welcome! Please create a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.