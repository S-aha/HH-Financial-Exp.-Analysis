import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans

# Set up the page layout
st.set_page_config(page_title="Household Financial Behavior Analysis", layout="wide")

# Apply custom styles
st.markdown("""
    <style>
        body { background-color: #FFFFFF; }
        .css-18ni7ap { background-color: #003366; color: #FFFFFF; font-weight: bold; }
        .css-1d391kg { background-color: #800000; color: #FFFFFF; font-weight: bold; }
        .stTabs [role="tab"] { background-color: #006400; color: #FFFFFF; font-weight: bold; border-radius: 8px; }
        .stTabs [role="tab"]:hover { background-color: #228B22; }
        .stTabs [role="tab"][aria-selected="true"] { background-color: #8B0000; color: #FFFFFF; }
        .stHeader { color: #00008B; font-weight: bold; }
        .stInput, .stButton, .stMultiselect { border-radius: 8px; padding: 5px; background-color: #F0F0F0; }
        html, body, [class*="css"] { font-family: "Arial", sans-serif; }
    </style>
""", unsafe_allow_html=True)

# File Upload Tab
with st.sidebar:
    st.title("Financial Behavior Analysis")
    uploaded_file = st.file_uploader("Upload your household data file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file to proceed.")
        st.stop()

# Standardize column names
data.columns = [col.strip().replace(' ', '_').lower() for col in data.columns]

# Handle missing values
numeric_columns = data.select_dtypes(include=["number"]).columns
categorical_columns = data.select_dtypes(include=["object"]).columns

data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])

# Encode categorical variables
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Calculate total expenditure and savings
expenditure_columns = [
    'staple_food_expenditure', 'meat_expenditure', 'seafood_expenditure',
    'leisure_expenditure', 'alcohol_expenditure', 'tobacco_expenditure',
    'medical_expenditure', 'transportation_expenditure', 'communication_expenditure',
    'utilities_expenditure', 'housing_tax', 'education_expenditure',
    'crop_farming_expenditure'
]
data['total_expenditure'] = data[expenditure_columns].sum(axis=1)
if 'total_household_income' in data.columns:
    data['savings'] = data['total_household_income'] - data['total_expenditure']
else:
    st.error("Missing 'total_household_income' column in the dataset.")
    st.stop()

# Allow the user to download the updated data
st.sidebar.download_button("Download Updated Data", data.to_csv(), "updated_data.csv", "text/csv")

# Tabs for analysis
tabs = st.tabs(["Descriptive Statistics", "Univariate Analysis", "Bivariate Analysis", "Machine Learning"])

# Descriptive Statistics Tab
with tabs[0]:
    st.header("Descriptive Statistics")
    st.write("""
    **Overview:** This tab summarizes your dataset using measures like mean, median, and standard deviation. 
    It helps you quickly assess central tendencies and variability, making it easier to spot potential issues 
    like outliers or skewed distributions.
    **Advantages:** Ideal for initial exploration of numerical data and ensuring correctness.
    """)
    st.write(data.describe())

# Univariate Analysis Tab
with tabs[1]:
    st.header("Univariate Analysis")
    st.write("""
    **Overview:** This tab allows you to analyze the distribution of individual variables through histograms and boxplots.
    Histograms reveal frequency distributions, while boxplots highlight outliers and spread. 
    **Advantages:** Useful for understanding the structure of a single variable before diving into relationships.
    """)
    numerical_vars = st.selectbox("Select a variable for visualization", numeric_columns)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(data[numerical_vars], bins=30, kde=True, ax=ax[0])
    sns.boxplot(x=data[numerical_vars], ax=ax[1])
    ax[0].set_title(f"Histogram of {numerical_vars}")
    ax[1].set_title(f"Boxplot of {numerical_vars}")
    st.pyplot(fig)

# Bivariate Analysis Tab
with tabs[2]:
    st.header("Bivariate Analysis")
    st.write("""
    **Overview:** This tab explores relationships between two variables using scatterplots. 
    Scatterplots help visualize trends, clusters, and potential correlations.
    **Advantages:** Effective for spotting linear or non-linear relationships between variables.
    """)
    x_var = st.selectbox("Select X variable", numeric_columns)
    y_var = st.selectbox("Select Y variable", numeric_columns)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data, x=x_var, y=y_var, alpha=0.6)
    ax.set_title(f"{y_var} vs {x_var}")
    st.pyplot(fig)

# Machine Learning Tab
with tabs[3]:
    st.header("Machine Learning Models")
    st.write("""
    **Overview:** This tab allows you to apply machine learning models to your dataset. 
    Features like correlation heatmaps help identify relationships, and models predict outcomes like savings. 
    Use Linear Regression for simple relationships, and ensemble methods like Random Forest for more complex data.
    **Advantages:** Facilitates data-driven decision-making by leveraging predictive algorithms.
    """)
    
    feature_options = numeric_columns.tolist()
    selected_features = st.multiselect("Select features for the model", feature_options, default=feature_options[:2])
    
    if len(selected_features) > 0:
        st.subheader("Correlation Heatmap")
        st.write("""
        **Overview:** This heatmap visualizes the correlation between selected features. 
        Strong correlations (positive or negative) can guide feature selection for models.
        **Advantages:** Ensures the model is built on meaningful relationships.
        """)
        corr_data = data[selected_features]
        corr_matrix = corr_data.corr()
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)

    model_option = st.selectbox("Select a Machine Learning Model", 
                                ["Linear Regression", "Random Forest", "Support Vector Machine", "Gradient Boosting"])
    target = 'savings'
    
    if len(selected_features) > 0:
        X = data[selected_features]
        y = data[target]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        if model_option == "Linear Regression":
            model = LinearRegression()
            st.write("""
            **Linear Regression**: A straightforward, interpretable algorithm suitable for linear relationships.
            """)
        elif model_option == "Random Forest":
            model = RandomForestRegressor(random_state=42)
            st.write("""
            **Random Forest**: A versatile, robust ensemble method effective for both linear and nonlinear data.
            """)
        elif model_option == "Support Vector Machine":
            model = SVR()
            st.write("""
            **Support Vector Machine**: Ideal for datasets with complex relationships and small size.
            """)
        elif model_option == "Gradient Boosting":
            model = GradientBoostingRegressor(random_state=42)
            st.write("""
            **Gradient Boosting**: Delivers high accuracy for structured data by minimizing errors iteratively.
            """)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        metrics_dict = {"RMSE": rmse, "R2 Score": r2, "Mean Absolute Error (MAE)": mae}
        
        st.subheader(f"{model_option} Model Metrics")
        st.write(metrics_dict)
        
        # Graphical Analysis: Predicted vs. Actual
        st.subheader("Graphical Analysis: Predicted vs. Actual Values")
        st.write("""
        **Overview:** The scatterplot shows predicted values against actual ones. 
        Points closer to the diagonal line indicate higher accuracy.
        **Advantages:** Quickly identifies under- or over-predictions by the model.
        """)
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, ax=ax1)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax1.set_title("Predicted vs. Actual Values")
        ax1.set_xlabel("Actual Values")
        ax1.set_ylabel("Predicted Values")
        st.pyplot(fig1)
        
        # Line Graph: Changes in Predicted vs. Actual
        st.subheader("Line Graph: Changes in Predicted vs. Actual Values")
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        indices = np.arange(len(y_test))
        ax2.plot(indices, y_test, label="Actual Values", marker='o', color='blue', alpha=0.7)
        ax2.plot(indices, y_pred, label="Predicted Values", marker='x', color='red', alpha=0.7)
        ax2.set_title("Comparison of Actual and Predicted Values")
        ax2.set_xlabel("Test Data Index")
        ax2.set_ylabel("Savings")
        ax2.legend()
        st.pyplot(fig2)

        # User Input Section for Prediction
        st.subheader("Predict Savings Based on Input Data")
        user_input = {}
        for feature in selected_features:
            user_input[feature] = st.number_input(f"Enter {feature}", value=float(data[feature].mean()))
        
        if st.button("Predict Savings"):
            input_df = pd.DataFrame([user_input])
            input_scaled = scaler.transform(input_df)
            predicted_savings = model.predict(input_scaled)
            st.success(f"Predicted Savings: {predicted_savings[0]:,.2f}")
    else:
        st.warning("Please select at least one feature for the model.")