import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components

# -------------------------------
# Step 1: Load Model and Scaler
# -------------------------------

st.set_page_config(layout="wide")

# Load trained XGBoost model
try:
    with open('best_xgboost_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("`best_xgboost_model.pkl` not found! Please ensure the model file is in the correct directory.")
    st.stop()

# Load MinMaxScaler
try:
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    st.error("`scaler.pkl` not found! Please ensure the scaler file is available.")
    st.stop()

# -------------------------------
# Step 2: Define Features
# -------------------------------

feature_names = [
    'vintage', 'age', 'gender', 'dependents', 'occupation', 'city',
    'customer_nw_category', 'current_balance', 'previous_month_end_balance',
    'average_monthly_balance_prevQ', 'average_monthly_balance_prevQ2',
    'current_month_credit', 'previous_month_credit', 'current_month_debit',
    'previous_month_debit', 'current_month_balance',
    'previous_month_balance', 'days_since_last_transaction'
]

# Columns requiring scaling
scale_vars = [
    'vintage', 'age', 'dependents', 'current_balance', 'previous_month_end_balance',
    'average_monthly_balance_prevQ', 'average_monthly_balance_prevQ2',
    'current_month_credit', 'previous_month_credit', 'current_month_debit',
    'previous_month_debit', 'current_month_balance',
    'previous_month_balance', 'days_since_last_transaction'
]


# Step 3: Sidebar User Inputs (With Defaults)

st.sidebar.image("Pic 1.png", use_container_width=True)  # Display Pic 1
st.sidebar.header("User Inputs")

# Default values for user input (ensures valid predictions)
default_values = {
    "vintage": 60, "age": 35, "gender": "Male", "dependents": 2, "occupation": "salaried",
    "city": "1020", "customer_nw_category": "1", "current_balance": 50000,
    "previous_month_end_balance": 45000, "average_monthly_balance_prevQ": 42000,
    "average_monthly_balance_prevQ2": 40000, "current_month_credit": 15000,
    "previous_month_credit": 13000, "current_month_debit": 12000,
    "previous_month_debit": 11000, "current_month_balance": 40000,
    "previous_month_balance": 43000, "days_since_last_transaction": 30
}

# Collect user inputs
user_inputs = {}
for feature in feature_names:
    if feature in scale_vars:
        user_inputs[feature] = st.sidebar.number_input(feature, value=default_values[feature], step=1)
    else:
        if feature == "gender":
            user_inputs[feature] = st.sidebar.selectbox("Gender", options=["Male", "Female"], index=0)
        elif feature == "occupation":
            user_inputs[feature] = st.sidebar.selectbox("Occupation", options=["salaried", "self-employed", "unemployed"], index=0)
        elif feature == "customer_nw_category":
            user_inputs[feature] = st.sidebar.selectbox("Customer NW Category", options=["1", "2", "3"], index=0)
        elif feature == "city":
            user_inputs[feature] = st.sidebar.selectbox("City", options=["1020", "1030"], index=0)
        else:
            user_inputs[feature] = st.sidebar.text_input(feature, value=default_values[feature])

# Convert to DataFrame
input_data = pd.DataFrame([user_inputs])

# Ensure categorical variables are properly encoded (manual encoding for simplicity)
input_data.replace({
    "gender": {"Male": 1, "Female": 0},
    "occupation": {"salaried": 1, "self-employed": 2, "unemployed": 3},
    "customer_nw_category": {"1": 1, "2": 2, "3": 3},
    "city": {"1020": 1020, "1030": 1030}
}, inplace=True)

# -------------------------------
# Step 4: Apply MinMaxScaler
# -------------------------------

# Ensure all features are numeric before scaling
try:
    input_data_scaled = input_data.copy()
    input_data_scaled[scale_vars] = scaler.transform(input_data[scale_vars])
except ValueError as ve:
    st.error(f"Invalid input! Please check the values of your fields. Details: {ve}")
    st.stop()

# -------------------------------
# Step 5: Prediction
# -------------------------------
st.image("Pic 2.png", use_container_width=True)  # Display Pic 2
st.title("Customer Churn Prediction")

# Page Layout
left_col, right_col = st.columns(2)

with left_col:
    st.header("Feature Importance")
    # Load feature importance data from the Excel file
    feature_importance_df = pd.read_excel("feature_importance.xlsx", usecols=["Feature", "Feature Importance Score"])
    # Plot the feature importance bar chart
    fig = px.bar(
        feature_importance_df.sort_values(by="Feature Importance Score", ascending=True),
        x="Feature Importance Score",
        y="Feature",
        orientation="h",
        title="Feature Importance",
        labels={"Feature Importance Score": "Importance", "Feature": "Features"},
        width=400,  # Set custom width
        height=500  # Set custom height
    )
    st.plotly_chart(fig)

# Right Page: Prediction
with right_col:
    st.header("Prediction")
    if st.button("Predict"):
        try:
            # Get predicted probabilities and label
            probabilities = model.predict_proba(input_data_scaled)[0]
            prediction = model.predict(input_data_scaled)[0]
    
            # Map prediction to label
            prediction_label = "Churned" if prediction == 1 else "Retained"
    
            # Display results
            st.subheader(f"Predicted Value: **{prediction_label}**")
            st.write(f"**Churn Probability:** {probabilities[1]:.2%}")
            st.write(f"**Retention Probability:** {probabilities[0]:.2%}")
    
            # Display result with color indication
            if prediction == 1:
                st.error("High risk of churn! Take action to retain this customer.")
            else:
                st.success("Customer is likely to stay!")
        
        except Exception as e:
            st.error(f" Error in prediction: {e}")

# Step 6: Display Tableau Dashboard
#st.header("Customer Insights Dashboard")
#tableau_url = "https://public.tableau.com/views/Churn_Dashboard_17397175233280/Dashboard1?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link"
#st.markdown(f'<iframe src="{tableau_url}" width="100%" height="600"></iframe>', unsafe_allow_html=True)

embed_code = '''
<div class='tableauPlaceholder' id='viz1739763468407' style='position: relative'><noscript><a href='#'><img alt='Dashboard 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ch&#47;Churn_Dashboard_17397175233280&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Churn_Dashboard_17397175233280&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ch&#47;Churn_Dashboard_17397175233280&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1739763468407');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1100px';vizElement.style.height='607px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1100px';vizElement.style.height='607px';} else { vizElement.style.width='100%';vizElement.style.height='1627px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
''' 
st.header("Customer Insights Dashboard")
components.html(embed_code, height=800)