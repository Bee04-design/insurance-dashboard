from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import logging
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import folium
from folium.plugins import HeatMap
import geopandas
from shapely.geometry import Point
from streamlit_folium import st_folium
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.utils import resample
import weasyprint

# Setup Logging with Version Control
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
MODEL_VERSION = "v1.0"
DATASET_VERSION = "2025-05-20"
MODEL_LAST_TRAINED = "2025-05-20 12:10:00"  # Updated to current time

# Define save_dir globally
save_dir = './'
os.makedirs(save_dir, exist_ok=True)

# Page Setup for Wide Layout
st.set_page_config(page_title="Insurance Risk Dashboard", page_icon="📊", layout="wide")

# Title and Version Info
st.title("Insurance Risk Streamlit Dashboard")
st.markdown(f"_Prototype v0.4.6 | Model: {MODEL_VERSION} | Dataset: {DATASET_VERSION} | Last Trained: {MODEL_LAST_TRAINED}_")

# Sidebar for File Upload
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Choose a file (eswatini_insurance_final_dataset.csv)")

if uploaded_file is None:
    st.info("Upload a file through config", icon="ℹ️")
    st.stop()

# Load Data
@st.cache_data
def load_data(path):
    logger.info("Loading data...")
    df = pd.read_csv(path)
    logger.info("Data loaded successfully")
    return df

try:
    df = load_data(uploaded_file)
except Exception as e:
    st.error(f"Dataset loading failed: {str(e)}")
    logger.error(f"Dataset loading failed: {str(e)}")
    st.stop()

# Data Preprocessing with Dynamic Customer Segmentation
missing_values = df.isna().sum().sum()
df['claim_risk'] = (df['claim_amount_SZL'] >= df['claim_amount_SZL'].quantile(0.75)).astype(int)
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna('Unknown', inplace=True)

# Convert date columns to numeric features
date_cols = ['policy_start_date', 'claim_date']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        df[f'{col}_day'] = df[col].dt.day
        df = df.drop(columns=[col])

# Dynamic Customer Segmentation using K-means
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
X_segment = df[numeric_cols].drop(columns=['claim_risk'], errors='ignore')
kmeans = KMeans(n_clusters=4, random_state=42)
df['customer_segment'] = kmeans.fit_predict(X_segment).astype(str)
categorical_cols = ['claim_type', 'gender', 'location', 'policy_type', 'insurance_provider', 'customer_segment']
for col in df.columns:
    if df[col].dtype == 'object' and col not in date_cols and col not in ['claim_amount_SZL', 'claim_risk']:
        if col not in categorical_cols:
            categorical_cols.append(col)
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

# Split features and target with balancing
X = df_encoded.drop(columns=['claim_amount_SZL', 'claim_risk'])
y = df_encoded['claim_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
train_data = pd.concat([X_train, y_train], axis=1)
majority = train_data[train_data.claim_risk == 0]
minority = train_data[train_data.claim_risk == 1]
minority_oversampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
train_data_balanced = pd.concat([majority, minority_oversampled])
X_train_balanced = train_data_balanced.drop(columns=['claim_risk'])
y_train_balanced = train_data_balanced['claim_risk']

# Train Random Forest Model
rf = RandomForestClassifier(
    n_estimators=300,
    class_weight={0: 1.0, 1: 2.5},
    max_depth=15,
    min_samples_leaf=5,
    random_state=42
)
rf.fit(X_train_balanced, y_train_balanced)
y_pred_rf = rf.predict(X_test)
logger.info("Random Forest model trained and evaluated.")

# Model Metrics
report = classification_report(y_test, y_pred_rf, output_dict=True)
recall_class_1 = report['1']['recall']
fpr, tpr, _ = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)
if recall_class_1 > 0.39:
    joblib.dump(rf, '/content/rf_model.pkl')
    !cp /content/rf_model.pkl ./  # Adjusted for Streamlit Cloud
    logger.info(f"Model saved. Recall for class 1: {recall_class_1}")
else:
    logger.info(f"Model not saved. Recall for class 1: {recall_class_1} (below 0.39 threshold)")

# KPI Cards
st.header("Key Performance Indicators")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
total_policies = len(df)
high_risk_percent = (df['claim_risk'].mean() * 100)
kpi1.metric("Total Policies", total_policies)
kpi2.metric("% High-Risk Policies", f"{high_risk_percent:.1f}%")
kpi3.metric("Model AUC", f"{roc_auc:.2f}")
kpi4.metric("Missing Values Imputed", missing_values)

# Map Functions with Segmentations
def init_map(center=(-26.5, 31.5), zoom_start=7, map_type="cartodbpositron"):
    return folium.Map(location=center, zoom_start=zoom_start, tiles=map_type)

def create_point_map(df):
    df[['Latitude', 'Longitude']] = df[['Latitude', 'Longitude']].apply(pd.to_numeric, errors='coerce')
    df['coordinates'] = df[['Latitude', 'Longitude']].values.tolist()
    df['coordinates'] = df['coordinates'].apply(Point)
    df = geopandas.GeoDataFrame(df, geometry='coordinates')
    df = df.dropna(subset=['Latitude', 'Longitude', 'coordinates'])
    return df

def plot_from_df(df, folium_map, selected_risk_levels, selected_regions, selected_segments):
    region_coords = {
        'Lubombo': (-26.3, 31.8),
        'Hhohho': (-26.0, 31.1),
        'Manzini': (-26.5, 31.4),
        'Shiselweni': (-27.0, 31.3)
    }
    # Aggregate risk by region and segment
    risk_by_region_segment = df.groupby(['location', 'customer_segment'])['claim_risk'].mean().reset_index()
    risk_by_region_segment = risk_by_region_segment[risk_by_region_segment['location'].isin(region_coords.keys())]
    risk_by_region_segment['Latitude'] = risk_by_region_segment['location'].map(lambda x: region_coords[x][0])
    risk_by_region_segment['Longitude'] = risk_by_region_segment['location'].map(lambda x: region_coords[x][1])
    risk_by_region_segment['risk_level'] = pd.qcut(risk_by_region_segment['claim_risk'], 3, labels=['Low', 'Medium', 'High'], duplicates='drop')

    # Apply filters
    if selected_risk_levels:
        risk_by_region_segment = risk_by_region_segment[risk_by_region_segment['risk_level'].isin(selected_risk_levels)]
    if selected_regions:
        risk_by_region_segment = risk_by_region_segment[risk_by_region_segment['location'].isin(selected_regions)]
    if selected_segments:
        risk_by_region_segment = risk_by_region_segment[risk_by_region_segment['customer_segment'].isin(selected_segments)]

    # Plot markers with segment-specific styling
    segment_styles = {
        '0': {'radius': 10, 'color': '#1f77b4'},
        '1': {'radius': 12, 'color': '#ff7f0e'},
        '2': {'radius': 14, 'color': '#2ca02c'},
        '3': {'radius': 16, 'color': '#d62728'}
    }
    for i, row in risk_by_region_segment.iterrows():
        style = segment_styles.get(row['customer_segment'], {'radius': 10, 'color': '#1f77b4'})
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=style['radius'],
            color=style['color'],
            fill=True,
            fill_color=style['color'],
            fill_opacity=0.7,
            tooltip=f"{row['location']} (Segment {row['customer_segment']}): {row['risk_level']} Risk ({row['claim_risk']*100:.1f}%)"
        ).add_to(folium_map)

    # Add heatmap for high-risk claims
    heat_data = [[row['Latitude'], row['Longitude']] for _, row in df.iterrows() if row['claim_risk'] == 1]
    HeatMap(heat_data, radius=15).add_to(folium_map)
    return folium_map

@st.cache_data
def load_map(df, selected_risk_levels, selected_regions, selected_segments):
    m = init_map()
    m = plot_from_df(df, m, selected_risk_levels, selected_regions, selected_segments)
    return m

# Section 1: Prediction
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.header("Predict Claim Risk")
    input_data = {}
    for col in df.columns:
        if col in ['claim_amount_SZL', 'claim_risk'] or col in categorical_cols or col in date_cols:
            continue
        try:
            if df[col].dtype in ['int64', 'float64']:
                input_data[col] = st.slider(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
            else:
                input_data[col] = st.selectbox(f"{col}", df[col].unique())
        except Exception as e:
            st.warning(f"Error with {col}: {str(e)}. Using default value.")
            input_data[col] = 0 if df[col].dtype in ['int64', 'float64'] else df[col].mode()[0]

    for col in categorical_cols:
        input_data[col] = st.selectbox(f"{col}", df[col].unique())

    for col in date_cols:
        if col in df.columns:
            continue
        input_data[f'{col}_year'] = st.slider(f"{col} Year", 2000, 2025, 2020)
        input_data[f'{col}_month'] = st.slider(f"{col} Month", 1, 12, 6)
        input_data[f'{col}_day'] = st.slider(f"{col} Day", 1, 31, 15)

    if st.button("Predict"):
        logger.info("Predict button clicked")
        try:
            input_df = pd.DataFrame([input_data])
            expected_features = rf.feature_names_in_ if hasattr(rf, 'feature_names_in_') else X_test.columns
            input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=False)
            for col in expected_features:
                if col not in input_df_encoded.columns:
                    input_df_encoded[col] = 0
            input_df_encoded = input_df_encoded[expected_features]
            pred = rf.predict(input_df_encoded)[0]
            prob = rf.predict_proba(input_df_encoded)[0][1]
            with col2:
                st.markdown(f"**Prediction**: {'High Risk' if pred == 1 else 'Low Risk'}")
            with col3:
                st.metric("Probability (High Risk)", f"{prob*100:.1f}%")
                st.progress(prob)
            logger.info(f"Prediction: {pred}, Probability: {prob}")
            pred_log = pd.DataFrame({
                'timestamp': [pd.Timestamp.now()],
                'prediction': ['High Risk' if pred == 1 else 'Low Risk'],
                'probability_high_risk': [prob]
            })
            log_file = os.path.join(save_dir, 'prediction_log.csv')
            if os.path.exists(log_file):
                pred_log.to_csv(log_file, mode='a', header=False, index=False)
            else:
                pred_log.to_csv(log_file, index=False)
            logger.info("Prediction saved to prediction_log.csv")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            logger.error(f"Prediction failed: {str(e)}")

# Section 2: Model Performance and Risk Trends
col4, col5, col6 = st.columns([1, 1, 1])
with col4:
    st.header("Model Performance")
    try:
        y_pred = rf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Risk', 'High Risk'], yticklabels=['Low Risk', 'High Risk'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        st.pyplot(fig_cm)
        logger.info("Confusion matrix rendered")
    except Exception as e:
        st.error(f"Performance plotting failed: {str(e)}")
        logger.error(f"Performance plotting failed: {str(e)}")

with col5:
    st.header("ROC Curve")
    try:
        fpr, tpr, _ = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
        fig_roc = px.line(x=fpr, y=tpr, title=f'ROC Curve (AUC = {roc_auc:.2f})', labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'})
        fig_roc.add_scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'))
        fig_roc.update_layout(height=300)
        st.plotly_chart(fig_roc, use_container_width=True)
        st.text(f"Recall for High Risk: {recall_class_1:.2f}")
        logger.info("ROC curve rendered")
    except Exception as e:
        st.error(f"ROC plotting failed: {str(e)}")
        logger.error(f"ROC plotting failed: {str(e)}")

with col6:
    st.header("Risk Trend Over Time")
    try:
        log_file = os.path.join(save_dir, 'prediction_log.csv')
        if os.path.exists(log_file):
            pred_log = pd.read_csv(log_file)
            pred_log['timestamp'] = pd.to_datetime(pred_log['timestamp'])
            fig_trend = px.line(pred_log, x='timestamp', y='probability_high_risk', title="High Risk Probability Trend", markers=True)
            fig_trend.update_layout(height=300)
            st.plotly_chart(fig_trend, use_container_width=True)
            logger.info("Risk trend plot rendered")
        else:
            st.info("No prediction history available to display trends.")
    except Exception as e:
        st.error(f"Risk trend plotting failed: {str(e)}")
        logger.error(f"Risk trend plotting failed: {str(e)}")

# Section 3: Feature Importance and Risk Distributions
col7, col8, col9 = st.columns([1, 1, 1])
with col7:
    st.header("Risk Driver Insights (SHAP)")
    with st.spinner("Computing SHAP values..."):
        try:
            explainer = shap.TreeExplainer(rf)
            sample_data = X_test.sample(50, random_state=42)
            sample_encoded = pd.get_dummies(sample_data, columns=[col for col in categorical_cols if col in sample_data.columns], drop_first=False)
            expected_features = rf.feature_names_in_ if hasattr(rf, 'feature_names_in_') else X_test.columns
            for col in expected_features:
                if col not in sample_encoded.columns:
                    sample_encoded[col] = 0
            sample_encoded = sample_encoded[expected_features].values
            shap_values = explainer.shap_values(sample_encoded)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            shap_values = np.array(shap_values).reshape(-1, len(expected_features))
            st.subheader("Features Used in SHAP Analysis")
            st.write(list(expected_features))
            fig_shap = plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, sample_encoded, feature_names=expected_features, max_display=5, show=False, plot_type="bar")
            if plt.gcf().axes:
                plt.title('Top Features for High Risk')
                plt.tight_layout()
                st.pyplot(fig_shap)
                plt.savefig('shap_plot.png')
            shap_df = pd.DataFrame({'Feature': expected_features, 'SHAP Value': np.abs(shap_values).mean(axis=0)}).sort_values(by='SHAP Value', ascending=False).head(5)
            st.session_state['shap_df'] = shap_df
            logger.info("SHAP plot rendered")
        except Exception as e:
            st.error(f"SHAP plot failed: {str(e)}")
            logger.error(f"SHAP plot failed: {str(e)}")

with col8:
    st.header("Risk by Location")
    try:
        risk_by_location = df.groupby('location')['claim_risk'].mean().reset_index()
        risk_by_location['claim_risk'] *= 100
        fig_loc = px.bar(risk_by_location, x='location', y='claim_risk', title="Average Risk by Location (%)", color='claim_risk', color_continuous_scale='Blues')
        fig_loc.update_layout(height=300)
        st.plotly_chart(fig_loc, use_container_width=True)
        logger.info("Risk by location plot rendered")
    except Exception as e:
        st.error(f"Risk by location plotting failed: {str(e)}")
        logger.error(f"Risk by location plotting failed: {str(e)}")

with col9:
    st.header("Risk by Claim Type")
    try:
        risk_by_claim_type = df.groupby('claim_type')['claim_risk'].mean().reset_index()
        risk_by_claim_type['claim_risk'] *= 100
        fig_claim = px.bar(risk_by_claim_type, x='claim_type', y='claim_risk', title="Average Risk by Claim Type (%)", color='claim_risk', color_continuous_scale='Blues')
        fig_claim.update_layout(height=300)
        st.plotly_chart(fig_claim, use_container_width=True)
        logger.info("Risk by claim type plot rendered")
    except Exception as e:
        st.error(f"Risk by claim type plotting failed: {str(e)}")
        logger.error(f"Risk by claim type plotting failed: {str(e)}")

# Section 4: Segmentation Drill-down
col10, col11 = st.columns([1, 1])
with col10:
    st.header("Customer Segment Drill-down")
    segment = st.selectbox("Select Customer Segment", df['customer_segment'].unique())
    segment_data = df[df['customer_segment'] == segment]
    try:
        # Check for variance in claim_amount_SZL to avoid binning issues
        if segment_data['claim_amount_SZL'].nunique() > 1:
            fig_segment_trend = px.histogram(segment_data, x='claim_amount_SZL', color='claim_risk', title=f"Claim Amount Distribution in {segment}", nbins=20)
        else:
            raise ValueError("Not enough variance in data for histogram")
        fig_segment_trend.update_layout(height=300)
        st.plotly_chart(fig_segment_trend, use_container_width=True)
    except ValueError as e:
        st.warning(f"Histogram failed: {str(e)}. Using bar plot instead.")
        fig_segment_trend = px.bar(segment_data.groupby('claim_risk').size().reset_index(name='Count'), x='claim_risk', y='Count', title=f"Risk Distribution in {segment}")
        fig_segment_trend.update_layout(height=300)
        st.plotly_chart(fig_segment_trend, use_container_width=True)
    logger.info(f"Segment trend plot rendered for {segment}")

with col11:
    st.header("Top Features for Segment")
    try:
        segment_encoded = pd.get_dummies(segment_data.drop(columns=['claim_amount_SZL', 'claim_risk']), columns=categorical_cols, drop_first=False)
        for col in expected_features:
            if col not in segment_encoded.columns:
                segment_encoded[col] = 0
        segment_encoded = segment_encoded[expected_features].values
        shap_values_segment = explainer.shap_values(segment_encoded)
        if isinstance(shap_values_segment, list):
            shap_values_segment = shap_values_segment[1]
        shap_values_segment = np.array(shap_values_segment).reshape(-1, len(expected_features))
        fig_shap_segment = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_segment, segment_encoded, feature_names=expected_features, max_display=5, show=False, plot_type="bar")
        if plt.gcf().axes:
            plt.title(f'Top Features for {segment}')
            plt.tight_layout()
            st.pyplot(fig_shap_segment)
        logger.info(f"SHAP plot for segment {segment} rendered")
    except Exception as e:
        st.error(f"SHAP plot for segment failed: {str(e)}")
        logger.error(f"SHAP plot for segment failed: {str(e)}")

# Section 5: Interactive Eswatini Risk Map with Segmentations
col12 = st.columns([3])[0]
with col12:
    st.header("Interactive Eswatini Risk Map with Segmentations")
    risk_levels = st.multiselect("Filter by Risk Level", ['Low', 'Medium', 'High'], default=['Low', 'Medium', 'High'])
    regions = st.multiselect("Filter by Region", ['Lubombo', 'Hhohho', 'Manzini', 'Shiselweni'], default=['Lubombo', 'Hhohho', 'Manzini', 'Shiselweni'])
    customer_segments = st.multiselect("Filter by Customer Segment", df['customer_segment'].unique(), default=df['customer_segment'].unique())
    try:
        m = load_map(df, risk_levels, regions, customer_segments)
        map_data = st_folium(m, height=500, width=1000, key="eswatini_map")
        selected_region = map_data.get('last_object_clicked_tooltip', '').split(':')[0].strip() if map_data.get('last_object_clicked_tooltip') else None

        if selected_region:
            st.subheader(f"Risk Analysis for {selected_region}")
            region_data = df[df['location'] == selected_region]
            if not region_data.empty:
                try:
                    if region_data['claim_amount_SZL'].nunique() > 1:
                        fig_region_dist = px.histogram(region_data, x='claim_amount_SZL', color='claim_risk', title=f"Claim Amount Distribution in {selected_region}", nbins=20)
                    else:
                        raise ValueError("Not enough variance in data for histogram")
                    fig_region_dist.update_layout(height=300)
                    st.plotly_chart(fig_region_dist, use_container_width=True)
                except ValueError as e:
                    st.warning(f"Histogram failed: {str(e)}. Using bar plot instead.")
                    fig_region_dist = px.bar(region_data.groupby('claim_risk').size().reset_index(name='Count'), x='claim_risk', y='Count', title=f"Risk Distribution in {selected_region}")
                    fig_region_dist.update_layout(height=300)
                    st.plotly_chart(fig_region_dist, use_container_width=True)
                sample_region = region_data.sample(min(20, len(region_data)), random_state=42)
                sample_encoded_region = pd.get_dummies(sample_region.drop(columns=['claim_amount_SZL', 'claim_risk']), columns=categorical_cols, drop_first=False)
                for col in expected_features:
                    if col not in sample_encoded_region.columns:
                        sample_encoded_region[col] = 0
                sample_encoded_region = sample_encoded_region[expected_features].values
                shap_values_region = explainer.shap_values(sample_encoded_region)
                if isinstance(shap_values_region, list):
                    shap_values_region = shap_values_region[1]
                shap_values_region = np.array(shap_values_region).reshape(-1, len(expected_features))
                fig_shap_region = plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values_region, sample_encoded_region, feature_names=expected_features, max_display=5, show=False, plot_type="bar")
                if plt.gcf().axes:
                    plt.title(f'Top Features for High Risk in {selected_region}')
                    plt.tight_layout()
                    st.pyplot(fig_shap_region)
            else:
                st.warning(f"No data available for {selected_region}.")
        logger.info("Interactive map and region analysis rendered")
    except Exception as e:
        st.error(f"Map rendering or analysis failed: {str(e)}")
        logger.error(f"Map rendering or analysis failed: {str(e)}")

# Section 6: Downloadable Reports and Data
st.header("Download Reports and Data")
col13, col14, col15, col16 = st.columns(4)
with col13:
    st.download_button("Download Cleaned Data", data=df.to_csv(index=False), file_name="cleaned_data.csv")
with col14:
    predictions_df = X_test.copy()
    predictions_df['Predicted_Risk'] = y_pred_rf
    st.download_button("Download Predictions", data=predictions_df.to_csv(index=False), file_name="predictions.csv")
with col15:
    if 'shap_df' in st.session_state:
        st.download_button("Download SHAP Analysis (CSV)", data=st.session_state['shap_df'].to_csv(index=False), file_name="shap_analysis.csv")
    if os.path.exists('shap_plot.png'):
        with open('shap_plot.png', 'rb') as f:
            st.download_button("Download SHAP Plot (PNG)", data=f, file_name="shap_plot.png")
with col16:
    try:
        html_content = f"""
        <h1>Insurance Risk Dashboard Report</h1>
        <p><strong>Total Policies:</strong> {total_policies}</p>
        <p><strong>% High-Risk Policies:</strong> {high_risk_percent:.1f}%</p>
        <p><strong>Model AUC:</strong> {roc_auc:.2f}</p>
        <h2>Risk by Location</h2>
        {risk_by_location.to_html()}
        """
        weasyprint.HTML(string=html_content).write_pdf('report.pdf')
        with open('report.pdf', 'rb') as f:
            st.download_button("Download Full Report (PDF)", data=f, file_name="report.pdf")
    except Exception as e:
        st.warning(f"PDF generation failed: {str(e)}. Ensure WeasyPrint is installed and configured.")
        logger.error(f"PDF generation failed: {str(e)}")

# Notes
st.markdown("**Note**: Ensure the dataset is available. Risk map uses claim risk data to highlight high-risk areas.", unsafe_allow_html=True)
st.markdown(f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", unsafe_allow_html=True)
