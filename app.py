import streamlit as st
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

hide_github_info = """
    <style>
    /* 隐藏侧边栏的 github 信息部分 */
    [data-testid="stSidebar"] a:first-child {
        display: none;
    }
    </style>
"""
st.markdown(hide_github_info, unsafe_allow_html=True)



default_values = [
    0.0, 0, 127.2635061594643, 10.60798788070679, 0.7694719318184242, 0.9660658108666664,
    258.5477265864002, 0.0014467183428345, 0.1334740300172912, -74.97466146546846, 0.0313649742330879
]

feature_names = [
    'gender', 'tumor_LN_metastasis',
    'original_shape_Maximum2DDiameterColumn',
    'log-sigma-3-0-mm-3D_firstorder_90Percentile',
    'original_shape_Sphericity', 'log-sigma-3-0-mm-3D_glcm_Idn',
    'wavelet-LHL_firstorder_Maximum', 'wavelet-HHL_gldm_SmallDependenceLowGrayLevelEmphasis',
    'wavelet-HHH_gldm_DependenceNonUniformityNormalized',
    'original_firstorder_Minimum',
    'log-sigma-5-0-mm-3D_ngtdm_Strength'
]

st.title("              ccRCC ISUP Grade Group Predictor               ")

col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("Gender (0=Female, 1=Male):", options=[0, 1])
    tumor_LN_metastasis = st.selectbox("Tumor LN Metastasis (0=No, 1=Yes, 2=Unknown):", options=[0, 1, 2], index=default_values[1])
    original_shape_Maximum2DDiameterColumn = st.number_input("Original_shape_Maximum2DDiameterColumn:", value=default_values[2])
    log_sigma_3_0_mm_3D_firstorder_90Percentile = st.number_input("log-sigma-3-0-mm-3D_firstorder_90Percentile:", value=default_values[3])

with col2:
    original_shape_Sphericity = st.number_input("Original_shape_Sphericity:", value=default_values[4])
    log_sigma_3_0_mm_3D_glcm_Idn = st.number_input("log-sigma-3-0-mm-3D_glcm_Idn:", value=default_values[5])
    wavelet_LHL_firstorder_Maximum = st.number_input("Wavelet-LHL_firstorder_Maximum:", value=default_values[6])
    wavelet_HHL_gldm_SmallDependenceLowGrayLevelEmphasis = st.number_input("Wavelet-HHL_gldm_SmallDependenceLowGrayLevelEmphasis:", value=default_values[7])

with col3:
    wavelet_HHH_gldm_DependenceNonUniformityNormalized = st.number_input("Wavelet-HHH_gldm_DependenceNonUniformityNormalized:", value=default_values[8])
    original_firstorder_Minimum = st.number_input("Original_firstorder_Minimum:", value=default_values[9])
    log_sigma_5_0_mm_3D_ngtdm_Strength = st.number_input("log-sigma-5-0-mm-3D_ngtdm_Strength:", value=default_values[10])

feature_values = [
    gender,
    tumor_LN_metastasis,
    original_shape_Maximum2DDiameterColumn,
    log_sigma_3_0_mm_3D_firstorder_90Percentile,
    original_shape_Sphericity,
    log_sigma_3_0_mm_3D_glcm_Idn,
    wavelet_LHL_firstorder_Maximum,
    wavelet_HHL_gldm_SmallDependenceLowGrayLevelEmphasis,
    wavelet_HHH_gldm_DependenceNonUniformityNormalized,
    original_firstorder_Minimum,
    log_sigma_5_0_mm_3D_ngtdm_Strength
]
features = np.array([feature_values])

if st.button("Predict"):
    model = joblib.load('save_lgb.pkl')

    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    print(predicted_proba)

    st.write(f"**Predicted Class:** {predicted_class}")
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = f"**The model predicts probability is {probability:.1f}%. The prediction is for +.    >>    isupGG 3-4**"
    else:
        advice = f"**The model predicts probability is {probability:.1f}%. The prediction is for -.    >>    isupGG 1-2**"
    st.write(advice)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(pd.DataFrame([feature_values], columns=feature_names))
    shap.plots.waterfall(shap_values[-1], max_display=12, show=True)
    plt.savefig("shap.png", bbox_inches='tight', format='png')
    st.image("shap.png")
