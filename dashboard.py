import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os
import sys

##page configuration
st.set_page_config(
    page_title = "Hospital Readmission Predictor",
    page_icon = "🏥",
    layout = "wide",
    initial_sidebar_state="expanded"
)

####title and header
st.title("AI-Powered Hospital Readmission Risk Predictor")
st.markdown("### Predict if a patient will be readmitted within 30 days")
st.markdown("---")

####load model
@st.cache_resource
def load_model():
    """Load the trained model and feature names"""
    try:
        #try different paths
        if os.path.exists('models/final_hospital_model.pkl'):
            model = joblib.load('models/final_hospital_model.pkl')
            feature_names = joblib.load('models/feature_names.pkl')
        else:
            st.error("Model not found! Please run model training first.")
            st.stop()
        return model, feature_names
    except Exception as e:
        st.error(f"Error loading model : {e}")
        st.stop()

#load the model
with st.spinner("Loading AI model..."):
    model, feature_names = load_model()
st.success("AI Model Ready!")

####side bar --- patient input form
st.sidebar.header("📝 Enter Patient Information")

with st.sidebar:
    st.markdown("### 🏥 Current Hospital Stay")

    col1, col2 = st.columns(2)
    with col1:
        time_in_hospital = st.number_input("Days in Hospital", min_value=1, max_value=30, value=3, help="Length of current stay")
        num_lab_procedures = st.number_input("Lab Procedures", min_value=0, max_value=100, value=45, help="Number of lab tests")

    with col2:
        num_procedures = st.number_input("Other Procedures", min_value=0, max_value=20, value=2, help="Number of procedures")
        num_medications = st.number_input("Medications", min_value=0, max_value=50, value=15, help="Number of medications")

    st.markdown("### 📅 Medical History (Last Year)")

    col3, col4 = st.columns(2)
    with col3:
        number_outpatient = st.number_input("Outpatient Visits", min_value=0, max_value=30, value=0)
        number_emergency = st.number_input("Emergency Visits", min_value=0, max_value=20, value=0)
    with col4:
        number_inpatient = st.number_input("Hospital Stays", min_value=0, max_value=15, value=0)
        number_diagnoses = st.number_input("Diagnoses Count", min_value=1, max_value=20, value=7)

    st.markdown("### 💊 Clinical Details")

    discharge_options = {
        1: "🏠 Home / Self Care",
        2: "🏥 Skilled Nursing Facility",
        3: "🏨 Hospice / Home",
        4: "🔄 Transfer to Another Facility",
        5: "🏡 Home with Home Health Services",
        6: "⚰️ Expired",
        7: "❓ Other / Unknown"
    }

    discharge_disposition = st.selectbox(
        "Discharge Disposition",
        options=list(discharge_options.keys()),
        format_func=lambda x: discharge_options.get(x, f"Code {x}")
    )

    col5, col6 = st.columns(2)
    with col5:
        diabetesMed = st.selectbox("Diabetes Medication", ["No", "Yes"])
    with col6:
        insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])


####predict button
predict_button = st.sidebar.button("🔮 PREDICT READMISSION RISK", type="primary", use_container_width=True)


### function to create feature vector
# def create_feature_vector():
#     """convert user inputs to model features"""

#     #calculate derived features
#     total_visits = number_outpatient + number_emergency + number_inpatient
#     meds_per_day = num_medications / (time_in_hospital + 0.1)
#     procedures_per_day = num_procedures / (time_in_hospital + 0.1)
#     lab_per_day = num_lab_procedures / (time_in_hospital + 0.1)
#     had_emergency = 1 if number_emergency > 0 else 0
#     had_inpatient = 1 if number_inpatient > 0 else 0

#     #create base features
#     features = {
#         'time_in_hospital': time_in_hospital,
#         'num_lab_procedures': num_lab_procedures,
#         'num_procedures': num_procedures,
#         'num_medications': num_medications,
#         'number_outpatient': number_outpatient,
#         'number_emergency': number_emergency,
#         'number_inpatient': number_inpatient,
#         'number_diagnoses': number_diagnoses,
#         'discharge_disposition_id': discharge_disposition,
#         'total_visits': total_visits,
#         'meds_per_day': meds_per_day,
#         'procedures_per_day': procedures_per_day,
#         'lab_per_day': lab_per_day,
#         'had_emergency': had_emergency,
#         'had_inpatient': had_inpatient,
#         'diabetesMed_Yes': 1 if diabetesMed == "Yes" else 0,
#         'insulin_No': 1 if insulin == "No" else 0,
#         'insulin_Steady': 1 if insulin == "Steady" else 0,
#         'insulin_Up': 1 if insulin == "Up" else 0,
#         'insulin_Down': 1 if insulin == "Down" else 0,
#     }

#     #create dataframe
#     df = pd.DataFrame([features])

#     #fill missing columns  -- for featueres not in our simplified inputs
#     for col in feature_names:
#         if col not in df.columns:
#             d[col] = 0

#     #ensure same column order as training
#     df = df[feature_names]

#     return df
def create_feature_vector():
    """Convert user inputs to model features"""
    
    # Calculate derived features
    total_visits = number_outpatient + number_emergency + number_inpatient
    meds_per_day = num_medications / (time_in_hospital + 0.1)
    procedures_per_day = num_procedures / (time_in_hospital + 0.1)
    lab_per_day = num_lab_procedures / (time_in_hospital + 0.1)
    had_emergency = 1 if number_emergency > 0 else 0
    had_inpatient = 1 if number_inpatient > 0 else 0
    
    # Create base features
    features = {
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'num_procedures': num_procedures,
        'num_medications': num_medications,
        'number_outpatient': number_outpatient,
        'number_emergency': number_emergency,
        'number_inpatient': number_inpatient,
        'number_diagnoses': number_diagnoses,
        'discharge_disposition_id': discharge_disposition,
        'total_visits': total_visits,
        'meds_per_day': meds_per_day,
        'procedures_per_day': procedures_per_day,
        'lab_per_day': lab_per_day,
        'had_emergency': had_emergency,
        'had_inpatient': had_inpatient,
        'diabetesMed_Yes': 1 if diabetesMed == "Yes" else 0,
        'insulin_No': 1 if insulin == "No" else 0,
        'insulin_Steady': 1 if insulin == "Steady" else 0,
        'insulin_Up': 1 if insulin == "Up" else 0,
        'insulin_Down': 1 if insulin == "Down" else 0,
    }
    
    # Create DataFrame
    df = pd.DataFrame([features])
    
    # Fill missing columns (for features not in our simplified input)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0  # FIXED: was 'd' instead of 'df'
    
    # Ensure same column order as training
    df = df[feature_names]
    
    return df

#### prediction function
def predict_risk(patient_data):
    """Make prediction and return results"""

    try:
        #get risk probability
        risk_score = model.predict_proba(patient_data)[0,1] * 100

        #determine risk level
        if risk_score >= 70:
            risk_level = "HIGH RISK"
            risk_color = "red"
            risk_icon = "🔴"
            action = "📞 Schedule follow-up within 3 days\n🏠 Arrange home health visit\n💊 Review all medications\n📋 Daily monitoring for 2 weeks"
        elif risk_score >= 40:
            risk_level = "MEDIUM RISK"
            risk_color = "orange"
            risk_icon = "🟡"
            action = "📞 Schedule follow-up within 7 days\n💊 Medication review\n📋 Monitor symptoms\n📞 Weekly check-in calls"
        else:
            risk_level = "LOW RISK"
            risk_color = "green"
            risk_icon = "🟢"
            action = "📋 Standard discharge procedure\n📞 Follow-up call within 14 days\n📝 Regular care plan"

        return risk_score, risk_level, risk_color, risk_icon, action
    except Exception as e:
        st.error(f"Prediction error : {e}")
        return None, None, None, None, None
    

##### display prediction results
if predict_button:
    st.markdown("---")
    st.subheader("📊 PREDICTION RESULTS")

    #create feature vector and predict
    with st.spinner("Analyzing patient data...."):
        patient_data = create_feature_vector()
        risk_score, risk_level, risk_color, risk_icon, action = predict_risk(patient_data)

    if risk_score is not None:
        #create 2 columns fro results
        col1, col2 = st.columns([1, 1])

        #column 1: gauge chart
        with col1:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = risk_score,
                title = {'text': "Readmission Risk Score", 'font': {'size': 20}},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "black"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 40], 'color': '#2ecc71'},
                        {'range': [40, 70], 'color': '#f39c12'},
                        {'range': [70, 100], 'color': '#e74c3c'}
                    ],
                    'threshold': {
                        'line':{'color': "black", 'width':4},
                        'thickness': 0.75,
                        'value' : risk_score

                    }
                }
            ))

            fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

        #column 2--- risk summary
        with col2:
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: {risk_color}10; border: 2px solid {risk_color};">
                <h2 style="text-align: center; margin: 0;">{risk_icon} {risk_level}</h2>
                <h1 style="text-align: center; font-size: 48px; margin: 10px 0;">{risk_score:.1f}%</h1>
                <p style="text-align: center; font-size: 16px;">
                    Out of 100 similar patients, <strong>{int(risk_score)}</strong> would be readmitted.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        #recommended actions
        st.markdown("---")
        st.subheader("📋 RECOMMENDED ACTIONS FOR HOSPITAL STAFF")
        
        st.info(f"""
        {action}
        """)
        
        #risk factors explanation
        st.markdown("---")
        st.subheader("🔍 WHY THIS PATIENT IS AT RISK")  

        #collect risk factors based on input
        risk_factors = []

        if number_inpatient > 0:
            risk_factors.append(f"⚠️ **Previous hospital stays:** {number_inpatient} time(s) - This is the STRONGEST risk factor")
        if number_emergency > 2:
            risk_factors.append(f"⚠️ **Frequent ER visits:** {number_emergency} visits in past year")
        if num_medications > 15:
            risk_factors.append(f"⚠️ **Multiple medications:** {num_medications} prescriptions - Complex care needs")
        if time_in_hospital > 7:
            risk_factors.append(f"⚠️ **Extended hospital stay:** {time_in_hospital} days - Serious condition")
        if discharge_disposition != 1:
            risk_factors.append(f"⚠️ **Discharge to facility:** Not going directly home - Needs extra support")
        if diabetesMed == "Yes":
            risk_factors.append(f"⚠️ **Diabetes:** Requires medication management")
        if insulin != "No":
            risk_factors.append(f"⚠️ **Insulin treatment:** Needs careful monitoring")
        if number_diagnoses > 10:
            risk_factors.append(f"⚠️ **Multiple conditions:** {number_diagnoses} diagnoses")

        if risk_factors:
            for factor in risk_factors:
                st.markdown(factor)
        else:
            st.markdown("✅ No major risk factors identified")
            st.markdown("📋 Standard care recommended")

        #patient summary card
        st.markdown("---")
        st.subheader("📄 PATIENT SUMMARY")

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:
            st.metric("Days in Hospital", time_in_hospital)
            st.metric("Medications", num_medications)
            st.metric("Lab Procedures", num_lab_procedures)
        
        with summary_col2:
            st.metric("Previous Hospital Stays", number_inpatient)
            st.metric("Previous ER Visits", number_emergency)
            st.metric("Previous Outpatient", number_outpatient)
        
        with summary_col3:
            st.metric("Diagnoses", number_diagnoses)
            st.metric("Diabetes Med", diabetesMed)
            st.metric("Insulin", insulin)

        #disclaimer
        st.markdown("---")
        st.caption("⚠️ **Medical Disclaimer:** This is an AI prediction tool. All clinical decisions should be made by qualified healthcare professionals based on complete patient evaluation.")

        #download report button
        report_data = {
            "Risk Score": f"{risk_score:.1f}%",
            "Risk Level": risk_level,
            "Days in Hospital": time_in_hospital,
            "Medications": num_medications,
            "Previous Hospital Stays": number_inpatient,
            "Previous ER Visits": number_emergency,
            "Recommended Action": action.replace("\n", " | ")
        }

        report_df = pd.DataFrame([report_data])
        csv = report_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Patient Report (CSV)",
            data=csv,
            file_name=f"patient_risk_report_{risk_score:.0f}%.csv",
            mime="text/csv",
            use_container_width=True
        )


###### sidebar - footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📖 Quick Guide

1. Enter patient data
2. Click **PREDICT**
3. View risk score
4. Follow recommendations

### 🎯 Risk Levels

| Level | Score | Action |
|-------|-------|--------|
| 🔴 High | 70-100% | 3 days |
| 🟡 Medium | 40-69% | 7 days |
| 🟢 Low | 0-39% | 14 days |

### 💡 Tips
- More accurate data = better predictions
- Include all previous visits
- Update medications
""")

####
#####main page ---- welcome---- whne no prediction still
if not predict_button:
    st.markdown("""
    ## 👋 Welcome to the AI-Powered Readmission Predictor
    
    ### How it works:
    
    1. **Enter patient information** in the sidebar → (left panel)
    2. **Click the PREDICT button** → 
    3. **Get instant risk assessment** → 
    4. **Receive actionable recommendations** → 
    
    ### Why this matters:
    
    - 🏥 **Reduce readmissions** by identifying high-risk patients
    - 💰 **Save costs** by preventing unnecessary hospital stays
    - ❤️ **Improve patient outcomes** with targeted interventions
    - 📊 **Data-driven decisions** using AI
    
    ### Ready to start?
    
    **👉 Fill out the patient information in the left sidebar and click PREDICT!**
    """)

    #show example
    with st.expander("📋 See Example Prediction"):
        st.markdown("""
        **Example Patient:**
        - 5 days in hospital
        - 12 medications
        - 2 previous hospital stays
        - 3 ER visits last year
        
        **Result:** 67% HIGH RISK
        **Action:** Follow-up within 3 days, home health visit
        """)