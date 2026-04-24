#  Hospital Readmission Risk Prediction System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace-blue)](https://huggingface.co/spaces/SamadhiDBS/hospital-readmission)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.ai/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> AI-powered machine learning system that predicts whether a patient will be readmitted to the hospital within **30 days of discharge**.

---

##  Overview

Hospital readmissions are a major challenge in healthcare systems. Many readmissions can be prevented through timely follow-up care, discharge planning, medication review, and patient support.

This project uses **Machine Learning + Healthcare Analytics** to identify high-risk patients **before they leave the hospital**.

###  Goals

- Reduce avoidable readmissions  
- Improve patient outcomes  
- Optimize hospital resources  
- Lower financial penalties  
- Prioritize high-risk patients  

---

##  Live Demo

👉 https://huggingface.co/spaces/SamadhiDBS/hospital-readmission

---

##  Key Results

| Metric | Value |
|--------|-------|
| Recall | **60.6%** |
| ROC-AUC | **0.67** |
| Precision | **17.8%** |
| Patients Analyzed | **101,766** |
| Prediction Time | **<1 second** |

---

##  Real World Impact

✅ Detects **6 out of 10** likely readmissions before discharge  
✅ Helps hospitals take early action  
✅ Potential annual savings of **$100M+** for large networks  
✅ Better quality of care and patient safety  

---

##  Application Features

- Interactive Streamlit dashboard  
- Instant prediction results  
- Risk score percentage  
- Low / Medium / High risk levels  
- Action recommendations  
- Fast and simple UI  

---



##  Tech Stack

| Category | Technologies |
|---------|--------------|
| Programming | Python |
| Machine Learning | XGBoost, Scikit-learn |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Explainability | SHAP |
| Deployment | Streamlit, Hugging Face |

---



##  Model Performance

### Confusion Matrix

| Actual / Predicted | No Readmission | Readmission |
|-------------------|---------------|------------|
| No Readmission | 17,611 | 9,512 |
| Readmission | 1,344 | 2,063 |

### Metrics

| Metric | Score |
|-------|------|
| Recall | 60.6% |
| Precision | 17.8% |
| F1 Score | 0.275 |
| ROC-AUC | 0.671 |

---

##  Top Important Features

1. Previous inpatient admissions  
2. Discharge disposition  
3. Total healthcare visits  
4. Insurance / payer status  
5. Number of diagnoses  

---

##  Example Predictions

| Patient Type | Risk Score | Risk Level |
|------------|-----------|-----------|
| Healthy | 26.2% | 🟢 Low |
| Mild | 54.1% | 🟡 Medium |
| Severe | 67.9% | 🟡 Medium |
| Critical | 73.2% | 🔴 High |

---

##  Who Can Use This?

- Doctors  
- Nurses  
- Case Managers  
- Hospital Administrators  
- Healthcare Analysts  

---

##  Limitations

- Moderate precision may create false positives  
- Based on one public dataset  
- Needs hospital validation before deployment  
- Supports clinicians, not replacement for doctors  

---

##  Future Improvements

- Deep learning models  
- Real-time EHR integration  
- Multi-hospital federated learning  
- Mobile dashboard for nurses  
- Better precision optimization  

---

##  Dataset

Public healthcare dataset containing:

- 101,766 patient encounters  
- Demographics  
- Diagnoses  
- Medication records  
- Readmission labels  

---

##  Contributing

1. Fork repository  
2. Create new branch  
3. Commit changes  
4. Push branch  
5. Open Pull Request  

---

##  License

MIT License

---

##  Author

**Samadhi DBS**

GitHub: https://github.com/SamadhiDBS  
LinkedIn: https://www.linkedin.com/in/sithumi-samadhi-0746b6292

---

##  Support

If you found this useful:

⭐ Star this repository  
🔁 Share with others  
💬 Give feedback  

