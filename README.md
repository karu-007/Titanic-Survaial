# Titanic Survival Predictor Dashboard

## Simple Overview
An interactive dashboard that predicts whether a passenger would survive the Titanic disaster based on user input using a machine learning model.

---

## Description
This project is a machine learning-based web application that allows users to input passenger details such as age, class, sex, fare, and more to predict survival outcomes from the Titanic dataset. The model is built using CatBoost and deployed through a Streamlit dashboard for an interactive user experience. The application also provides survival probability and a visual pie chart to help users better understand the prediction. This project demonstrates end-to-end data science workflow including data preprocessing, model building, and deployment.

---

## Getting Started

### Dependencies
Before running the project, make sure you have:

- Python 3.8 or higher  
- Windows 10 / macOS / Linux  
- Required libraries:
  - streamlit  
  - catboost  
  - pandas  
  - plotly  

Install dependencies using:

```bash
pip install streamlit catboost pandas plotly
```

## Executing Program
```bash
streamlit run app.py
```

## Help
### Common Issues
- Steamlit is not recognized
    - Fix: 
        ```bash
        python -m streamlit run app.py
        ```
- Model file not found
    - Fix: Make sure the catboost cbm file is inside project folder

## Authors
- Karunya Jaghni
- Github: https://github.com/karu-007





