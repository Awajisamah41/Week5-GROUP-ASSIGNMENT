"""
Patient Readmission Prediction System
Complete implementation for predicting 30-day hospital readmissions
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA GENERATION (Simulated EHR Data)
# ============================================================================

def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic patient data for demonstration
    In production, this would come from actual EHR systems
    """
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(18, 90, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'num_prior_admissions': np.random.poisson(2, n_samples),
        'length_of_stay': np.random.randint(1, 30, n_samples),
        'num_diagnoses': np.random.randint(1, 10, n_samples),
        'num_medications': np.random.randint(1, 20, n_samples),
        'num_procedures': np.random.randint(0, 6, n_samples),
        'emergency_admission': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'diabetes': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'heart_disease': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'kidney_disease': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'lab_result_1': np.random.normal(100, 20, n_samples),  # e.g., glucose
        'lab_result_2': np.random.normal(7, 1.5, n_samples),   # e.g., HbA1c
        'socioeconomic_score': np.random.randint(1, 6, n_samples),  # 1-5 scale
    }
    
    df = pd.DataFrame(data)
    
    # Generate target variable with realistic correlations
    readmit_prob = (
        0.1 +
        0.02 * (df['age'] > 65) +
        0.05 * df['num_prior_admissions'] +
        0.03 * (df['length_of_stay'] > 7) +
        0.04 * df['emergency_admission'] +
        0.03 * df['diabetes'] +
        0.04 * df['heart_disease'] +
        0.05 * df['kidney_disease'] +
        0.02 * (df['num_medications'] > 10) +
        np.random.normal(0, 0.1, n_samples)
    )
    
    df['readmitted_30days'] = (readmit_prob > 0.3).astype(int)
    
    # Introduce some missing values (realistic scenario)
    missing_mask = np.random.random(n_samples) < 0.05
    df.loc[missing_mask, 'lab_result_1'] = np.nan
    
    return df