# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================

class DataPreprocessor:
    """Handle all data preprocessing steps"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def preprocess(self, df, fit=True):
        """
        Complete preprocessing pipeline
        """
        df = df.copy()
        
        # 1. Handle missing values
        df['lab_result_1'].fillna(df['lab_result_1'].median(), inplace=True)
        
        # 2. Encode categorical variables
        if fit:
            self.label_encoders['gender'] = LabelEncoder()
            df['gender_encoded'] = self.label_encoders['gender'].fit_transform(df['gender'])
        else:
            df['gender_encoded'] = self.label_encoders['gender'].transform(df['gender'])
        
        # 3. Feature Engineering
        df['comorbidity_score'] = (
            df['diabetes'] + 
            df['heart_disease'] + 
            df['kidney_disease']
        )
        
        df['high_risk_flag'] = (
            (df['age'] > 65) & 
            (df['num_prior_admissions'] > 2)
        ).astype(int)
        
        df['medication_complexity'] = df['num_medications'] / (df['length_of_stay'] + 1)
        
        # 4. Select features for modeling
        feature_cols = [
            'age', 'gender_encoded', 'num_prior_admissions', 'length_of_stay',
            'num_diagnoses', 'num_medications', 'num_procedures',
            'emergency_admission', 'diabetes', 'heart_disease', 'kidney_disease',
            'lab_result_1', 'lab_result_2', 'socioeconomic_score',
            'comorbidity_score', 'high_risk_flag', 'medication_complexity'
        ]
        
        X = df[feature_cols]
        y = df['readmitted_30days'] if 'readmitted_30days' in df.columns else None
        
        # 5. Scale numeric features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.feature_names = feature_cols
        else:
            X_scaled = self.scaler.transform(X)
        
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
        
        return X_scaled, y
    
    def save(self, filepath='preprocessor.pkl'):
        """Save preprocessor for deployment"""
        joblib.dump(self, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath='preprocessor.pkl'):
        """Load preprocessor"""
        return joblib.load(filepath)
