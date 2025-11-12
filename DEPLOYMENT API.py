# ============================================================================
# 5. DEPLOYMENT API
# ============================================================================

class ReadmissionAPI:
    """
    Deployment-ready API for real-time predictions
    """
    
    def __init__(self, model_path='readmission_model.pkl', 
                 preprocessor_path='preprocessor.pkl'):
        """Initialize API with trained model and preprocessor"""
        self.model = ReadmissionPredictor.load(model_path)
        self.preprocessor = DataPreprocessor.load(preprocessor_path)
        print("API initialized successfully")
    
    def predict_single_patient(self, patient_data):
        """
        Predict readmission risk for a single patient
        
        Args:
            patient_data: dict with patient features
        
        Returns:
            dict with prediction and risk score
        """
        # Convert to DataFrame
        df = pd.DataFrame([patient_data])
        
        # Preprocess
        X, _ = self.preprocessor.preprocess(df, fit=False)
        
        # Predict
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0, 1]
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            'patient_id': patient_data.get('patient_id', 'N/A'),
            'readmission_predicted': bool(prediction),
            'readmission_probability': float(probability),
            'risk_level': risk_level,
            'timestamp': datetime.now().isoformat(),
            'recommendations': self._generate_recommendations(probability, patient_data)
        }
    
    def _generate_recommendations(self, probability, patient_data):
        """Generate clinical recommendations based on risk"""
        recommendations = []
        
        if probability > 0.5:
            recommendations.append("Schedule follow-up appointment within 7 days")
            recommendations.append("Assign care coordinator for discharge planning")
            
        if patient_data.get('num_medications', 0) > 10:
            recommendations.append("Medication reconciliation required")
            
        if patient_data.get('comorbidity_score', 0) > 2:
            recommendations.append("Multi-disciplinary care team consultation")
            
        if patient_data.get('socioeconomic_score', 5) < 3:
            recommendations.append("Social services referral for support")
        
        return recommendations
