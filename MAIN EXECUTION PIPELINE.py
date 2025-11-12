# ============================================================================
# 6. MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """
    Complete workflow execution
    """
    print("="*60)
    print("PATIENT READMISSION PREDICTION SYSTEM")
    print("="*60)
    
    # 1. Generate/Load Data
    print("\n1. Loading data...")
    df = generate_synthetic_data(n_samples=1000)
    print(f"Dataset shape: {df.shape}")
    print(f"Readmission rate: {df['readmitted_30days'].mean():.2%}")
    
    # 2. Preprocess Data
    print("\n2. Preprocessing data...")
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess(df, fit=True)
    
    # 3. Split Data (70% train, 15% val, 15% test)
    print("\n3. Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 4. Train Model
    print("\n4. Training model...")
    predictor = ReadmissionPredictor()
    predictor.train(X_train, y_train, X_val, y_val, tune_hyperparameters=False)
    
    # 5. Evaluate Model
    print("\n5. Evaluating model...")
    metrics = evaluate_model(predictor, X_test, y_test)
    
    # 6. Save Model and Preprocessor
    print("\n6. Saving model and preprocessor...")
    predictor.save('readmission_model.pkl')
    preprocessor.save('preprocessor.pkl')
    
    # 7. Deployment Example
    print("\n7. Testing deployment API...")
    api = ReadmissionAPI('readmission_model.pkl', 'preprocessor.pkl')
    
    # Test prediction on a sample patient
    test_patient = {
        'patient_id': 'P12345',
        'age': 72,
        'gender': 'M',
        'num_prior_admissions': 3,
        'length_of_stay': 10,
        'num_diagnoses': 5,
        'num_medications': 15,
        'num_procedures': 2,
        'emergency_admission': 1,
        'diabetes': 1,
        'heart_disease': 1,
        'kidney_disease': 0,
        'lab_result_1': 120,
        'lab_result_2': 8.5,
        'socioeconomic_score': 2,
        'comorbidity_score': 2
    }
    
    result = api.predict_single_patient(test_patient)
    
    print("\n" + "="*60)
    print("SAMPLE PREDICTION RESULT")
    print("="*60)
    print(f"Patient ID: {result['patient_id']}")
    print(f"Readmission Predicted: {result['readmission_predicted']}")
    print(f"Readmission Probability: {result['readmission_probability']:.2%}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"\nRecommendations:")
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print("\nFiles created:")
    print("  - readmission_model.pkl (trained model)")
    print("  - preprocessor.pkl (data preprocessor)")
    print("  - model_evaluation.png (evaluation plots)")
    print("\nThe system is ready for deployment!")

if __name__ == "__main__":
    main()
