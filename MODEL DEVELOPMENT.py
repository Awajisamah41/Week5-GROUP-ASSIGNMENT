# ============================================================================
# 3. MODEL DEVELOPMENT
# ============================================================================

class ReadmissionPredictor:
    """Gradient Boosting Model for readmission prediction"""
    
    def __init__(self):
        self.model = None
        self.best_params = None
        self.training_date = None
        
    def train(self, X_train, y_train, X_val, y_val, tune_hyperparameters=True):
        """
        Train the Gradient Boosting model
        """
        print("Training Gradient Boosting Model...")
        
        if tune_hyperparameters:
            print("Performing hyperparameter tuning...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5],
                'subsample': [0.8, 1.0]
            }
            
            base_model = GradientBoostingClassifier(random_state=42)
            grid_search = GridSearchCV(
                base_model, 
                param_grid, 
                cv=3, 
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            print(f"Best parameters: {self.best_params}")
        else:
            # Use default good parameters
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=5,
                subsample=0.8,
                random_state=42
            )
            self.model.fit(X_train, y_train)
        
        self.training_date = datetime.now()
        
        # Validation performance
        val_pred = self.model.predict(X_val)
        val_score = f1_score(y_val, val_pred)
        print(f"Validation F1 Score: {val_score:.4f}")
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def save(self, filepath='readmission_model.pkl'):
        """Save model for deployment"""
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load(filepath='readmission_model.pkl'):
        """Load trained model"""
        return joblib.load(filepath)
