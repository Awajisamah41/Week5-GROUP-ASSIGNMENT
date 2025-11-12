# ============================================================================
# 4. MODEL EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test, save_plots=True):
    """
    Comprehensive model evaluation
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print(f"\nActual Readmit & Predicted Readmit (True Positive): {cm[1,1]}")
    print(f"Actual Readmit & Predicted No Readmit (False Negative): {cm[1,0]}")
    print(f"Actual No Readmit & Predicted Readmit (False Positive): {cm[0,1]}")
    print(f"Actual No Readmit & Predicted No Readmit (True Negative): {cm[0,0]}")
    
    # Metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n{'Metric':<20} {'Value':<10}")
    print("-" * 30)
    print(f"{'Precision:':<20} {precision:.4f}")
    print(f"{'Recall:':<20} {recall:.4f}")
    print(f"{'F1 Score:':<20} {f1:.4f}")
    print(f"{'ROC AUC:':<20} {roc_auc:.4f}")
    
    print("\n" + classification_report(y_test, y_pred, 
                                       target_names=['Not Readmitted', 'Readmitted']))
    
    if save_plots:
        # Confusion Matrix Plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Readmit', 'Readmit'],
                    yticklabels=['No Readmit', 'Readmit'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # ROC Curve
        plt.subplot(1, 2, 2)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        print("\nEvaluation plots saved as 'model_evaluation.png'")
        plt.show()
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }
