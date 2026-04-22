import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

def evaluate_all(models_dict: dict, X_test, y_test):
    results = []

    for name, model in models_dict.items():
        if hasattr(model, 'predict'):
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]
        else:
            preds = model
            proba = model.astype(float)

        report = classification_report(y_test, preds, output_dict=True)
        auc = roc_auc_score(y_test, proba)

        results.append({
            'model': name,
            'precision': round(report['1']['precision'], 3),
            'recall': round(report['1']['recall'], 3),
            'f1': round(report['1']['f1-score'], 3),
            'roc_auc': round(auc, 3)
        })

        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.title(f'Confusion Matrix — {name}')
        plt.savefig(f'outputs/figures/cm_{name.lower().replace(" ", "_")}.png', dpi=150)
        plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv('outputs/reports/model_comparison.csv', index=False)
    print(results_df.to_string())

    plt.figure(figsize=(8, 6))
    for name, model in models_dict.items():
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)[:, 1]
        else:
            proba = model.astype(float)
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=name)
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('outputs/figures/roc_curve.png', dpi=150)
    plt.close()
    print("Grafikler kaydedildi.")

    return results_df

def plot_feature_importance(rf_model, feature_names):
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(feature_names)), importances[indices])
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45)
    plt.title('Feature Importance — Random Forest')
    plt.tight_layout()
    plt.savefig('outputs/figures/feature_importance.png', dpi=150)
    plt.close()
    print("Feature importance grafiği kaydedildi.")