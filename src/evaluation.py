import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score
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

def failure_analysis(model, X_test, y_test, feature_names):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    df = X_test.copy()
    df['y_true'] = y_test.values
    df['y_pred'] = preds
    df['waste_prob'] = proba

    fp = df[(df['y_pred'] == 1) & (df['y_true'] == 0)]
    fn = df[(df['y_pred'] == 0) & (df['y_true'] == 1)]
    tp = df[(df['y_pred'] == 1) & (df['y_true'] == 1)]
    tn = df[(df['y_pred'] == 0) & (df['y_true'] == 0)]

    print(f"\n=== FAILURE ANALYSIS ===")
    print(f"TP: {len(tp)} | TN: {len(tn)} | FP: {len(fp)} | FN: {len(fn)}")

    print(f"\nFalse Positives ({len(fp)}) — predicted waste, actually useful:")
    print(fp[feature_names].mean().round(3))

    print(f"\nFalse Negatives ({len(fn)}) — missed waste:")
    print(fn[feature_names].mean().round(3))

    # CSV kaydet
    summary = pd.DataFrame({
        'FP_mean': fp[feature_names].mean(),
        'FN_mean': fn[feature_names].mean(),
        'TP_mean': tp[feature_names].mean(),
        'TN_mean': tn[feature_names].mean()
    })
    summary.to_csv('outputs/reports/failure_analysis.csv')
    print("Failure analysis kaydedildi.")

    # PR curve
    precision, recall, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)

    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision, color='steelblue', lw=2, label=f'AP={ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve — Random Forest')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/figures/precision_recall.png', dpi=150)
    plt.close()
    print("PR curve kaydedildi.")

    return fp, fn