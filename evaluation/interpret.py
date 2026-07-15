"""SHAP-based interpretability for the XGBoost model.

Uses a 1,000-row sample of the validation set for the SHAP pass rather than
the full ~8,700 validation rows - shap.Explainer on the complete set is slow
for not much extra insight, and this is the same sampling approach used
during exploration.
"""

import matplotlib.pyplot as plt
import shap


def explain_model(model, X_val, out_path: str = "outputs/shap_summary.png", sample_size: int = 1000):
    X_sample = X_val.iloc[:sample_size]

    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)

    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    importance = shap_values.abs.mean(0).values
    ranked = sorted(zip(X_sample.columns, importance), key=lambda x: -x[1])
    return ranked
