# ============================================================
# remediation.py — Bias Remediation Toolkit
# FairScan India — Google Solution Challenge Edition
# ============================================================
# Each function is self-contained and documented.
# Install: pip install -r requirements.txt
# ============================================================

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator


# ────────────────────────────────────────────────────────────
# 1. REMOVE PROXY FEATURES — Fastest fix, no retraining
# ────────────────────────────────────────────────────────────
def remove_proxy_features(df: pd.DataFrame, protected_attrs: list,
                           threshold: float = 0.30) -> tuple:
    """
    Detect and remove features correlated with protected attributes.
    Returns (cleaned_df, list_of_removed_features).

    India context: Removes PIN code (caste proxy), surname (caste/religion proxy),
    employer name, neighbourhood columns, etc.

    Example:
        clean_df, removed = remove_proxy_features(df, ['caste', 'gender'])
        print("Removed proxy features:", [r['feature'] for r in removed])
    """
    removed = []
    cols_to_drop = set()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for attr in protected_attrs:
        if attr not in df.columns:
            continue
        encoded, _ = pd.factorize(df[attr])

        for col in numeric_cols:
            if col in protected_attrs:
                continue
            try:
                r = float(np.corrcoef(df[col].fillna(0), encoded)[0, 1])
                if abs(r) >= threshold:
                    cols_to_drop.add(col)
                    removed.append({
                        "feature":     col,
                        "attribute":   attr,
                        "correlation": round(abs(r), 4),
                        "severity":    "critical" if abs(r) >= 0.6 else "warning",
                    })
            except Exception:
                pass

    cleaned = df.drop(columns=list(cols_to_drop))
    return cleaned, removed


# ────────────────────────────────────────────────────────────
# 2. THRESHOLD OPTIMIZER — Post-processing, no retraining
# ────────────────────────────────────────────────────────────
def apply_threshold_optimizer(model, X_train, y_train,
                               sensitive_train, X_test, sensitive_test,
                               constraint: str = "equalized_odds"):
    """
    Equalize outcomes across groups without retraining the model.
    Uses Fairlearn's ThresholdOptimizer.

    Constraints:
        'equalized_odds'           — equalize TPR and FPR (recommended)
        'demographic_parity'       — equalize approval rates
        'true_positive_rate_parity'— equalize TPR only

    India context: Use 'equalized_odds' for loans (RBI compliance),
    'demographic_parity' for hiring (Article 16 compliance).

    Example:
        y_fair = apply_threshold_optimizer(
            model, X_train, y_train, df_train['caste'],
            X_test, df_test['caste'], constraint='equalized_odds'
        )
    """
    from fairlearn.postprocessing import ThresholdOptimizer

    optimizer = ThresholdOptimizer(
        estimator=model,
        constraints=constraint,
        predict_method="predict_proba",
        objective="balanced_accuracy_score",
    )
    optimizer.fit(X_train, y_train, sensitive_features=sensitive_train)
    return optimizer.predict(X_test, sensitive_features=sensitive_test)


# ────────────────────────────────────────────────────────────
# 3. REWEIGHING — Pre-processing (before retraining)
# ────────────────────────────────────────────────────────────
def apply_reweighing(df: pd.DataFrame, label_col: str,
                     protected_attr: str,
                     privileged_val, unprivileged_val) -> np.ndarray:
    """
    Compute instance weights using IBM AIF360 Reweighing.
    Upweights underrepresented groups so training sees a balanced world.

    Returns sample_weights array — pass to model.fit(..., sample_weight=w)

    India context: Use to reweight SC/ST applicants in loan training data.

    Example:
        w = apply_reweighing(df, 'outcome', 'caste', 'General', 'SC')
        model.fit(X_train, y_train, sample_weight=w)
    """
    from aif360.datasets import BinaryLabelDataset
    from aif360.algorithms.preprocessing import Reweighing

    aif_dataset = BinaryLabelDataset(
        df=df,
        label_names=[label_col],
        protected_attribute_names=[protected_attr],
    )
    privileged_groups   = [{protected_attr: privileged_val}]
    unprivileged_groups = [{protected_attr: unprivileged_val}]

    rw = Reweighing(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)
    rw.fit(aif_dataset)
    transformed = rw.transform(aif_dataset)
    return transformed.instance_weights


# ────────────────────────────────────────────────────────────
# 4. EXPONENTIATED GRADIENT — In-processing (retraining)
# ────────────────────────────────────────────────────────────
def train_with_exponentiated_gradient(base_model, X_train, y_train,
                                       sensitive_features,
                                       constraint: str = "EqualizedOdds"):
    """
    Train a fairer model using Fairlearn's ExponentiatedGradient.
    Wraps any sklearn estimator with a fairness constraint during training.

    Constraints: 'EqualizedOdds', 'DemographicParity', 'TruePositiveRateParity'

    Example:
        from sklearn.tree import DecisionTreeClassifier
        model = train_with_exponentiated_gradient(
            DecisionTreeClassifier(max_depth=4),
            X_train, y_train, df_train['caste']
        )
    """
    from fairlearn.reductions import (ExponentiatedGradient,
                                       EqualizedOdds, DemographicParity,
                                       TruePositiveRateParity)

    constraint_map = {
        "EqualizedOdds":          EqualizedOdds(),
        "DemographicParity":      DemographicParity(),
        "TruePositiveRateParity": TruePositiveRateParity(),
    }
    constraint_obj = constraint_map.get(constraint, EqualizedOdds())

    mitigator = ExponentiatedGradient(base_model, constraint_obj)
    mitigator.fit(X_train, y_train, sensitive_features=sensitive_features)
    return mitigator


# ────────────────────────────────────────────────────────────
# 5. ADVERSARIAL DEBIASING — In-processing (TensorFlow)
# ────────────────────────────────────────────────────────────
def train_adversarial_debiasing(aif_dataset_train, protected_attr: str,
                                  privileged_val, unprivileged_val,
                                  num_epochs: int = 50, batch_size: int = 128):
    """
    Train model using AIF360 AdversarialDebiasing.
    Adds an adversarial head that penalizes predictions correlated with
    the protected attribute.

    India context: Best long-term fix for caste/gender bias in loan models.

    Example:
        model = train_adversarial_debiasing(aif_train, 'caste', 'General', 'SC')
        predictions = model.predict(aif_dataset_test)
    """
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    from aif360.algorithms.inprocessing import AdversarialDebiasing

    privileged_groups   = [{protected_attr: privileged_val}]
    unprivileged_groups = [{protected_attr: unprivileged_val}]

    sess  = tf.Session()
    model = AdversarialDebiasing(
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups,
        scope_name="debiased_classifier",
        debias=True,
        sess=sess,
        num_epochs=num_epochs,
        batch_size=batch_size,
        classifier_num_hidden_units=200,
    )
    model.fit(aif_dataset_train)
    return model


# ────────────────────────────────────────────────────────────
# 6. SMOTE OVERSAMPLING — Data balancing
# ────────────────────────────────────────────────────────────
def oversample_minority_groups(X: pd.DataFrame, y: np.ndarray,
                                 sensitive: pd.Series) -> tuple:
    """
    Use SMOTENC to oversample underrepresented group+label combinations.
    Returns (X_resampled, y_resampled, sensitive_resampled).

    India context: Useful when SC/ST applicants are <10% of training data.

    Example:
        X_bal, y_bal, s_bal = oversample_minority_groups(
            X_train, y_train, df_train['caste']
        )
        model.fit(X_bal, y_bal)
    """
    from imblearn.over_sampling import SMOTENC

    X_combined = X.copy()
    X_combined["__sensitive__"] = sensitive.values
    categorical_cols = list(X_combined.select_dtypes(include="object").columns)
    cat_indices      = [X_combined.columns.get_loc(c) for c in categorical_cols]

    smote   = SMOTENC(categorical_features=cat_indices, random_state=42)
    X_res, y_res = smote.fit_resample(X_combined, y)
    s_res   = X_res["__sensitive__"]
    X_res   = X_res.drop(columns=["__sensitive__"])
    return X_res, y_res, s_res


# ────────────────────────────────────────────────────────────
# 7. SHAP BIAS AUDIT — Feature importance check
# ────────────────────────────────────────────────────────────
def compute_shap_bias_audit(model, X: pd.DataFrame,
                              protected_attrs: list) -> dict:
    """
    Use SHAP to check if protected attributes or their proxies
    are high-importance features in model decisions.

    Returns dict with feature importances and bias warnings.

    India context: Detects if 'pin_code' or 'surname' are top predictors
    (indicating they act as caste/religion proxies inside the model).

    Example:
        result = compute_shap_bias_audit(model, X_test, ['caste', 'gender'])
        if result['warning']:
            print("Protected attributes influence model decisions!")
    """
    import shap

    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
    except Exception:
        sample = shap.sample(X, min(100, len(X)))
        explainer = shap.KernelExplainer(model.predict_proba, sample)
        shap_vals = explainer.shap_values(X, nsamples=100)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

    mean_abs = np.abs(shap_vals).mean(axis=0)
    result   = dict(zip(X.columns, mean_abs.tolist()))
    flagged  = {k: v for k, v in result.items() if k in protected_attrs}

    return {
        "all_importances":             {k: round(v, 4) for k, v in sorted(result.items(), key=lambda x: -x[1])},
        "protected_attr_importances":  {k: round(v, 4) for k, v in flagged.items()},
        "warning":                     bool(len(flagged) > 0 and any(v > 0.05 for v in flagged.values())),
        "top_10_features":             list(sorted(result, key=result.get, reverse=True))[:10],
    }


# ────────────────────────────────────────────────────────────
# QUICK DEMO — Run this file directly to see output
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from bias_scanner import BiasScanner
    import json

    print("=" * 60)
    print("FairScan India — Remediation Demo")
    print("=" * 60)

    # Generate synthetic biased Indian loan dataset
    np.random.seed(42)
    n      = 3000
    caste  = np.random.choice(["General","OBC","SC","ST"], n, p=[0.45,0.28,0.17,0.10])
    gender = np.random.choice(["Male","Female"], n)
    income = np.random.normal(55000, 20000, n).clip(10000, 200000)
    credit = np.random.normal(680, 80, n).clip(300, 900)
    pin    = np.where(np.isin(caste, ["SC","ST"]), 400100, 400001) + np.random.randint(0,99,n)

    # Inject caste + gender bias
    prob = 1 / (1 + np.exp(-(credit - 650) / 60))
    prob[caste == "SC"]     *= 0.72
    prob[caste == "ST"]     *= 0.65
    prob[gender == "Female"] *= 0.83
    outcome = (np.random.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "income": income, "credit_score": credit, "pin_code": pin,
        "caste": caste, "gender": gender, "outcome": outcome,
        "prediction": outcome
    })

    print("\n1. Detecting proxy features...")
    clean_df, removed = remove_proxy_features(df, ["caste", "gender"])
    for r in removed:
        print(f"   ⚠  {r['feature']} → {r['attribute']} (r={r['correlation']}, {r['severity']})")

    print("\n2. Running bias scan...")
    scanner = BiasScanner(df=df, y_true=df["outcome"].values,
                          y_pred=df["prediction"].values,
                          protected_attrs=["caste","gender"],
                          reference_group={"caste":"General","gender":"Male"},
                          domain="loans")
    results = scanner.run_full_audit()
    print(f"   Risk Score: {results['bias_score']['score']}/100 — {results['bias_score']['risk_level']}")
    print(f"   India violations: {results['india_compliance']['n_violations']}")
    print(f"   Recommendations: {len(results['recommendations'])}")
    for r in results["recommendations"]:
        print(f"   [{r['priority'].upper()}] {r['title']}")
