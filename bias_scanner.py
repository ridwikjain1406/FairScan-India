# ============================================================
# bias_scanner.py — Core Bias Analysis Engine
# FairScan India — Google Solution Challenge Edition
# ============================================================

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from itertools import combinations
from typing import Optional, Dict, List, Any
import warnings
warnings.filterwarnings("ignore")


class BiasScanner:
    """
    Full fairness audit engine with India-specific compliance checks.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset including protected attribute columns.
    y_true : np.ndarray
        Ground truth binary labels (0/1).
    y_pred : np.ndarray
        Model predictions (0/1).
    y_prob : np.ndarray, optional
        Model probability scores [0,1].
    protected_attrs : list
        Column names of protected attributes to analyze.
    reference_group : dict, optional
        e.g. {"caste": "General", "gender": "Male"}
        If None, the most frequent value per attribute is used.
    domain : str
        One of: 'loans', 'hiring', 'healthcare', 'education'
        Used to select India-specific legal compliance rules.
    """

    # Standard fairness thresholds
    FAIRNESS_THRESHOLD_DI         = 0.80   # 4/5 rule (RBI, EEOC)
    FAIRNESS_THRESHOLD_PARITY     = 0.05   # 5% max parity gap
    FAIRNESS_THRESHOLD_EO         = 0.05   # 5% equal opportunity diff
    PROXY_CORRELATION_THRESHOLD   = 0.30   # flag proxy if |r| > 0.30

    # India-specific protected categories per domain
    INDIA_PROTECTED = {
        "loans":      ["caste", "religion", "gender", "region", "state"],
        "hiring":     ["caste", "gender", "religion", "age", "region", "disability"],
        "healthcare": ["caste", "gender", "region", "age"],
        "education":  ["caste", "gender", "region", "economic_background"],
    }

    # Indian laws triggered per domain
    INDIA_LAWS = {
        "loans": [
            {"law": "DPDP Act 2023", "section": "§7, §8", "penalty": "Up to ₹250 crore", "url": "https://www.meity.gov.in/"},
            {"law": "RBI Fair Practices Code", "section": "Para 3", "penalty": "Operational restrictions", "url": "https://www.rbi.org.in/"},
            {"law": "Constitution of India — Article 15", "section": "Art. 15(1)", "penalty": "Constitutional challenge", "url": ""},
            {"law": "Equal Remuneration Act 1976", "section": "§4", "penalty": "Criminal liability", "url": ""},
        ],
        "hiring": [
            {"law": "Equal Remuneration Act 1976", "section": "§4, §5", "penalty": "Criminal liability", "url": ""},
            {"law": "Constitution of India — Article 16", "section": "Art. 16(1)", "penalty": "Constitutional challenge", "url": ""},
            {"law": "DPDP Act 2023", "section": "§7", "penalty": "Up to ₹250 crore", "url": ""},
            {"law": "Rights of Persons with Disabilities Act 2016", "section": "§20", "penalty": "Up to ₹5 lakh", "url": ""},
        ],
        "healthcare": [
            {"law": "DPDP Act 2023", "section": "§8", "penalty": "Up to ₹250 crore", "url": ""},
            {"law": "Constitution of India — Article 21", "section": "Art. 21", "penalty": "Constitutional challenge", "url": ""},
        ],
        "education": [
            {"law": "Constitution of India — Article 15(4)", "section": "Art. 15(4)", "penalty": "Constitutional challenge", "url": ""},
            {"law": "Right to Education Act 2009", "section": "§3", "penalty": "Regulatory action", "url": ""},
        ],
    }

    def __init__(self, df, y_true, y_pred, y_prob=None,
                 protected_attrs=None, reference_group=None, domain="loans"):
        self.df = df.copy().reset_index(drop=True)
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_prob = np.array(y_prob) if y_prob is not None else None
        self.protected_attrs = protected_attrs or []
        self.reference_group = reference_group or {}
        self.domain = domain

        # Auto-detect reference groups (most frequent value)
        for attr in self.protected_attrs:
            if attr not in self.reference_group and attr in self.df.columns:
                self.reference_group[attr] = self.df[attr].value_counts().idxmax()

    # ──────────────────────────────────────────────────────
    # PUBLIC: Run everything
    # ──────────────────────────────────────────────────────
    def run_full_audit(self) -> Dict[str, Any]:
        results = {
            "summary":              self._summary(),
            "group_metrics":        {},
            "fairness_tests":       {},
            "proxy_features":       self._detect_proxy_features(),
            "dataset_composition":  self._dataset_composition(),
            "intersectional":       {},
            "india_compliance":     self._india_compliance_check(),
            "recommendations":      [],
        }

        for attr in self.protected_attrs:
            if attr not in self.df.columns:
                continue
            results["group_metrics"][attr]  = self._group_metrics(attr)
            results["fairness_tests"][attr] = self._fairness_tests(attr)

        # Intersectional analysis for attribute pairs
        for a1, a2 in combinations(self.protected_attrs, 2):
            if a1 in self.df.columns and a2 in self.df.columns:
                key = f"{a1}_x_{a2}"
                results["intersectional"][key] = self._intersectional(a1, a2)

        results["recommendations"] = self._generate_recommendations(results)
        results["bias_score"]      = self._compute_bias_score(results)
        return results

    # ──────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────
    def _summary(self) -> Dict:
        n = len(self.y_true)
        overall_approval = float(self.y_pred.mean())
        overall_accuracy = float(accuracy_score(self.y_true, self.y_pred))
        auc = float(roc_auc_score(self.y_true, self.y_prob)) if self.y_prob is not None else None
        return {
            "n_samples":                    n,
            "overall_approval_rate":        round(overall_approval, 4),
            "overall_accuracy":             round(overall_accuracy, 4),
            "auc_roc":                      round(auc, 4) if auc else None,
            "protected_attributes_analyzed": self.protected_attrs,
            "reference_groups":             self.reference_group,
            "domain":                       self.domain,
        }

    # ──────────────────────────────────────────────────────
    # GROUP METRICS — per protected attribute
    # ──────────────────────────────────────────────────────
    def _group_metrics(self, attr: str) -> Dict:
        groups = {}
        for group_val in self.df[attr].unique():
            mask = self.df[attr] == group_val
            if mask.sum() < 10:
                continue
            yt = self.y_true[mask]
            yp = self.y_pred[mask]
            n  = int(mask.sum())

            approval_rate = float(yp.mean())
            try:
                tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
            except ValueError:
                tn, fp, fn, tp = 0, 0, 0, n

            tpr       = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr       = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            groups[str(group_val)] = {
                "n":                    n,
                "approval_rate":        round(approval_rate, 4),
                "true_positive_rate":   round(tpr, 4),
                "false_positive_rate":  round(fpr, 4),
                "false_negative_rate":  round(fnr, 4),
                "precision":            round(precision, 4),
            }
        return groups

    # ──────────────────────────────────────────────────────
    # FAIRNESS TESTS — per protected attribute
    # ──────────────────────────────────────────────────────
    def _fairness_tests(self, attr: str) -> Dict:
        ref_val  = self.reference_group.get(attr)
        ref_mask = self.df[attr] == ref_val
        ref_rate = float(self.y_pred[ref_mask].mean()) if ref_mask.sum() > 0 else None
        ref_yt   = self.y_true[ref_mask]
        ref_yp   = self.y_pred[ref_mask]
        ref_tpr  = self._tpr(ref_yt, ref_yp)
        ref_fpr  = self._fpr(ref_yt, ref_yp)

        tests = {
            "demographic_parity": {},
            "disparate_impact":   {},
            "equal_opportunity":  {},
            "equalized_odds":     {},
        }

        max_parity_gap = 0.0
        min_di_ratio   = 1.0
        max_eo_diff    = 0.0

        for group_val in self.df[attr].unique():
            if str(group_val) == str(ref_val):
                continue
            mask = self.df[attr] == group_val
            if mask.sum() < 10:
                continue
            gyt  = self.y_true[mask]
            gyp  = self.y_pred[mask]
            rate = float(gyp.mean())
            g    = str(group_val)

            # Demographic Parity
            gap = abs(rate - ref_rate) if ref_rate is not None else None
            tests["demographic_parity"][g] = {
                "group_rate":     round(rate, 4),
                "reference_rate": round(ref_rate, 4) if ref_rate else None,
                "gap":            round(gap, 4) if gap is not None else None,
                "pass":           bool(gap < self.FAIRNESS_THRESHOLD_PARITY) if gap is not None else None,
            }
            if gap is not None:
                max_parity_gap = max(max_parity_gap, gap)

            # Disparate Impact (4/5 rule)
            di = (rate / ref_rate) if (ref_rate and ref_rate > 0) else None
            tests["disparate_impact"][g] = {
                "ratio": round(di, 4) if di is not None else None,
                "pass":  bool(di >= self.FAIRNESS_THRESHOLD_DI) if di is not None else None,
                "threshold": self.FAIRNESS_THRESHOLD_DI,
            }
            if di is not None:
                min_di_ratio = min(min_di_ratio, di)

            # Equal Opportunity (TPR parity)
            g_tpr  = self._tpr(gyt, gyp)
            eo_diff = abs(g_tpr - ref_tpr)
            tests["equal_opportunity"][g] = {
                "group_tpr":    round(g_tpr, 4),
                "reference_tpr": round(ref_tpr, 4),
                "diff":         round(eo_diff, 4),
                "pass":         bool(eo_diff < self.FAIRNESS_THRESHOLD_EO),
            }
            max_eo_diff = max(max_eo_diff, eo_diff)

            # Equalized Odds (TPR + FPR parity)
            g_fpr   = self._fpr(gyt, gyp)
            fpr_diff = abs(g_fpr - ref_fpr)
            tests["equalized_odds"][g] = {
                "tpr_diff": round(eo_diff, 4),
                "fpr_diff": round(fpr_diff, 4),
                "pass":     bool(eo_diff < self.FAIRNESS_THRESHOLD_EO and fpr_diff < self.FAIRNESS_THRESHOLD_EO),
            }

        tests["_summary"] = {
            "max_parity_gap":             round(max_parity_gap, 4),
            "min_disparate_impact":       round(min_di_ratio, 4),
            "max_equal_opportunity_diff": round(max_eo_diff, 4),
            "reference_group":            str(ref_val),
            "reference_approval_rate":    round(ref_rate, 4) if ref_rate else None,
        }
        return tests

    # ──────────────────────────────────────────────────────
    # PROXY FEATURE DETECTION
    # ──────────────────────────────────────────────────────
    def _detect_proxy_features(self) -> List[Dict]:
        proxies = []
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        for attr in self.protected_attrs:
            if attr not in self.df.columns:
                continue
            encoded, _ = pd.factorize(self.df[attr])

            for col in numeric_cols:
                if col in self.protected_attrs:
                    continue
                try:
                    r = np.corrcoef(self.df[col].fillna(0), encoded)[0, 1]
                    if abs(r) >= self.PROXY_CORRELATION_THRESHOLD:
                        proxies.append({
                            "feature":        col,
                            "attribute":      attr,
                            "correlation":    round(float(r), 4),
                            "severity":       "critical" if abs(r) >= 0.6 else "warning",
                            "recommendation": f"Remove or decorrelate '{col}' — acts as proxy for '{attr}'.",
                        })
                except Exception:
                    pass

        proxies.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return proxies

    # ──────────────────────────────────────────────────────
    # INDIA COMPLIANCE CHECK — NEW
    # ──────────────────────────────────────────────────────
    def _india_compliance_check(self) -> Dict:
        """
        Check compliance with Indian-specific laws based on domain.
        Returns structured violations and warnings.
        """
        laws = self.INDIA_LAWS.get(self.domain, self.INDIA_LAWS["loans"])
        violations = []
        warnings_list = []

        # Run preliminary checks to determine violations
        # (Full metrics run later; here we do quick estimates)
        for attr in self.protected_attrs:
            if attr not in self.df.columns:
                continue
            ref_val  = self.reference_group.get(attr)
            ref_mask = self.df[attr] == ref_val
            if ref_mask.sum() == 0:
                continue
            ref_rate = float(self.y_pred[ref_mask].mean())

            for group_val in self.df[attr].unique():
                if str(group_val) == str(ref_val):
                    continue
                mask = self.df[attr] == group_val
                if mask.sum() < 10:
                    continue
                rate = float(self.y_pred[mask].mean())
                di   = rate / ref_rate if ref_rate > 0 else 1.0
                gap  = abs(rate - ref_rate)

                if di < 0.80:
                    for law in laws[:2]:  # Top 2 laws for domain
                        violations.append({
                            "law":        law["law"],
                            "section":    law["section"],
                            "penalty":    law["penalty"],
                            "reason":     f"Disparate impact {di:.2f} for '{group_val}' in '{attr}' — below 0.80 threshold",
                            "group":      str(group_val),
                            "attribute":  attr,
                        })
                elif gap > 0.05:
                    warnings_list.append({
                        "law":       laws[0]["law"] if laws else "DPDP Act 2023",
                        "reason":    f"Parity gap of {gap*100:.1f}% for '{group_val}' in '{attr}' — monitor closely",
                        "group":     str(group_val),
                        "attribute": attr,
                    })

        # EU AI Act warning for high-risk domains
        if self.domain in ["loans", "hiring"]:
            warnings_list.append({
                "law":    "EU AI Act (Aug 2026 deadline)",
                "reason": f"'{self.domain}' AI systems are classified HIGH RISK under EU AI Act. Bias testing and human oversight required if serving European customers.",
                "group":  "all",
                "attribute": "all",
            })

        return {
            "domain":      self.domain,
            "laws_checked": [l["law"] for l in laws],
            "violations":  violations[:10],   # cap at 10
            "warnings":    warnings_list[:5],
            "n_violations": len(violations),
            "n_warnings":   len(warnings_list),
        }

    # ──────────────────────────────────────────────────────
    # DATASET COMPOSITION AUDIT
    # ──────────────────────────────────────────────────────
    def _dataset_composition(self) -> Dict:
        composition = {}
        for attr in self.protected_attrs:
            if attr not in self.df.columns:
                continue
            counts = self.df[attr].value_counts()
            pcts   = (counts / len(self.df) * 100).round(2)
            composition[attr] = {
                str(k): {"count": int(counts[k]), "pct": float(pcts[k])}
                for k in counts.index
            }
        return composition

    # ──────────────────────────────────────────────────────
    # INTERSECTIONAL ANALYSIS
    # ──────────────────────────────────────────────────────
    def _intersectional(self, a1: str, a2: str) -> Dict:
        result = {}
        for v1 in self.df[a1].unique():
            for v2 in self.df[a2].unique():
                mask = (self.df[a1] == v1) & (self.df[a2] == v2)
                if mask.sum() < 20:
                    continue
                rate = float(self.y_pred[mask].mean())
                key  = f"{v1}|{v2}"
                result[key] = {"n": int(mask.sum()), "approval_rate": round(rate, 4)}
        return result

    # ──────────────────────────────────────────────────────
    # BIAS SCORE (0–100, higher = more biased)
    # ──────────────────────────────────────────────────────
    def _compute_bias_score(self, results: Dict) -> Dict:
        penalties = 0.0
        reasons   = []

        for attr, tests in results["fairness_tests"].items():
            s   = tests.get("_summary", {})
            di  = s.get("min_disparate_impact", 1.0)
            gap = s.get("max_parity_gap", 0.0)
            eo  = s.get("max_equal_opportunity_diff", 0.0)

            if di < 0.80:
                p = (0.80 - di) / 0.80 * 40
                penalties += p
                reasons.append(f"{attr}: disparate impact {di:.2f} (severity: {p:.0f}pts)")
            if gap > 0.05:
                p = min((gap - 0.05) / 0.15 * 25, 25)
                penalties += p
                reasons.append(f"{attr}: parity gap {gap*100:.1f}% (severity: {p:.0f}pts)")
            if eo > 0.05:
                p = min((eo - 0.05) / 0.15 * 20, 20)
                penalties += p
                reasons.append(f"{attr}: equal opportunity diff {eo*100:.1f}% (severity: {p:.0f}pts)")

        critical_proxies = [p for p in results.get("proxy_features", []) if p["severity"] == "critical"]
        penalties += len(critical_proxies) * 5

        raw_score  = min(int(penalties), 100)
        risk_level = "LOW" if raw_score < 30 else "MEDIUM" if raw_score < 60 else "HIGH"
        return {
            "score":      raw_score,
            "risk_level": risk_level,
            "reasons":    reasons,
        }

    # ──────────────────────────────────────────────────────
    # RECOMMENDATIONS ENGINE
    # ──────────────────────────────────────────────────────
    def _generate_recommendations(self, results: Dict) -> List[Dict]:
        recs = []

        # 1. Remove proxy features (fastest win, no retraining)
        for proxy in results.get("proxy_features", []):
            if proxy["severity"] == "critical":
                recs.append({
                    "priority": "critical",
                    "title":    f"Remove proxy feature: '{proxy['feature']}'",
                    "detail":   (f"'{proxy['feature']}' has correlation {proxy['correlation']:.2f} with "
                                 f"'{proxy['attribute']}' — it acts as an indirect discriminator. "
                                 f"Remove it or replace with a less correlated alternative. "
                                 f"No model retraining required."),
                    "effort":   "low",
                    "impact":   "high",
                    "code":     f"# Remove proxy feature — no retraining needed\ndf = df.drop(columns=['{proxy['feature']}'])\nX_train = X_train.drop(columns=['{proxy['feature']}'])",
                    "india_law": "DPDP Act 2023, RBI Fair Practices Code",
                })

        # 2. Threshold optimizer (no retraining)
        for attr, tests in results["fairness_tests"].items():
            s  = tests.get("_summary", {})
            di = s.get("min_disparate_impact", 1.0)
            if di < self.FAIRNESS_THRESHOLD_DI:
                recs.append({
                    "priority": "critical",
                    "title":    f"Apply ThresholdOptimizer for '{attr}' (DI: {di:.2f})",
                    "detail":   ("Use Fairlearn's ThresholdOptimizer to post-process predictions and "
                                 "equalize outcomes across groups. No model retraining needed. "
                                 "Can be deployed as a wrapper in 1–2 days."),
                    "effort":   "low",
                    "impact":   "high",
                    "code": (
                        "from fairlearn.postprocessing import ThresholdOptimizer\n\n"
                        f"optimizer = ThresholdOptimizer(\n"
                        f"    estimator=model,\n"
                        f"    constraints='equalized_odds',\n"
                        f"    predict_method='predict_proba',\n"
                        f"    objective='balanced_accuracy_score'\n"
                        f")\n"
                        f"optimizer.fit(X_train, y_train, sensitive_features=df_train['{attr}'])\n"
                        f"y_pred_fair = optimizer.predict(X_test, sensitive_features=df_test['{attr}'])"
                    ),
                    "india_law": "RBI Fair Practices Code, Article 15",
                })

        # 3. Data reweighing (before retraining)
        for attr, comp in results["dataset_composition"].items():
            total = sum(v["count"] for v in comp.values())
            for group, info in comp.items():
                if info["pct"] < 10 and total > 1000:
                    recs.append({
                        "priority": "high",
                        "title":    f"Reweigh underrepresented group: '{group}' in '{attr}'",
                        "detail":   (f"'{group}' is only {info['pct']}% of training data. "
                                     "Use AIF360 Reweighing to correct data imbalance before retraining."),
                        "effort":   "medium",
                        "impact":   "high",
                        "code": (
                            "from aif360.algorithms.preprocessing import Reweighing\n"
                            "from aif360.datasets import BinaryLabelDataset\n\n"
                            "aif_ds = BinaryLabelDataset(\n"
                            "    df=df, label_names=['outcome'],\n"
                            f"    protected_attribute_names=['{attr}']\n"
                            ")\n"
                            f"rw = Reweighing(\n"
                            f"    unprivileged_groups=[{{'{attr}': '{group}'}}],\n"
                            f"    privileged_groups=[{{'{attr}': reference_val}}]\n"
                            ")\n"
                            "dataset_rw = rw.fit_transform(aif_ds)\n"
                            "# Use dataset_rw.instance_weights in model.fit(..., sample_weight=w)"
                        ),
                        "india_law": "DPDP Act 2023",
                    })

        # 4. Adversarial debiasing (deep fix, retraining needed)
        recs.append({
            "priority": "medium",
            "title":    "Retrain with Adversarial Debiasing (root-cause fix)",
            "detail":   ("Retrain using AIF360 AdversarialDebiasing — an in-processing technique "
                         "that penalizes the model for making predictions correlated with protected attributes. "
                         "Achieves the most durable bias reduction but requires 6–8 weeks."),
            "effort":   "high",
            "impact":   "high",
            "code": (
                "from aif360.algorithms.inprocessing import AdversarialDebiasing\n"
                "import tensorflow.compat.v1 as tf\n"
                "tf.disable_eager_execution()\n\n"
                "sess = tf.Session()\n"
                "debiased_model = AdversarialDebiasing(\n"
                "    privileged_groups=privileged_groups,\n"
                "    unprivileged_groups=unprivileged_groups,\n"
                "    scope_name='debiased_classifier',\n"
                "    debias=True, sess=sess,\n"
                "    num_epochs=50, batch_size=128\n"
                ")\n"
                "debiased_model.fit(aif_dataset_train)"
            ),
            "india_law": "All — structural fix",
        })

        return recs

    # ──────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────
    @staticmethod
    def _tpr(y_true, y_pred):
        positives = y_true == 1
        return float((y_pred[positives] == 1).mean()) if positives.sum() > 0 else 0.0

    @staticmethod
    def _fpr(y_true, y_pred):
        negatives = y_true == 0
        return float((y_pred[negatives] == 1).mean()) if negatives.sum() > 0 else 0.0
