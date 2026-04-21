# ============================================================
# FairScan India — FastAPI Backend
# main.py — Application Entry Point
# Google Solution Challenge Edition
# ============================================================
# Install: pip install -r requirements.txt
# Run:     uvicorn main:app --reload --port 8000
# Docs:    http://localhost:8000/docs
# ============================================================

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import pandas as pd
import numpy as np
import json
import io
import os
import joblib
from typing import Optional
from datetime import datetime

from bias_scanner import BiasScanner
from remediation import remove_proxy_features, compute_shap_bias_audit
from report_generator import generate_report

app = FastAPI(
    title="FairScan India API",
    description=(
        "AI Bias Detection and Fairness Auditing System — India Edition.\n\n"
        "Compliant with: DPDP Act 2023, RBI Fair Practices Code, "
        "Constitution Articles 15/16, Equal Remuneration Act 1976, EU AI Act.\n\n"
        "SDG Alignment: SDG 10 (Reduced Inequalities), SDG 8 (Decent Work)."
    ),
    version="2.0.0",
    contact={"name": "FairScan India", "url": "https://fairscan.india"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict to your frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ────────────────────────────────────────────────────────────
# HEALTH CHECK
# ────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    return {
        "status":    "ok",
        "version":   "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "features":  ["bias_scan", "india_compliance", "proxy_detection",
                      "remediation", "report_export", "shap_audit"],
    }


# ────────────────────────────────────────────────────────────
# DATASET PREVIEW — Show columns, types, sample rows
# ────────────────────────────────────────────────────────────
@app.post("/api/preview", tags=["Data"])
async def preview_dataset(dataset: UploadFile = File(...)):
    """
    Upload a CSV and get back column names, types, sample rows,
    null counts, and basic stats. Use this to validate the file
    before running a full scan.
    """
    content = await dataset.read()
    try:
        if dataset.filename.endswith(".parquet"):
            df = pd.read_parquet(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse file: {str(e)}")

    # Auto-detect likely protected attribute columns
    india_keywords = {
        "caste": ["caste", "category", "sc", "st", "obc", "reservation"],
        "gender": ["gender", "sex"],
        "religion": ["religion", "faith", "community"],
        "region": ["region", "state", "city", "zone"],
        "age": ["age", "dob"],
        "outcome": ["outcome", "decision", "approved", "result", "label", "target"],
    }
    detected = {}
    for attr, keywords in india_keywords.items():
        for col in df.columns:
            if any(kw in col.lower() for kw in keywords):
                detected[attr] = col
                break

    return {
        "columns":          df.dtypes.apply(str).to_dict(),
        "shape":            list(df.shape),
        "sample":           df.head(5).fillna("").to_dict(orient="records"),
        "nulls":            df.isnull().sum().to_dict(),
        "numeric_stats":    df.describe().fillna(0).to_dict(),
        "detected_columns": detected,
        "filename":         dataset.filename,
    }


# ────────────────────────────────────────────────────────────
# FULL BIAS SCAN — Main endpoint
# ────────────────────────────────────────────────────────────
@app.post("/api/scan", tags=["Scan"])
async def full_scan(
    dataset:              UploadFile = File(..., description="CSV file with data + predictions"),
    label_column:         str        = Form("outcome", description="Ground truth column name (0/1)"),
    prediction_column:    str        = Form("prediction", description="Model prediction column name (0/1)"),
    protected_attributes: str        = Form("caste,gender,religion", description="Comma-separated protected attribute column names"),
    reference_group:      Optional[str] = Form(None, description='JSON dict e.g. {"caste":"General","gender":"Male"}'),
    domain:               str        = Form("loans", description="Domain: loans | hiring | healthcare | education"),
    model_file:           Optional[UploadFile] = File(None, description="Optional: joblib model file to generate predictions"),
):
    """
    Run a complete AI bias audit.

    Returns:
    - Bias risk score (0–100)
    - Group-level fairness metrics (demographic parity, disparate impact, equal opportunity)
    - Proxy feature detection
    - India-specific compliance check (DPDP Act, RBI, Article 15, etc.)
    - Intersectional analysis (e.g. SC women vs General men)
    - Prioritized remediation recommendations with code snippets
    """
    # ── Load dataset ──
    content = await dataset.read()
    try:
        df = pd.read_parquet(io.BytesIO(content)) if dataset.filename.endswith(".parquet") \
             else pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse dataset: {str(e)}")

    attr_list = [a.strip() for a in protected_attributes.split(",") if a.strip()]

    # ── Validate columns ──
    required = [label_column]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Required columns not found in file: {missing}. Available: {list(df.columns)}")

    # ── Get predictions ──
    y_pred = None
    y_prob = None

    if model_file and model_file.filename:
        # Load sklearn model from uploaded joblib file
        try:
            model_bytes = await model_file.read()
            model       = joblib.load(io.BytesIO(model_bytes))
            feature_cols = [c for c in df.columns if c not in attr_list + [label_column, prediction_column]]
            X            = df[feature_cols].fillna(0)
            y_pred       = model.predict(X)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X)[:, 1]
        except Exception as e:
            raise HTTPException(400, f"Could not load model: {str(e)}")
    elif prediction_column in df.columns:
        y_pred = df[prediction_column].values
    else:
        # Auto-generate predictions using a simple rule for demo/preview
        # (In production, always supply a model or prediction column)
        y_pred = df[label_column].values
        import warnings
        warnings.warn("No prediction column or model found. Using ground truth as predictions — results will not reflect model bias.")

    # Normalize labels/predictions to 0/1
    def normalize_binary(arr):
        arr = pd.Series(arr)
        if arr.dtype == object:
            pos_vals = {"1", "yes", "approved", "accept", "true", "positive"}
            return (arr.astype(str).str.lower().isin(pos_vals)).astype(int).values
        return (arr > 0.5).astype(int).values

    y_true = normalize_binary(df[label_column].values)
    y_pred = normalize_binary(y_pred)

    # ── Parse reference group ──
    ref = None
    if reference_group:
        try:
            ref = json.loads(reference_group)
        except json.JSONDecodeError:
            raise HTTPException(400, "reference_group must be valid JSON, e.g. {\"caste\": \"General\"}")

    # ── Run full audit ──
    try:
        scanner = BiasScanner(
            df=df, y_true=y_true, y_pred=y_pred,
            y_prob=y_prob, protected_attrs=attr_list,
            reference_group=ref, domain=domain
        )
        results = scanner.run_full_audit()
    except Exception as e:
        raise HTTPException(500, f"Scan engine error: {str(e)}")

    # ── Add metadata ──
    results["meta"] = {
        "filename":    dataset.filename,
        "scanned_at":  datetime.utcnow().isoformat(),
        "domain":      domain,
        "n_rows":      len(df),
        "n_cols":      len(df.columns),
        "api_version": "2.0.0",
        "sdg_alignment": ["SDG 10 — Reduced Inequalities", "SDG 8 — Decent Work and Economic Growth"],
    }

    return JSONResponse(content=results)


# ────────────────────────────────────────────────────────────
# QUICK METRICS — Pass data directly as JSON
# ────────────────────────────────────────────────────────────
@app.post("/api/metrics", tags=["Scan"])
async def compute_metrics(payload: dict):
    """
    Lightweight endpoint — pass data as JSON (no file upload).

    Expected payload:
    {
      "y_true": [0, 1, 0, 1, ...],
      "y_pred": [0, 1, 1, 0, ...],
      "protected": {"caste": ["General","SC","OBC",...], "gender": [...]},
      "reference_group": {"caste": "General", "gender": "Male"},
      "domain": "loans"
    }
    """
    try:
        y_true    = np.array(payload["y_true"])
        y_pred    = np.array(payload["y_pred"])
        protected = payload.get("protected", {})
        ref       = payload.get("reference_group", None)
        domain    = payload.get("domain", "loans")

        df = pd.DataFrame(protected)
        scanner = BiasScanner(
            df=df, y_true=y_true, y_pred=y_pred,
            protected_attrs=list(protected.keys()),
            reference_group=ref, domain=domain
        )
        return JSONResponse(content=scanner.run_full_audit())
    except KeyError as e:
        raise HTTPException(400, f"Missing field: {e}")
    except Exception as e:
        raise HTTPException(500, str(e))


# ────────────────────────────────────────────────────────────
# PROXY DETECTION — Standalone endpoint
# ────────────────────────────────────────────────────────────
@app.post("/api/proxy-check", tags=["Analysis"])
async def proxy_check(
    dataset:              UploadFile = File(...),
    protected_attributes: str        = Form("caste,gender,religion"),
    threshold:            float      = Form(0.30),
):
    """
    Detect proxy features in a dataset — columns correlated with
    protected attributes (like PIN code correlating with caste).
    """
    content  = await dataset.read()
    df       = pd.read_csv(io.BytesIO(content))
    attr_list = [a.strip() for a in protected_attributes.split(",")]
    missing   = [c for c in attr_list if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Columns not found: {missing}")

    cleaned, removed = remove_proxy_features(df, attr_list, threshold)
    return {
        "proxy_features_found": len(removed),
        "proxies":              removed,
        "columns_after_removal": list(cleaned.columns),
        "recommendation":       "Remove listed columns before training/deploying your model.",
    }


# ────────────────────────────────────────────────────────────
# SHAP AUDIT — Feature importance bias check
# ────────────────────────────────────────────────────────────
@app.post("/api/shap-audit", tags=["Analysis"])
async def shap_audit(
    dataset:              UploadFile = File(...),
    model_file:           UploadFile = File(...),
    protected_attributes: str        = Form("caste,gender,religion"),
    label_column:         str        = Form("outcome"),
):
    """
    Run SHAP analysis to check if the model is using protected
    attributes or their proxies as key decision factors.
    """
    content     = await dataset.read()
    model_bytes = await model_file.read()
    df    = pd.read_csv(io.BytesIO(content))
    model = joblib.load(io.BytesIO(model_bytes))
    attr_list    = [a.strip() for a in protected_attributes.split(",")]
    feature_cols = [c for c in df.columns if c not in attr_list + [label_column]]
    X            = df[feature_cols].fillna(0)

    try:
        result = compute_shap_bias_audit(model, X, attr_list)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(500, f"SHAP analysis failed: {str(e)}")


# ────────────────────────────────────────────────────────────
# REPORT EXPORT — Generate HTML compliance report
# ────────────────────────────────────────────────────────────
@app.post("/api/report", tags=["Export"])
async def export_report(scan_results: dict):
    """
    Generate a self-contained HTML compliance report from scan results.
    Returns the HTML as a string (save it as .html to open in browser).
    """
    try:
        report_html = generate_report(scan_results)
        return JSONResponse({"html": report_html, "length": len(report_html)})
    except Exception as e:
        raise HTTPException(500, f"Report generation failed: {str(e)}")


@app.get("/api/report/sample", tags=["Export"], response_class=HTMLResponse)
async def sample_report():
    """Returns a sample HTML report with demo data — useful for UI preview."""
    sample_results = {
        "bias_score": {"score": 67, "risk_level": "HIGH", "reasons": ["caste: disparate impact 0.61", "gender: parity gap 18.2%"]},
        "summary":    {"n_samples": 48230, "overall_approval_rate": 0.683, "overall_accuracy": 0.812, "auc_roc": 0.76,
                       "protected_attributes_analyzed": ["caste", "gender", "religion"],
                       "reference_groups": {"caste": "General", "gender": "Male"}, "domain": "loans"},
        "proxy_features": [
            {"feature": "pin_code", "attribute": "caste", "correlation": 0.78, "severity": "critical", "recommendation": "Remove pin_code"},
            {"feature": "surname",  "attribute": "caste", "correlation": 0.71, "severity": "critical", "recommendation": "Remove surname"},
        ],
        "recommendations": [
            {"priority": "critical", "title": "Remove proxy feature: pin_code", "detail": "78% correlated with caste.", "effort": "low", "impact": "high", "code": "df = df.drop(columns=['pin_code'])"},
            {"priority": "critical", "title": "Apply ThresholdOptimizer for 'caste'", "detail": "No retraining needed.", "effort": "low", "impact": "high", "code": "optimizer = ThresholdOptimizer(...)"},
        ],
        "india_compliance": {
            "domain": "loans", "n_violations": 3, "n_warnings": 2,
            "violations": [
                {"law": "DPDP Act 2023", "penalty": "Up to ₹250 crore", "reason": "DI 0.61 for SC/ST"},
                {"law": "RBI Fair Practices Code", "penalty": "Operational restrictions", "reason": "Discriminatory loan decisions"},
            ],
            "warnings": [{"law": "EU AI Act", "reason": "High-risk system — must comply by Aug 2026"}],
        },
    }
    return HTMLResponse(content=generate_report(sample_results))


# ────────────────────────────────────────────────────────────
# AI ASSISTANT PROXY — Keeps Anthropic API key server-side
# ────────────────────────────────────────────────────────────
import httpx
from pydantic import BaseModel
from typing import List

class ChatMessage(BaseModel):
    role: str      # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    scan_context: Optional[dict] = None   # Optional: inject real scan results into system prompt

FAIRSCAN_SYSTEM_PROMPT = """You are FairScan India's AI assistant — an expert on AI bias detection, Indian laws, and fairness in machine learning.

Key Indian laws to know:
- DPDP Act 2023: Digital Personal Data Protection Act — fines up to ₹250 crore
- RBI Fair Practices Code: requires transparent, non-discriminatory loan decisions
- Article 15 of the Constitution: no discrimination on caste, religion, sex, race
- Equal Remuneration Act 1976: equal pay and treatment regardless of gender
- EU AI Act: loan approval AI is HIGH RISK — Indian exporters must comply by Aug 2026

IMPORTANT RULES:
- Answer in plain simple English that non-technical people can understand
- Keep answers concise (3-5 sentences max unless the question needs more detail)
- Always relate answers to the Indian context — mention Indian laws, Indian examples
- If asked in Hindi, answer in Hindi
- Be direct and actionable — tell them exactly what to do
- Do NOT use markdown, bullet points, or headers in your response — plain flowing text only"""

@app.post("/api/chat", tags=["AI Assistant"])
async def chat(request: ChatRequest):
    """
    Proxy endpoint for the AI assistant chat.
    Keeps the Anthropic API key secure on the server — never exposed to the browser.

    Optionally pass scan_context (the results from /api/scan) to inject
    real scan findings into the system prompt for accurate, data-driven answers.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(500, "ANTHROPIC_API_KEY not set on server. Add it to your .env file.")

    # Build dynamic system prompt — inject real scan results if provided
    system = FAIRSCAN_SYSTEM_PROMPT
    if request.scan_context:
        sc = request.scan_context
        score = sc.get("bias_score", {})
        summary = sc.get("summary", {})
        compliance = sc.get("india_compliance", {})
        proxies = sc.get("proxy_features", [])
        system += f"""

Your user's actual scan results:
- Bias Risk Score: {score.get('score', 'N/A')}/100 ({score.get('risk_level', 'N/A')} RISK)
- Domain: {summary.get('domain', 'N/A')}
- Records scanned: {summary.get('n_samples', 'N/A'):,}
- Overall approval rate: {round(summary.get('overall_approval_rate', 0) * 100, 1)}%
- Indian law violations: {compliance.get('n_violations', 0)}
- Proxy features detected: {', '.join(p['feature'] for p in proxies) if proxies else 'none'}
- Score reasons: {'; '.join(score.get('reasons', []))}

Always refer to these real numbers when answering questions about their specific model."""

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type":         "application/json",
                    "x-api-key":            api_key,
                    "anthropic-version":    "2023-06-01",
                },
                json={
                    "model":      "claude-sonnet-4-20250514",
                    "max_tokens": 1000,
                    "system":     system,
                    "messages":   [{"role": m.role, "content": m.content} for m in request.messages],
                },
            )
        if response.status_code != 200:
            raise HTTPException(502, f"Anthropic API error: {response.text}")
        data = response.json()
        reply = data["content"][0]["text"] if data.get("content") else "Sorry, I could not generate a response."
        return {"reply": reply}

    except httpx.TimeoutException:
        raise HTTPException(504, "AI assistant timed out. Please try again.")
    except Exception as e:
        raise HTTPException(500, f"AI assistant error: {str(e)}")


# ────────────────────────────────────────────────────────────
# SAMPLE DATA DOWNLOAD — for demo/testing
# ────────────────────────────────────────────────────────────
@app.get("/api/sample-data", tags=["Data"])
def sample_data(domain: str = Query("loans", description="Domain: loans | hiring | healthcare | education"),
                n: int = Query(500, ge=100, le=5000, description="Number of rows"),
                seed: int = Query(-1, description="Random seed (-1 = random each call)")):
    """
    Download a synthetic biased dataset for testing FairScan.
    The dataset has injected bias — SC/ST and women are approved less often.
    Pass seed= for reproducibility, or omit for a fresh random dataset each time.
    """
    rng_seed = seed if seed >= 0 else np.random.randint(0, 999999)
    rng = np.random.default_rng(rng_seed)
    castes    = rng.choice(["General", "OBC", "SC", "ST"], n, p=[0.45, 0.28, 0.17, 0.10])
    genders   = rng.choice(["Male", "Female"], n)
    religions = rng.choice(["Hindu", "Muslim", "Christian", "Sikh", "Other"], n, p=[0.80, 0.14, 0.02, 0.02, 0.02])
    states    = rng.choice(["Maharashtra", "Delhi", "UP", "Bihar", "Karnataka", "TN", "WB"], n)
    ages      = rng.integers(22, 58, n)
    incomes   = rng.normal(55000, 20000, n).clip(15000, 200000).round(-3)
    credits   = rng.normal(680, 80, n).clip(300, 900).round()

    # PIN codes — correlated with caste (proxy feature)
    pin_base  = np.where(np.isin(castes, ["SC","ST"]), 400100, 400001)
    pins      = pin_base + rng.integers(0, 99, n)

    # Inject bias into approval probability
    prob = 1 / (1 + np.exp(-(credits - 650) / 60))
    prob[castes == "SC"]      *= 0.72
    prob[castes == "ST"]      *= 0.65
    prob[castes == "OBC"]     *= 0.87
    prob[genders == "Female"] *= 0.83
    prob[religions == "Muslim"] *= 0.89

    outcome    = (rng.random(n) < prob).astype(int)
    prediction = outcome.copy()
    # Add some model noise
    noise_mask = rng.random(n) < 0.08
    prediction[noise_mask] = 1 - prediction[noise_mask]

    df = pd.DataFrame({
        "applicant_id":   np.arange(1001, 1001 + n),
        "age":            ages,
        "gender":         genders,
        "caste":          castes,
        "religion":       religions,
        "state":          states,
        "monthly_income": incomes.astype(int),
        "credit_score":   credits.astype(int),
        "pin_code":       pins,
        "outcome":        outcome,
        "prediction":     prediction,
    })

    from fastapi.responses import StreamingResponse
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=fairscan_sample_{domain}_{n}rows.csv"}
    )
