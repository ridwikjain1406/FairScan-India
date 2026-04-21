# ============================================================
# report_generator.py — HTML Compliance Report Generator
# FairScan India — Google Solution Challenge Edition
# ============================================================

from datetime import datetime
from typing import Dict, Any


def generate_report(scan_results: Dict[str, Any]) -> str:
    """
    Generate a self-contained, print-ready HTML compliance report
    from scan_results returned by BiasScanner.run_full_audit().

    Includes:
    - Executive summary with risk score
    - Group-level fairness metrics table
    - India law violations and penalties
    - Proxy feature findings
    - Remediation recommendations with code
    - SDG alignment section
    """
    now      = datetime.now().strftime("%B %d, %Y at %H:%M")
    score    = scan_results.get("bias_score", {})
    summary  = scan_results.get("summary", {})
    recs     = scan_results.get("recommendations", [])
    proxies  = scan_results.get("proxy_features", [])
    india    = scan_results.get("india_compliance", {})
    meta     = scan_results.get("meta", {})

    risk       = score.get("risk_level", "UNKNOWN")
    raw_score  = score.get("score", 0)
    risk_color = {"LOW": "#137333", "MEDIUM": "#B06000", "HIGH": "#C5221F"}.get(risk, "#666")
    risk_bg    = {"LOW": "#E6F4EA", "MEDIUM": "#FEF7E0", "HIGH": "#FCE8E6"}.get(risk, "#F1F3F4")
    domain     = summary.get("domain", meta.get("domain", "loans"))
    n_samples  = summary.get("n_samples", meta.get("n_rows", "N/A"))
    scanned_at = meta.get("scanned_at", now)

    # ── Recommendations HTML ──
    rec_rows = ""
    priority_colors = {"critical": "#C5221F", "high": "#B06000", "medium": "#1A73E8"}
    for i, r in enumerate(recs, 1):
        pc   = priority_colors.get(r.get("priority","medium"), "#1A73E8")
        code = f"<pre style='background:#f8f9fa;border:1px solid #e8eaed;padding:12px;border-radius:6px;font-size:11px;overflow-x:auto;font-family:monospace;'>{r.get('code','')}</pre>" if r.get("code") else ""
        law_badge = f"<span style='font-size:10px;background:#E8F0FE;color:#1A73E8;padding:2px 8px;border-radius:10px;margin-top:6px;display:inline-block;'>⚖️ {r.get('india_law','')}</span>" if r.get("india_law") else ""
        rec_rows += f"""
        <div style='border-left:4px solid {pc};background:#fff;border-radius:0 8px 8px 0;padding:16px 20px;margin-bottom:12px;box-shadow:0 1px 3px rgba(0,0,0,.08);'>
          <div style='display:flex;align-items:center;gap:10px;margin-bottom:8px;'>
            <span style='background:{pc};color:white;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;'>{r.get('priority','?').upper()}</span>
            <strong style='font-size:14px;color:#202124;'>{i}. {r.get('title','')}</strong>
          </div>
          <p style='font-size:13px;color:#5F6368;line-height:1.7;margin:0 0 8px;'>{r.get('detail','')}</p>
          {code}
          {law_badge}
          <div style='font-size:11px;color:#9AA0A6;margin-top:8px;'>Effort: {r.get('effort','?')} &nbsp;·&nbsp; Impact: {r.get('impact','?')}</div>
        </div>"""

    # ── Violations table ──
    violations_html = ""
    for v in india.get("violations", []):
        violations_html += f"""
        <tr>
          <td style='font-weight:500;'>{v.get('law','')}</td>
          <td>{v.get('reason','')}</td>
          <td style='color:#C5221F;font-weight:600;'>{v.get('penalty','')}</td>
          <td><span style='background:#FCE8E6;color:#C5221F;padding:2px 10px;border-radius:10px;font-size:11px;'>VIOLATED</span></td>
        </tr>"""

    for w in india.get("warnings", []):
        violations_html += f"""
        <tr>
          <td style='font-weight:500;'>{w.get('law','')}</td>
          <td>{w.get('reason','')}</td>
          <td style='color:#B06000;font-weight:600;'>Compliance required</td>
          <td><span style='background:#FEF7E0;color:#B06000;padding:2px 10px;border-radius:10px;font-size:11px;'>WARNING</span></td>
        </tr>"""

    # ── Proxy features table ──
    proxy_rows = ""
    for p in proxies:
        sev_color = "#C5221F" if p.get("severity") == "critical" else "#B06000"
        proxy_rows += f"""
        <tr>
          <td style='font-family:monospace;font-weight:500;'>{p.get('feature','')}</td>
          <td>{p.get('attribute','')}</td>
          <td style='font-family:monospace;color:{sev_color};font-weight:600;'>{p.get('correlation','')}</td>
          <td style='color:{sev_color};font-weight:600;'>{p.get('severity','').upper()}</td>
          <td style='font-size:12px;color:#5F6368;'>{p.get('recommendation','')}</td>
        </tr>"""

    # ── Score reasons ──
    reason_html = "".join(f"<li style='margin-bottom:4px;'>{r}</li>" for r in score.get("reasons", []))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>FairScan India — AI Bias Compliance Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #F8F9FA; color: #202124; }}
  .page {{ max-width: 960px; margin: 0 auto; padding: 40px 24px 80px; }}
  .stripe {{ height: 8px; background: linear-gradient(90deg,#4285F4 25%,#EA4335 25%,#EA4335 50%,#FBBC04 50%,#FBBC04 75%,#34A853 75%); }}
  h1 {{ font-size: 26px; font-weight: 700; color: #202124; margin-bottom: 4px; }}
  h2 {{ font-size: 17px; font-weight: 600; color: #202124; margin: 36px 0 14px; padding-bottom: 8px; border-bottom: 2px solid #E8EAED; }}
  .meta {{ font-size: 12px; color: #9AA0A6; font-family: monospace; margin-top: 4px; }}
  .risk-box {{ background: {risk_bg}; border: 2px solid {risk_color}; border-radius: 12px; padding: 24px 28px; margin: 20px 0; display: flex; align-items: center; gap: 24px; }}
  .risk-score {{ font-size: 64px; font-weight: 700; color: {risk_color}; line-height: 1; letter-spacing: -2px; }}
  .risk-label {{ font-size: 18px; font-weight: 700; color: {risk_color}; }}
  .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 20px 0; }}
  .metric {{ background: white; border: 1px solid #E8EAED; border-radius: 10px; padding: 16px; text-align: center; }}
  .metric-val {{ font-size: 28px; font-weight: 700; letter-spacing: -1px; }}
  .metric-lbl {{ font-size: 11px; color: #9AA0A6; margin-top: 4px; text-transform: uppercase; letter-spacing: .05em; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  th {{ background: #F8F9FA; padding: 11px 14px; text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: #5F6368; font-weight: 600; border-bottom: 1px solid #E8EAED; }}
  td {{ padding: 11px 14px; border-bottom: 1px solid #E8EAED; color: #202124; vertical-align: top; }}
  tr:last-child td {{ border-bottom: none; }}
  .sdg {{ background: white; border: 1px solid #E8EAED; border-radius: 10px; padding: 20px 24px; margin-top: 20px; display: flex; gap: 20px; align-items: flex-start; }}
  .sdg-badge {{ background: #E8272A; color: white; font-size: 24px; font-weight: 900; width: 56px; height: 56px; border-radius: 8px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }}
  .footer {{ margin-top: 48px; padding-top: 20px; border-top: 1px solid #E8EAED; font-size: 11px; color: #9AA0A6; font-family: monospace; }}
  @media print {{ body {{ background: white; }} .page {{ padding: 20px; }} }}
  @media (max-width: 600px) {{ .metrics {{ grid-template-columns: 1fr 1fr; }} .risk-box {{ flex-direction: column; }} }}
</style>
</head>
<body>
<div class="stripe"></div>
<div class="page">

  <!-- Header -->
  <div style="margin: 28px 0 20px; display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
    <div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
        <div style="display:flex;gap:4px;">
          <div style="width:10px;height:10px;border-radius:50%;background:#4285F4;"></div>
          <div style="width:10px;height:10px;border-radius:50%;background:#EA4335;"></div>
          <div style="width:10px;height:10px;border-radius:50%;background:#FBBC04;"></div>
          <div style="width:10px;height:10px;border-radius:50%;background:#34A853;"></div>
        </div>
        <span style="font-size:13px;font-weight:700;color:#4285F4;">FairScan India</span>
        <span style="font-size:11px;background:#E8F0FE;color:#1A73E8;padding:2px 8px;border-radius:10px;">AI Bias Compliance Report</span>
      </div>
      <h1>AI Bias Audit — {domain.title()} Domain</h1>
      <div class="meta">Scanned: {scanned_at[:10] if len(str(scanned_at)) > 10 else scanned_at} &nbsp;|&nbsp; {n_samples:,} records analyzed &nbsp;|&nbsp; Reference: {summary.get('reference_groups', {})} &nbsp;|&nbsp; Report ID: FSI-{abs(hash(str(scanned_at)))%100000:05d}</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:11px;color:#9AA0A6;">SDG Aligned</div>
      <div style="font-size:13px;font-weight:600;color:#E8272A;">🎯 SDG 10 · SDG 8</div>
    </div>
  </div>

  <!-- Risk Score -->
  <div class="risk-box">
    <div class="risk-score">{raw_score}</div>
    <div>
      <div class="risk-label">{risk} RISK — /100</div>
      <div style="font-size:13px;color:#5F6368;margin-top:6px;line-height:1.6;">
        {india.get('n_violations', 0)} Indian law violation(s) found &nbsp;·&nbsp; {india.get('n_warnings', 0)} warning(s)
      </div>
      <ul style="margin-top:10px;padding-left:18px;font-size:12px;color:#5F6368;line-height:1.8;">
        {reason_html}
      </ul>
    </div>
  </div>

  <!-- Key Metrics -->
  <div class="metrics">
    <div class="metric">
      <div class="metric-val" style="color:#4285F4;">{summary.get('overall_approval_rate', 0)*100:.1f}%</div>
      <div class="metric-lbl">Overall Approval Rate</div>
    </div>
    <div class="metric">
      <div class="metric-val" style="color:#34A853;">{summary.get('overall_accuracy', 0)*100:.1f}%</div>
      <div class="metric-lbl">Model Accuracy</div>
    </div>
    <div class="metric">
      <div class="metric-val" style="color:#EA4335;">{india.get('n_violations', 0)}</div>
      <div class="metric-lbl">Law Violations</div>
    </div>
    <div class="metric">
      <div class="metric-val" style="color:#FBBC04;">{len(proxies)}</div>
      <div class="metric-lbl">Proxy Features</div>
    </div>
  </div>

  <!-- India Law Compliance -->
  <h2>⚖️ Indian Law Compliance</h2>
  <p style="font-size:13px;color:#5F6368;margin-bottom:14px;">Domain: <strong>{domain.upper()}</strong> &nbsp;·&nbsp; Laws checked: {', '.join(india.get('laws_checked', []))}</p>
  {"<table><thead><tr><th>Law</th><th>Violation Reason</th><th>Penalty</th><th>Status</th></tr></thead><tbody>" + violations_html + "</tbody></table>" if violations_html else '<p style="color:#137333;font-size:13px;">✅ No violations detected.</p>'}

  <!-- Proxy Features -->
  <h2>🔍 Proxy Feature Findings</h2>
  <p style="font-size:13px;color:#5F6368;margin-bottom:14px;">Features correlated with protected attributes act as indirect discriminators even when the protected attribute itself is not used.</p>
  {"<table><thead><tr><th>Feature</th><th>Proxies For</th><th>Correlation</th><th>Severity</th><th>Action</th></tr></thead><tbody>" + proxy_rows + "</tbody></table>" if proxy_rows else '<p style="color:#137333;font-size:13px;">✅ No proxy features detected.</p>'}

  <!-- Remediation -->
  <h2>🔧 Remediation Recommendations</h2>
  <p style="font-size:13px;color:#5F6368;margin-bottom:16px;">{len(recs)} action(s) recommended, ordered by priority. Critical and High items can be completed without model retraining.</p>
  {rec_rows if rec_rows else '<p style="color:#5F6368;font-size:13px;">No recommendations generated.</p>'}

  <!-- SDG Alignment -->
  <h2>🌍 UN SDG Alignment</h2>
  <div class="sdg">
    <div class="sdg-badge">10</div>
    <div>
      <div style="font-size:14px;font-weight:600;margin-bottom:4px;">SDG 10 — Reduced Inequalities</div>
      <div style="font-size:13px;color:#5F6368;line-height:1.7;">FairScan India directly addresses SDG 10 by detecting and eliminating algorithmic discrimination based on caste, gender, religion, and region — ensuring equitable access to loans, jobs, healthcare, and education for all citizens regardless of background.</div>
    </div>
  </div>
  <div class="sdg" style="margin-top:12px;">
    <div class="sdg-badge" style="background:#1A73E8;">8</div>
    <div>
      <div style="font-size:14px;font-weight:600;margin-bottom:4px;">SDG 8 — Decent Work and Economic Growth</div>
      <div style="font-size:13px;color:#5F6368;line-height:1.7;">By ensuring AI hiring and lending systems are fair, FairScan India enables SC/ST, women, and minority applicants to access economic opportunities on merit — contributing to inclusive economic growth.</div>
    </div>
  </div>

  <!-- Footer -->
  <div class="footer">
    FairScan India — AI Bias Detection Platform &nbsp;|&nbsp; Google Solution Challenge &nbsp;|&nbsp;
    Protected attributes: {', '.join(summary.get('protected_attributes_analyzed', []))} &nbsp;|&nbsp;
    Thresholds: DI &gt; 0.80 (4/5 rule) · Parity gap &lt; 5% · EO diff &lt; 5% &nbsp;|&nbsp;
    Generated: {now}
  </div>

</div>
</body>
</html>"""
