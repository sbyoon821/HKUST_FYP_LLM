"""
Study Session Discrepancy Analysis Agent

Compares model-based concentration prediction (CCoT output) with user self-report,
quantifies discrepancy, and uses Snowflake Cortex LLM to reason about potential causes.
"""

import argparse
import json
import os
from datetime import datetime

from dotenv import load_dotenv
from snowflake.snowpark import Session


def score_to_label(score):
    """Map a 0-10 focus score to categorical label."""
    if score >= 7.0:
        return "GOOD"
    if score >= 4.0:
        return "MODERATE"
    return "POOR"


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def normalize_inputs(payload):
    """
    Support both nested and flat input payload formats.

    Preferred format:
    {
      "prediction": {...},
      "self_report": {...}
    }
    """
    prediction = payload.get("prediction")
    self_report = payload.get("self_report")

    if prediction is None:
        prediction = {
            "pred_focus_score": payload.get("pred_focus_score"),
            "pred_label": payload.get("pred_label"),
            "reasoning_summary": payload.get("reasoning_summary", ""),
            "key_evidence": payload.get("key_evidence", []),
        }

    if self_report is None:
        self_report = {
            "self_focus_score": payload.get("self_focus_score"),
            "agreement": payload.get("agreement"),
            "report_confidence": payload.get("report_confidence"),
            "reason_tag": payload.get("reason_tag", ""),
            "feedback_text": payload.get("feedback_text", ""),
        }

    return prediction, self_report


def validate_payload(prediction, self_report):
    """Basic schema validation for required fields and ranges."""
    required_pred = ["pred_focus_score", "pred_label", "reasoning_summary", "key_evidence"]
    required_self = ["self_focus_score", "agreement", "report_confidence", "reason_tag", "feedback_text"]

    for key in required_pred:
        if key not in prediction:
            raise ValueError(f"Missing prediction field: {key}")
    for key in required_self:
        if key not in self_report:
            raise ValueError(f"Missing self_report field: {key}")

    prediction["pred_focus_score"] = float(prediction["pred_focus_score"])
    self_report["self_focus_score"] = float(self_report["self_focus_score"])
    self_report["agreement"] = int(self_report["agreement"])
    self_report["report_confidence"] = float(self_report["report_confidence"])

    prediction["pred_focus_score"] = clamp(prediction["pred_focus_score"], 0.0, 10.0)
    self_report["self_focus_score"] = clamp(self_report["self_focus_score"], 0.0, 10.0)
    self_report["agreement"] = int(clamp(self_report["agreement"], 1, 5))
    self_report["report_confidence"] = clamp(self_report["report_confidence"], 0.0, 100.0)

    prediction["pred_label"] = str(prediction["pred_label"]).upper().strip()
    if prediction["pred_label"] not in {"GOOD", "MODERATE", "POOR"}:
        prediction["pred_label"] = score_to_label(prediction["pred_focus_score"])

    if not isinstance(prediction["key_evidence"], list):
        prediction["key_evidence"] = [str(prediction["key_evidence"])]

    prediction["reasoning_summary"] = str(prediction["reasoning_summary"])
    self_report["reason_tag"] = str(self_report["reason_tag"])
    self_report["feedback_text"] = str(self_report["feedback_text"])


def compute_discrepancy_metrics(prediction, self_report):
    """
    Compute deterministic discrepancy metrics before sending to LLM.

    Returns:
        dict with gap scores and derived consistency metrics
    """
    pred_score = prediction["pred_focus_score"]
    self_score = self_report["self_focus_score"]
    pred_label = prediction["pred_label"]
    self_label = score_to_label(self_score)

    score_gap = abs(pred_score - self_score)
    normalized_gap = score_gap / 10.0

    label_mismatch = 1 if pred_label != self_label else 0

    # Agreement scale: 1 (strong disagreement) to 5 (strong agreement)
    agreement = self_report["agreement"]
    agreement_discrepancy = (5 - agreement) / 4.0

    confidence = self_report["report_confidence"] / 100.0

    discrepancy_score = (
        0.50 * normalized_gap +
        0.25 * label_mismatch +
        0.15 * agreement_discrepancy +
        0.10 * confidence
    ) * 100.0

    discrepancy_score = clamp(discrepancy_score, 0.0, 100.0)

    if discrepancy_score >= 70:
        level = "HIGH"
    elif discrepancy_score >= 40:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "predicted_label_from_score": score_to_label(pred_score),
        "self_report_label": self_label,
        "score_gap": round(score_gap, 3),
        "normalized_gap": round(normalized_gap, 4),
        "label_mismatch": bool(label_mismatch),
        "agreement_discrepancy": round(agreement_discrepancy, 4),
        "confidence_factor": round(confidence, 4),
        "discrepancy_score": round(discrepancy_score, 2),
        "discrepancy_level": level,
    }


def extract_json_object(text):
    """Best-effort JSON extraction from model text output."""
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except Exception:
        pass

    first = stripped.find("{")
    last = stripped.rfind("}")
    if first != -1 and last != -1 and first < last:
        snippet = stripped[first:last + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return None
    return None


def build_discrepancy_prompt(prediction, self_report, metrics):
    """Create a strict JSON-output prompt for discrepancy reasoning."""
    return f"""You are a Study Session Discrepancy Analysis Agent.
Analyze discrepancy between model prediction (CCoT) and user self-report.

CCoT Prediction:
{json.dumps(prediction, ensure_ascii=False, indent=2)}

User Self Report:
{json.dumps(self_report, ensure_ascii=False, indent=2)}

Precomputed Metrics:
{json.dumps(metrics, ensure_ascii=False, indent=2)}

Your task:
1) Determine likely reasons for discrepancy.
2) Consider measurement noise, user perception bias, model blind spots, and contextual factors.
3) Keep reasoning practical and concise.
4) Propose actionable next steps to reduce future discrepancy.

Return STRICT JSON only (no markdown, no extra text) in this schema:
{{
  "summary": "short overall discrepancy interpretation",
  "root_causes": ["cause 1", "cause 2", "cause 3"],
  "evidence_alignment": {{
    "model_supports": ["..."] ,
    "user_supports": ["..."]
  }},
  "reliability_assessment": {{
    "model_reliability": "LOW|MEDIUM|HIGH",
    "self_report_reliability": "LOW|MEDIUM|HIGH",
    "confidence_note": "..."
  }},
  "recommended_actions": ["action 1", "action 2", "action 3"],
  "follow_up_questions": ["question 1", "question 2"]
}}
"""


def run_cortex_complete(session, model, prompt):
    """Call SNOWFLAKE.CORTEX.COMPLETE and return raw response text."""
    escaped_prompt = prompt.replace("'", "''")

    query = f"""
    SELECT SNOWFLAKE.CORTEX.COMPLETE(
        '{model}',
        '{escaped_prompt}'
    ) AS response
    """

    result = session.sql(query).collect()
    if not result:
        raise RuntimeError("No response from SNOWFLAKE.CORTEX.COMPLETE")
    return result[0]["RESPONSE"]


def analyze_discrepancy(prediction, self_report, model="claude-3-5-sonnet", session=None):
    """
    Main discrepancy analysis pipeline.

    Returns:
        dict containing deterministic metrics + model reasoning
    """
    validate_payload(prediction, self_report)
    metrics = compute_discrepancy_metrics(prediction, self_report)

    prompt = build_discrepancy_prompt(prediction, self_report, metrics)
    llm_raw = run_cortex_complete(session=session, model=model, prompt=prompt)

    llm_json = extract_json_object(llm_raw)
    if llm_json is None:
        llm_json = {
            "summary": "Model returned non-JSON output.",
            "root_causes": [],
            "evidence_alignment": {"model_supports": [], "user_supports": []},
            "reliability_assessment": {
                "model_reliability": "MEDIUM",
                "self_report_reliability": "MEDIUM",
                "confidence_note": "Unable to parse strict JSON from model output."
            },
            "recommended_actions": [],
            "follow_up_questions": [],
            "raw_response": llm_raw,
        }

    return {
        "prediction": prediction,
        "self_report": self_report,
        "discrepancy_metrics": metrics,
        "llm_analysis": llm_json,
        "raw_llm_response": llm_raw,
    }


def process_study_session_discrepancy(data_path, model="claude-3-5-sonnet", output_path="discrepancy_analysis_results.json"):
    """
    Load payload JSON, run discrepancy analysis, and save result JSON.

    Input JSON examples:
    1) Nested:
    {
      "prediction": {...},
      "self_report": {...}
    }

    2) Flat:
    {
      "pred_focus_score": ...,
      ...,
      "self_focus_score": ...,
      ...
    }
    """
    load_dotenv()

    connection_params = {
        "account": os.environ.get("SNOWFLAKE_ACCOUNT"),
        "user": os.environ.get("SNOWFLAKE_USER"),
        "password": os.environ.get("SNOWFLAKE_USER_PASSWORD"),
        "role": os.environ.get("SNOWFLAKE_ROLE"),
        "database": os.environ.get("SNOWFLAKE_DATABASE"),
        "schema": os.environ.get("SNOWFLAKE_SCHEMA"),
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE"),
    }

    print("=" * 100)
    print("Study Session Discrepancy Analysis Agent")
    print("=" * 100)
    print("Connecting to Snowflake...")

    session = Session.builder.configs(connection_params).create()
    print("✓ Connected to Snowflake")

    try:
        print(f"Loading input data from: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        prediction, self_report = normalize_inputs(payload)

        print(f"Running discrepancy analysis with model: {model}")
        result = analyze_discrepancy(
            prediction=prediction,
            self_report=self_report,
            model=model,
            session=session,
        )

        output = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model": model,
            "agent": "StudySessionDiscrepancyAnalyst",
            "result": result,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print("✓ Analysis complete")
        print(f"✓ Results saved to: {output_path}")

        metrics = result["discrepancy_metrics"]
        print("\nDiscrepancy Summary:")
        print(f"  Score Gap: {metrics['score_gap']}")
        print(f"  Discrepancy Score: {metrics['discrepancy_score']}")
        print(f"  Level: {metrics['discrepancy_level']}")

        return output

    finally:
        session.close()
        print("✓ Snowflake session closed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze discrepancy between CCoT prediction and user self-report"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to input JSON (prediction + self_report)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-sonnet",
        help="Cortex model name",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="discrepancy_analysis_results.json",
        help="Output JSON path",
    )

    args = parser.parse_args()

    process_study_session_discrepancy(
        data_path=args.data,
        model=args.model,
        output_path=args.output,
    )
