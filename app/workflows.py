"""The specialized "assistant" workflows (project summary, README generation,
model validation, explainability, repository review, interview questions).

Every one of these is the same shape: retrieve broad context for a fixed
query, build a prompt, make one LLM call. The previous version of this
project had six separate classes (~600 lines total) that each hand-rolled
that same sequence. Here it's one data table plus one runner.
"""
from dataclasses import dataclass
from typing import Optional

import prompts
from llm import chat
from retrieval import build_context, retrieve
from vector_store import VectorStore


@dataclass
class Workflow:
    label: str
    query: str
    prompt_template: str
    k: int = 12


WORKFLOWS: dict[str, Workflow] = {
    "project_summary": Workflow(
        label="Project Summary",
        query="project overview architecture components technologies data pipeline model training",
        prompt_template=prompts.PROJECT_SUMMARY_PROMPT,
    ),
    "readme": Workflow(
        label="Generate README",
        query="README documentation setup installation usage instructions",
        prompt_template=prompts.README_PROMPT,
    ),
    "architecture_doc": Workflow(
        label="Architecture Documentation",
        query="architecture design system components modules data flow",
        prompt_template=prompts.ARCHITECTURE_DOC_PROMPT,
    ),
    "model_card": Workflow(
        label="Model Card",
        query="model training evaluation metrics performance intended use",
        prompt_template=prompts.MODEL_CARD_PROMPT,
    ),
    "validation_report": Workflow(
        label="Model Validation Report",
        query="model training validation evaluation data preprocessing leakage overfitting",
        prompt_template=prompts.VALIDATION_REPORT_PROMPT,
        k=15,
    ),
    "risk_analysis": Workflow(
        label="Risk Analysis",
        query="risk bias fairness ethical concerns security compliance",
        prompt_template=prompts.RISK_ANALYSIS_PROMPT,
        k=15,
    ),
    "explain_shap": Workflow(
        label="Explain SHAP Analysis",
        query="SHAP feature importance explainability values",
        prompt_template=prompts.EXPLAIN_SHAP_PROMPT,
    ),
    "explain_feature_importance": Workflow(
        label="Explain Feature Importance",
        query="feature importance feature selection ranking",
        prompt_template=prompts.EXPLAIN_FEATURE_IMPORTANCE_PROMPT,
    ),
    "explain_metrics": Workflow(
        label="Explain Evaluation Metrics",
        query="evaluation metrics accuracy precision recall F1 ROC AUC",
        prompt_template=prompts.EXPLAIN_METRICS_PROMPT,
    ),
    "code_review": Workflow(
        label="Code Quality Review",
        query="code structure modules functions classes design patterns",
        prompt_template=prompts.CODE_REVIEW_PROMPT,
        k=15,
    ),
    "engineering_review": Workflow(
        label="Engineering Practices Review",
        query="testing CI/CD version control documentation error handling security",
        prompt_template=prompts.ENGINEERING_REVIEW_PROMPT,
        k=15,
    ),
    "interview_questions": Workflow(
        label="Interview Questions",
        query="project implementation algorithms techniques design decisions",
        prompt_template=prompts.INTERVIEW_QUESTIONS_PROMPT,
    ),
}


def run_workflow(key: str, store: VectorStore, role: Optional[str] = None) -> dict:
    """Run a named assistant workflow: one retrieval call + one LLM call."""
    workflow = WORKFLOWS[key]
    results = retrieve(workflow.query, store, k=workflow.k)

    if not results:
        return {"text": "No indexed content yet -- upload and process a project first.", "sources": []}

    context, sources = build_context(results)
    system_prompt = workflow.prompt_template.format(context=context, role=role or "Data Scientist")
    answer = chat(system_prompt, [{"role": "user", "content": f"Please complete the {workflow.label} task."}])

    return {"text": answer, "sources": sources}
