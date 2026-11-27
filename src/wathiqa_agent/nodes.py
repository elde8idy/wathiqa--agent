
from __future__ import annotations

import json
import re

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.types import interrupt

from .config import Settings
from .policies import search_policies, category_limit
from .schemas import DOC_TYPES, ExtractedFields
from .state import ReviewState
from .tools import INVESTIGATION_TOOLS, TOOL_MAP, parse_document


def _parse_json(text: str) -> dict:
    """Tolerant JSON extraction from an LLM reply."""

    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else {}

    except json.JSONDecodeError:
        decoder = json.JSONDecoder()

        for match in re.finditer(r"\{", text):
            try:
                result, _ = decoder.raw_decode(text[match.start():])

                if isinstance(result, dict):
                    return result

            except json.JSONDecodeError:
                continue

    return {}


def make_nodes(llm: BaseChatModel, settings: Settings):
    """Return the node callables for the review graph."""

    def intake(state: ReviewState) -> dict:
        """Load the document text from a path or passed-in text."""

        if state.get("document_text"):
            return {}

        if state.get("document_path"):
            return {
                "document_text": parse_document.invoke(
                    {"path": state["document_path"]}
                )
            }

        raise ValueError(
            "intake: provide document_text or document_path"
        )

    def classify(state: ReviewState) -> dict:
        """Classify the document type with the LLM."""

        prompt = (
            "TASK=classify_document\n"
            "Classify the document into exactly one of: "
            + ", ".join(DOC_TYPES)
            + ".\nReply with only the label.\n"
            f"DOCUMENT:\n{state['document_text']}"
        )

        reply = llm.invoke([HumanMessage(content=prompt)])

        label = str(reply.content).strip().lower()

        if label not in DOC_TYPES:
            label = "unknown"

        return {"doc_type": label}

    def extract(state: ReviewState) -> dict:
        """Extract structured fields from the document with the LLM."""

        prompt = (
            "TASK=extract_fields\n"
            "Extract these fields from the document and reply with "
            "a JSON object only: "
            "vendor (string), amount (gross total, number), "
            "currency (string), date (string), "
            "category (one of: meals, travel, software, "
            "office_supplies, training, other), "
            "net_amount (number), vat_amount (number). "
            "Use null for anything not stated.\n"
            f"DOCUMENT:\n{state['document_text']}"
        )

        reply = llm.invoke([HumanMessage(content=prompt)])

        fields = ExtractedFields.model_validate(
            _parse_json(str(reply.content))
        )

        return {
            "extraction": fields.model_dump(),
            "confidence": fields.completeness(),
        }

    def retrieve_policies(state: ReviewState) -> dict:
        """Pull the policy rules relevant to this document."""

        extraction = state.get("extraction", {})

        query = " ".join(
            part
            for part in [
                state.get("doc_type", ""),
                extraction.get("category") or "",
                "expense policy rules limits required fields",
            ]
        )

        return {
            "policies": search_policies(query, top_k=3)
        }

    def validate(state: ReviewState) -> dict:
        """Perform deterministic compliance checks."""

        fields = ExtractedFields.model_validate(
            state.get("extraction", {})
        )

        flags: list[str] = []

        # Check required fields.
        for required in (
            "vendor",
            "date",
            "currency",
            "category",
            "amount",
        ):
            if getattr(fields, required) is None:
                flags.append(f"missing_fields:{required}")

        # Check expense category limit.
        limit = category_limit(fields.category)

        if (
            limit is not None
            and fields.amount is not None
            and fields.amount > limit
        ):
            flags.append(
                f"over_limit:{fields.category}:"
                f"{fields.amount}>{limit:g}"
            )

        # Check VAT calculation.
        if (
            fields.net_amount is not None
            and fields.vat_amount is not None
        ):
            expected_vat = round(
                fields.net_amount * settings.vat_rate, 2
            )

            if abs(fields.vat_amount - expected_vat) > 1.0:
                flags.append(
                    f"vat_mismatch:stated={fields.vat_amount}:"
                    f"expected={expected_vat}"
                )

            # Check gross total.
            if fields.amount is not None:
                expected_gross = round(
                    fields.net_amount + fields.vat_amount, 2
                )

                if abs(fields.amount - expected_gross) > 1.0:
                    flags.append(
                        f"total_mismatch:stated={fields.amount}:"
                        f"expected={expected_gross}"
                    )

        return {"flags": flags}

    def investigate(state: ReviewState) -> dict:
        """Agentic tool-calling loop to investigate flagged issues."""

        line_items = "\n".join(
            line
            for line in state["document_text"].splitlines()
            if re.search(r"\d+(?:\.\d+)?\s*x\s+.*@", line)
        )

        system = SystemMessage(
            content=(
                "TASK=plan_investigation\n"
                "You are verifying flagged issues on an expense "
                "document. Call a tool to double-check, or reply "
                "with a short findings summary when done."
            )
        )

        human = HumanMessage(
            content=(
                f"FLAGS: {json.dumps(state.get('flags', []))}\n"
                f"EXTRACTION: "
                f"{json.dumps(state.get('extraction', {}))}\n"
                f"LINE ITEMS:\n{line_items or '(none)'}\n"
                f"DOCUMENT:\n{state['document_text']}"
            )
        )

        agent = llm.bind_tools(INVESTIGATION_TOOLS)

        messages: list = [system, human]
        log: list[str] = []

        for _ in range(settings.max_investigation_steps):
            reply: AIMessage = agent.invoke(messages)

            messages.append(reply)

            if not getattr(reply, "tool_calls", None):
                if reply.content:
                    log.append(str(reply.content))

                break

            for call in reply.tool_calls:
                tool = TOOL_MAP[call["name"]]

                result = tool.invoke(call["args"])

                log.append(
                    f"tool {call['name']}({call['args']}) -> {result}"
                )

                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=call["id"],
                    )
                )

        return {"investigation": log}

    def gate(state: ReviewState) -> dict:
        """Decide whether the document can auto-approve."""

        fields = ExtractedFields.model_validate(
            state.get("extraction", {})
        )

        reasons: list[str] = []

        if state.get("flags"):
            reasons.append(
                "policy flags raised: "
                + ", ".join(state["flags"])
            )

        if (
            state.get("confidence", 0.0)
            < settings.confidence_threshold
        ):
            reasons.append(
                f"extraction confidence "
                f"{state.get('confidence', 0.0)} below "
                f"threshold {settings.confidence_threshold}"
            )

        if (
            fields.amount is not None
            and fields.amount > settings.auto_approve_limit
        ):
            reasons.append(
                f"amount {fields.amount} above auto-approve limit "
                f"{settings.auto_approve_limit}"
            )

        requires_human = bool(reasons)

        return {
            "requires_human": requires_human,
            "decision_reasons": reasons,
            "decision": (
                "pending_review"
                if requires_human
                else "auto_approved"
            ),
        }

    def human_review(state: ReviewState) -> dict:
        """Pause the graph until a human approves or rejects."""

        feedback = interrupt(
            {
                "type": "review_request",
                "flags": state.get("flags", []),
                "reasons": state.get("decision_reasons", []),
                "extraction": state.get("extraction", {}),
                "investigation": state.get("investigation", []),
            }
        )

        feedback_text = str(feedback).strip().lower()

        if feedback_text in {"approve", "approved"}:
            decision = "approved_after_review"

        elif feedback_text in {"reject", "rejected"}:
            decision = "rejected_after_review"

        else:
            decision = "rejected_after_review"

        return {
            "human_feedback": str(feedback),
            "decision": decision,
        }

    def finalize(state: ReviewState) -> dict:
        """Compose the final decision report."""

        fields = ExtractedFields.model_validate(
            state.get("extraction", {})
        )

        policies = state.get("policies", [])

        lines = [
            "# WATHQA Review Report",
            "",
            f"- Decision: **{state.get('decision', 'unknown')}**",
            f"- Document type: {state.get('doc_type', 'unknown')}",
            f"- Vendor: {fields.vendor or '-'}",
            f"- Date: {fields.date or '-'}",
            f"- Category: {fields.category or '-'}",
            f"- Net: "
            f"{fields.net_amount if fields.net_amount is not None else '-'}",
            f"- VAT: "
            f"{fields.vat_amount if fields.vat_amount is not None else '-'}",
            f"- Total: "
            f"{fields.amount if fields.amount is not None else '-'} "
            f"{fields.currency or ''}".rstrip(),
            f"- Extraction confidence: "
            f"{state.get('confidence', 0.0)}",
            "",
            "## Policy checks",
        ]

        if state.get("flags"):
            lines += [
                f"- FLAG: {flag}"
                for flag in state["flags"]
            ]
        else:
            lines.append("- all checks passed")

        lines.append("")
        lines.append("## Applicable policies")

        lines += [
            f"- {policy['id']} {policy['title']}: {policy['text']}"
            for policy in policies
        ] or ["- none"]

        if state.get("investigation"):
            lines.append("")
            lines.append("## Investigation log")

            lines += [
                f"- {entry}"
                for entry in state["investigation"]
            ]

        if state.get("decision_reasons"):
            lines.append("")
            lines.append("## Decision reasons")

            lines += [
                f"- {reason}"
                for reason in state["decision_reasons"]
            ]

        if state.get("human_feedback"):
            lines.append("")
            lines.append(
                "## Human reviewer\n"
                f"- feedback: {state['human_feedback']}"
            )

        return {
            "report": "\n".join(lines)
        }

    return {
        "intake": intake,
        "classify": classify,
        "extract": extract,
        "retrieve_policies": retrieve_policies,
        "validate": validate,
        "investigate": investigate,
        "gate": gate,
        "human_review": human_review,
        "finalize": finalize,
    }


def route_after_validate(state: ReviewState) -> str:
    """Route the graph after validation."""

    return (
        "investigate"
        if state.get("flags")
        else "gate"
    )


def route_after_gate(state: ReviewState) -> str:
    """Route the graph after the approval gate."""

    return (
        "human_review"
        if state.get("requires_human")
        else "finalize"
    )
