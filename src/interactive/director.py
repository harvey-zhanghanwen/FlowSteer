"""Inference-time Qwen Flow-Director loop over the strict AgentGraph Canvas."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from itertools import permutations, product
import json
import os
import random
import socket
import time
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .agent_workflow_env import AgentWorkflowEnv, AgentWorkflowStepResult
from .model_registry import ModelRegistry
from .tool_runtime import ToolRegistry
from .scientific_sampling import (
    GenerationPhase,
    SCIENTIFIC_SAMPLING_ALGORITHM,
    ScientificSamplingCoordinate,
    derive_generation_seed,
)


LEGACY_DIRECTOR_SYSTEM_PROMPT_V8 = """You are the Flow-Director. Incrementally build an executable AgentGraph. Follow the latest Canvas observation and return exactly one JSON object each turn.

Actions:
{"action":"add_subgraph","agents":[{"agent_id":"...","model_id":"...","contract":"...","role_family":"...","allowed_tools":[],"execution_mode":"reasoning|react|coding","artifact_type":"text","completion_condition":"..."}],"relations":[{"source_id":"...","target_id":"...","source_to_target":true,"target_to_source":false}],"output_agent_id":"..."}
{"action":"modify_agent","agent_id":"...","model_id":"...","contract":"...","role_family":"...","allowed_tools":[],"execution_mode":"reasoning|react|coding","artifact_type":"...","completion_condition":"..."}
{"action":"delete_agent","agent_id":"..."}
{"action":"set_relation","source_id":"...","target_id":"...","source_to_target":true,"target_to_source":false}
{"action":"set_output","agent_id":"..."}
{"action":"finish"}

An add_subgraph action adds one functional subgraph of one to three Agents and is executed once after the whole action is accepted. relations may be an empty array; output_agent_id is optional and belongs only at the top level of add_subgraph, never inside an Agent object. Use model_id values only from model_catalog. Every allowed_tools entry must be an exact tool_id from tool_catalog; action_names are Executor actions, not allowed_tools identifiers. execution_mode is execution semantics, not a fixed role; use reasoning unless a listed tool or environment requires react or coding. A directed relation routes the source artifact to the target; a bidirectional relation is one bounded two-Agent exchange. Describe each Agent's objective, required inputs, output artifact, and completion condition in concise ordinary text. Keep every contract faithful to the task's original relation, qualifiers, comparison criterion, and answer type; require source-grounded evidence when the answer depends on multiple facts. A completed semantic-answer artifact states one explicit bare answer span, not a sentence or question restatement, in the requested answer type and preserves its evidence-aligned lexical form, units, qualifiers, date, and full proper name. Independent evidence branches may merge at one semantic-answer Agent, but the Format Agent must remain a separate sink with one semantic predecessor. role_family is optional metadata, not a fixed Operator type. Inspect execution feedback and Canvas issues before selecting the next action. Use a distinct role_family "format" Output Agent only when the observation requires the exact-answer terminal protocol; it extracts one routed semantic answer and does not solve the task. Do not assume a fixed workflow topology or an unlisted Skill."""


# SkillFlow keeps the Supervisor instruction short, while FlowSteer exposes
# legal edits and execution feedback through the progressive Canvas.  This
# prompt therefore defines only the policy/environment boundary.  Task-solving
# recipes belong in evidence-gated Skills or graph-authored Agent contracts.
LEGACY_DIRECTOR_SYSTEM_PROMPT_V9 = """You are the Flow-Director. Incrementally edit the executable AgentGraph from the latest Canvas observation. Return exactly one valid JSON action each turn and no other text.

Legal actions are add_subgraph, modify_agent, delete_agent, set_relation, set_output, and finish. add_subgraph adds one functional subgraph of one to three Agents as one transaction. Use only model_id values from model_catalog and exact tool_id values from tool_catalog. execution_mode is reasoning, react, or coding.

A directed relation routes the source artifact to the target. A bidirectional relation performs one bounded two-Agent exchange. Each accepted edit is executed once, and its Canvas validation and execution feedback appear in the next observation. Inspect that state before choosing the next action. Use finish only when finish_admissibility is present and admissible. Do not assume a fixed workflow topology or an unlisted Skill."""

DIRECTOR_SYSTEM_PROMPT = """You are the Flow-Director. Incrementally edit the executable AgentGraph from the latest Canvas observation. Return exactly one valid JSON action each turn and no other text.

Use only action types listed in admissible_action_types, model_id values from model_catalog, and exact tool_id values from tool_catalog. add_subgraph adds one functional subgraph of one to three Agents as one transaction. execution_mode is reasoning, react, or coding.

A directed relation routes the source artifact to the target. A bidirectional relation performs one bounded two-Agent exchange. Each accepted edit is executed once, and its Canvas validation and execution feedback appear in the next observation. Inspect that state before choosing the next action. Use finish only when finish_admissibility is present and admissible. Do not assume a fixed workflow topology or an unlisted Skill."""

DIRECTOR_PROMPT_VERSION = "agentgraph.director.minimal-neutral.v10"
LEGACY_DIRECTOR_PROMPT_VERSION_V9 = "agentgraph.director.minimal-neutral.v9"
LEGACY_DIRECTOR_PROMPT_VERSION_V8 = "agentgraph.director.minimal-neutral.v8"
HOTPOTQA_DIRECTOR_PROMPT_VERSION = (
    "agentgraph.director.hotpotqa-semantic-recovery.v22"
)
QA_DIRECTOR_PROMPT_VERSION = "agentgraph.director.qa-semantic-recovery.v1"
LEGACY_HOTPOTQA_DIRECTOR_PROMPT_VERSION_V21 = (
    "agentgraph.director.hotpotqa-semantic-recovery.v21"
)
LEGACY_HOTPOTQA_DIRECTOR_PROMPT_VERSION_V20 = (
    "agentgraph.director.hotpotqa-semantic-recovery.v20"
)
LEGACY_HOTPOTQA_DIRECTOR_PROMPT_VERSION_V19 = (
    "agentgraph.director.hotpotqa-semantic-recovery.v19"
)
LEGACY_HOTPOTQA_DIRECTOR_PROMPT_VERSION_V18 = (
    "agentgraph.director.hotpotqa-semantic-recovery.v18"
)
LEGACY_HOTPOTQA_DIRECTOR_PROMPT_VERSION_V17 = (
    "agentgraph.director.hotpotqa-semantic-recovery.v17"
)
LEGACY_HOTPOTQA_DIRECTOR_PROMPT_VERSION_V16 = (
    "agentgraph.director.hotpotqa-semantic-recovery.v16"
)
LEGACY_HOTPOTQA_DIRECTOR_PROMPT_VERSION_V15 = (
    "agentgraph.director.hotpotqa-semantic-recovery.v15"
)
LEGACY_HOTPOTQA_DIRECTOR_PROMPT_VERSION_V14 = (
    "agentgraph.director.hotpotqa-semantic-recovery.v14"
)
LEGACY_HOTPOTQA_DIRECTOR_PROMPT_VERSION_V13 = (
    "agentgraph.director.hotpotqa-semantic-recovery.v13"
)
LEGACY_HOTPOTQA_DIRECTOR_PROMPT_VERSION_V12 = (
    "agentgraph.director.hotpotqa-semantic-recovery.v12"
)
LEGACY_HOTPOTQA_DIRECTOR_PROMPT_VERSION_V11 = (
    "agentgraph.director.hotpotqa-semantic-recovery.v11"
)
HOTPOTQA_SEMANTIC_PROTOCOL = "hotpotqa_verified_answer_slot_v1"
HOTPOTQA_SEMANTIC_LINEAGE_PROTOCOL = "hotpotqa_semantic_lineage_v2"
HOTPOTQA_ROLE_CONDITIONAL_PROTOCOL = (
    "hotpotqa_role_conditional_capabilities_v1"
)
QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL = "qa_verified_answer_lineage_v2"
_VERIFIED_QA_SEMANTIC_PROTOCOLS = frozenset(
    {
        HOTPOTQA_SEMANTIC_PROTOCOL,
        HOTPOTQA_SEMANTIC_LINEAGE_PROTOCOL,
        HOTPOTQA_ROLE_CONDITIONAL_PROTOCOL,
        QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL,
    }
)
PRESERVE_DIAGNOSE_REPAIR_AUGMENT_POLICY = (
    "preserve_diagnose_repair_augment"
)

# This is an explicitly selected HotpotQA policy.  The neutral v10 prompt above
# remains the default for every other dataset and for existing callers.
HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V11 = """You are the Flow-Director. Incrementally edit the executable AgentGraph from the latest Canvas observation. Return exactly one valid JSON action each turn and no other text.

Use only action types listed in admissible_action_types, model_id values from model_catalog, and exact tool_id values from tool_catalog. add_subgraph adds one functional subgraph of one to three Agents as one transaction. role_family names a semantic responsibility; execution_mode is only the execution schedule reasoning, react, or coding. Never define ReAct as an Agent role. When tools are needed, execution_mode react follows one bounded Thought -> Action(tool) -> Observation -> Thought -> Final schedule.

Use the strict FlowSteer Canvas action shapes below. Do not mix fields from different actions, and every relation contains both endpoint identifiers and both direction flags. output_agent_id is optional in add_subgraph: omit it until the complete terminal semantic lineage exists. If a workflow needs more than three Agents, add it through multiple accepted edits; never place more than three Agents in one add_subgraph.
{"action":"add_subgraph","agents":[{"agent_id":"...","model_id":"...","contract":"...","role_family":"...","allowed_tools":[],"execution_mode":"reasoning|react|coding","artifact_type":"text","completion_condition":"..."}],"relations":[{"source_id":"...","target_id":"...","source_to_target":true,"target_to_source":false}],"output_agent_id":"..."}
{"action":"modify_agent","agent_id":"...","model_id":"...","contract":"...","role_family":"...","allowed_tools":[],"execution_mode":"reasoning|react|coding","artifact_type":"...","completion_condition":"..."}
{"action":"delete_agent","agent_id":"..."}
{"action":"set_relation","source_id":"...","target_id":"...","source_to_target":true,"target_to_source":false}
{"action":"set_output","agent_id":"..."}
{"action":"finish"}

For the HotpotQA semantic protocol, preserve the original question scope, relation, qualifiers, comparison criterion, and answer type in every contract. Never introduce a narrower qualifier such as "singles" unless it is present in the question. Use role_family reasoner for the Agent that owns the semantic answer, verifier for the Agent that checks it, and format only for the terminal Formatter. A Reasoner must align each retrieved database fact with both (a) its proposition structure--subject/entity, predicate/relation, object/attribute value, and qualifiers--and (b) the answer slot actually requested by the question. The Reasoner alone determines the semantic answer and emits Question scope, Answer slot, Evidence propositions, Multi-hop chain, Candidate answer, and Evidence fields. Completion requires at least one successful non-empty qa-retrieval read. The Reasoner must declare allowed_tools ["qa-retrieval"] with execution_mode react; an additional retrieval Agent may augment evidence later but must not replace this capability or own the semantic answer. Route the Reasoner's receipt-bearing artifact directly into the Verifier. The Verifier checks that the candidate has explicit database evidence, the entity-to-attribute binding is correct, every required hop is complete, and the question scope is unchanged. It copies the identical candidate and emits Candidate answer, Evidence supported, Entity attribute binding correct, Multi-hop complete, Scope preserved, and Verification status fields; it must not select, replace, or invent a different candidate. A terminal Formatter receives only one passed Verifier artifact, never the original question, and copies the Candidate answer value exactly into the required output wrapper. It must not reason, verify, canonicalize, or reselect an answer.

For a comparison, if both retrieved values are unexpectedly equal, do not conclude a tie immediately. Recheck the original scope, both entity bindings, retrieved evidence, and whether any upstream contract narrowed the question before determining the candidate.

Recover from failures in this order: preserve -> diagnose -> repair -> augment. Preserve valid evidence, semantic answers, and working relations. Diagnose execution_mode, Tool capability, relation, and contract faults; repair the existing node or edge first, then augment with a repair, retrieval, or Verifier Agent if needed. Do not delete an Agent merely because it failed. Delete only when the node itself is unusable, a replacement has already taken over its artifact, and deletion cannot break semantic lineage. Inspect Canvas validation and execution feedback before every edit, and use finish only when finish_admissibility is present and admissible. Do not hard-code a benchmark sample, accepted answer, fixed evidence, or Ground Truth, and do not assume an unlisted Skill."""

HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V13 = HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V11.replace(
    '{"action":"set_output","agent_id":"..."}',
    "For set_relation only, setting both direction flags to false removes the "
    "existing relation between those endpoints; add_subgraph relations must keep "
    "at least one direction true. Remove only the faulty edge and preserve every "
    "working relation.\n"
    '{"action":"set_output","agent_id":"..."}',
    1,
).replace(
    "an additional retrieval Agent may augment evidence later but must not replace "
    "this capability or own the semantic answer. Route the Reasoner's receipt-bearing",
    "an additional retrieval Agent may augment evidence later but must not replace "
    "this capability or own the semantic answer; route that evidence into the "
    "Reasoner, never directly into the Verifier. Route the Reasoner's receipt-bearing",
    1,
)

HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V14 = HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V13.replace(
    "Use role_family reasoner for the Agent that owns the semantic answer, verifier",
    "Use exactly one role_family reasoner as the semantic-answer owner, verifier",
    1,
).replace(
    "an additional retrieval Agent may augment evidence later but must not replace "
    "this capability or own the semantic answer; route that evidence into the "
    "Reasoner, never directly into the Verifier.",
    "an additional retrieval Agent may augment evidence later but must use "
    "role_family evidence_retriever, must not replace this capability or own the "
    "semantic answer, and must route its evidence into the Reasoner, never directly "
    "into the Verifier. Before searching, resolve entity aliases and coreference "
    "from the supplied passages and retain that entity binding through every hop.",
    1,
).replace(
    "output_agent_id is optional in add_subgraph: omit it until the complete terminal "
    "semantic lineage exists.",
    "output_agent_id is optional in add_subgraph: omit it until the complete terminal "
    "semantic lineage exists. Once a Format output is selected, later augmentation "
    "must omit output_agent_id and preserve the selected output.",
    1,
).replace(
    "Inspect Canvas validation and execution feedback before every edit,",
    "When finish_admissibility exposes failure_attribution, repair its responsible "
    "Agent before augmentation and preserve every listed artifact. Inspect Canvas "
    "validation and execution feedback before every edit,",
    1,
)

HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V15 = HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V14.replace(
    "A Reasoner must align each retrieved database fact with both",
    "The original wh-word fixes the answer type: a Which-comparison returns the "
    "compared entity, not the comparison value; a who-question returns the person "
    "entity, not a possessive attribute phrase. A Reasoner must align each retrieved "
    "database fact with both",
    1,
)

HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V16 = HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V15.replace(
    "Do not delete an Agent merely because it failed.",
    "Do not delete an Agent merely because it failed. When recovery_state reports "
    "an active_semantic_lineage and redundant_after_replacement_takeover_agent_ids, "
    "replacement takeover is complete: remove only those reported disconnected "
    "duplicates and preserve the active lineage.",
    1,
)

# v17 keeps SkillFlow's Supervisor instruction compact and moves legality into
# the request-scoped constrained-decoding schema projected by the latest
# FlowSteer Canvas.  The semantic responsibilities below are task invariants,
# not a fixed workflow template: the Director still chooses the graph size,
# model assignment, evidence branches, and directed/reciprocal relations.
HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V17 = """You are the Flow-Director. Incrementally edit the executable AgentGraph from the latest Canvas observation. Return exactly one JSON action and no other text.

Use only the current admissible_action_types, action_target_domains, model_catalog model_id values, and exact tool_catalog tool_id values. add_subgraph adds one functional subgraph of one to three Agents and then executes it once. modify_agent repairs one existing Agent; set_relation changes one directed or reciprocal communication edge; set_output selects the terminal Agent; finish submits the current executed result. role_family is a semantic responsibility. execution_mode is only reasoning, react, or coding. ReAct is never a role; with a Tool it is the bounded Thought -> Action(tool) -> Observation -> Thought -> Final schedule.

For HotpotQA, preserve the original question scope, relation, qualifiers, comparison criterion, answer type, and answer cardinality. Exactly one Reasoner owns the semantic answer. It may retrieve with qa-retrieval in execution_mode react or consume explicit read evidence from a direct Retriever predecessor; in either case it resolves aliases and coreference, represents each retrieved fact as a subject--predicate--object/attribute proposition with qualifiers, aligns the requested answer slot to one proposition argument, completes every required hop, and selects one minimal evidence-supported answer surface. For a single-value slot it must not return an alias list, appositive gloss, or a redundant answer-type noun. When an identity bridge links an alias or stage name to a canonical person/entity name, select the surface that fills the question's semantic type; keep the alias as bridge evidence. A Which-comparison returns the compared entity, not its value. A who-question returns the person, not a possessive attribute. If compared values are unexpectedly equal, recheck the unchanged question scope, both entity bindings, retrieved evidence, and contract qualifiers before deciding.

The Verifier consumes only the Reasoner's semantic artifact. It checks explicit database evidence, entity--attribute and alias binding, answer-slot type/cardinality, complete multi-hop support, minimal answer surface, and unchanged scope. It copies the identical candidate and never selects or invents another answer. The terminal Formatter consumes only one supported Verifier artifact and copies that candidate into the required wrapper; it never reasons, verifies, canonicalizes, or reselects.

Recover in this order: preserve -> diagnose -> repair -> augment. Preserve valid evidence, semantic answers, working relations, and Output identity. Repair the failure_attribution responsible Agent or relation before adding another Agent. For a transient provider failure, keep the failed Agent's role, contract, tools, and relations and modify only its model_id to a catalog model on a different provider when available. Delete only an Agent listed as deletable after a replacement artifact has taken over its downstream responsibility. Never hard-code a sample, answer, evidence span, Ground Truth, or evaluator result, and never assume an unlisted Skill."""


# v18 adds only one Canvas-legality condition exposed by the real progressive
# edit boundary.  It does not prescribe a graph template or task solution.
HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V18 = HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V17.replace(
    "add_subgraph adds one functional subgraph of one to three Agents and then "
    "executes it once.",
    "add_subgraph adds one functional subgraph of one to three Agents and then "
    "executes it once. Every relation endpoint and non-null output_agent_id in "
    "add_subgraph must name an Agent already on the Canvas or declared in that "
    "same action; never reference a future Agent.",
    1,
)

# v19 keeps the v18 policy and makes the runtime declarations required by the
# HotpotQA semantic protocol explicit.  ReAct remains an execution schedule,
# not an Agent role, and FINISH follows the progressive Canvas terminal gate.
HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V19 = HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V18.replace(
    "role_family is a semantic responsibility. execution_mode is only reasoning, "
    "react, or coding.",
    "Every Agent declaration includes role_family, execution_mode, and "
    "allowed_tools. role_family is a semantic responsibility. execution_mode is "
    "only reasoning, react, or coding.",
    1,
).replace(
    "It may retrieve with qa-retrieval in execution_mode react or consume explicit "
    "read evidence from a direct Retriever predecessor; in either case it resolves",
    "It uses execution_mode react and declares qa-retrieval in allowed_tools; it "
    "resolves",
    1,
).replace(
    "Never hard-code a sample, answer, evidence span, Ground Truth, or evaluator "
    "result, and never assume an unlisted Skill.",
    "When finish_admissibility.admissible is true, submit finish. Never hard-code "
    "a sample, answer, evidence span, Ground Truth, or evaluator result, and never "
    "assume an unlisted Skill.",
    1,
)

# v20 keeps the neutral progressive-Canvas search space and makes two measured
# recovery boundaries explicit: contracts cannot precommit a semantic answer
# before execution, and non-provider ReAct failures repair the current
# contract/completion condition while retaining public Action--Observation
# state.  Neither rule supplies a workflow template or task answer.
HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V20 = HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V19.replace(
    "Exactly one Reasoner owns the semantic answer.",
    "Exactly one Reasoner owns the semantic answer. Before execution, an Agent "
    "contract states only the unchanged scope, requested relation and answer-slot "
    "and evidence obligations; it must not predict or embed a concrete candidate "
    "answer, alias, value, or evidence span.",
    1,
).replace(
    "For a transient provider failure, keep the failed Agent's role, contract, "
    "tools, and relations and modify only its model_id to a catalog model on a "
    "different provider when available.",
    "For a transient provider failure, keep the failed Agent's role, contract, "
    "tools, and relations and modify only its model_id to a catalog model on a "
    "different provider when available. For ReAct exhaustion or a semantic "
    "completion-schema error, preserve the public Action--Observation history "
    "and Tool receipts and repair the responsible Agent's contract or completion "
    "condition before changing models or adding a replacement.",
    1,
)

# v21 retains v20's neutral FlowSteer action search space.  Its only prompt
# delta exposes the generic SkillFlow public repair instruction already carried
# by Runtime feedback; Canvas admission remains authoritative and answer-free.
HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V21 = HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V20.replace(
    "condition before changing models or adding a replacement.",
    "condition before changing models or adding a replacement. When a public "
    "repair_instruction is present, apply it only as a generic output-schema "
    "obligation for that Agent; do not copy any task candidate, value, alias, "
    "or evidence span into the contract.",
    1,
)

# v22 exposes only the newly authoritative state-conditioned recovery domains.
# It does not prescribe a graph template: retrieval/repair fan-in and reciprocal
# Reasoner--Verifier communication remain sampled when Canvas admits them.
HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V22 = HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V21.replace(
    "Delete only an Agent listed as deletable after a replacement artifact has "
    "taken over its downstream responsibility.",
    "The current action mask is authoritative: when it exposes only modify_agent, "
    "repair that responsible Agent before augmentation; when add_subgraph is "
    "available, sample new semantic roles only from admitted_new_role_families. "
    "Delete only an Agent listed as deletable after a replacement artifact has "
    "taken over its downstream responsibility.",
    1,
)

# NECESSARY_ADAPTATION: TriviaQA uses the same FlowSteer Canvas and the same
# HotpotQA-tested semantic/recovery policy.  Only the benchmark name is
# generalized here; spelling/alias/query retry mechanics remain in the
# SkillFlow-compatible retrieval Action--Observation contract rather than in
# the Director prompt.  This keeps the Director instruction short and avoids
# prescribing a fixed graph topology.
QA_DIRECTOR_SYSTEM_PROMPT_V1 = (
    HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V22.replace(
        "For HotpotQA,",
        "For evidence-grounded question answering,",
        1,
    ).replace(
        "A who-question returns the person, not a possessive attribute.",
        "A who-question returns the evidence-supported answer-bearing entity, "
        "which may be a person or organization, not a possessive attribute.",
        1,
    )
)


def verified_qa_semantic_protocol(value: object) -> bool:
    """Return whether the shared evidence-lineage Canvas policy is active."""

    return value in _VERIFIED_QA_SEMANTIC_PROTOCOLS


def flexible_hotpotqa_semantic_protocol(value: object) -> bool:
    """Return whether HotpotQA terminal lineage is topology-neutral."""

    return value in {
        HOTPOTQA_SEMANTIC_LINEAGE_PROTOCOL,
        HOTPOTQA_ROLE_CONDITIONAL_PROTOCOL,
    }


def role_conditional_hotpotqa_protocol(value: object) -> bool:
    """Return whether semantic roles are optional per-Agent capabilities."""

    return value == HOTPOTQA_ROLE_CONDITIONAL_PROTOCOL


def director_system_prompt_for_version(prompt_version: str) -> str:
    """Resolve one explicitly versioned Director policy without changing v10."""

    if not isinstance(prompt_version, str) or not prompt_version.strip():
        raise ValueError("Director prompt_version must be non-empty text")
    normalized = prompt_version.strip()
    by_version = {
        DIRECTOR_PROMPT_VERSION: DIRECTOR_SYSTEM_PROMPT,
        LEGACY_DIRECTOR_PROMPT_VERSION_V9: LEGACY_DIRECTOR_SYSTEM_PROMPT_V9,
        LEGACY_DIRECTOR_PROMPT_VERSION_V8: LEGACY_DIRECTOR_SYSTEM_PROMPT_V8,
        # These are the two historical v8 experiment labels still present in
        # checked-in evaluation configs.  Their exact transcript policy is the
        # canonical v8 prompt above.
        "agentgraph.director.constrained-action.skillflow-qa.v8": (
            LEGACY_DIRECTOR_SYSTEM_PROMPT_V8
        ),
        "agentgraph.director.skillflow_continuation_v8": (
            LEGACY_DIRECTOR_SYSTEM_PROMPT_V8
        ),
        HOTPOTQA_DIRECTOR_PROMPT_VERSION: HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V22,
        QA_DIRECTOR_PROMPT_VERSION: QA_DIRECTOR_SYSTEM_PROMPT_V1,
        LEGACY_HOTPOTQA_DIRECTOR_PROMPT_VERSION_V21: (
            HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V21
        ),
        LEGACY_HOTPOTQA_DIRECTOR_PROMPT_VERSION_V20: (
            HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V20
        ),
        LEGACY_HOTPOTQA_DIRECTOR_PROMPT_VERSION_V19: (
            HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V19
        ),
        LEGACY_HOTPOTQA_DIRECTOR_PROMPT_VERSION_V18: (
            HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V18
        ),
        LEGACY_HOTPOTQA_DIRECTOR_PROMPT_VERSION_V17: (
            HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V17
        ),
        LEGACY_HOTPOTQA_DIRECTOR_PROMPT_VERSION_V16: (
            HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V16
        ),
        LEGACY_HOTPOTQA_DIRECTOR_PROMPT_VERSION_V15: (
            HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V15
        ),
        LEGACY_HOTPOTQA_DIRECTOR_PROMPT_VERSION_V14: (
            HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V14
        ),
        LEGACY_HOTPOTQA_DIRECTOR_PROMPT_VERSION_V13: (
            HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V13
        ),
        LEGACY_HOTPOTQA_DIRECTOR_PROMPT_VERSION_V12: (
            HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V11
        ),
        LEGACY_HOTPOTQA_DIRECTOR_PROMPT_VERSION_V11: (
            HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V11
        ),
    }
    # Older experiment and unit-test receipts used arbitrary version labels
    # (for example ``prompt-v1``) while executing the then-current default
    # prompt.  Preserve that metadata compatibility by resolving unrecognized
    # legacy labels to neutral v10; HotpotQA v11 is selected only by its exact
    # version above.
    return by_version.get(normalized, DIRECTOR_SYSTEM_PROMPT)


_SUPPORTED_DIRECTOR_SYSTEM_PROMPTS = frozenset(
    {
        DIRECTOR_SYSTEM_PROMPT,
        HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V11,
        HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V13,
        HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V14,
        HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V15,
        HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V16,
        HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V17,
        HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V18,
        HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V19,
        HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V20,
        HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V21,
        HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V22,
        QA_DIRECTOR_SYSTEM_PROMPT_V1,
        LEGACY_DIRECTOR_SYSTEM_PROMPT_V9,
        LEGACY_DIRECTOR_SYSTEM_PROMPT_V8,
    }
)


_NON_EMPTY_STRING_SCHEMA = {"type": "string", "minLength": 1}
# Qwen3.5/SGLang returns the sampled EOS token in ``text`` even when the
# JSON-Schema grammar has already closed the object.  Keep that token in the
# exact generation receipt, but admit it as the sole transport-level suffix
# when a hierarchical phase parses its JSON payload.
_QWEN_JSON_EOS_TEXT = "<|endoftext|>"
_AGENT_SPEC_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["agent_id", "model_id", "contract"],
    "properties": {
        "agent_id": _NON_EMPTY_STRING_SCHEMA,
        "model_id": _NON_EMPTY_STRING_SCHEMA,
        "contract": _NON_EMPTY_STRING_SCHEMA,
        "role_family": _NON_EMPTY_STRING_SCHEMA,
        "allowed_tools": {
            "type": "array",
            "items": _NON_EMPTY_STRING_SCHEMA,
            "uniqueItems": True,
        },
        "execution_mode": {"enum": ["reasoning", "react", "coding"]},
        "artifact_type": _NON_EMPTY_STRING_SCHEMA,
        "completion_condition": _NON_EMPTY_STRING_SCHEMA,
    },
}
_RELATION_SPEC_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "source_id",
        "target_id",
        "source_to_target",
        "target_to_source",
    ],
    "properties": {
        "source_id": _NON_EMPTY_STRING_SCHEMA,
        "target_id": _NON_EMPTY_STRING_SCHEMA,
        "source_to_target": {"type": "boolean"},
        "target_to_source": {"type": "boolean"},
    },
    "anyOf": [
        {"properties": {"source_to_target": {"const": True}}},
        {"properties": {"target_to_source": {"const": True}}},
    ],
}
_MUTABLE_AGENT_PROPERTIES = {
    "model_id": _NON_EMPTY_STRING_SCHEMA,
    "contract": _NON_EMPTY_STRING_SCHEMA,
    "role_family": _NON_EMPTY_STRING_SCHEMA,
    "allowed_tools": {
        "type": "array",
        "items": _NON_EMPTY_STRING_SCHEMA,
        "uniqueItems": True,
    },
    "execution_mode": {"enum": ["reasoning", "react", "coding"]},
    "artifact_type": _NON_EMPTY_STRING_SCHEMA,
    "completion_condition": _NON_EMPTY_STRING_SCHEMA,
}
DIRECTOR_ACTION_JSON_SCHEMA = {
    "type": "object",
    "oneOf": [
        {
            "additionalProperties": False,
            "required": ["action", "agents", "relations"],
            "properties": {
                "action": {"const": "add_subgraph"},
                "agents": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": _AGENT_SPEC_JSON_SCHEMA,
                },
                "relations": {
                    "type": "array",
                    "items": _RELATION_SPEC_JSON_SCHEMA,
                },
                "output_agent_id": {
                    "anyOf": [_NON_EMPTY_STRING_SCHEMA, {"type": "null"}]
                },
            },
        },
        {
            "additionalProperties": False,
            "required": ["action", "agent_id", "model_id", "contract"],
            "properties": {
                "action": {"const": "add_agent"},
                "agent_id": _NON_EMPTY_STRING_SCHEMA,
                "model_id": _NON_EMPTY_STRING_SCHEMA,
                "contract": _NON_EMPTY_STRING_SCHEMA,
                "role_family": _NON_EMPTY_STRING_SCHEMA,
                "allowed_tools": _MUTABLE_AGENT_PROPERTIES["allowed_tools"],
                "execution_mode": _MUTABLE_AGENT_PROPERTIES["execution_mode"],
                "artifact_type": _NON_EMPTY_STRING_SCHEMA,
                "completion_condition": _NON_EMPTY_STRING_SCHEMA,
            },
        },
        {
            "additionalProperties": False,
            "required": ["action", "agent_id"],
            "properties": {
                "action": {"const": "modify_agent"},
                "agent_id": _NON_EMPTY_STRING_SCHEMA,
                **_MUTABLE_AGENT_PROPERTIES,
            },
            "anyOf": [
                {"required": [field_name]}
                for field_name in _MUTABLE_AGENT_PROPERTIES
            ],
        },
        {
            "additionalProperties": False,
            "required": ["action", "agent_id"],
            "properties": {
                "action": {"const": "delete_agent"},
                "agent_id": _NON_EMPTY_STRING_SCHEMA,
            },
        },
        {
            "additionalProperties": False,
            "required": [
                "action",
                "source_id",
                "target_id",
                "source_to_target",
                "target_to_source",
            ],
            "properties": {
                "action": {"const": "set_relation"},
                **_RELATION_SPEC_JSON_SCHEMA["properties"],
            },
        },
        {
            "additionalProperties": False,
            "required": ["action", "agent_id"],
            "properties": {
                "action": {"const": "set_output"},
                "agent_id": _NON_EMPTY_STRING_SCHEMA,
            },
        },
        {
            "additionalProperties": False,
            "required": ["action"],
            "properties": {"action": {"const": "finish"}},
        },
    ],
}
DIRECTOR_ACTION_SCHEMA_VERSION = "agentgraph.canvas-action-json-schema.v1"
DIRECTOR_SGLANG_SAMPLING_SCHEMA_VERSION = (
    "agentgraph.sglang-flat-action-sampling-schema.v1"
)
DIRECTOR_PROGRESSIVE_ACTION_MASK_PROFILE = (
    "progressive_add_subgraph_then_finish"
)
DIRECTOR_MODEL_ADMISSIBLE_ACTION_MASK_PROFILE = (
    "model_admissible_canvas_actions"
)
DIRECTOR_STATE_CONDITIONED_ACTION_SCHEMA_VERSION = (
    "agentgraph.state-conditioned-action-mask.v2"
)
DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V1 = (
    "agentgraph.model-admissible-action-mask.v1"
)
DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION = (
    "agentgraph.model-admissible-action-mask.v2"
)
DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3 = (
    "agentgraph.model-admissible-action-mask.v3"
)
DIRECTOR_ACTION_TARGET_DOMAIN_SCHEMA_VERSION = (
    "agentgraph.live-action-target-domains.v5"
)
DIRECTOR_ACTION_JSON_SCHEMA_TEXT = json.dumps(
    DIRECTOR_ACTION_JSON_SCHEMA,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
)


def director_action_json_schema_text(actions: Sequence[str]) -> str:
    """Render the existing parser schema for one configured Canvas profile."""

    if isinstance(actions, (str, bytes)) or not actions:
        raise ValueError("Canvas actions must be a non-empty sequence")
    by_name = {
        branch["properties"]["action"]["const"]: branch
        for branch in DIRECTOR_ACTION_JSON_SCHEMA["oneOf"]
    }
    normalized = tuple(actions)
    if (
        any(not isinstance(action, str) or action not in by_name for action in normalized)
        or len(normalized) != len(set(normalized))
    ):
        raise ValueError("Canvas actions contain an unknown or duplicate action")
    return json.dumps(
        {
            "type": "object",
            "oneOf": [by_name[action] for action in normalized],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def director_sglang_sampling_json_schema_text(actions: Sequence[str]) -> str:
    """Render the evaluation-only constrained-decoding schema for SGLang.

    NECESSARY_ADAPTATION (SGLang 0.5.15): the deployed constrained decoder
    merges mutually exclusive top-level ``oneOf`` branches. Sampling therefore
    uses one flat top-level object. The unchanged ``AgentActionParser`` remains
    authoritative for action-specific required fields and semantics.
    """

    strict_profile = json.loads(director_action_json_schema_text(actions))
    branches = strict_profile["oneOf"]
    properties: dict[str, Any] = {
        "action": {
            "enum": [branch["properties"]["action"]["const"] for branch in branches]
        }
    }
    for branch in branches:
        for field_name, field_schema in branch["properties"].items():
            if field_name == "action":
                continue
            existing = properties.get(field_name)
            if existing is not None and existing != field_schema:
                raise ValueError(
                    f"Canvas actions disagree on sampling schema for {field_name}"
                )
            properties[field_name] = field_schema
    return json.dumps(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["action"],
            "properties": properties,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def director_state_conditioned_sampling_json_schema_text(action: str) -> str:
    """Render one exact SGLang action branch without JSON-Schema intersections.

    The deployed xgrammar converter does not preserve an object-level
    ``required``/``properties`` intersection with the relation schema's
    direction ``anyOf``.  It consequently admitted relation objects that
    contained only one direction flag.  This evaluation-only compatibility
    schema expresses the same relation invariant as two self-contained object
    alternatives.  The unchanged strict ``AgentActionParser`` remains the
    authoritative post-generation validator.
    """

    by_name = {
        branch["properties"]["action"]["const"]: branch
        for branch in DIRECTOR_ACTION_JSON_SCHEMA["oneOf"]
    }
    if action not in by_name:
        raise ValueError("state-conditioned sampling received an unknown action")
    # A JSON round trip makes a request-local copy without changing the strict
    # parser schema shared by the rest of the runtime.
    branch = json.loads(json.dumps(by_name[action]))
    if action == "add_subgraph":
        relation_schema = branch["properties"]["relations"]["items"]
        relation_properties = relation_schema["properties"]
        relation_required = relation_schema["required"]
        branch["properties"]["relations"]["items"] = {
            # ``anyOf`` preserves the bidirectional case, which satisfies both
            # complete object alternatives.  ``oneOf`` would incorrectly
            # reject a reciprocal relation.
            "anyOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": relation_required,
                    "properties": {
                        **relation_properties,
                        "source_to_target": {"const": True},
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": relation_required,
                    "properties": {
                        **relation_properties,
                        "target_to_source": {"const": True},
                    },
                },
            ]
        }
    return json.dumps(
        branch,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def director_model_admissible_sampling_json_schema_text_v1(
    actions: Sequence[str],
) -> str:
    """Render the legacy v1 current-Canvas sampling schema.

    A singleton domain can retain the strict action-specific schema.  Multiple
    actions use the historical flat SGLang compatibility schema.  This function
    is retained only so persisted v1 receipts remain exactly replayable.
    """

    normalized = tuple(actions)
    if not normalized:
        raise ValueError("model-admissible Canvas actions must be non-empty")
    if len(normalized) == 1:
        return director_state_conditioned_sampling_json_schema_text(normalized[0])
    return director_sglang_sampling_json_schema_text(normalized)


def director_model_admissible_sampling_json_schema_text(
    actions: Sequence[str],
) -> str:
    """Render the v2 first-stage action-discriminator schema.

    SGLang 0.5.15 does not preserve branch-local fields in a multi-action JSON
    Schema union.  V2 therefore samples only the legal action discriminator in
    stage one; the native receipt client then samples the complete action under
    that action's exact singleton schema.  No sampled field is rewritten.
    """

    normalized = tuple(actions)
    # Reuse the strict renderer for unknown/empty/duplicate validation.
    director_action_json_schema_text(normalized)
    return json.dumps(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["action"],
            "properties": {"action": {"enum": list(normalized)}},
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def director_model_admissible_sampling_json_schema_text_v3(
    actions: Sequence[str],
) -> str:
    """Render the v3 live-domain action discriminator.

    The first phase is intentionally byte-equivalent to the v2 discriminator;
    its new version applies only to the later request-local parameter phases.
    Keeping a separate renderer makes persisted v2 receipts unambiguous.
    """

    return director_model_admissible_sampling_json_schema_text(actions)


def director_live_action_target_domains_json(
    actions: Sequence[str],
    action_target_domains: Mapping[str, Any],
) -> str:
    """Canonicalize the exact current-Canvas domains carried by a v3 request."""

    normalized_actions = tuple(actions)
    director_model_admissible_sampling_json_schema_text_v3(normalized_actions)
    if not isinstance(action_target_domains, Mapping):
        raise ValueError("live action target domains must be an object")
    if set(action_target_domains) != set(normalized_actions):
        raise ValueError("live action target domains must match admitted actions")
    selected: dict[str, Any] = {}
    for action in normalized_actions:
        domain = action_target_domains.get(action)
        if not isinstance(domain, Mapping):
            raise ValueError(f"live target domain for {action} must be an object")
        selected[action] = dict(domain)
    try:
        # The round trip rejects non-JSON runtime objects and gives receipts a
        # deterministic representation without changing any sampled action.
        normalized = json.loads(
            json.dumps(
                selected,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("live action target domains must be JSON serializable") from exc
    director_validate_live_action_target_domains(normalized_actions, normalized)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


DIRECTOR_MODIFY_AGENT_FIELDS = tuple(_MUTABLE_AGENT_PROPERTIES)


def director_modify_agent_field_selector_json_schema_text(
    fields: Optional[Sequence[str]] = None,
) -> str:
    """Render the v2 atomic MODIFY-field selector."""

    admitted_fields = DIRECTOR_MODIFY_AGENT_FIELDS if fields is None else tuple(fields)
    if (
        not admitted_fields
        or any(field not in DIRECTOR_MODIFY_AGENT_FIELDS for field in admitted_fields)
        or len(admitted_fields) != len(set(admitted_fields))
    ):
        raise ValueError("modify_agent field domain is empty or invalid")

    return json.dumps(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "field"],
            "properties": {
                "action": {"const": "modify_agent"},
                "field": {"enum": list(admitted_fields)},
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _live_string_domain(value: Any, *, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"{label} must be a non-empty string domain")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must not contain duplicates")
    return result


def _live_role_agent_schema(
    required_fields: Sequence[str],
    role_family: str,
    constraint: Mapping[str, Any],
    model_ids: Sequence[str],
    *,
    agent_id: Optional[str] = None,
    execution_mode: Optional[str] = None,
    allowed_tools: Optional[Sequence[str]] = None,
) -> Mapping[str, Any]:
    execution_modes = _live_string_domain(
        constraint.get("execution_modes"),
        label=f"{role_family}.execution_modes",
    )
    if any(mode not in {"reasoning", "react", "coding"} for mode in execution_modes):
        raise ValueError("live role constraint contains an unknown execution mode")
    raw_allowed_tool_sets = constraint.get("allowed_tools")
    if (
        not isinstance(raw_allowed_tool_sets, (list, tuple))
        or not raw_allowed_tool_sets
        or any(
            not isinstance(tool_set, (list, tuple))
            or any(not isinstance(tool_id, str) or not tool_id for tool_id in tool_set)
            for tool_set in raw_allowed_tool_sets
        )
    ):
        raise ValueError(f"{role_family}.allowed_tools must contain Tool-ID lists")
    normalized_tool_sets = [list(tool_set) for tool_set in raw_allowed_tool_sets]
    if len({json.dumps(item, separators=(",", ":")) for item in normalized_tool_sets}) != len(
        normalized_tool_sets
    ):
        raise ValueError(f"{role_family}.allowed_tools must not contain duplicates")
    properties = json.loads(json.dumps(_AGENT_SPEC_JSON_SCHEMA["properties"]))
    if agent_id is not None:
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("live Agent ID must be non-empty text")
        properties["agent_id"] = {"const": agent_id}
    properties["model_id"] = {"enum": list(model_ids)}
    properties["role_family"] = {"const": role_family}
    if (execution_mode is None) != (allowed_tools is None):
        raise ValueError(
            "live role execution_mode and allowed_tools must be conditioned together"
        )
    if execution_mode is None:
        properties["execution_mode"] = {"enum": list(execution_modes)}
        properties["allowed_tools"] = {"enum": normalized_tool_sets}
    else:
        normalized_tools = list(allowed_tools or ())
        if execution_mode not in execution_modes:
            raise ValueError("live conditioned execution mode is outside its role")
        if normalized_tools not in normalized_tool_sets:
            raise ValueError("live conditioned Tool set is outside its role")
        properties["execution_mode"] = {"const": execution_mode}
        properties["allowed_tools"] = {"const": normalized_tools}
    raw_contracts = constraint.get("contracts")
    if raw_contracts is not None:
        properties["contract"] = {
            "enum": list(
                _live_string_domain(
                    raw_contracts,
                    label=f"{role_family}.contracts",
                )
            )
        }
    raw_artifact_types = constraint.get("artifact_types")
    if raw_artifact_types is not None:
        properties["artifact_type"] = {
            "enum": list(
                _live_string_domain(
                    raw_artifact_types,
                    label=f"{role_family}.artifact_types",
                )
            )
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(required_fields),
        "properties": properties,
    }


def _live_execution_profiles(
    value: Any,
    *,
    label: str,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Validate correlated Runtime execution profiles from the live Canvas."""

    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a non-empty profile domain")
    profiles: list[tuple[str, tuple[str, ...]]] = []
    for raw_profile in value:
        if not isinstance(raw_profile, Mapping) or set(raw_profile) != {
            "execution_mode",
            "allowed_tools",
        }:
            raise ValueError(f"{label} contains a malformed execution profile")
        execution_mode = raw_profile.get("execution_mode")
        allowed_tools = raw_profile.get("allowed_tools")
        if execution_mode not in {"reasoning", "react", "coding"}:
            raise ValueError(f"{label} contains an unknown execution mode")
        if (
            not isinstance(allowed_tools, (list, tuple))
            or any(
                not isinstance(tool_id, str)
                or not tool_id
                or tool_id != tool_id.strip()
                for tool_id in allowed_tools
            )
            or len(allowed_tools) != len(set(allowed_tools))
        ):
            raise ValueError(f"{label} contains an invalid Tool-ID set")
        profile = (execution_mode, tuple(allowed_tools))
        if profile in profiles:
            raise ValueError(f"{label} contains duplicate execution profiles")
        profiles.append(profile)
    return tuple(profiles)


def _live_role_execution_tool_pairs(
    semantic_protocol: object,
    role_family: str,
    constraint: Mapping[str, Any],
) -> Optional[tuple[tuple[str, tuple[str, ...]], ...]]:
    """Project v2 semantic capabilities as correlated runtime declarations.

    ``execution_modes`` and ``allowed_tools`` remain readable marginal domains
    in the Canvas receipt.  Their Runtime-published ``execution_profiles`` are
    the correlated authority, so constrained decoding cannot create a
    reasoning+Tool pair, an unknown Tool, or an unregistered coding profile.
    Semantic roles remain independent of execution mode and graph topology.

    The earlier flexible HotpotQA protocol retains its historical correlated
    role table; non-flexible protocols and unconstrained auxiliary roles keep
    their prior independent domains.
    """

    if not flexible_hotpotqa_semantic_protocol(semantic_protocol):
        return None
    if not role_conditional_hotpotqa_protocol(semantic_protocol):
        expected_by_role: dict[
            str, tuple[tuple[str, tuple[str, ...]], ...]
        ] = {
            "reasoner": (
                ("react", ("qa-retrieval",)),
                ("reasoning", ()),
            ),
            "verifier": (("reasoning", ()),),
            "format": (("reasoning", ()),),
        }
        expected = expected_by_role.get(role_family)
        if expected is None:
            return None
        execution_modes = _live_string_domain(
            constraint.get("execution_modes"),
            label=f"{role_family}.execution_modes",
        )
        raw_tool_sets = constraint.get("allowed_tools")
        if not isinstance(raw_tool_sets, (list, tuple)) or any(
            not isinstance(tool_set, (list, tuple)) for tool_set in raw_tool_sets
        ):
            raise ValueError(
                f"{role_family}.allowed_tools must contain Tool-ID lists"
            )
        admitted = tuple(
            (mode, tools)
            for mode, tools in expected
            if mode in execution_modes
            and tools in tuple(tuple(tool_set) for tool_set in raw_tool_sets)
        )
        if not admitted:
            raise ValueError(
                f"{role_family} has no protocol-valid execution_mode/allowed_tools pair"
            )
        if role_family == "format":
            contracts = _live_string_domain(
                constraint.get("contracts"),
                label="format.contracts",
            )
            if len(contracts) != 1:
                raise ValueError(
                    "format contract domain must contain one exact pure contract"
                )
        return admitted
    profiles = _live_execution_profiles(
        constraint.get("execution_profiles"),
        label=f"{role_family}.execution_profiles",
    )
    execution_modes = _live_string_domain(
        constraint.get("execution_modes"),
        label=f"{role_family}.execution_modes",
    )
    raw_tool_sets = constraint.get("allowed_tools")
    if (
        not isinstance(raw_tool_sets, (list, tuple))
        or not raw_tool_sets
        or any(not isinstance(tool_set, (list, tuple)) for tool_set in raw_tool_sets)
    ):
        raise ValueError(f"{role_family}.allowed_tools must contain Tool-ID lists")
    tool_sets = tuple(tuple(tool_set) for tool_set in raw_tool_sets)
    if any(
        mode not in execution_modes or tools not in tool_sets
        for mode, tools in profiles
    ):
        raise ValueError(
            f"{role_family} execution profile is outside its marginal domains"
        )
    if set(execution_modes) != {mode for mode, _ in profiles}:
        raise ValueError(
            f"{role_family}.execution_modes does not match execution_profiles"
        )
    if set(tool_sets) != {tools for _, tools in profiles}:
        raise ValueError(
            f"{role_family}.allowed_tools does not match execution_profiles"
        )
    if role_family == "format":
        contracts = _live_string_domain(
            constraint.get("contracts"),
            label="format.contracts",
        )
        if len(contracts) != 1:
            raise ValueError(
                "format contract domain must contain one exact pure contract"
            )
    return profiles


def _live_role_agent_schema_branches(
    required_fields: Sequence[str],
    semantic_protocol: object,
    role_family: str,
    constraint: Mapping[str, Any],
    model_ids: Sequence[str],
    *,
    agent_id: str,
) -> tuple[Mapping[str, Any], ...]:
    conditioned_pairs = _live_role_execution_tool_pairs(
        semantic_protocol,
        role_family,
        constraint,
    )
    if conditioned_pairs is None:
        return (
            _live_role_agent_schema(
                required_fields,
                role_family,
                constraint,
                model_ids,
                agent_id=agent_id,
            ),
        )
    return tuple(
        _live_role_agent_schema(
            required_fields,
            role_family,
            constraint,
            model_ids,
            agent_id=agent_id,
            execution_mode=execution_mode,
            allowed_tools=allowed_tools,
        )
        for execution_mode, allowed_tools in conditioned_pairs
    )


def _live_new_agent_ids(
    existing_agent_ids: Sequence[str],
    max_agents: int,
) -> tuple[str, ...]:
    """Derive unique neutral node IDs from the current Canvas.

    FlowSteer's ``WorkflowGraph`` assigns ``node_N`` identifiers when an
    Operator is added.  AgentGraph keeps the sampled declaration in the exact
    action receipt, so the constrained schema supplies those same neutral IDs
    as positional constants instead of asking the Director to invent graph
    identifiers.  Roles, contracts, models, Tools, count, and topology remain
    sampled choices.
    """

    if (
        isinstance(existing_agent_ids, (str, bytes))
        or any(not isinstance(agent_id, str) or not agent_id for agent_id in existing_agent_ids)
    ):
        raise ValueError("existing Agent IDs are invalid")
    if type(max_agents) is not int or max_agents < 1:
        raise ValueError("new Agent ID count must be positive")
    used = set(existing_agent_ids)
    result: list[str] = []
    index = 1
    while len(result) < max_agents:
        candidate = f"node_{index}"
        index += 1
        if candidate in used:
            continue
        used.add(candidate)
        result.append(candidate)
    return tuple(result)


def _live_existing_agent_roles(
    domain: Mapping[str, Any],
    role_constraints: Mapping[str, Any],
) -> dict[str, str]:
    """Validate the HotpotQA Canvas endpoint-to-role receipt."""

    existing_ids = tuple(domain.get("existing_agent_ids", ()))
    raw_agents = domain.get("existing_agents")
    if not isinstance(raw_agents, (list, tuple)) or len(raw_agents) != len(
        existing_ids
    ):
        raise ValueError(
            "add_subgraph HotpotQA existing-Agent role domain is incomplete"
        )
    roles: dict[str, str] = {}
    ordered_ids: list[str] = []
    for raw_agent in raw_agents:
        if not isinstance(raw_agent, Mapping) or set(raw_agent) != {
            "agent_id",
            "role_family",
        }:
            raise ValueError(
                "add_subgraph HotpotQA existing-Agent role entry is malformed"
            )
        agent_id = raw_agent.get("agent_id")
        role_family = raw_agent.get("role_family")
        if (
            not isinstance(agent_id, str)
            or not agent_id
            or not isinstance(role_family, str)
            or role_family not in role_constraints
            or agent_id in roles
        ):
            raise ValueError(
                "add_subgraph HotpotQA existing-Agent role entry is invalid"
            )
        ordered_ids.append(agent_id)
        roles[agent_id] = role_family
    if tuple(ordered_ids) != existing_ids:
        raise ValueError(
            "add_subgraph HotpotQA existing-Agent roles changed Canvas order"
        )
    return roles


def _live_admitted_new_role_families(
    domain: Mapping[str, Any],
    role_constraints: Mapping[str, Any],
) -> tuple[str, ...]:
    """Validate the state-conditioned HotpotQA augmentation-role domain."""

    raw_roles = domain.get("admitted_new_role_families")
    if (
        not verified_qa_semantic_protocol(domain.get("semantic_protocol"))
        and raw_roles is None
    ):
        return tuple(role_constraints)
    if not isinstance(raw_roles, (list, tuple)) or not raw_roles:
        raise ValueError(
            "add_subgraph admitted new-Agent role domain is missing"
        )
    roles = tuple(raw_roles)
    if (
        len(roles) != len(set(roles))
        or any(
            not isinstance(role_family, str)
            or role_family not in role_constraints
            for role_family in roles
        )
    ):
        raise ValueError(
            "add_subgraph admitted new-Agent role domain is invalid"
        )
    return roles


def _live_hotpotqa_output_domain(
    domain: Mapping[str, Any],
    roles: Mapping[str, str],
) -> Optional[str]:
    """Validate the revision-local HotpotQA Output ownership receipt."""

    if role_conditional_hotpotqa_protocol(domain.get("semantic_protocol")):
        allowed_output_roles = _live_string_domain(
            domain.get("output_role_families"),
            label="add_subgraph.output_role_families",
        )
    else:
        if domain.get("output_role_family") != "format":
            raise ValueError("add_subgraph HotpotQA Output role domain is invalid")
        allowed_output_roles = ("format",)
    if "current_output_agent_id" not in domain:
        raise ValueError("add_subgraph HotpotQA current Output receipt is missing")
    current_output = domain.get("current_output_agent_id")
    if current_output is None:
        return None
    if (
        not isinstance(current_output, str)
        or current_output not in roles
        or roles[current_output] not in allowed_output_roles
    ):
        raise ValueError("add_subgraph HotpotQA current Output receipt is invalid")
    return current_output


def _live_hotpotqa_output_role_families(
    domain: Mapping[str, Any],
) -> tuple[str, ...]:
    if role_conditional_hotpotqa_protocol(domain.get("semantic_protocol")):
        return _live_string_domain(
            domain.get("output_role_families"),
            label="add_subgraph.output_role_families",
        )
    return ("format",)


def _live_defer_output_assignment(domain: Mapping[str, Any]) -> bool:
    """Return the Canvas-authored progressive Output-assignment boundary."""

    value = domain.get("defer_output_assignment", False)
    if type(value) is not bool:
        raise ValueError(
            "add_subgraph defer_output_assignment must be boolean"
        )
    return value


def _live_distinct_new_roles(domain: Mapping[str, Any]) -> bool:
    """Return whether this capability-construction edit forbids duplicates."""

    value = domain.get("distinct_new_role_families", False)
    if type(value) is not bool:
        raise ValueError(
            "add_subgraph distinct_new_role_families must be boolean"
        )
    return value


def _live_explicit_output_assignment_required(
    domain: Mapping[str, Any],
) -> bool:
    value = domain.get("explicit_output_assignment_required", False)
    if type(value) is not bool:
        raise ValueError(
            "add_subgraph explicit_output_assignment_required must be boolean"
        )
    return value


def _live_terminal_only_role_families(
    role_constraints: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return optional roles that are valid only as the selected Output."""

    terminal_only: list[str] = []
    for role_family, constraint in role_constraints.items():
        if not isinstance(role_family, str) or not isinstance(constraint, Mapping):
            raise ValueError("add_subgraph role constraints are malformed")
        value = constraint.get("must_be_output_agent", False)
        if type(value) is not bool:
            raise ValueError(
                f"{role_family}.must_be_output_agent must be boolean"
            )
        if value:
            terminal_only.append(role_family)
    return tuple(terminal_only)


def _hotpotqa_directed_role_relation_allowed(
    source_role: str,
    target_role: str,
) -> bool:
    """Mirror the incremental HotpotQA semantic-edge validator."""

    if source_role == "format":
        return False
    if target_role == "verifier":
        return source_role == "reasoner"
    if target_role == "format":
        return source_role == "verifier"
    return True


def _live_add_subgraph_isolated_boundary(
    domain: Mapping[str, Any],
) -> bool:
    """Validate an explicit isolated ADD boundary supplied by the Canvas."""

    has_relations = "relations" in domain
    has_output = "output_agent_id" in domain
    if has_relations != has_output:
        raise ValueError(
            "add_subgraph isolated boundary must declare relations and "
            "output_agent_id together"
        )
    if not has_relations:
        return False
    if domain.get("relations") != [] or domain.get("output_agent_id") is not None:
        raise ValueError(
            "add_subgraph isolated boundary requires relations=[] and "
            "output_agent_id=null"
        )
    return True


def director_live_add_subgraph_agent_declarations_json_schema_text(
    action_target_domains: Mapping[str, Any],
    *,
    selected_agent_roles: Optional[Sequence[Mapping[str, Any]]] = None,
) -> str:
    """Render the first v3 ADD phase for exact Agent declarations.

    JSON Schema cannot express that a relation endpoint equals an ``agent_id``
    sampled elsewhere in the same object.  V3 therefore samples declarations
    first, then uses their exact IDs to render the complete ADD action schema.
    The final action is still sampled in full; no phase output is spliced into
    or substituted for the Canvas action.
    """

    domain = action_target_domains.get("add_subgraph")
    if not isinstance(domain, Mapping):
        raise ValueError("add_subgraph live target domain is missing")
    _live_add_subgraph_isolated_boundary(domain)
    min_agents = domain.get("min_new_agents")
    max_agents = domain.get("max_new_agents")
    if (
        type(min_agents) is not int
        or type(max_agents) is not int
        or not 1 <= min_agents <= max_agents <= 3
    ):
        raise ValueError("add_subgraph live Agent-count domain is invalid")
    required_fields = domain.get("required_agent_fields")
    required_minimum = {
        "agent_id",
        "model_id",
        "contract",
        "role_family",
        "allowed_tools",
        "execution_mode",
    }
    if (
        not isinstance(required_fields, (list, tuple))
        or any(
            field not in _AGENT_SPEC_JSON_SCHEMA["properties"]
            for field in required_fields
        )
        or not required_minimum.issubset(required_fields)
        or len(required_fields) != len(set(required_fields))
    ):
        raise ValueError("add_subgraph required Agent fields are incomplete")
    model_ids = _live_string_domain(
        domain.get("model_ids"),
        label="add_subgraph.model_ids",
    )
    existing_agent_ids = domain.get("existing_agent_ids")
    if not isinstance(existing_agent_ids, (list, tuple)) or any(
        not isinstance(agent_id, str) or not agent_id
        for agent_id in existing_agent_ids
    ):
        raise ValueError("add_subgraph existing Agent IDs are invalid")
    if len(existing_agent_ids) != len(set(existing_agent_ids)):
        raise ValueError("add_subgraph existing Agent IDs contain duplicates")
    endpoint_scope = domain.get("endpoint_scope")
    expected_endpoint_sources = {"existing_agent_ids", "same_action_agent_ids"}
    if not isinstance(endpoint_scope, Mapping) or any(
        set(endpoint_scope.get(key, ())) != expected_endpoint_sources
        for key in ("relation_endpoint_sources", "output_agent_id_sources")
    ):
        raise ValueError("add_subgraph endpoint scope is incomplete")
    role_constraints = domain.get("role_constraints")
    if not isinstance(role_constraints, Mapping) or not role_constraints:
        raise ValueError("add_subgraph role constraints are missing")
    if role_conditional_hotpotqa_protocol(domain.get("semantic_protocol")):
        registered_profiles = set(
            _live_execution_profiles(
                domain.get("registered_execution_profiles"),
                label="add_subgraph.registered_execution_profiles",
            )
        )
        for role_family, constraint in role_constraints.items():
            if not isinstance(role_family, str) or not isinstance(
                constraint,
                Mapping,
            ):
                raise ValueError("add_subgraph role constraints are malformed")
            role_profiles = set(
                _live_execution_profiles(
                    constraint.get("execution_profiles"),
                    label=f"{role_family}.execution_profiles",
                )
            )
            if not role_profiles <= registered_profiles:
                raise ValueError(
                    f"{role_family} exposes an execution profile not registered "
                    "by the current Runtime"
                )
    admitted_new_roles = _live_admitted_new_role_families(
        domain,
        role_constraints,
    )
    if verified_qa_semantic_protocol(domain.get("semantic_protocol")):
        existing_roles = _live_existing_agent_roles(domain, role_constraints)
        _live_hotpotqa_output_domain(domain, existing_roles)
    new_agent_ids = _live_new_agent_ids(existing_agent_ids, max_agents)
    selected_roles: Optional[tuple[str, ...]] = None
    if selected_agent_roles is not None:
        if (
            not isinstance(selected_agent_roles, (list, tuple))
            or not min_agents <= len(selected_agent_roles) <= max_agents
        ):
            raise ValueError("add_subgraph selected Agent roles have invalid count")
        normalized_roles: list[str] = []
        for position, value in enumerate(selected_agent_roles):
            if not isinstance(value, Mapping) or set(value) != {
                "agent_id",
                "role_family",
            }:
                raise ValueError("add_subgraph selected Agent role is malformed")
            if value.get("agent_id") != new_agent_ids[position]:
                raise ValueError(
                    "add_subgraph selected Agent role changed its Canvas node ID"
                )
            role_family = value.get("role_family")
            if (
                not isinstance(role_family, str)
                or role_family not in admitted_new_roles
            ):
                raise ValueError(
                    "add_subgraph selected Agent role is outside the live domain"
                )
            normalized_roles.append(role_family)
        selected_roles = tuple(normalized_roles)
    positional_agent_schemas: list[Mapping[str, Any]] = []
    positional_count = (
        len(selected_roles) if selected_roles is not None else max_agents
    )
    for position, agent_id in enumerate(new_agent_ids[:positional_count]):
        admitted_roles = (
            ((selected_roles[position], role_constraints[selected_roles[position]]),)
            if selected_roles is not None
            else tuple(
                (role_family, role_constraints[role_family])
                for role_family in admitted_new_roles
            )
        )
        if any(
            not isinstance(role_family, str)
            or not role_family
            or not isinstance(constraint, Mapping)
            for role_family, constraint in admitted_roles
        ):
            raise ValueError("add_subgraph role constraints are malformed")
        role_branches = [
            schema
            for role_family, constraint in admitted_roles
            for schema in _live_role_agent_schema_branches(
                required_fields,
                domain.get("semantic_protocol"),
                role_family,
                constraint,
                model_ids,
                agent_id=agent_id,
            )
        ]
        positional_agent_schemas.append({"anyOf": role_branches})
    admitted_counts = (
        (len(selected_roles),)
        if selected_roles is not None
        else tuple(range(min_agents, max_agents + 1))
    )
    agent_count_branches = [
        {
            "type": "array",
            "minItems": count,
            "maxItems": count,
            "prefixItems": positional_agent_schemas[:count],
            "items": False,
        }
        for count in admitted_counts
    ]
    return json.dumps(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "agents"],
            "properties": {
                "action": {"const": "add_subgraph"},
                "agents": {
                    "oneOf": agent_count_branches,
                },
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def director_live_add_subgraph_role_selection_json_schema_text(
    action_target_domains: Mapping[str, Any],
) -> str:
    """Select ADD Agent count and semantic roles before free contracts.

    This is the same inference-only hierarchical factorization used for the
    action discriminator and MODIFY target.  The role choice remains sampled;
    it is merely committed before a role-conditioned declaration schema lets
    the Director write each free contract.
    """

    declaration_schema = json.loads(
        director_live_add_subgraph_agent_declarations_json_schema_text(
            action_target_domains
        )
    )
    domain = action_target_domains["add_subgraph"]
    min_agents = domain["min_new_agents"]
    max_agents = domain["max_new_agents"]
    existing_agent_ids = domain["existing_agent_ids"]
    role_constraints = domain["role_constraints"]
    admitted_new_roles = _live_admitted_new_role_families(
        domain,
        role_constraints,
    )
    new_agent_ids = _live_new_agent_ids(existing_agent_ids, max_agents)
    role_families = admitted_new_roles
    if not role_families:
        raise ValueError("add_subgraph role domain is empty")
    positional_roles = [
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["agent_id", "role_family"],
            "properties": {
                "agent_id": {"const": agent_id},
                "role_family": {"enum": list(role_families)},
            },
        }
        for agent_id in new_agent_ids
    ]
    terminal_only_roles = _live_terminal_only_role_families(role_constraints)
    if _live_distinct_new_roles(domain):
        count_branches = [
            {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "prefixItems": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["agent_id", "role_family"],
                        "properties": {
                            "agent_id": {"const": new_agent_ids[position]},
                            "role_family": {"const": role_family},
                        },
                    }
                    for position, role_family in enumerate(role_sequence)
                ],
                "items": False,
            }
            for count in range(min_agents, max_agents + 1)
            for role_sequence in permutations(role_families, count)
        ]
    elif terminal_only_roles:
        count_branches = [
            {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "prefixItems": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["agent_id", "role_family"],
                        "properties": {
                            "agent_id": {"const": new_agent_ids[position]},
                            "role_family": {"const": role_family},
                        },
                    }
                    for position, role_family in enumerate(role_sequence)
                ],
                "items": False,
            }
            for count in range(min_agents, max_agents + 1)
            for role_sequence in product(role_families, repeat=count)
            if all(
                role_sequence.count(role_family) <= 1
                for role_family in terminal_only_roles
            )
        ]
    else:
        count_branches = [
            {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "prefixItems": positional_roles[:count],
                "items": False,
            }
            for count in range(min_agents, max_agents + 1)
        ]
    # The declaration render above is intentionally evaluated first so this
    # smaller selector cannot accept a malformed live domain that the complete
    # role-conditioned phase would later reject.
    if not declaration_schema:
        raise ValueError("add_subgraph declaration schema is empty")
    return json.dumps(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "agents"],
            "properties": {
                "action": {"const": "add_subgraph"},
                "agents": {"oneOf": count_branches},
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def director_live_add_subgraph_role_selection_from_text(
    text: str,
    action_target_domains: Mapping[str, Any],
) -> tuple[dict[str, str], ...]:
    """Parse one exact ADD count/role phase without repairing sampled text."""

    if not isinstance(text, str):
        raise ValueError("add_subgraph Agent role selection must be text")
    stripped = text.strip()
    try:
        payload, end = json.JSONDecoder().raw_decode(stripped)
    except (TypeError, ValueError) as exc:
        raise ValueError("add_subgraph Agent role selection is not JSON") from exc
    trailing = stripped[end:].strip()
    if trailing and trailing != _QWEN_JSON_EOS_TEXT:
        raise ValueError(
            "add_subgraph Agent role selection contains trailing text: "
            f"{trailing[:80]!r}"
        )
    if not isinstance(payload, Mapping) or set(payload) != {"action", "agents"}:
        raise ValueError("add_subgraph Agent role selection fields are invalid")
    if payload.get("action") != "add_subgraph":
        raise ValueError("add_subgraph Agent role selection changed its action")
    agents = payload.get("agents")
    domain = action_target_domains.get("add_subgraph")
    if not isinstance(domain, Mapping):
        raise ValueError("add_subgraph live target domain is missing")
    min_agents = domain.get("min_new_agents")
    max_agents = domain.get("max_new_agents")
    if (
        not isinstance(agents, list)
        or type(min_agents) is not int
        or type(max_agents) is not int
        or not min_agents <= len(agents) <= max_agents
    ):
        raise ValueError("add_subgraph selected Agent roles have invalid count")
    # Render once to apply the complete domain validation before accepting the
    # smaller selector receipt.
    director_live_add_subgraph_role_selection_json_schema_text(
        action_target_domains
    )
    expected_ids = _live_new_agent_ids(domain["existing_agent_ids"], max_agents)
    role_constraints = domain["role_constraints"]
    admitted_new_roles = _live_admitted_new_role_families(
        domain,
        role_constraints,
    )
    normalized: list[dict[str, str]] = []
    for position, value in enumerate(agents):
        if not isinstance(value, Mapping) or set(value) != {
            "agent_id",
            "role_family",
        }:
            raise ValueError("add_subgraph selected Agent role is malformed")
        agent_id = value.get("agent_id")
        role_family = value.get("role_family")
        if agent_id != expected_ids[position]:
            raise ValueError(
                "add_subgraph selected Agent role changed its Canvas node ID"
            )
        if (
            not isinstance(role_family, str)
            or role_family not in admitted_new_roles
        ):
            raise ValueError(
                "add_subgraph selected Agent role is outside the live domain"
            )
        normalized.append(
            {"agent_id": agent_id, "role_family": role_family}
        )
    if _live_distinct_new_roles(domain) and len(
        {item["role_family"] for item in normalized}
    ) != len(normalized):
        raise ValueError(
            "add_subgraph selected Agent roles must be distinct at the "
            "current capability-construction boundary"
        )
    terminal_only_roles = _live_terminal_only_role_families(role_constraints)
    if any(
        sum(item["role_family"] == role_family for item in normalized) > 1
        for role_family in terminal_only_roles
    ):
        raise ValueError(
            "add_subgraph terminal-only role may be selected at most once"
        )
    return tuple(normalized)


def _live_add_subgraph_agents(
    action_target_domains: Mapping[str, Any],
    agents: Sequence[Mapping[str, Any]],
    *,
    selected_agent_roles: Optional[Sequence[Mapping[str, Any]]] = None,
) -> tuple[dict[str, Any], ...]:
    """Validate sampled declarations against the exact first-phase schema."""

    declaration_schema = json.loads(
        director_live_add_subgraph_agent_declarations_json_schema_text(
            action_target_domains,
            selected_agent_roles=selected_agent_roles,
        )
    )
    domain = action_target_domains["add_subgraph"]
    min_agents = domain["min_new_agents"]
    max_agents = domain["max_new_agents"]
    if (
        not isinstance(agents, (list, tuple))
        or not min_agents <= len(agents) <= max_agents
    ):
        raise ValueError("add_subgraph sampled Agent declarations have invalid count")
    existing_ids = set(domain["existing_agent_ids"])
    expected_new_ids = _live_new_agent_ids(
        domain["existing_agent_ids"],
        max_agents,
    )
    role_constraints = domain["role_constraints"]
    admitted_new_roles = _live_admitted_new_role_families(
        domain,
        role_constraints,
    )
    model_ids = set(domain["model_ids"])
    required_fields = set(domain["required_agent_fields"])
    known_fields = set(_AGENT_SPEC_JSON_SCHEMA["properties"])
    normalized: list[dict[str, Any]] = []
    new_ids: set[str] = set()
    for position, raw_agent in enumerate(agents):
        if not isinstance(raw_agent, Mapping):
            raise ValueError("add_subgraph Agent declaration must be an object")
        agent = dict(raw_agent)
        if not required_fields.issubset(agent) or not set(agent).issubset(known_fields):
            raise ValueError("add_subgraph Agent declaration fields are invalid")
        agent_id = agent.get("agent_id")
        model_id = agent.get("model_id")
        role_family = agent.get("role_family")
        contract = agent.get("contract")
        execution_mode = agent.get("execution_mode")
        allowed_tools = agent.get("allowed_tools")
        if (
            not isinstance(agent_id, str)
            or not agent_id
            or agent_id != agent_id.strip()
            or agent_id != expected_new_ids[position]
            or agent_id in existing_ids
            or agent_id in new_ids
        ):
            raise ValueError(
                "add_subgraph new Agent IDs must match the unique neutral IDs "
                "assigned by the current Canvas"
            )
        if not isinstance(model_id, str) or model_id != model_id.strip() or model_id not in model_ids:
            raise ValueError("add_subgraph Agent model_id is outside the live catalog")
        if not isinstance(contract, str) or not contract or contract != contract.strip():
            raise ValueError("add_subgraph Agent contract must be non-empty")
        if (
            not isinstance(role_family, str)
            or not role_family
            or role_family != role_family.strip()
        ):
            raise ValueError("add_subgraph Agent role is outside the live domain")
        constraint = role_constraints.get(role_family)
        if (
            role_family not in admitted_new_roles
            or not isinstance(constraint, Mapping)
        ):
            raise ValueError("add_subgraph Agent role is outside the live domain")
        contract_domain = constraint.get("contracts")
        if contract_domain is not None and contract not in _live_string_domain(
            contract_domain,
            label=f"{role_family}.contracts",
        ):
            raise ValueError("add_subgraph Agent contract violates its role")
        artifact_type_domain = constraint.get("artifact_types")
        if (
            artifact_type_domain is not None
            and agent.get("artifact_type")
            not in _live_string_domain(
                artifact_type_domain,
                label=f"{role_family}.artifact_types",
            )
        ):
            raise ValueError(
                "add_subgraph Agent artifact_type violates its role"
            )
        if (
            execution_mode not in constraint.get("execution_modes", ())
        ):
            raise ValueError("add_subgraph Agent execution mode violates its role")
        if not isinstance(allowed_tools, list) or allowed_tools not in [
            list(tool_set) for tool_set in constraint.get("allowed_tools", ())
        ]:
            raise ValueError("add_subgraph Agent Tool set violates its role")
        if any(tool_id != tool_id.strip() for tool_id in allowed_tools):
            raise ValueError("add_subgraph Agent Tool IDs must be canonical")
        conditioned_pairs = _live_role_execution_tool_pairs(
            domain.get("semantic_protocol"),
            role_family,
            constraint,
        )
        if conditioned_pairs is not None and (
            execution_mode,
            tuple(allowed_tools),
        ) not in conditioned_pairs:
            raise ValueError(
                "add_subgraph Agent execution mode and Tool set violate its role"
            )
        for optional_text in ("artifact_type", "completion_condition"):
            value = agent.get(optional_text)
            if value is not None and (
                not isinstance(value, str)
                or not value
                or value != value.strip()
            ):
                raise ValueError(
                    f"add_subgraph Agent {optional_text} must be non-empty text"
                )
        new_ids.add(agent_id)
        normalized.append(agent)
    if _live_distinct_new_roles(domain) and len(
        {item["role_family"] for item in normalized}
    ) != len(normalized):
        raise ValueError(
            "add_subgraph Agent declarations must use distinct roles at the "
            "current capability-construction boundary"
        )
    terminal_only_roles = _live_terminal_only_role_families(role_constraints)
    if any(
        sum(item["role_family"] == role_family for item in normalized) > 1
        for role_family in terminal_only_roles
    ):
        raise ValueError(
            "add_subgraph terminal-only role may be declared at most once"
        )
    if selected_agent_roles is not None:
        expected_roles = tuple(
            (value["agent_id"], value["role_family"])
            for value in selected_agent_roles
        )
        actual_roles = tuple(
            (value["agent_id"], value["role_family"])
            for value in normalized
        )
        if actual_roles != expected_roles:
            raise ValueError(
                "add_subgraph Agent declarations changed their selected roles"
            )
    return tuple(normalized)


def director_live_add_subgraph_relation_candidates(
    action_target_domains: Mapping[str, Any],
    agents: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Project exact role-valid relation encodings for one sampled subgraph.

    The Canvas remains authoritative.  Legacy semantic protocols retain their
    exact role-edge ordering.  The topology-neutral HotpotQA lineage protocol
    admits every directed or reciprocal communication edge allowed by the
    terminal Formatter-sink invariant; its semantic order is validated from
    actual routed artifacts at FINISH, rather than imposed as a graph template.
    A one-way relation is always encoded as its actual sender ``source_id`` to
    receiver ``target_id`` with ``(true,false)`` instead of the directionally
    equivalent but ambiguous ``(false,true)``.  No relation is required, so
    the Director still selects the graph topology.
    """

    normalized_agents = _live_add_subgraph_agents(
        action_target_domains,
        agents,
    )
    domain = action_target_domains["add_subgraph"]
    if not verified_qa_semantic_protocol(domain.get("semantic_protocol")):
        return ()
    if _live_add_subgraph_isolated_boundary(domain):
        return ()
    role_constraints = domain["role_constraints"]
    roles = _live_existing_agent_roles(domain, role_constraints)
    for agent in normalized_agents:
        roles[agent["agent_id"]] = agent["role_family"]
    endpoint_ids = [*domain["existing_agent_ids"]]
    endpoint_ids.extend(agent["agent_id"] for agent in normalized_agents)
    candidates: list[dict[str, Any]] = []
    if flexible_hotpotqa_semantic_protocol(domain.get("semantic_protocol")):
        role_conditional = role_conditional_hotpotqa_protocol(
            domain.get("semantic_protocol")
        )
        for source_index, source_id in enumerate(endpoint_ids):
            for target_id in endpoint_ids[source_index + 1 :]:
                source_role = roles[source_id]
                target_role = roles[target_id]
                source_to_target = source_role != "format" and not (
                    role_conditional
                    and target_role in {"verifier", "format"}
                    and source_role
                    in {"evidence_retriever", "format", "output"}
                )
                target_to_source = target_role != "format" and not (
                    role_conditional
                    and source_role in {"verifier", "format"}
                    and target_role
                    in {"evidence_retriever", "format", "output"}
                )
                if source_to_target:
                    candidates.append(
                        {
                            "source_id": source_id,
                            "target_id": target_id,
                            "source_to_target": True,
                            "target_to_source": False,
                        }
                    )
                if target_to_source:
                    candidates.append(
                        {
                            "source_id": target_id,
                            "target_id": source_id,
                            "source_to_target": True,
                            "target_to_source": False,
                        }
                    )
                if source_to_target and target_to_source:
                    candidates.append(
                        {
                            "source_id": source_id,
                            "target_id": target_id,
                            "source_to_target": True,
                            "target_to_source": True,
                        }
                    )
        required_ingress_ids = tuple(
            domain.get("required_ingress_consumer_agent_ids", ())
        )
        if required_ingress_ids:
            if (
                len(required_ingress_ids) != len(set(required_ingress_ids))
                or any(
                    not isinstance(agent_id, str)
                    or agent_id not in domain["existing_agent_ids"]
                    for agent_id in required_ingress_ids
                )
            ):
                raise ValueError(
                    "add_subgraph required ingress consumer domain is invalid"
                )
            new_agent_ids = {agent["agent_id"] for agent in normalized_agents}

            def supplies_required_ingress(candidate: Mapping[str, Any]) -> bool:
                return any(
                    (
                        candidate["source_id"] in new_agent_ids
                        and candidate["target_id"] == consumer_id
                        and candidate["source_to_target"] is True
                    )
                    or (
                        candidate["target_id"] in new_agent_ids
                        and candidate["source_id"] == consumer_id
                        and candidate["target_to_source"] is True
                    )
                    for consumer_id in required_ingress_ids
                )

            candidates = [
                candidate
                for candidate in candidates
                if supplies_required_ingress(candidate)
            ]
            if (
                domain.get("exact_relation_count") == 1
                and tuple(domain.get("admitted_new_role_families", ()))
                == ("evidence_retriever",)
            ):
                candidates = [
                    candidate
                    for candidate in candidates
                    if candidate["source_id"] in new_agent_ids
                    and candidate["target_id"] in required_ingress_ids
                    and candidate["source_to_target"] is True
                    and candidate["target_to_source"] is False
                ]
        required_reachability_output = domain.get(
            "required_reachability_output_agent_id"
        )
        if required_reachability_output is not None:
            if required_ingress_ids:
                raise ValueError(
                    "add_subgraph cannot combine required ingress and atomic "
                    "Output-reachability domains"
                )
            if (
                not isinstance(required_reachability_output, str)
                or required_reachability_output
                != domain.get("current_output_agent_id")
                or required_reachability_output
                not in domain.get("existing_agent_ids", ())
                or roles.get(required_reachability_output) != "output"
                or domain.get("exact_relation_count") != 1
            ):
                raise ValueError(
                    "add_subgraph atomic Output-reachability domain is invalid"
                )
            new_agent_ids = {agent["agent_id"] for agent in normalized_agents}
            candidates = [
                candidate
                for candidate in candidates
                if candidate["source_id"] in new_agent_ids
                and candidate["target_id"] == required_reachability_output
                and candidate["source_to_target"] is True
                and candidate["target_to_source"] is False
            ]
        return tuple(candidates)
    semantic_dataflow_pairs = {
        ("evidence_retriever", "reasoner"),
        ("repair", "reasoner"),
        ("reasoner", "verifier"),
        ("verifier", "format"),
    }
    for source_index, source_id in enumerate(endpoint_ids):
        for target_id in endpoint_ids[source_index + 1 :]:
            source_role = roles[source_id]
            target_role = roles[target_id]
            source_to_target = _hotpotqa_directed_role_relation_allowed(
                source_role,
                target_role,
            )
            target_to_source = _hotpotqa_directed_role_relation_allowed(
                target_role,
                source_role,
            )
            forward_is_source_to_target = (source_role, target_role) in (
                semantic_dataflow_pairs
            )
            forward_is_target_to_source = (target_role, source_role) in (
                semantic_dataflow_pairs
            )
            if forward_is_source_to_target or forward_is_target_to_source:
                if forward_is_source_to_target and source_to_target:
                    sender_id, receiver_id = source_id, target_id
                elif forward_is_target_to_source and target_to_source:
                    sender_id, receiver_id = target_id, source_id
                else:
                    sender_id = receiver_id = None
                if sender_id is not None and receiver_id is not None:
                    candidates.append(
                        {
                            "source_id": sender_id,
                            "target_id": receiver_id,
                            "source_to_target": True,
                            "target_to_source": False,
                        }
                    )
                if source_to_target and target_to_source:
                    candidates.append(
                        {
                            "source_id": source_id,
                            "target_id": target_id,
                            "source_to_target": True,
                            "target_to_source": True,
                        }
                    )
                continue
            if source_to_target:
                candidates.append(
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "source_to_target": True,
                        "target_to_source": False,
                    }
                )
            if target_to_source:
                candidates.append(
                    {
                        "source_id": target_id,
                        "target_id": source_id,
                        "source_to_target": True,
                        "target_to_source": False,
                    }
                )
            if source_to_target and target_to_source:
                candidates.append(
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "source_to_target": True,
                        "target_to_source": True,
                    }
                )
    return tuple(candidates)


def director_live_add_subgraph_agent_declarations_from_text(
    text: str,
    action_target_domains: Mapping[str, Any],
    *,
    selected_agent_roles: Optional[Sequence[Mapping[str, Any]]] = None,
) -> tuple[dict[str, Any], ...]:
    """Parse one exact declaration-phase object without repairing its text."""

    if not isinstance(text, str):
        raise ValueError("add_subgraph Agent declarations must be text")
    stripped = text.strip()
    try:
        payload, end = json.JSONDecoder().raw_decode(stripped)
    except (TypeError, ValueError) as exc:
        raise ValueError("add_subgraph Agent declarations are not JSON") from exc
    trailing = stripped[end:].strip()
    if trailing and trailing != _QWEN_JSON_EOS_TEXT:
        raise ValueError(
            "add_subgraph Agent declarations contain trailing text: "
            f"{trailing[:80]!r}"
        )
    if not isinstance(payload, Mapping) or set(payload) != {"action", "agents"}:
        raise ValueError("add_subgraph Agent declaration fields are invalid")
    if payload.get("action") != "add_subgraph":
        raise ValueError("add_subgraph Agent declarations changed their action")
    return _live_add_subgraph_agents(
        action_target_domains,
        payload.get("agents"),
        selected_agent_roles=selected_agent_roles,
    )


def _live_discrete_values(
    candidate: Mapping[str, Any],
    field_name: str,
) -> Optional[tuple[Any, ...]]:
    raw_domains = candidate.get("discrete_value_domains", {})
    if not isinstance(raw_domains, Mapping):
        raise ValueError("modify_agent discrete_value_domains must be an object")
    if field_name not in raw_domains:
        return None
    raw_values = raw_domains[field_name]
    if not isinstance(raw_values, (list, tuple)) or not raw_values:
        raise ValueError("modify_agent discrete value domain must be non-empty")
    normalized: list[Any] = []
    identities: set[str] = set()
    for value in raw_values:
        if field_name in {
            "model_id",
            "contract",
            "role_family",
            "artifact_type",
            "completion_condition",
        } and (
            not isinstance(value, str)
            or not value
            or value != value.strip()
        ):
            raise ValueError(
                f"modify_agent {field_name} discrete values must be canonical text"
            )
        if field_name == "execution_mode" and value not in {
            "reasoning",
            "react",
            "coding",
        }:
            raise ValueError("modify_agent execution_mode domain is invalid")
        if field_name == "allowed_tools" and (
            not isinstance(value, list)
            or any(
                not isinstance(tool_id, str)
                or not tool_id
                or tool_id != tool_id.strip()
                for tool_id in value
            )
            or len(value) != len(set(value))
        ):
            raise ValueError("modify_agent allowed_tools domain is invalid")
        try:
            identity = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "modify_agent discrete value is not JSON serializable"
            ) from exc
        if identity in identities:
            raise ValueError("modify_agent discrete value domain has duplicates")
        identities.add(identity)
        normalized.append(value)
    return tuple(normalized)


def _live_modify_agent_candidates(
    action_target_domains: Mapping[str, Any],
    field_name: str,
) -> tuple[Mapping[str, Any], ...]:
    domain = action_target_domains.get("modify_agent")
    if not isinstance(domain, Mapping):
        raise ValueError("modify_agent live target domain is missing")
    global_fields = _live_string_domain(
        domain.get("mutable_fields"),
        label="modify_agent.mutable_fields",
    )
    if field_name not in global_fields or field_name not in DIRECTOR_MODIFY_AGENT_FIELDS:
        raise ValueError("modify_agent field is outside the live domain")
    candidates = domain.get("per_agent_candidates")
    if not isinstance(candidates, (list, tuple)):
        raise ValueError("modify_agent per-Agent candidates are missing")
    admitted: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("modify_agent candidate must be an object")
        agent_id = candidate.get("agent_id")
        fields = candidate.get("mutable_fields")
        if (
            not isinstance(agent_id, str)
            or not agent_id
            or agent_id in seen_ids
            or not isinstance(fields, (list, tuple))
            or any(field not in DIRECTOR_MODIFY_AGENT_FIELDS for field in fields)
            or len(fields) != len(set(fields))
        ):
            raise ValueError("modify_agent candidate is malformed")
        seen_ids.add(agent_id)
        raw_discrete = candidate.get("discrete_value_domains", {})
        if not isinstance(raw_discrete, Mapping) or any(
            discrete_field not in fields for discrete_field in raw_discrete
        ):
            raise ValueError("modify_agent discrete value domain has an invalid field")
        for discrete_field in raw_discrete:
            _live_discrete_values(candidate, discrete_field)
        raw_profiles = candidate.get("execution_profiles")
        if raw_profiles is not None:
            profiles = _live_execution_profiles(
                raw_profiles,
                label=f"modify_agent.{agent_id}.execution_profiles",
            )
            if "execution_mode" not in fields:
                raise ValueError(
                    "modify_agent execution profiles require execution_mode "
                    "in the mutable field domain"
                )
            if any(
                profile_field in raw_discrete
                for profile_field in ("execution_mode", "allowed_tools")
            ):
                raise ValueError(
                    "modify_agent correlated execution profiles must not be "
                    "split into independent discrete value domains"
                )
            if not profiles:
                raise ValueError(
                    "modify_agent execution profile domain must be non-empty"
                )
        if field_name in fields:
            admitted.append(candidate)
    if not admitted:
        raise ValueError("modify_agent field has no live Agent target")
    return tuple(admitted)


def director_live_modify_agent_selector_json_schema_text(
    action_target_domains: Mapping[str, Any],
    field_name: str,
) -> str:
    """Render the v3 Agent selector for one already-selected atomic field."""

    candidates = _live_modify_agent_candidates(action_target_domains, field_name)
    return json.dumps(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "agent_id"],
            "properties": {
                "action": {"const": "modify_agent"},
                "agent_id": {
                    "enum": [candidate["agent_id"] for candidate in candidates]
                },
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def director_live_action_parameter_json_schema_text(
    action: str,
    action_target_domains: Mapping[str, Any],
    *,
    add_agents: Optional[Sequence[Mapping[str, Any]]] = None,
    modify_field: Optional[str] = None,
    modify_agent_id: Optional[str] = None,
    relation_candidate_index: Optional[int] = None,
) -> str:
    """Render one exact v3 parameter phase from current Canvas domains.

    The function only projects domains supplied by ``AgentWorkflowEnv``.  It
    does not infer missing targets, repair a sample, or replace parser/Canvas
    validation.
    """

    if not isinstance(action_target_domains, Mapping):
        raise ValueError("live action target domains must be an object")
    domain = action_target_domains.get(action)
    if not isinstance(domain, Mapping):
        raise ValueError(f"missing live target domain for {action}")

    if action == "add_subgraph":
        if add_agents is None:
            raise ValueError(
                "add_subgraph v3 parameter phase requires sampled Agent declarations"
            )
        normalized_agents = _live_add_subgraph_agents(
            action_target_domains,
            add_agents,
        )
        isolated_boundary = _live_add_subgraph_isolated_boundary(domain)
        endpoint_ids = list(domain["existing_agent_ids"]) + [
            agent["agent_id"] for agent in normalized_agents
        ]
        schema = json.loads(
            director_state_conditioned_sampling_json_schema_text("add_subgraph")
        )
        schema["properties"]["agents"] = {"const": list(normalized_agents)}
        if verified_qa_semantic_protocol(domain.get("semantic_protocol")):
            relation_candidates = director_live_add_subgraph_relation_candidates(
                action_target_domains,
                normalized_agents,
            )
            if relation_candidates:
                exact_relation_count = domain.get("exact_relation_count")
                if exact_relation_count is not None and (
                    isinstance(exact_relation_count, bool)
                    or not isinstance(exact_relation_count, int)
                    or exact_relation_count < 1
                ):
                    raise ValueError(
                        "add_subgraph exact_relation_count must be a positive integer"
                    )
                max_relations = (
                    exact_relation_count
                    if exact_relation_count is not None
                    else (
                        2
                        if flexible_hotpotqa_semantic_protocol(
                            domain.get("semantic_protocol")
                        )
                        else 1
                    )
                )
                schema["properties"]["relations"] = {
                    "type": "array",
                    # xgrammar supports exact candidate branches but JSON
                    # Schema has no portable unique-by-unordered-endpoint-pair
                    # constraint.  The topology-neutral HotpotQA protocol
                    # admits one bounded two-edge functional block; receipt
                    # validation still rejects a repeated unordered pair.
                    "maxItems": max_relations,
                    "minItems": (
                        exact_relation_count
                        if exact_relation_count is not None
                        else (
                            1
                            if domain.get(
                                "required_ingress_consumer_agent_ids", ()
                            )
                            else 0
                        )
                    ),
                    "uniqueItems": True,
                    "items": {
                        "anyOf": [
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "required": [
                                    "source_id",
                                    "target_id",
                                    "source_to_target",
                                    "target_to_source",
                                ],
                                "properties": {
                                    key: {"const": value}
                                    for key, value in candidate.items()
                                },
                            }
                            for candidate in relation_candidates
                        ]
                    },
                }
            else:
                schema["properties"]["relations"] = {
                    "type": "array",
                    "maxItems": 0,
                }
            roles = _live_existing_agent_roles(
                domain,
                domain["role_constraints"],
            )
            current_output_agent_id = _live_hotpotqa_output_domain(
                domain,
                roles,
            )
            roles.update(
                {
                    agent["agent_id"]: agent["role_family"]
                    for agent in normalized_agents
                }
            )
            allowed_output_roles = set(
                _live_hotpotqa_output_role_families(domain)
            )
            terminal_only_roles = set(
                _live_terminal_only_role_families(domain["role_constraints"])
            )
            selected_terminal_output_ids = [
                agent["agent_id"]
                for agent in normalized_agents
                if agent["role_family"] in terminal_only_roles
            ]
            if len(selected_terminal_output_ids) > 1:
                raise ValueError(
                    "add_subgraph selected more than one terminal-only role"
                )
            output_ids = [
                agent_id
                for agent_id in endpoint_ids
                if roles[agent_id] in allowed_output_roles
            ]
            explicit_output_assignment = (
                _live_explicit_output_assignment_required(domain)
                or bool(selected_terminal_output_ids)
            )
            if selected_terminal_output_ids:
                output_ids = selected_terminal_output_ids
            if isolated_boundary and explicit_output_assignment:
                raise ValueError(
                    "add_subgraph isolated replacement boundary cannot require "
                    "an explicit Output assignment"
                )
            if explicit_output_assignment:
                if not output_ids:
                    raise ValueError(
                        "add_subgraph explicit Output domain has no legal target"
                    )
                if "output_agent_id" not in schema["required"]:
                    schema["required"].append("output_agent_id")
                schema["properties"]["output_agent_id"] = {
                    "enum": output_ids
                }
            else:
                schema["properties"]["output_agent_id"] = (
                {"type": "null"}
                if (
                    isolated_boundary
                    or (
                        current_output_agent_id is not None
                    )
                    or _live_defer_output_assignment(domain)
                )
                else
                {
                    "anyOf": [
                        {"enum": output_ids},
                        {"type": "null"},
                    ]
                }
                if output_ids
                else {"type": "null"}
                )
        else:
            relation_items = schema["properties"]["relations"]["items"]
            for branch in relation_items["anyOf"]:
                branch["properties"]["source_id"] = {"enum": endpoint_ids}
                branch["properties"]["target_id"] = {"enum": endpoint_ids}
            schema["properties"]["output_agent_id"] = {
                "anyOf": [{"enum": endpoint_ids}, {"type": "null"}]
            }
    elif action == "modify_agent":
        if modify_field not in DIRECTOR_MODIFY_AGENT_FIELDS:
            raise ValueError("modify_agent v3 parameter phase requires a live field")
        candidates = _live_modify_agent_candidates(
            action_target_domains,
            modify_field,
        )
        by_id = {candidate["agent_id"]: candidate for candidate in candidates}
        if not isinstance(modify_agent_id, str) or modify_agent_id not in by_id:
            raise ValueError(
                "modify_agent v3 parameter phase requires a live Agent target"
            )
        selected_candidate = by_id[modify_agent_id]
        raw_profiles = selected_candidate.get("execution_profiles")
        if modify_field == "execution_mode" and raw_profiles is not None:
            profiles = _live_execution_profiles(
                raw_profiles,
                label=(
                    f"modify_agent.{modify_agent_id}.execution_profiles"
                ),
            )
            schema = {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "action",
                    "agent_id",
                    "execution_mode",
                    "allowed_tools",
                ],
                "properties": {
                    "action": {"const": "modify_agent"},
                    "agent_id": {"const": modify_agent_id},
                    "execution_mode": (
                        {"const": profiles[0][0]}
                        if len(profiles) == 1
                        else {
                            "enum": list(
                                dict.fromkeys(mode for mode, _ in profiles)
                            )
                        }
                    ),
                    "allowed_tools": (
                        {"const": list(profiles[0][1])}
                        if len(profiles) == 1
                        else {
                            "enum": [
                                list(tool_ids) for _, tool_ids in profiles
                            ]
                        }
                    ),
                },
            }
            if len(profiles) > 1:
                schema["oneOf"] = [
                    {
                        "properties": {
                            "execution_mode": {"const": execution_mode},
                            "allowed_tools": {"const": list(allowed_tools)},
                        }
                    }
                    for execution_mode, allowed_tools in profiles
                ]
        else:
            schema = json.loads(
                director_modify_agent_field_sampling_json_schema_text(
                    modify_field
                )
            )
            schema["properties"]["agent_id"] = {"const": modify_agent_id}
            discrete_values = _live_discrete_values(
                selected_candidate,
                modify_field,
            )
            if discrete_values is not None:
                schema["properties"][modify_field] = {
                    "enum": list(discrete_values)
                }
    elif action in {"delete_agent", "set_output"}:
        agent_ids = _live_string_domain(
            domain.get("agent_ids"),
            label=f"{action}.agent_ids",
        )
        schema = json.loads(director_state_conditioned_sampling_json_schema_text(action))
        schema["properties"]["agent_id"] = {"enum": list(agent_ids)}
    elif action == "set_relation":
        candidates = domain.get("candidates")
        if not isinstance(candidates, (list, tuple)) or not candidates:
            raise ValueError("set_relation exact live candidates are missing")
        if (
            type(relation_candidate_index) is not int
            or not 0 <= relation_candidate_index < len(candidates)
        ):
            raise ValueError("set_relation candidate index is outside the live domain")
        candidate = candidates[relation_candidate_index]
        required = (
            "source_id",
            "target_id",
            "source_to_target",
            "target_to_source",
        )
        if not isinstance(candidate, Mapping) or set(candidate) != set(required):
            raise ValueError("set_relation live candidate is malformed")
        source_id = candidate.get("source_id")
        target_id = candidate.get("target_id")
        source_to_target = candidate.get("source_to_target")
        target_to_source = candidate.get("target_to_source")
        if (
            not isinstance(source_id, str)
            or not source_id
            or not isinstance(target_id, str)
            or not target_id
            or source_id == target_id
            or type(source_to_target) is not bool
            or type(target_to_source) is not bool
        ):
            raise ValueError("set_relation live candidate violates relation semantics")
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", *required],
            "properties": {
                "action": {"const": "set_relation"},
                "source_id": {"const": source_id},
                "target_id": {"const": target_id},
                "source_to_target": {"const": source_to_target},
                "target_to_source": {"const": target_to_source},
            },
        }
    elif action == "finish":
        if domain.get("admissible") is not True:
            raise ValueError("finish is outside the live terminal domain")
        schema = json.loads(director_state_conditioned_sampling_json_schema_text("finish"))
    else:
        raise ValueError("live parameter schema received an unknown action")
    return json.dumps(
        schema,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def director_live_relation_candidate_selector_json_schema_text(
    action_target_domains: Mapping[str, Any],
) -> str:
    """Render the v3 selector for exact non-self, non-no-op relation candidates."""

    domain = action_target_domains.get("set_relation")
    if not isinstance(domain, Mapping):
        raise ValueError("set_relation live target domain is missing")
    candidates = domain.get("candidates")
    if not isinstance(candidates, (list, tuple)) or not candidates:
        raise ValueError("set_relation exact live candidates are missing")
    # Validate every candidate through the exact parameter renderer before
    # exposing its index to constrained decoding.
    for index in range(len(candidates)):
        director_live_action_parameter_json_schema_text(
            "set_relation",
            action_target_domains,
            relation_candidate_index=index,
        )
    return json.dumps(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "candidate_index"],
            "properties": {
                "action": {"const": "set_relation"},
                "candidate_index": {"enum": list(range(len(candidates)))},
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def director_validate_live_action_target_domains(
    actions: Sequence[str],
    action_target_domains: Mapping[str, Any],
) -> None:
    """Fail closed on every v3 domain before the first generation request."""

    normalized_actions = tuple(actions)
    if set(action_target_domains) != set(normalized_actions):
        raise ValueError("live action target domains must match admitted actions")
    for action in normalized_actions:
        if action == "add_subgraph":
            director_live_add_subgraph_agent_declarations_json_schema_text(
                action_target_domains
            )
        elif action == "modify_agent":
            domain = action_target_domains.get("modify_agent")
            if not isinstance(domain, Mapping):
                raise ValueError("modify_agent live target domain is missing")
            fields = _live_string_domain(
                domain.get("mutable_fields"),
                label="modify_agent.mutable_fields",
            )
            if any(field not in DIRECTOR_MODIFY_AGENT_FIELDS for field in fields):
                raise ValueError("modify_agent mutable field domain is invalid")
            candidates = domain.get("per_agent_candidates")
            if not isinstance(candidates, (list, tuple)):
                raise ValueError("modify_agent per-Agent candidates are missing")
            admitted_fields = tuple(
                field
                for field in fields
                if any(
                    isinstance(candidate, Mapping)
                    and field in candidate.get("mutable_fields", ())
                    for candidate in candidates
                )
            )
            director_modify_agent_field_selector_json_schema_text(admitted_fields)
            for field in admitted_fields:
                director_live_modify_agent_selector_json_schema_text(
                    action_target_domains,
                    field,
                )
        elif action == "set_relation":
            director_live_relation_candidate_selector_json_schema_text(
                action_target_domains
            )
        else:
            director_live_action_parameter_json_schema_text(
                action,
                action_target_domains,
            )


def director_modify_agent_field_sampling_json_schema_text(field_name: str) -> str:
    """Render one exact atomic ``modify_agent`` field branch."""

    if field_name not in _MUTABLE_AGENT_PROPERTIES:
        raise ValueError("modify_agent field selector returned an unknown field")
    return json.dumps(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "agent_id", field_name],
            "properties": {
                "action": {"const": "modify_agent"},
                "agent_id": _NON_EMPTY_STRING_SCHEMA,
                field_name: _MUTABLE_AGENT_PROPERTIES[field_name],
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def director_model_admissible_schema_branch_v1(actions: Sequence[str]) -> str:
    """Encode one legacy v1 model-admissible receipt label."""

    normalized = tuple(actions)
    director_model_admissible_sampling_json_schema_text_v1(normalized)
    return "admissible:" + "|".join(normalized)


def director_model_admissible_schema_branch(actions: Sequence[str]) -> str:
    """Encode one v2 branch-exact model-admissible receipt label."""

    normalized = tuple(actions)
    director_model_admissible_sampling_json_schema_text(normalized)
    return "admissible-v2:" + "|".join(normalized)


def director_model_admissible_schema_branch_v3(actions: Sequence[str]) -> str:
    """Encode one v3 action discriminator paired with live target domains."""

    normalized = tuple(actions)
    director_model_admissible_sampling_json_schema_text_v3(normalized)
    return "admissible-v3:" + "|".join(normalized)


def director_actions_from_admissible_schema_branch(
    branch: str,
) -> Tuple[str, ...]:
    """Decode and validate one v1, v2, or v3 action-domain receipt label."""

    if not isinstance(branch, str):
        raise ValueError("model-admissible schema branch has an invalid prefix")
    if branch.startswith("admissible-v3:"):
        prefix = "admissible-v3:"
        renderer = director_model_admissible_sampling_json_schema_text_v3
    elif branch.startswith("admissible-v2:"):
        prefix = "admissible-v2:"
        renderer = director_model_admissible_sampling_json_schema_text
    elif branch.startswith("admissible:"):
        prefix = "admissible:"
        renderer = director_model_admissible_sampling_json_schema_text_v1
    else:
        raise ValueError("model-admissible schema branch has an invalid prefix")
    actions = tuple(branch[len(prefix) :].split("|"))
    renderer(actions)
    return actions


DIRECTOR_TRANSCRIPT_SCHEMA = "flowsteer.director.transcript.v1"
DIRECTOR_TRANSCRIPT_HEADER = "Flow-Director chat transcript"


def encode_director_transcript(
    messages: Sequence[Mapping[str, str]],
) -> str:
    """Serialize the exact multi-turn Director messages into a receipt string."""

    normalized: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError("Director transcript has an unsupported role")
        if not isinstance(content, str) or not content:
            raise ValueError("Director transcript messages require non-empty content")
        normalized.append({"role": role, "content": content})
    if (
        len(normalized) < 2
        or normalized[0]["role"] != "system"
        or normalized[0]["content"] not in _SUPPORTED_DIRECTOR_SYSTEM_PROMPTS
    ):
        raise ValueError(
            "Director transcript must start with a supported versioned system prompt"
        )
    if normalized[1]["role"] != "user":
        raise ValueError("Director transcript must start with a user task message")
    payload = {
        "schema_version": DIRECTOR_TRANSCRIPT_SCHEMA,
        "messages": normalized,
    }
    return DIRECTOR_TRANSCRIPT_HEADER + "\n\n" + json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def decode_director_transcript(
    prompt: str,
) -> Optional[Tuple[Mapping[str, str], ...]]:
    """Decode a canonical transcript, or return ``None`` for a legacy prompt."""

    if not isinstance(prompt, str) or not prompt.startswith(
        DIRECTOR_TRANSCRIPT_HEADER + "\n\n"
    ):
        return None
    _, _, raw_payload = prompt.partition("\n\n")
    try:
        payload = json.loads(raw_payload)
    except (TypeError, ValueError) as exc:
        raise DirectorError("Director transcript is not valid JSON") from exc
    if not isinstance(payload, Mapping) or payload.get(
        "schema_version"
    ) != DIRECTOR_TRANSCRIPT_SCHEMA:
        raise DirectorError("Director transcript has an unsupported schema")
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        raise DirectorError("Director transcript has no message list")
    try:
        canonical = encode_director_transcript(raw_messages)
    except (TypeError, ValueError) as exc:
        raise DirectorError("Director transcript violates its message contract") from exc
    if canonical != prompt:
        raise DirectorError("Director transcript is not canonical")
    return tuple(dict(message) for message in raw_messages)


class DirectorError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DirectorResponse:
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class DirectorClient(Protocol):
    async def propose(
        self,
        prompt: str,
        *,
        seed: Optional[int] = None,
        action_json_schema: Optional[str] = None,
        action_json_schema_version: Optional[str] = None,
        action_schema_branch: Optional[str] = None,
        action_target_domains_json: Optional[str] = None,
        action_target_domain_version: Optional[str] = None,
    ) -> DirectorResponse:
        ...


class OpenAIDirectorClient:
    """OpenAI-compatible chat client for the local Qwen3.5-9B endpoint."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8015/v1",
        model: str = "supervisor_theta",
        api_key_env: Optional[str] = None,
        policy_version: str = "qwen3.5-9b-sglang-unversioned",
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 768,
        timeout_seconds: float = 180.0,
        max_retries: int = 2,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be absolute HTTP(S)")
        if urlsplit(base_url).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Flow-Director must use the local Qwen3.5-9B endpoint")
        if model != "supervisor_theta":
            raise ValueError("Flow-Director model must be supervisor_theta")
        if not model.strip() or not policy_version.strip():
            raise ValueError("model and policy_version must be non-empty")
        if temperature < 0 or not 0 < top_p <= 1:
            raise ValueError("Director temperature/top_p are invalid")
        if max_tokens <= 0 or timeout_seconds <= 0 or max_retries < 0:
            raise ValueError("Director token, timeout, and retry limits are invalid")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.policy_version = policy_version
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)

    async def propose(
        self,
        prompt: str,
        *,
        seed: Optional[int] = None,
        action_json_schema: Optional[str] = None,
        action_json_schema_version: Optional[str] = None,
        action_schema_branch: Optional[str] = None,
        action_target_domains_json: Optional[str] = None,
        action_target_domain_version: Optional[str] = None,
    ) -> DirectorResponse:
        if any(
            value is not None
            for value in (
                action_json_schema,
                action_json_schema_version,
                action_schema_branch,
                action_target_domains_json,
                action_target_domain_version,
            )
        ):
            raise DirectorError(
                "state-conditioned action schemas require the native SGLang client"
            )
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        ):
            raise ValueError("Director seed must be a non-negative integer or None")
        api_key = "EMPTY"
        if self.api_key_env:
            api_key = os.getenv(self.api_key_env, "")
            if not api_key:
                raise DirectorError(f"missing Director credential environment variable: {self.api_key_env}")
        messages = decode_director_transcript(prompt)
        payload = {
            "model": self.model,
            "messages": (
                list(messages)
                if messages is not None
                else [
                    {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
            ),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if seed is not None:
            # SkillFlow sends the generation seed through the provider payload.
            payload["seed"] = seed
        last_error: BaseException | None = None
        started_at = time.monotonic()
        for attempt in range(self.max_retries + 1):
            try:
                value = await asyncio.to_thread(self._post, api_key, payload)
                parsed = self._parse(value)
                metadata = dict(parsed.metadata)
                metadata.update(
                    {
                        "latency_ms": max(
                            (time.monotonic() - started_at) * 1000.0,
                            0.0,
                        ),
                        "attempt_count": attempt + 1,
                        "generation_seed": seed,
                    }
                )
                return DirectorResponse(parsed.text, metadata)
            except HTTPError as exc:
                last_error = exc
                if not (exc.code in {408, 409, 425, 429} or exc.code >= 500):
                    break
            except (URLError, TimeoutError, socket.timeout) as exc:
                last_error = exc
            if attempt < self.max_retries:
                await asyncio.sleep(min(2.0**attempt, 4.0))
        detail = f"HTTP {last_error.code}" if isinstance(last_error, HTTPError) else type(last_error).__name__
        raise DirectorError(f"Director request failed: {detail}") from last_error

    def _post(self, api_key: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        request = Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "FlowSteer-Director/1",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            value = json.load(response)
        if not isinstance(value, dict):
            raise DirectorError("Director returned a non-object response")
        return value

    def _parse(self, value: Mapping[str, Any]) -> DirectorResponse:
        choices = value.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise DirectorError("Director response has no completion choice")
        message = choices[0].get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise DirectorError("Director response has no text content")
        usage = value.get("usage") if isinstance(value.get("usage"), dict) else {}
        return DirectorResponse(
            text=message["content"],
            metadata={
                "policy_version": self.policy_version,
                "model": value.get("model", self.model),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "request_id": value.get("id"),
            },
        )


@dataclass(frozen=True, slots=True)
class DirectorTurn:
    turn_index: int
    prompt: str
    response: DirectorResponse
    canvas_result: AgentWorkflowStepResult


@dataclass(frozen=True, slots=True)
class OrchestrationResult:
    final_answer: Optional[str]
    turns: Tuple[DirectorTurn, ...]
    final_graph: Mapping[str, Any]
    termination_reason: str
    explicit_finish: bool
    valid_lineage_fallback_used: bool = False
    valid_lineage_fallback_receipt: Mapping[str, Any] = field(
        default_factory=dict
    )


class AgentGraphOrchestrator:
    def __init__(
        self,
        registry: ModelRegistry,
        client: DirectorClient,
        *,
        max_rounds: int = 20,
        seed: int = 42,
        catalog_order_seed: int | str | None = None,
        history_window: int = 4,
        sampling_base_seed: int | None = None,
        sampling_coordinate: ScientificSamplingCoordinate | None = None,
        tool_registry: Optional[ToolRegistry] = None,
        sampling_action_profile: Optional[str] = None,
        sampling_action_schema_version: str = (
            DIRECTOR_STATE_CONDITIONED_ACTION_SCHEMA_VERSION
        ),
        system_prompt: Optional[str] = None,
        prompt_version: str = DIRECTOR_PROMPT_VERSION,
        semantic_protocol: str = "none",
        recovery_policy: str = "default",
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        if isinstance(history_window, bool) or not isinstance(history_window, int) or history_window < 1:
            raise ValueError("history_window must be a positive integer")
        self.registry = registry
        self.client = client
        self.max_rounds = max_rounds
        self.seed = seed
        if (sampling_base_seed is None) != (sampling_coordinate is None):
            raise ValueError(
                "sampling_base_seed and sampling_coordinate must be supplied together"
            )
        if sampling_base_seed is not None and (
            type(sampling_base_seed) is not int
            or not 0 <= sampling_base_seed < 2**64
        ):
            raise ValueError("sampling_base_seed must be an unsigned 64-bit integer")
        self.sampling_base_seed = sampling_base_seed
        self.sampling_coordinate = sampling_coordinate
        # Sampling varies across rollouts, while a same-task/same-condition
        # group must see the same catalog presentation in its exact prompt.
        self.catalog_order_seed = seed if catalog_order_seed is None else catalog_order_seed
        self.history_window = history_window
        self.tool_registry = tool_registry
        if not isinstance(prompt_version, str) or not prompt_version.strip():
            raise ValueError("Director prompt_version must be non-empty text")
        self.prompt_version = prompt_version.strip()
        expected_system_prompt = director_system_prompt_for_version(
            self.prompt_version
        )
        if system_prompt is None:
            self.system_prompt = expected_system_prompt
        elif not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("Director system_prompt must be non-empty text")
        elif system_prompt != expected_system_prompt:
            raise ValueError(
                "Director system_prompt does not match its prompt_version"
            )
        else:
            self.system_prompt = system_prompt
        if semantic_protocol != "none" and not verified_qa_semantic_protocol(
            semantic_protocol
        ):
            raise ValueError("unsupported Director semantic_protocol")
        if recovery_policy not in {
            "default",
            PRESERVE_DIAGNOSE_REPAIR_AUGMENT_POLICY,
        }:
            raise ValueError("unsupported Director recovery_policy")
        self.semantic_protocol = semantic_protocol
        self.recovery_policy = recovery_policy
        if sampling_action_profile not in {
            None,
            DIRECTOR_PROGRESSIVE_ACTION_MASK_PROFILE,
            DIRECTOR_MODEL_ADMISSIBLE_ACTION_MASK_PROFILE,
        }:
            raise ValueError("unsupported Director sampling action profile")
        if (
            not isinstance(sampling_action_schema_version, str)
            or not sampling_action_schema_version.strip()
        ):
            raise ValueError("sampling_action_schema_version must be non-empty")
        self.sampling_action_profile = sampling_action_profile
        self.sampling_action_schema_version = (
            sampling_action_schema_version.strip()
        )
        if (
            self.sampling_action_profile
            == DIRECTOR_MODEL_ADMISSIBLE_ACTION_MASK_PROFILE
            and self.sampling_action_schema_version
            not in {
                DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V1,
                DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION,
                DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3,
            }
        ):
            raise ValueError("unsupported model-admissible action schema version")

    def action_schema_request(
        self,
        env: AgentWorkflowEnv,
    ) -> Mapping[str, str]:
        """Return the evaluation-only constrained action branch for this state.

        FlowSteer's progressive Canvas executes every accepted structural ADD
        before asking the policy for the next edit.  The v2 model-admissible
        profile exposes the legal action discriminator; the native SGLang
        client then samples the selected action under its exact singleton
        schema.  The strict parser remains authoritative and no sampled action
        is repaired.
        """

        if self.sampling_action_profile is None:
            return {}
        if (
            self.sampling_action_profile
            == DIRECTOR_MODEL_ADMISSIBLE_ACTION_MASK_PROFILE
        ):
            actions = env.model_admissible_action_types()
            legacy_v1 = (
                self.sampling_action_schema_version
                == DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V1
            )
            live_v3 = (
                self.sampling_action_schema_version
                == DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3
            )
            if live_v3:
                target_domains = env.model_admissible_action_targets()
                return {
                    "action_json_schema": (
                        director_model_admissible_sampling_json_schema_text_v3(actions)
                    ),
                    "action_json_schema_version": self.sampling_action_schema_version,
                    "action_schema_branch": (
                        director_model_admissible_schema_branch_v3(actions)
                    ),
                    "action_target_domains_json": (
                        director_live_action_target_domains_json(
                            actions,
                            target_domains,
                        )
                    ),
                    "action_target_domain_version": (
                        DIRECTOR_ACTION_TARGET_DOMAIN_SCHEMA_VERSION
                    ),
                }
            return {
                "action_json_schema": (
                    director_model_admissible_sampling_json_schema_text_v1(actions)
                    if legacy_v1
                    else director_model_admissible_sampling_json_schema_text(actions)
                ),
                "action_json_schema_version": self.sampling_action_schema_version,
                "action_schema_branch": (
                    director_model_admissible_schema_branch_v1(actions)
                    if legacy_v1
                    else director_model_admissible_schema_branch(actions)
                ),
            }
        finish_admissible = env.finish_admissibility().get("admissible") is True
        action_branch = "finish" if finish_admissible else "add_subgraph"
        return {
            "action_json_schema": director_state_conditioned_sampling_json_schema_text(
                action_branch
            ),
            "action_json_schema_version": self.sampling_action_schema_version,
            "action_schema_branch": action_branch,
        }

    def generation_seed(self, round_index: int) -> int:
        """Return the exact Director action seed for one zero-based Canvas round."""

        if type(round_index) is not int or round_index < 0:
            raise ValueError("round_index must be a non-negative integer")
        if self.sampling_coordinate is None:
            return self.seed + round_index
        assert self.sampling_base_seed is not None
        return derive_generation_seed(
            base_seed=self.sampling_base_seed,
            coordinate=self.sampling_coordinate,
            step_index=round_index + 1,
            phase=GenerationPhase.ACTION,
        )

    @property
    def sampling_receipt(self) -> Mapping[str, Any]:
        """Return the trajectory-level SkillFlow scientific sampling receipt."""

        if self.sampling_coordinate is None:
            return {}
        assert self.sampling_base_seed is not None
        return {
            "algorithm": SCIENTIFIC_SAMPLING_ALGORITHM,
            "base_seed": self.sampling_base_seed,
            "coordinate": self.sampling_coordinate.to_value(),
            "phase": GenerationPhase.ACTION.value,
        }

    def _model_catalog(self) -> list[dict[str, Any]]:
        # Present the frozen set in a deterministic per-condition order.  The
        # previous sorted order made the alphabetically first family the de
        # facto default after the preferred-model hint was removed.  This does
        # not select a model; every action still names the Director's choice.
        catalog_model_ids = list(self.registry.model_ids)
        random.Random(self.catalog_order_seed).shuffle(catalog_model_ids)
        catalog = [
            {
                "model_id": model_id,
                "selection_weight": self.registry.require_model(model_id).selection_weight,
                "cheap_weight": self.registry.require_model(model_id).cheap_weight,
                "fast_weight": self.registry.require_model(model_id).fast_weight,
                "routing_metadata": {
                    key: value
                    for key, value in self.registry.require_model(model_id).metadata.items()
                    if key
                    in {
                        "family",
                        "profile",
                        "text_qa_canary",
                        "canary_source",
                    }
                },
            }
            for model_id in catalog_model_ids
        ]
        if verified_qa_semantic_protocol(self.semantic_protocol):
            for item in catalog:
                item["provider_id"] = self.registry.provider_for(
                    str(item["model_id"])
                ).provider_id
        return catalog

    def _tool_catalog(self, env: AgentWorkflowEnv) -> list[dict[str, object]]:
        if self.tool_registry is None:
            return []
        if env.runtime.tool_registry is not self.tool_registry:
            raise DirectorError(
                "Director and AgentRuntime must share the same ToolRegistry"
            )
        dataset_id = env.runtime.dataset_id
        return [
            capability.to_value()
            for capability in self.tool_registry.capabilities
            if dataset_id is None or capability.supports_dataset(dataset_id)
        ]

    def _canvas_observation(
        self,
        env: AgentWorkflowEnv,
        *,
        include_task_context: bool,
        skills: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        # FlowSteer applies terminal completeness at FINISH.  Intermediate
        # Canvas observations expose only mutation-safety validation.
        partial_validation = env.graph.validate(
            self.registry,
            require_complete=False,
        )
        snapshot = env.snapshot()
        directed_edges = [
            {"from": source_id, "to": target_id}
            for relation in env.graph.relations
            for source_id, target_id in relation.directed_edges()
        ]
        payload: dict[str, Any] = {
            "current_graph": env.graph.to_dict(),
            "topology_statistics": env.graph.topology_statistics(),
            "canvas_feedback": snapshot.last_feedback,
            "admissible_action_types": list(
                env.model_admissible_action_types()
            ),
            # These are existing admission constraints enforced by
            # AgentWorkflowEnv, not a role or topology template.  Surfacing
            # them lets the minimal policy observe its terminal boundary.
            "terminal_constraints": {
                "explicit_finish_required": True,
                "require_exact_answer_tag": env.require_exact_answer_tag,
                "require_format_agent": env.require_format_agent,
                "required_tool_id": env.required_tool_id,
            },
        }
        if verified_qa_semantic_protocol(self.semantic_protocol):
            payload["action_target_domains"] = (
                env.model_admissible_action_targets()
            )
            pending_ingress_ids = (
                env._pending_role_conditional_ingress_consumer_ids()
            )
            if pending_ingress_ids:
                payload["pending_routed_artifact_inputs"] = {
                    "consumer_agent_ids": list(pending_ingress_ids),
                    "repair_actions": list(
                        env.model_admissible_action_types()
                    ),
                }
            recent_rejections: list[dict[str, Any]] = []
            for entry in reversed(env.history):
                if entry.accepted:
                    continue
                action_value = (
                    {} if entry.action is None else entry.action.to_dict()
                )
                target = action_value.get("agent_id")
                if target is None and action_value.get("source_id") is not None:
                    target = {
                        "source_id": action_value.get("source_id"),
                        "target_id": action_value.get("target_id"),
                    }
                reason = " ".join(entry.feedback.split())
                recent_rejections.append(
                    {
                        "revision": entry.revision,
                        "action": action_value.get("action"),
                        "target": target,
                        "reason": reason[:360],
                    }
                )
                if len(recent_rejections) >= 3:
                    break
            if recent_rejections:
                payload["recent_rejected_actions"] = list(
                    reversed(recent_rejections)
                )
        if env.required_evidence_tool_id is not None:
            # The HotpotQA semantic gate distinguishes an evidence-bearing
            # read receipt from environment-native action Tools.  Expose that
            # existing admission constraint to the Director so the selected
            # Reasoner can declare the exact capability in its first Canvas
            # edit; this does not prescribe an Agent count or topology.
            payload["terminal_constraints"]["required_evidence_tool_id"] = (
                env.required_evidence_tool_id
            )
        if self.semantic_protocol != "none":
            payload["semantic_protocol"] = self.semantic_protocol
            if role_conditional_hotpotqa_protocol(self.semantic_protocol):
                # Role families remain part of the Director's open search
                # space.  These receipts describe capability-specific
                # validation only; none is an existence, ordering, or FINISH
                # prerequisite.
                payload["optional_role_capabilities"] = {
                    "question_scope": {
                        "preserve_original": True,
                        "unrequested_qualifier_allowed": False,
                    },
                    "reasoner": {
                        "aligns_fact_predicate_arguments_to_answer_slot": True,
                        "determines_semantic_answer_when_selected": True,
                        "unexpected_equal_comparison_recheck": [
                            "original_question_scope",
                            "entity_attribute_binding",
                            "explicit_evidence",
                            "upstream_contract_scope",
                        ],
                    },
                    "verifier": {
                        "checks_when_selected": [
                            "explicit_evidence",
                            "entity_attribute_binding",
                            "multi_hop_completeness",
                            "question_scope_preservation",
                        ],
                        "candidate_reselection_allowed": False,
                    },
                    "format": {
                        "operation_when_selected": (
                            "serialize_routed_semantic_answer"
                        ),
                        "original_question_visible": False,
                        "reasoning_allowed": False,
                        "candidate_reselection_allowed": False,
                    },
                    "react": {
                        "kind": "execution_mode",
                        "schedule": (
                            "Thought -> Action(tool) -> Observation -> "
                            "Thought -> Final"
                        ),
                    },
                    "terminal_artifact_constraints": {
                        "required_evidence_tool_id": (
                            env.required_evidence_tool_id
                        ),
                        "exact_output_syntax_required": (
                            env.require_exact_answer_tag
                        ),
                    },
                    "topology_and_role_multiplicity_source": (
                        "action_target_domains"
                    ),
                    "max_agents_per_add_subgraph": (
                        env.max_agents_per_subgraph
                    ),
                }
            elif flexible_hotpotqa_semantic_protocol(self.semantic_protocol):
                # Keep SkillFlow's Supervisor instruction neutral and expose
                # the task contract through the progressive FlowSteer Canvas.
                # These are terminal artifact capabilities, not a prescribed
                # role count, direct edge list, or workflow template.
                payload["semantic_lineage_constraints"] = {
                    "question_scope": {
                        "preserve_original": True,
                        "unrequested_qualifier_allowed": False,
                    },
                    "reasoner_capability": {
                        "determines_semantic_answer": True,
                        "aligns_fact_predicate_arguments_to_answer_slot": True,
                        "unexpected_equal_comparison_recheck": [
                            "original_question_scope",
                            "entity_attribute_binding",
                            "explicit_evidence",
                            "upstream_contract_scope",
                        ],
                    },
                    "verifier_capability": {
                        "checks": [
                            "explicit_evidence",
                            "entity_attribute_binding",
                            "multi_hop_completeness",
                            "question_scope_preservation",
                        ],
                        "candidate_reselection_allowed": False,
                    },
                    "formatter_capability": {
                        "operation": "serialize_verified_semantic_answer",
                        "original_question_visible": False,
                        "reasoning_allowed": False,
                        "candidate_reselection_allowed": False,
                    },
                    "react": {
                        "kind": "execution_mode",
                        "schedule": (
                            "Thought -> Action(tool) -> Observation -> "
                            "Thought -> Final"
                        ),
                    },
                    "terminal_artifact_constraints": {
                        "required_evidence_tool_id": (
                            env.required_evidence_tool_id
                        ),
                        "supported_verifier_artifact_required": True,
                        "formatter_exact_copy_required": True,
                    },
                    "topology_and_role_multiplicity_source": (
                        "action_target_domains"
                    ),
                    "max_agents_per_add_subgraph": (
                        env.max_agents_per_subgraph
                    ),
                }
            else:
                # Persist the legacy exact-lineage receipt for old protocol
                # versions and their replay/evaluation artifacts.
                payload["semantic_lineage_constraints"] = {
                    "semantic_answer_owner_role_family": "reasoner",
                    "required_evidence_tool_id": env.required_evidence_tool_id,
                    "required_evidence_tool_owner": (
                        "reasoner_or_direct_reasoner_predecessor"
                    ),
                    "required_evidence_execution_mode": "react",
                    "verifier_execution_mode": "reasoning",
                    "formatter_execution_mode": "reasoning",
                    "required_direct_role_edges": [
                        ["reasoner", "verifier"],
                        ["verifier", "format"],
                    ],
                    "output_role_family": "format",
                    "formatter_original_question_visible": False,
                    "formatter_answer_reselection_allowed": False,
                    "semantic_answer_owner_count": 1,
                    "max_agents_per_add_subgraph": (
                        env.max_agents_per_subgraph
                    ),
                    "output_agent_id_optional_until_lineage_complete": True,
                }
        if self.recovery_policy != "default":
            payload["recovery_policy"] = self.recovery_policy
        if directed_edges:
            # The two-bit relation remains the canonical mutation receipt.  A
            # direct edge view avoids making the Director mentally invert a
            # relation after AgentGraph canonicalizes endpoint order.
            payload["directed_edges"] = directed_edges
        if partial_validation.issues:
            payload["structural_issues"] = [
                {
                    "code": issue.code,
                    "message": issue.message,
                }
                for issue in partial_validation.issues
            ]
        if env.graph.output_agent_id is not None:
            format_issue = env.format_agent_issue()
            if format_issue is not None:
                payload["terminal_format_issue"] = format_issue
        # FlowSteer returns terminal-constraint state to the policy, while
        # SkillFlow accepts completion only after validation.  Expose the
        # revision-local gate and its first measured failure stage so the
        # Director repairs the responsible semantic node instead of probing
        # FINISH or repeatedly modifying the Formatter.
        payload["finish_admissibility"] = env.finish_admissibility()
        if include_task_context:
            payload.update(
                {
                    "task": env.problem,
                    "model_catalog": self._model_catalog(),
                }
            )
            tool_catalog = self._tool_catalog(env)
            if tool_catalog:
                payload["tool_catalog"] = tool_catalog
            if env.max_agents is not None:
                payload["max_agents"] = env.max_agents
        if skills:
            # The MD's signal-isolation contract distinguishes a forced
            # exploration condition from an evidence-gated Skill prior.  Both
            # are prompt context only, but they must remain separate in the
            # exact Director observation and trajectory receipt.
            available_skills: list[dict[str, Any]] = []
            exploration_conditions: list[dict[str, Any]] = []
            for item in skills:
                value = dict(item)
                if value.get("application_mode") == "forced_probe_condition":
                    exploration_conditions.append(value)
                else:
                    available_skills.append(value)
            if available_skills:
                payload["available_skills"] = available_skills
            if exploration_conditions:
                payload["exploration_conditions"] = exploration_conditions
        return payload

    @staticmethod
    def _observation_message(payload: Mapping[str, Any]) -> str:
        return (
            "Canvas observation. Choose exactly one next action from the defined "
            "action space using only the state below.\n\n"
            + json.dumps(
                dict(payload),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    def build_prompt(
        self,
        env: AgentWorkflowEnv,
        turn_index: int,
        skills: Sequence[Mapping[str, Any]],
    ) -> str:
        """Start one SkillFlow-style persistent Director conversation."""

        if isinstance(turn_index, bool) or not isinstance(turn_index, int) or turn_index < 0:
            raise ValueError("turn_index must be a non-negative integer")
        initial = self._canvas_observation(
            env,
            include_task_context=True,
            skills=skills,
        )
        return encode_director_transcript(
            (
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self._observation_message(initial)},
            )
        )

    def continue_prompt(
        self,
        previous_prompt: str,
        assistant_content: str,
        env: AgentWorkflowEnv,
        skills: Sequence[Mapping[str, Any]],
    ) -> str:
        """Append the real sampled action and current Canvas observation."""

        messages = decode_director_transcript(previous_prompt)
        if messages is None:
            raise DirectorError("cannot continue a legacy single-user Director prompt")
        if not isinstance(assistant_content, str) or not assistant_content:
            raise DirectorError("Director continuation requires sampled assistant content")
        observation = self._canvas_observation(
            env,
            include_task_context=False,
            skills=skills,
        )
        if (
            verified_qa_semantic_protocol(self.semantic_protocol)
            and env.history
            and env.history[-1].accepted is False
        ):
            # SkillFlow keeps the sampled invalid Action in the trajectory but
            # presents only its canonical failure Observation to the next
            # policy turn. Replace the pre-action Canvas observation in place
            # so a rejected answer-bearing contract cannot become an imitation
            # target or semantic anchor in the persistent Director transcript.
            redacted = list(messages)
            redacted[-1] = {
                "role": "user",
                "content": self._observation_message(observation),
            }
            return encode_director_transcript(tuple(redacted))
        continuation = list(messages[2:])
        continuation.extend(
            (
                {"role": "assistant", "content": assistant_content},
                {
                    "role": "user",
                    "content": self._observation_message(observation),
                },
            )
        )
        # Keep the immutable task/catalog context and a bounded real message
        # continuation.  Unlike the former reconstructed history JSON, these
        # are the exact assistant actions and Canvas observations seen by Qwen.
        continuation = continuation[-2 * self.history_window :]
        return encode_director_transcript(
            (messages[0], messages[1], *continuation)
        )

    @staticmethod
    def consumed_assistant_content(
        response: DirectorResponse,
        canvas: AgentWorkflowStepResult,
    ) -> str:
        action = canvas.action
        if action is None:
            return response.text
        return response.text[: action.consumed_end]

    async def run(
        self,
        env: AgentWorkflowEnv,
        problem: str,
        *,
        skills: Sequence[Mapping[str, Any]] = (),
    ) -> OrchestrationResult:
        env.reset(problem)
        turns: list[DirectorTurn] = []
        prompt = self.build_prompt(env, 0, skills)
        for index in range(self.max_rounds):
            if (
                self.sampling_action_profile
                == DIRECTOR_MODEL_ADMISSIBLE_ACTION_MASK_PROFILE
                and not env.model_admissible_action_types()
            ):
                # Constrained decoding cannot encode an empty JSON-schema
                # branch. Treat a measured empty Canvas domain as an explicit
                # orchestration terminal state rather than a collection error;
                # no action is invented and no model request is issued.
                return OrchestrationResult(
                    final_answer=None,
                    turns=tuple(turns),
                    final_graph=env.graph.to_dict(),
                    termination_reason="no_admissible_action",
                    explicit_finish=False,
                )
            schema_request = self.action_schema_request(env)
            response = await self.client.propose(
                prompt,
                seed=self.generation_seed(index),
                **schema_request,
            )
            canvas = await env.step(response.text)
            turns.append(DirectorTurn(index, prompt, response, canvas))
            if canvas.done and canvas.final_answer is not None:
                return OrchestrationResult(
                    final_answer=canvas.final_answer,
                    turns=tuple(turns),
                    final_graph=env.graph.to_dict(),
                    termination_reason="finish",
                    explicit_finish=True,
                )
            prompt = self.continue_prompt(
                prompt,
                self.consumed_assistant_content(response, canvas),
                env,
                skills,
            )
        # DIRECT_REUSE + NECESSARY_ADAPTATION: upstream FlowSteer retains the
        # last executed solver result when its edit budget is exhausted.  The
        # shared QA environment tightens that boundary: only an atomic graph
        # revision that already passed the complete evidence/semantic/format
        # FINISH gate is eligible.  It remains a max-rounds policy failure and
        # is never represented as an explicit FINISH.
        lineage = env.last_valid_evidence_lineage
        if lineage is not None:
            return OrchestrationResult(
                final_answer=lineage.answer,
                turns=tuple(turns),
                final_graph=lineage.graph_snapshot.to_dict(),
                termination_reason="max_rounds",
                explicit_finish=False,
                valid_lineage_fallback_used=True,
                valid_lineage_fallback_receipt={
                    "graph_revision": lineage.graph_revision,
                    "graph_snapshot_id": lineage.graph_snapshot.snapshot_id,
                    "admission": "complete_finish_gate",
                },
            )
        return OrchestrationResult(
            final_answer=None,
            turns=tuple(turns),
            final_graph=env.graph.to_dict(),
            termination_reason="max_rounds",
            explicit_finish=False,
        )


__all__ = [
    "AgentGraphOrchestrator",
    "DIRECTOR_ACTION_JSON_SCHEMA",
    "DIRECTOR_ACTION_JSON_SCHEMA_TEXT",
    "DIRECTOR_ACTION_SCHEMA_VERSION",
    "DIRECTOR_ACTION_TARGET_DOMAIN_SCHEMA_VERSION",
    "DIRECTOR_MODEL_ADMISSIBLE_ACTION_MASK_PROFILE",
    "DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION",
    "DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V1",
    "DIRECTOR_MODEL_ADMISSIBLE_ACTION_SCHEMA_VERSION_V3",
    "DIRECTOR_MODIFY_AGENT_FIELDS",
    "DIRECTOR_PROGRESSIVE_ACTION_MASK_PROFILE",
    "DIRECTOR_SGLANG_SAMPLING_SCHEMA_VERSION",
    "DIRECTOR_STATE_CONDITIONED_ACTION_SCHEMA_VERSION",
    "DIRECTOR_SYSTEM_PROMPT",
    "DIRECTOR_PROMPT_VERSION",
    "HOTPOTQA_DIRECTOR_PROMPT_VERSION",
    "HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V14",
    "HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V15",
    "HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V16",
    "HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V17",
    "HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V18",
    "HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V19",
    "HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V20",
    "HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V21",
    "HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V22",
    "HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V11",
    "HOTPOTQA_DIRECTOR_SYSTEM_PROMPT_V13",
    "HOTPOTQA_SEMANTIC_LINEAGE_PROTOCOL",
    "HOTPOTQA_SEMANTIC_PROTOCOL",
    "QA_DIRECTOR_PROMPT_VERSION",
    "QA_DIRECTOR_SYSTEM_PROMPT_V1",
    "QA_VERIFIED_ANSWER_LINEAGE_PROTOCOL",
    "PRESERVE_DIAGNOSE_REPAIR_AUGMENT_POLICY",
    "LEGACY_DIRECTOR_SYSTEM_PROMPT_V8",
    "LEGACY_DIRECTOR_SYSTEM_PROMPT_V9",
    "LEGACY_DIRECTOR_PROMPT_VERSION_V8",
    "LEGACY_DIRECTOR_PROMPT_VERSION_V9",
    "DIRECTOR_TRANSCRIPT_SCHEMA",
    "DirectorClient",
    "DirectorError",
    "DirectorResponse",
    "DirectorTurn",
    "OpenAIDirectorClient",
    "OrchestrationResult",
    "decode_director_transcript",
    "director_action_json_schema_text",
    "director_actions_from_admissible_schema_branch",
    "director_model_admissible_sampling_json_schema_text",
    "director_model_admissible_sampling_json_schema_text_v1",
    "director_model_admissible_sampling_json_schema_text_v3",
    "director_model_admissible_schema_branch",
    "director_model_admissible_schema_branch_v1",
    "director_model_admissible_schema_branch_v3",
    "director_live_add_subgraph_agent_declarations_from_text",
    "director_live_add_subgraph_agent_declarations_json_schema_text",
    "director_live_add_subgraph_role_selection_from_text",
    "director_live_add_subgraph_role_selection_json_schema_text",
    "director_live_add_subgraph_relation_candidates",
    "director_live_action_parameter_json_schema_text",
    "director_live_action_target_domains_json",
    "director_live_modify_agent_selector_json_schema_text",
    "director_live_relation_candidate_selector_json_schema_text",
    "director_validate_live_action_target_domains",
    "director_modify_agent_field_sampling_json_schema_text",
    "director_modify_agent_field_selector_json_schema_text",
    "director_system_prompt_for_version",
    "director_sglang_sampling_json_schema_text",
    "director_state_conditioned_sampling_json_schema_text",
    "verified_qa_semantic_protocol",
    "flexible_hotpotqa_semantic_protocol",
    "role_conditional_hotpotqa_protocol",
    "encode_director_transcript",
]
