from __future__ import annotations

import json
import unittest

from src.interactive.agent_graph import AgentGraph, AgentNode
from src.interactive.agent_runtime import AgentRuntime
from src.interactive.agent_workflow_env import (
    AgentWorkflowEnv,
    AgentWorkflowStateError,
)
from src.interactive.healthbench_professional_adapter import (
    render_model_visible_conversation,
)
from src.interactive.model_registry import (
    ModelRegistry,
    ModelSpec,
    ProviderSpec,
)


def _registry() -> ModelRegistry:
    return ModelRegistry(
        [ProviderSpec("fake", kind="test")],
        [ModelSpec("model", "fake")],
    )


def _add_contract(contract: str, *, agent_id: str = "node_1") -> str:
    return json.dumps(
        {
            "action": "add_subgraph",
            "agents": [
                {
                    "agent_id": agent_id,
                    "model_id": "model",
                    "contract": contract,
                }
            ],
            "relations": [],
        }
    )


class ScopeNeutralContractAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_healthbench_rejects_explicit_safety_scope_deletion(self) -> None:
        registry = _registry()
        runtime = AgentRuntime(
            registry,
            object(),
            dataset_id="healthbench_professional",
        )
        env = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem="Explain appropriate management and safety considerations.",
            require_scope_neutral_contracts=True,
        )

        rejected = await env.step(
            _add_contract(
                "Provide definitive management while excluding disclaimers or "
                "warnings."
            )
        )

        self.assertFalse(rejected.accepted)
        self.assertIn("safety-bearing part", rejected.feedback)
        self.assertEqual((), env.graph.nodes)

    async def test_healthbench_rejects_direct_without_warning_contract(self) -> None:
        registry = _registry()
        runtime = AgentRuntime(
            registry,
            object(),
            dataset_id="healthbench_professional",
        )
        env = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem="Explain appropriate management and safety considerations.",
            require_scope_neutral_contracts=True,
            require_explicit_safety_scope_preservation=True,
        )

        rejected = await env.step(
            _add_contract(
                "Return concise treatment bullets without disclaimers or warnings."
            )
        )

        self.assertFalse(rejected.accepted)
        self.assertIn("safety-bearing part", rejected.feedback)
        self.assertEqual((), env.graph.nodes)

    async def test_healthbench_allows_benign_output_deduplication(self) -> None:
        registry = _registry()
        runtime = AgentRuntime(
            registry,
            object(),
            dataset_id="healthbench_professional",
        )
        env = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem="Explain appropriate management and safety considerations.",
            require_scope_neutral_contracts=True,
        )

        accepted = await env.step(
            _add_contract(
                "Draft a complete response without repeating the same paragraph."
            )
        )

        self.assertTrue(accepted.accepted)

    async def test_healthbench_allows_without_when_it_preserves_safety(self) -> None:
        registry = _registry()
        runtime = AgentRuntime(
            registry,
            object(),
            dataset_id="healthbench_professional",
        )
        env = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem="Explain appropriate management and safety considerations.",
            require_scope_neutral_contracts=True,
        )

        accepted = await env.step(
            _add_contract(
                "Explain the requested management without increasing medication "
                "risks and preserve the safety warnings."
            )
        )

        self.assertTrue(accepted.accepted)

    async def test_canvas_agent_id_is_not_a_clinical_numeric_literal(self) -> None:
        env = AgentWorkflowEnv(
            _registry(),
            gateway=object(),
            problem="Determine the appropriate disposition from the supplied case.",
            require_scope_neutral_contracts=True,
        )

        accepted = await env.step(
            _add_contract(
                "Use the completed evidence artifact from node_1 and produce "
                "the complete user-facing response.",
                agent_id="node_2",
            )
        )

        self.assertTrue(accepted.accepted)

    async def test_healthbench_plan_numbers_are_grounded_from_public_messages(
        self,
    ) -> None:
        problem = render_model_visible_conversation(
            (
                {
                    "role": "user",
                    "content": (
                        "Translate this plan.\n1. Check blood tests.\n"
                        "2. Arrange combined care.\n3. Arrange dental review."
                    ),
                },
            )
        )
        registry = _registry()
        runtime = AgentRuntime(
            registry,
            object(),
            dataset_id="healthbench_professional",
        )
        env = AgentWorkflowEnv(
            registry,
            runtime=runtime,
            problem=problem,
            require_scope_neutral_contracts=True,
        )

        accepted = await env.step(
            _add_contract(
                "Return the translated plan while preserving points 1, 2, "
                "and 3 exactly as supplied."
            )
        )

        self.assertTrue(accepted.accepted, accepted.feedback)

    async def test_rejects_unsupported_clinical_level_on_add_and_modify(self) -> None:
        problem = (
            "A patient with prior spinal fusion asks where to enter for an "
            "SCS trial."
        )
        add_env = AgentWorkflowEnv(
            _registry(),
            gateway=object(),
            problem=problem,
            require_scope_neutral_contracts=True,
        )
        rejected_add = await add_env.step(
            _add_contract(
                "Recommend L2-L4 as the correct entry range and draft the response."
            )
        )
        self.assertFalse(rejected_add.accepted)
        self.assertIn("scope-neutral rewrite", rejected_add.feedback)
        self.assertIn("L2-L4", rejected_add.feedback)
        self.assertEqual((), add_env.graph.nodes)

        graph = AgentGraph(
            [AgentNode("node_1", "model", "Derive a supported response.")]
        )
        modify_env = AgentWorkflowEnv(
            _registry(),
            gateway=object(),
            graph=graph,
            problem=problem,
            require_scope_neutral_contracts=True,
        )
        rejected_modify = await modify_env.step(
            json.dumps(
                {
                    "action": "modify_agent",
                    "agent_id": "node_1",
                    "contract": (
                        "Conclude L2-L4 is the preferred entry range and draft "
                        "the response."
                    ),
                }
            )
        )
        self.assertFalse(rejected_modify.accepted)
        self.assertIn("scope-neutral rewrite", rejected_modify.feedback)
        self.assertEqual(
            "Derive a supported response.",
            modify_env.graph.get_node("node_1").contract,
        )

    async def test_allows_literal_present_in_initial_task(self) -> None:
        env = AgentWorkflowEnv(
            _registry(),
            gateway=object(),
            problem=(
                "The supplied operative note explicitly identifies L2-L4 as "
                "the proposed entry range. Assess that proposal."
            ),
            require_scope_neutral_contracts=True,
        )
        accepted = await env.step(
            _add_contract(
                "Verify L2-L4 against the supplied note and return a "
                "supported response."
            )
        )
        self.assertTrue(accepted.accepted)

    async def test_allows_operational_format_count(self) -> None:
        env = AgentWorkflowEnv(
            _registry(),
            gateway=object(),
            problem="Explain the supported care plan clearly.",
            require_scope_neutral_contracts=True,
        )
        accepted = await env.step(
            _add_contract(
                "Return 3 bullet points that summarize only supported care-plan facts."
            )
        )
        self.assertTrue(accepted.accepted)

    async def test_allows_literal_after_public_agent_artifact_observation(self) -> None:
        graph = AgentGraph(
            [AgentNode("source", "model", "Retrieve authoritative evidence.")]
        )
        env = AgentWorkflowEnv(
            _registry(),
            gateway=object(),
            graph=graph,
            problem=(
                "A patient with prior spinal fusion asks where to enter for an "
                "SCS trial."
            ),
            require_scope_neutral_contracts=True,
        )
        env._progressive_outputs["source"] = (
            "The observed evidence explicitly supports L2-L4 as the entry range."
        )
        accepted = await env.step(
            _add_contract(
                "Verify L2-L4 against the observed evidence and draft the response.",
                agent_id="node_2",
            )
        )
        self.assertTrue(accepted.accepted)

    async def test_allows_literal_after_successful_public_tool_observation(
        self,
    ) -> None:
        graph = AgentGraph(
            [AgentNode("source", "model", "Retrieve authoritative evidence.")]
        )
        env = AgentWorkflowEnv(
            _registry(),
            gateway=object(),
            graph=graph,
            problem=(
                "A patient with prior spinal fusion asks where to enter for an "
                "SCS trial."
            ),
            require_scope_neutral_contracts=True,
        )
        env._progressive_output_metadata["source"] = {
            "tool_receipts": [
                {
                    "tool_id": "healthbench-authoritative.search",
                    "error_type": None,
                    "result": {
                        "completed": True,
                        "value": {
                            "evidence": [
                                {
                                    "excerpt": (
                                        "The retrieved passage explicitly supports "
                                        "L2-L4 as the entry range."
                                    )
                                }
                            ]
                        },
                    },
                }
            ]
        }
        accepted = await env.step(
            _add_contract(
                "Verify L2-L4 against the Tool observation and draft the response.",
                agent_id="node_2",
            )
        )
        self.assertTrue(accepted.accepted)

    async def test_rejects_unsupported_named_diagnosis_precommitment(self) -> None:
        env = AgentWorkflowEnv(
            _registry(),
            gateway=object(),
            problem="Review the symptoms and determine a supported diagnosis.",
            require_scope_neutral_contracts=True,
        )
        rejected = await env.step(
            _add_contract(
                "Recommend Acute Myeloid Leukemia as the final diagnosis."
            )
        )
        self.assertFalse(rejected.accepted)
        self.assertIn("Acute Myeloid Leukemia", rejected.feedback)

    async def test_rejects_unsupported_entity_slot_substitution(self) -> None:
        env = AgentWorkflowEnv(
            _registry(),
            gateway=object(),
            problem=(
                "A patient with prior PSIF asks about feasibility of an SCS "
                "trial. Preserve unresolved abbreviations."
            ),
            require_scope_neutral_contracts=True,
        )

        rejected = await env.step(
            _add_contract(
                "Search current literature for patients with pseudoathletes "
                "and produce a concise evidence artifact."
            )
        )

        self.assertFalse(rejected.accepted)
        self.assertIn("pseudoathletes", rejected.feedback)
        self.assertEqual((), env.graph.nodes)

    async def test_allows_entity_slot_only_after_public_alias_evidence(self) -> None:
        graph = AgentGraph(
            [AgentNode("source", "model", "Retrieve authoritative evidence.")]
        )
        env = AgentWorkflowEnv(
            _registry(),
            gateway=object(),
            graph=graph,
            problem=(
                "A patient with prior PSIF asks about feasibility of an SCS "
                "trial. Preserve unresolved abbreviations."
            ),
            require_scope_neutral_contracts=True,
        )
        env._progressive_outputs["source"] = (
            "A public source explicitly uses pseudoathletes for the queried "
            "population term."
        )

        accepted = await env.step(
            _add_contract(
                "Search current literature for patients with pseudoathletes "
                "and return the resulting evidence artifact.",
                agent_id="node_2",
            )
        )

        self.assertTrue(accepted.accepted)

    async def test_allows_scope_neutral_contract_that_preserves_anchor(self) -> None:
        env = AgentWorkflowEnv(
            _registry(),
            gateway=object(),
            problem="Clarify unresolved PSIF terminology before responding.",
            require_scope_neutral_contracts=True,
        )

        accepted = await env.step(
            _add_contract(
                "Search for evidence about PSIF terminology and return only "
                "the public evidence artifact."
            )
        )

        self.assertTrue(accepted.accepted)

    async def test_allows_generic_population_search_without_alias(self) -> None:
        env = AgentWorkflowEnv(
            _registry(),
            gateway=object(),
            problem="Clarify unresolved PSIF terminology before responding.",
            require_scope_neutral_contracts=True,
        )

        accepted = await env.step(
            _add_contract(
                "Search public evidence for patients with similar "
                "presentations and retain PSIF as unresolved terminology."
            )
        )

        self.assertTrue(accepted.accepted)

    async def test_rejects_unsupported_hypothesis_and_proposed_level(self) -> None:
        env = AgentWorkflowEnv(
            _registry(),
            gateway=object(),
            problem=(
                "Review an unresolved procedural question and determine the "
                "supported anatomical level."
            ),
            require_scope_neutral_contracts=True,
        )
        rejected = await env.step(
            _add_contract(
                "Identify the likely condition, hypothesize C6 involvement, "
                "and propose C4-C5 as the initial level."
            )
        )
        self.assertFalse(rejected.accepted)
        self.assertIn("scope-neutral rewrite", rejected.feedback)
        self.assertIn("C4-C5", rejected.feedback)

    async def test_allows_operational_hypothesis_count(self) -> None:
        env = AgentWorkflowEnv(
            _registry(),
            gateway=object(),
            problem="Compare plausible explanations without choosing one in advance.",
            require_scope_neutral_contracts=True,
        )
        accepted = await env.step(
            _add_contract(
                "Propose 2 candidate diagnoses for later evidence checking "
                "without selecting a conclusion."
            )
        )
        self.assertTrue(accepted.accepted)

    async def test_allows_plain_responsibility_without_answer_literal(self) -> None:
        env = AgentWorkflowEnv(
            _registry(),
            gateway=object(),
            problem="Review a clinical conversation and provide a useful response.",
            require_scope_neutral_contracts=True,
        )
        accepted = await env.step(
            _add_contract(
                "Compare the conversation with available evidence and draft a "
                "complete response that preserves uncertainty."
            )
        )
        self.assertTrue(accepted.accepted)

    async def test_default_is_legacy_open_and_fork_preserves_explicit_gate(
        self,
    ) -> None:
        legacy = AgentWorkflowEnv(
            _registry(),
            gateway=object(),
            problem="A patient asks about an SCS trial entry level.",
        )
        accepted = await legacy.step(
            _add_contract("Recommend L2-L4 as the correct entry range.")
        )
        self.assertTrue(accepted.accepted)

        configured = AgentWorkflowEnv(
            _registry(),
            gateway=object(),
            problem="A patient asks about an SCS trial entry level.",
            require_scope_neutral_contracts=True,
            require_explicit_safety_scope_preservation=True,
        )
        self.assertTrue(configured.fork().require_scope_neutral_contracts)
        self.assertTrue(
            configured.fork().require_explicit_safety_scope_preservation
        )

    def test_gate_option_requires_bool(self) -> None:
        with self.assertRaisesRegex(
            AgentWorkflowStateError,
            "require_scope_neutral_contracts must be bool",
        ):
            AgentWorkflowEnv(
                _registry(),
                gateway=object(),
                require_scope_neutral_contracts=1,  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
