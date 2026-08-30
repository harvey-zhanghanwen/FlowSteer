# TriviaQA fact-memory result analysis

- Status: `partial_or_protocol_invalid`
- Fixed denominator: `128`
- AgentGraph EM: `N/A`
- AgentGraph F1: `N/A`
- Terminal failures: `18`
- Protocol valid: `False`
- Materialization manifest valid: `False`
- 100% semantic rewrite checks: `{'record_count': True, 'unique_source_count': True, 'strict_semantic_paraphrase_count': True, 'semantic_question_rewrite_count': False, 'lexical_or_phrase_replacement_count': True, 'semantic_admission_checked_count': True, 'fact_self_containment_checked_count': False, 'fact_self_containment_pass_count': False, 'fact_text_count': True}`
- Fact-only index checks: `{'fact_self_containment_admission_version': False, 'embedding_text_field_is_fact_text': True, 'fact_projection_fields_exact': True, 'original_question_indexed_false': True, 'canonical_answer_indexed_false': True, 'accepted_answers_indexed_false': True, 'paraphrase_question_indexed_false': True}`
- Materialized facts: `76523` / `76523`
- Final index memories: `76523`
- Selected Top-K: `5`
- Final index frozen Top-K: `5`
- Top-K receipt/index match: `True`
- Director Tool calls: `0`
- Director request allowed_tools: `[]`
- Director requests Tool-free: `True`
- Director data-plane isolated: `True`
- Worker search/read: `189` / `939`
- First data-plane Action is search: `True`
- Complete Top-K rank reads: `True`
- Fact artifact explicit-relation route: `True`
- Retrieval artifact relation route: `False`
- Output inbox receipt lineage: `False`
- Output lineage: `False`
- Web Search count: `0`
- Agent-facing data-plane violations: `0`
- Real wrong demos: `5`

> Formal EM/F1 is withheld because the fixed snapshot or protocol is incomplete.
