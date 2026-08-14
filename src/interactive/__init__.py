from .workflow_graph import (
    WorkflowGraph,
    WorkflowNode,
    NodeType,
    VALID_OPERATORS,
    create_workflow_from_dsl,
)

from .action_parser import (
    ActionParser,
    ParsedAction,
    ActionType,
    StructureType,
    parse_action,
    extract_actions,
)

from .workflow_env import (
    InteractiveWorkflowEnv,
    StepResult,
    create_env,
)

from .workflow_builder import (
    Trajectory,
    TurnRecord,
    create_action_mask,
    merge_trajectory_masks,
    InteractiveWorkflowBuilder,
    BatchWorkflowBuilder,
    create_aflow_executor_wrapper,
    create_builder,
)

from .trajectory_reward import (
    TrajectoryRewardCalculator,
    TrajectoryRewardResult,
    EfficiencyConfig,
    create_reward_calculator,
)

try:
    from .grpo_trainer import (
        InteractiveGRPOTrainer,
        InteractiveGRPOConfig,
        create_interactive_trainer,
    )
except Exception as _e:
    InteractiveGRPOTrainer = None
    InteractiveGRPOConfig = None

    def create_interactive_trainer(*args, **kwargs):
        raise ImportError("Missing training dependencies for `interactive.grpo_trainer`.") from _e

try:
    from .batch_inference import (
        BatchInferenceConfig,
        BatchGenerationManager,
        BatchInteractiveLoopManager,
        OptimizedAsyncLLMClient,
        create_batch_generator,
        create_optimized_client,
    )
except Exception as _e:
    BatchInferenceConfig = None
    BatchGenerationManager = None
    BatchInteractiveLoopManager = None
    OptimizedAsyncLLMClient = None

    def create_batch_generator(*args, **kwargs):
        raise ImportError("Missing optional dependencies for `interactive.batch_inference`.") from _e

    def create_optimized_client(*args, **kwargs):
        raise ImportError("Missing optional dependencies for `interactive.batch_inference`.") from _e

from .prompt_templates import (
    PromptConfig,
    InteractivePromptBuilder,
    CompactPromptBuilder,
    SYSTEM_PROMPT,
    ACTION_EXAMPLES,
    PROBLEM_TYPE_HINTS,
    create_prompt_builder,
    get_problem_type_hint,
)

# AgentGraph v1 is additive: explicit names avoid collisions with the legacy
# Operator-DSL Trajectory/TurnRecord API above.
from .agent_graph import AgentGraph, AgentGraphValidator, AgentNode, RelationBits
from .agent_action_parser import AgentAction, AgentActionParser, AgentActionType
from .agent_runtime import AgentRuntime, AgentRuntimeResult
from .agent_workflow_env import AgentWorkflowEnv, AgentWorkflowStepResult
from .director import AgentGraphOrchestrator, OpenAIDirectorClient
from .openai_gateway import OpenAICompatibleGateway
from .model_registry import ModelRegistry, ModelSpec, ProviderSpec
from .records import TrajectoryRecord as AgentTrajectoryRecord, TurnRecord as AgentTurnRecord
from .grpo_objective import GRPOTrajectory, action_masked_one_pass_loss
from .exploration import BayesianLinearPosterior, DisjointLinUCB, MACEFeatureExtractor
from .skills import SkillEvidenceGate, SkillLifecycleManager, SkillRecord, SkillStore

__all__ = [
    'WorkflowGraph',
    'WorkflowNode',
    'NodeType',
    'VALID_OPERATORS',
    'create_workflow_from_dsl',
    'ActionParser',
    'ParsedAction',
    'ActionType',
    'StructureType',
    'parse_action',
    'extract_actions',
    'InteractiveWorkflowEnv',
    'StepResult',
    'create_env',
    'Trajectory',
    'TurnRecord',
    'create_action_mask',
    'merge_trajectory_masks',
    'InteractiveWorkflowBuilder',
    'BatchWorkflowBuilder',
    'create_aflow_executor_wrapper',
    'create_builder',
    'TrajectoryRewardCalculator',
    'TrajectoryRewardResult',
    'EfficiencyConfig',
    'create_reward_calculator',
    'InteractiveGRPOTrainer',
    'InteractiveGRPOConfig',
    'create_interactive_trainer',
    'BatchInferenceConfig',
    'BatchGenerationManager',
    'BatchInteractiveLoopManager',
    'OptimizedAsyncLLMClient',
    'create_batch_generator',
    'create_optimized_client',
    'PromptConfig',
    'InteractivePromptBuilder',
    'CompactPromptBuilder',
    'SYSTEM_PROMPT',
    'ACTION_EXAMPLES',
    'PROBLEM_TYPE_HINTS',
    'create_prompt_builder',
    'get_problem_type_hint',
    'AgentGraph',
    'AgentGraphValidator',
    'AgentNode',
    'RelationBits',
    'AgentAction',
    'AgentActionParser',
    'AgentActionType',
    'AgentRuntime',
    'AgentRuntimeResult',
    'AgentWorkflowEnv',
    'AgentWorkflowStepResult',
    'AgentGraphOrchestrator',
    'OpenAIDirectorClient',
    'OpenAICompatibleGateway',
    'ModelRegistry',
    'ModelSpec',
    'ProviderSpec',
    'AgentTrajectoryRecord',
    'AgentTurnRecord',
    'GRPOTrajectory',
    'action_masked_one_pass_loss',
    'BayesianLinearPosterior',
    'DisjointLinUCB',
    'MACEFeatureExtractor',
    'SkillEvidenceGate',
    'SkillLifecycleManager',
    'SkillRecord',
    'SkillStore',
]
