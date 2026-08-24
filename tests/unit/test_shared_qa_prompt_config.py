from __future__ import annotations

import unittest

from src.interactive.config_loader import (
    ConfigurationError,
    load_yaml,
    validate_agent_graph_config,
)
from src.interactive.director import (
    DIRECTOR_PROMPT_VERSION,
    LEGACY_QA_DIRECTOR_PROMPT_VERSION_V1,
    LEGACY_QA_DIRECTOR_PROMPT_VERSION_V3,
    LEGACY_QA_DIRECTOR_PROMPT_VERSION_V4,
    LEGACY_QA_DIRECTOR_PROMPT_VERSION_V5,
    QA_DIRECTOR_PROMPT_VERSION,
)


class SharedQADirectorPromptConfigTests(unittest.TestCase):
    def _config(self):
        return load_yaml(
            "config/evaluation_triviaqa_unified_architecture_v2_fixed128.yaml"
        )

    def test_current_neutral_qa_prompt_is_admitted(self) -> None:
        config = self._config()
        self.assertEqual(
            QA_DIRECTOR_PROMPT_VERSION,
            config["experiment"]["prompt_version"],
        )
        config["experiment"]["prompt_version"] = QA_DIRECTOR_PROMPT_VERSION

        validate_agent_graph_config(config)

    def test_legacy_v3_qa_prompt_remains_replayable(self) -> None:
        config = self._config()
        config["experiment"]["prompt_version"] = (
            LEGACY_QA_DIRECTOR_PROMPT_VERSION_V3
        )

        validate_agent_graph_config(config)

    def test_legacy_v4_qa_prompt_remains_replayable(self) -> None:
        config = self._config()
        config["experiment"]["prompt_version"] = (
            LEGACY_QA_DIRECTOR_PROMPT_VERSION_V4
        )

        validate_agent_graph_config(config)

    def test_legacy_v5_qa_prompt_remains_replayable(self) -> None:
        config = self._config()
        config["experiment"]["prompt_version"] = (
            LEGACY_QA_DIRECTOR_PROMPT_VERSION_V5
        )

        validate_agent_graph_config(config)

    def test_legacy_v1_qa_prompt_remains_replayable(self) -> None:
        config = self._config()
        config["experiment"]["prompt_version"] = (
            LEGACY_QA_DIRECTOR_PROMPT_VERSION_V1
        )

        validate_agent_graph_config(config)

    def test_generic_prompt_is_rejected_for_shared_qa_protocol(self) -> None:
        config = self._config()
        config["experiment"]["prompt_version"] = DIRECTOR_PROMPT_VERSION

        with self.assertRaises(ConfigurationError):
            validate_agent_graph_config(config)


if __name__ == "__main__":
    unittest.main()
