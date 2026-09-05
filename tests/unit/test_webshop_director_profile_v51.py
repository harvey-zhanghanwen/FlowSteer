import unittest

from src.interactive.director import (
    STEPWISE_SCALAR_DIRECTOR_PROMPT_VERSION_V2,
    STEPWISE_SCALAR_DIRECTOR_PROMPT_VERSION_V3,
    STEPWISE_SCALAR_DIRECTOR_SYSTEM_PROMPT_V2,
    STEPWISE_SCALAR_DIRECTOR_SYSTEM_PROMPT_V3,
    director_system_prompt_for_version,
    scalar_director_prompt_version,
)


class WebShopDirectorProfileV51Tests(unittest.TestCase):
    def test_v3_uses_scalar_domains_without_changing_the_v16_prompt(self):
        self.assertEqual(STEPWISE_SCALAR_DIRECTOR_SYSTEM_PROMPT_V2,
                         director_system_prompt_for_version(STEPWISE_SCALAR_DIRECTOR_PROMPT_VERSION_V2))
        self.assertEqual(STEPWISE_SCALAR_DIRECTOR_SYSTEM_PROMPT_V3,
                         director_system_prompt_for_version(STEPWISE_SCALAR_DIRECTOR_PROMPT_VERSION_V3))
        self.assertTrue(scalar_director_prompt_version(STEPWISE_SCALAR_DIRECTOR_PROMPT_VERSION_V3))

    def test_v3_describes_capabilities_not_a_fixed_workflow(self):
        prompt = STEPWISE_SCALAR_DIRECTOR_SYSTEM_PROMPT_V3
        self.assertIn("Agents without tools can analyze the shared public environment state", prompt)
        self.assertIn("Only the environment Tool owner changes its state", prompt)
        self.assertIn("do not assume a fixed number or sequence of Agents", prompt)
        self.assertIn("ReAct is an execution mode, not an Agent role", prompt)
