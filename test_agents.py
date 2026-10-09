"""test_agents.py - Unit tests for agents.py prompts, subagents configuration, and limits."""
import unittest
from unittest.mock import MagicMock

from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    TodoListMiddleware,
    ToolCallLimitMiddleware,
)

import agents
from tools import SOURCE_TOOLS, web_fetch


class TestAgentsLimits(unittest.TestCase):
    """Test middleware limits for lead and subagents."""

    def test_lead_limits_configuration(self):
        self.assertEqual(len(agents.LEAD_LIMITS), 2)
        model_limit = next(m for m in agents.LEAD_LIMITS if isinstance(m, ModelCallLimitMiddleware))
        tool_limit = next(m for m in agents.LEAD_LIMITS if isinstance(m, ToolCallLimitMiddleware))
        self.assertEqual(model_limit.run_limit, 150)
        self.assertEqual(model_limit.exit_behavior, "end")
        self.assertEqual(tool_limit.run_limit, 300)

    def test_subagent_limits_configuration(self):
        self.assertEqual(len(agents.SUB_LIMITS), 2)
        model_limit = next(m for m in agents.SUB_LIMITS if isinstance(m, ModelCallLimitMiddleware))
        tool_limit = next(m for m in agents.SUB_LIMITS if isinstance(m, ToolCallLimitMiddleware))
        self.assertEqual(model_limit.run_limit, 40)
        self.assertEqual(model_limit.exit_behavior, "end")
        self.assertEqual(tool_limit.run_limit, 60)


class TestPrompts(unittest.TestCase):
    """Test prompts completeness against RUBRIC and GUIDE requirements."""

    def test_lead_prompt_paths_and_instructions(self):
        prompt = agents.LEAD_PROMPT
        # Paths interpolated properly
        self.assertIn(agents.WORKDIR, prompt)
        self.assertIn(agents.NOTES_DIR, prompt)
        self.assertIn(agents.SOURCES_PATH, prompt)
        self.assertIn(agents.VALIDATOR_PATH, prompt)
        self.assertIn(agents.FINALIZER_PATH, prompt)
        self.assertIn(agents.REPORT_PATH, prompt)

        # Rubric 2.3: planning with write_todos, subagent delegation with task, review
        self.assertIn("write_todos", prompt)
        self.assertIn("task", prompt)
        self.assertIn("researcher", prompt)
        self.assertIn("citation-checker", prompt)
        self.assertIn("execute", prompt)

        # Rubric 2.2: multi-source requirement
        self.assertIn("arxiv", prompt)
        self.assertIn("hf-daily", prompt)
        self.assertIn("hf-search", prompt)
        self.assertIn("web", prompt)
        self.assertIn("3 of the 4", prompt)

        # Section reference instruction
        self.assertIn("References", prompt)

    def test_researcher_prompt_requirements(self):
        prompt = agents.RESEARCHER_PROMPT
        # Rubric 1.2: tools listed
        self.assertIn("arxiv_search", prompt)
        self.assertIn("hf_daily_papers", prompt)
        self.assertIn("hf_search_papers", prompt)
        self.assertIn("web_search", prompt)
        self.assertIn("web_fetch", prompt)

        # Rubric 2.4: untrusted web data, no memory hallucination, note format
        self.assertIn("UNTRUSTED", prompt)
        self.assertIn("NO RESULTS", prompt)
        self.assertIn("ERROR", prompt)
        self.assertIn("key_findings", prompt)
        self.assertIn("source:", prompt)

    def test_checker_prompt_requirements(self):
        prompt = agents.CHECKER_PROMPT
        self.assertIn("web_fetch", prompt)
        self.assertIn("SUPPORTED", prompt)
        self.assertIn("UNSUPPORTED", prompt)
        self.assertIn("UNTRUSTED", prompt)


class TestSubagentsBuilding(unittest.TestCase):
    """Test subagents structure and configuration."""

    def test_build_subagents_structure(self):
        subagents = agents.build_subagents()
        self.assertEqual(len(subagents), 2)

        subagent_map = {s["name"]: s for s in subagents}
        self.assertIn("researcher", subagent_map)
        self.assertIn("citation-checker", subagent_map)

        # Verify researcher
        researcher = subagent_map["researcher"]
        self.assertTrue(len(researcher["description"]) > 20)
        self.assertEqual(researcher["system_prompt"], agents.RESEARCHER_PROMPT)
        self.assertEqual(len(researcher["tools"]), len(SOURCE_TOOLS))
        self.assertEqual(researcher["middleware"], agents.SUB_LIMITS)

        # Verify citation-checker
        checker = subagent_map["citation-checker"]
        self.assertTrue(len(checker["description"]) > 20)
        self.assertEqual(checker["system_prompt"], agents.CHECKER_PROMPT)
        self.assertEqual(checker["tools"], [web_fetch])
        self.assertEqual(checker["middleware"], agents.SUB_LIMITS)


class TestBuildLeadAgent(unittest.TestCase):
    """Test lead agent compilation with mock model and backend."""

    def test_build_lead_agent_creation(self):
        from deepagents.backends import StateBackend
        from langchain_core.language_models.chat_models import BaseChatModel

        backend = StateBackend()
        mock_model = MagicMock(spec=BaseChatModel)
        mock_model.profile = None

        agent = agents.build_lead_agent(backend, mock_model)
        self.assertIsNotNone(agent)


if __name__ == "__main__":
    unittest.main()
