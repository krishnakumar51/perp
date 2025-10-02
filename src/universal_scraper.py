
import asyncio
import uuid
import json
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
import traceback
from typing import List, TypedDict, Dict, Any, Optional, Annotated
from dataclasses import dataclass
from enum import Enum

from playwright.sync_api import sync_playwright, Page, Browser
from PIL import Image
import base64
import io

# LangGraph imports
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from src.enhanced_llm import MultiAgentCoordinator, LLMProvider, ActionExecutor
from src.config import SCREENSHOTS_DIR

class TaskState(str, Enum):
    INITIALIZING = "initializing"
    PLANNING = "planning"
    ANALYZING = "analyzing"
    EXECUTING = "executing"
    VALIDATING = "validating"
    RESEARCHING = "researching"
    COMPLETED = "completed"
    FAILED = "failed"
    RECOVERING = "recovering"

class ScraperState(TypedDict):
    """State for the universal scraper workflow"""
    # Core state
    task_id: str
    objective: str
    target_url: str
    current_state: TaskState

    # Execution context
    browser_session: Optional[Dict[str, Any]]
    current_page_url: str
    screenshots: List[str]  # Base64 encoded screenshots

    # Agent outputs
    workflow_plan: Optional[Dict[str, Any]]
    visual_analysis: List[Dict[str, Any]]
    executed_actions: List[Dict[str, Any]]
    validation_results: List[Dict[str, Any]]
    research_findings: List[Dict[str, Any]]

    # Results
    extracted_data: Dict[str, Any]
    success: bool
    error_log: List[str]

    # Metadata
    start_time: float
    step_count: int
    retry_count: int

@dataclass
class ScraperConfig:
    """Configuration for the universal scraper"""
    max_steps: int = 50
    max_retries: int = 3
    timeout_seconds: int = 30
    screenshot_quality: str = "medium"
    headless: bool = True
    provider: LLMProvider = LLMProvider.ANTHROPIC
    enable_learning: bool = True

class UniversalWebScraper:
    """
    Universal web scraper with multi-agent coordination and visual understanding
    """

    def __init__(self, config: ScraperConfig = None):
        self.config = config or ScraperConfig()
        self.coordinator = MultiAgentCoordinator(self.config.provider)
        self.memory = MemorySaver()
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow for web scraping
        """
        # Define the state graph
        workflow = StateGraph(ScraperState)

        # Add nodes (agents)
        workflow.add_node("initialize", self._initialize_node)
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("execute", self._execute_node)
        workflow.add_node("validate", self._validate_node)
        workflow.add_node("research", self._research_node)
        workflow.add_node("recover", self._recover_node)
        workflow.add_node("finalize", self._finalize_node)

        # Define entry point
        workflow.set_entry_point("initialize")

        # Add conditional edges based on state
        workflow.add_conditional_edges(
            "initialize",
            self._route_after_init,
            {
                "plan": "plan",
                "failed": END
            }
        )

        workflow.add_conditional_edges(
            "plan",
            self._route_after_plan,
            {
                "analyze": "analyze",
                "research": "research",
                "failed": END
            }
        )

        workflow.add_conditional_edges(
            "analyze",
            self._route_after_analyze,
            {
                "execute": "execute",
                "research": "research",
                "recover": "recover",
                "failed": END
            }
        )

        workflow.add_conditional_edges(
            "execute",
            self._route_after_execute,
            {
                "validate": "validate",
                "analyze": "analyze",  # Re-analyze after action
                "recover": "recover",
                "failed": END
            }
        )

        workflow.add_conditional_edges(
            "validate",
            self._route_after_validate,
            {
                "completed": "finalize",
                "analyze": "analyze",  # Continue with next step
                "recover": "recover",
                "failed": END
            }
        )

        workflow.add_conditional_edges(
            "research",
            self._route_after_research,
            {
                "analyze": "analyze",
                "plan": "plan",
                "recover": "recover"
            }
        )

        workflow.add_conditional_edges(
            "recover",
            self._route_after_recover,
            {
                "plan": "plan",
                "analyze": "analyze",
                "failed": END
            }
        )

        workflow.add_edge("finalize", END)

        return workflow.compile(checkpointer=self.memory)

    def scrape(self, url: str, objective: str) -> Dict[str, Any]:
        """
        Main entry point for universal web scraping
        """
        task_id = str(uuid.uuid4())

        initial_state = ScraperState(
            task_id=task_id,
            objective=objective,
            target_url=url,
            current_state=TaskState.INITIALIZING,
            browser_session=None,
            current_page_url="",
            screenshots=[],
            workflow_plan=None,
            visual_analysis=[],
            executed_actions=[],
            validation_results=[],
            research_findings=[],
            extracted_data={},
            success=False,
            error_log=[],
            start_time=time.time(),
            step_count=0,
            retry_count=0
        )

        try:
            # Execute the workflow
            result = self.workflow.invoke(
                initial_state,
                config={"configurable": {"thread_id": task_id}}
            )

            return {
                "task_id": task_id,
                "success": result.get("success", False),
                "extracted_data": result.get("extracted_data", {}),
                "screenshots": result.get("screenshots", []),
                "execution_time": time.time() - result.get("start_time", time.time()),
                "steps_completed": result.get("step_count", 0),
                "error_log": result.get("error_log", [])
            }

        except Exception as e:
            return {
                "task_id": task_id,
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    # Node implementations
    def _initialize_node(self, state: ScraperState) -> ScraperState:
        """Initialize browser session and basic setup"""
        try:
            # Start browser session
            playwright = sync_playwright().start()
            browser = playwright.chromium.launch(headless=self.config.headless)
            page = browser.new_page()

            # Store browser session info
            state["browser_session"] = {
                "playwright": playwright,
                "browser": browser,
                "page": page
            }

            state["current_state"] = TaskState.PLANNING
            state["step_count"] += 1

            return state

        except Exception as e:
            state["error_log"].append(f"Initialization failed: {str(e)}")
            state["current_state"] = TaskState.FAILED
            return state

    def _plan_node(self, state: ScraperState) -> ScraperState:
        """Plan the workflow strategy"""
        try:
            result = self.coordinator.plan_workflow(
                state["objective"],
                state["target_url"],
                {"step_count": state["step_count"]}
            )

            if result["success"] and result["parsed"]:
                state["workflow_plan"] = result["parsed"]
                state["current_state"] = TaskState.ANALYZING
            else:
                state["error_log"].append(f"Planning failed: {result.get('error', 'Unknown error')}")
                state["current_state"] = TaskState.RESEARCHING

            state["step_count"] += 1
            return state

        except Exception as e:
            state["error_log"].append(f"Planning error: {str(e)}")
            state["current_state"] = TaskState.FAILED
            return state

    def _analyze_node(self, state: ScraperState) -> ScraperState:
        """Analyze current page visually"""
        try:
            page = state["browser_session"]["page"]

            # Navigate to target URL if not already there
            if page.url != state["target_url"] and not state["current_page_url"]:
                page.goto(state["target_url"])
                state["current_page_url"] = state["target_url"]
                page.wait_for_timeout(2000)  # Allow page to load

            # Take screenshot
            screenshot = page.screenshot()
            screenshot_b64 = base64.b64encode(screenshot).decode()
            state["screenshots"].append(screenshot_b64)

            # Analyze with vision agent
            current_task = state["workflow_plan"].get("workflow_steps", [{}])[0].get("action", state["objective"])

            result = self.coordinator.analyze_screenshot(
                screenshot_b64,
                current_task,
                page.url
            )

            if result["success"]:
                state["visual_analysis"].append(result)

                # Determine next action based on analysis
                if result.get("parsed", {}).get("page_state") == "ready":
                    state["current_state"] = TaskState.EXECUTING
                elif result.get("parsed", {}).get("page_state") == "loading":
                    # Wait and re-analyze
                    page.wait_for_timeout(3000)
                    state["current_state"] = TaskState.ANALYZING
                else:
                    state["current_state"] = TaskState.RESEARCHING
            else:
                state["error_log"].append(f"Visual analysis failed: {result.get('error', 'Unknown error')}")
                state["current_state"] = TaskState.RESEARCHING

            state["step_count"] += 1
            return state

        except Exception as e:
            state["error_log"].append(f"Analysis error: {str(e)}")
            state["current_state"] = TaskState.RECOVERING
            return state

    def _execute_node(self, state: ScraperState) -> ScraperState:
        """Execute browser actions"""
        try:
            page = state["browser_session"]["page"]
            executor = ActionExecutor(self.coordinator)

            # Get latest visual analysis
            if not state["visual_analysis"]:
                state["current_state"] = TaskState.ANALYZING
                return state

            latest_analysis = state["visual_analysis"][-1]

            # Get recommended action from visual analysis
            recommended_action = latest_analysis.get("parsed", {}).get("recommended_action", "")

            if not recommended_action:
                state["current_state"] = TaskState.RESEARCHING
                return state

            # Prepare action specification
            action_spec = {
                "action_type": "click",  # Default action
                "description": recommended_action,
                "objective": state["objective"],
                "expected_outcome": "progress toward objective"
            }

            # Execute action with visual feedback
            execution_result = executor.execute_with_visual_feedback(
                action_spec,
                state["browser_session"]["browser"],
                page
            )

            state["executed_actions"].append(execution_result)

            if execution_result["success"]:
                state["current_state"] = TaskState.VALIDATING
            else:
                state["retry_count"] += 1
                if state["retry_count"] < self.config.max_retries:
                    state["current_state"] = TaskState.RECOVERING
                else:
                    state["current_state"] = TaskState.FAILED

            state["step_count"] += 1
            return state

        except Exception as e:
            state["error_log"].append(f"Execution error: {str(e)}")
            state["current_state"] = TaskState.RECOVERING
            return state

    def _validate_node(self, state: ScraperState) -> ScraperState:
        """Validate progress and results"""
        try:
            # Get current results
            current_data = {}  # Extract from page or previous actions

            result = self.coordinator.validate_progress(
                state["objective"],
                current_data,
                state["workflow_plan"]
            )

            state["validation_results"].append(result)

            if result["success"] and result.get("parsed", {}).get("overall_success", False):
                completion_pct = result.get("parsed", {}).get("completion_percentage", 0)

                if completion_pct >= 90:
                    state["current_state"] = TaskState.COMPLETED
                    state["success"] = True
                else:
                    state["current_state"] = TaskState.ANALYZING  # Continue
            else:
                state["current_state"] = TaskState.RECOVERING

            state["step_count"] += 1
            return state

        except Exception as e:
            state["error_log"].append(f"Validation error: {str(e)}")
            state["current_state"] = TaskState.RECOVERING
            return state

    def _research_node(self, state: ScraperState) -> ScraperState:
        """Research navigation guidance"""
        try:
            # Determine what guidance is needed
            challenge = "navigation guidance"
            if state["error_log"]:
                challenge = state["error_log"][-1]

            website_info = {
                "url": state["target_url"],
                "domain": urlparse(state["target_url"]).netloc
            }

            result = self.coordinator.research_guidance(
                challenge,
                website_info,
                state["objective"]
            )

            state["research_findings"].append(result)

            # After research, go back to planning or analysis
            state["current_state"] = TaskState.ANALYZING
            state["step_count"] += 1

            return state

        except Exception as e:
            state["error_log"].append(f"Research error: {str(e)}")
            state["current_state"] = TaskState.RECOVERING
            return state

    def _recover_node(self, state: ScraperState) -> ScraperState:
        """Handle error recovery"""
        try:
            state["retry_count"] += 1

            if state["retry_count"] >= self.config.max_retries:
                state["current_state"] = TaskState.FAILED
                return state

            # Reset to analysis state for retry
            state["current_state"] = TaskState.ANALYZING
            state["step_count"] += 1

            # Add recovery delay
            if state["browser_session"] and state["browser_session"]["page"]:
                state["browser_session"]["page"].wait_for_timeout(2000)

            return state

        except Exception as e:
            state["error_log"].append(f"Recovery error: {str(e)}")
            state["current_state"] = TaskState.FAILED
            return state

    def _finalize_node(self, state: ScraperState) -> ScraperState:
        """Finalize and cleanup"""
        try:
            # Extract final data from page
            if state["browser_session"] and state["browser_session"]["page"]:
                page = state["browser_session"]["page"]

                # Basic data extraction
                state["extracted_data"] = {
                    "final_url": page.url,
                    "page_title": page.title(),
                    "completion_time": time.time() - state["start_time"],
                    "total_screenshots": len(state["screenshots"])
                }

            # Cleanup browser session
            if state["browser_session"]:
                state["browser_session"]["browser"].close()
                state["browser_session"]["playwright"].stop()

            state["current_state"] = TaskState.COMPLETED
            return state

        except Exception as e:
            state["error_log"].append(f"Finalization error: {str(e)}")
            state["current_state"] = TaskState.FAILED
            return state

    # Routing functions
    def _route_after_init(self, state: ScraperState) -> str:
        if state["current_state"] == TaskState.FAILED:
            return "failed"
        return "plan"

    def _route_after_plan(self, state: ScraperState) -> str:
        if state["current_state"] == TaskState.FAILED:
            return "failed"
        elif state["current_state"] == TaskState.RESEARCHING:
            return "research"
        return "analyze"

    def _route_after_analyze(self, state: ScraperState) -> str:
        if state["current_state"] == TaskState.FAILED:
            return "failed"
        elif state["current_state"] == TaskState.EXECUTING:
            return "execute"
        elif state["current_state"] == TaskState.RESEARCHING:
            return "research"
        elif state["current_state"] == TaskState.RECOVERING:
            return "recover"
        return "failed"

    def _route_after_execute(self, state: ScraperState) -> str:
        if state["current_state"] == TaskState.FAILED:
            return "failed"
        elif state["current_state"] == TaskState.VALIDATING:
            return "validate"
        elif state["current_state"] == TaskState.ANALYZING:
            return "analyze"
        elif state["current_state"] == TaskState.RECOVERING:
            return "recover"
        return "failed"

    def _route_after_validate(self, state: ScraperState) -> str:
        if state["current_state"] == TaskState.COMPLETED:
            return "completed"
        elif state["current_state"] == TaskState.ANALYZING:
            return "analyze"
        elif state["current_state"] == TaskState.RECOVERING:
            return "recover"
        return "failed"

    def _route_after_research(self, state: ScraperState) -> str:
        if state["current_state"] == TaskState.ANALYZING:
            return "analyze"
        elif state["current_state"] == TaskState.PLANNING:
            return "plan"
        return "recover"

    def _route_after_recover(self, state: ScraperState) -> str:
        if state["current_state"] == TaskState.FAILED:
            return "failed"
        elif state["current_state"] == TaskState.ANALYZING:
            return "analyze"
        return "plan"

# Usage example
if __name__ == "__main__":
    scraper = UniversalWebScraper()

    result = scraper.scrape(
        "https://www.flipkart.com",
        "Find the top 5 latest samsung smartphones under 50000 with user ratings above 4 stars"
    )

    print(json.dumps(result, indent=2))
