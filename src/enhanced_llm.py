import re
import json
import base64
import random
import string
import asyncio
from enum import Enum
from pathlib import Path
from typing import List, Union, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup as bs
from src.config import (
    anthropic_client, groq_client, openai_client,
    ANTHROPIC_MODEL, GROQ_MODEL, OPENAI_MODEL
)

class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    OPENAI = "openai"

class AgentType(str, Enum):
    PLANNER = "planner"
    VISION = "vision"
    EXECUTOR = "executor"
    VALIDATOR = "validator"
    RESEARCH = "research"

# Enhanced Multi-Agent Prompt Templates
AGENT_PROMPTS = {
    "planner": """
You are the Planning Agent in a universal web scraper system. Your role is to decompose complex user objectives into executable steps.

**User Objective:** "{query}"
**Target URL:** {url}
**Website Analysis:** {website_context}

Your Tasks:
1. **Analyze the Objective:** Break down what the user wants to accomplish
2. **Plan Strategy:** Create a step-by-step approach considering:
   - Website type and structure
   - Required data extraction
   - Navigation complexity
   - Potential obstacles

3. **Generate Workflow:** Output a structured plan with:
   - High-level steps
   - Success criteria for each step
   - Fallback strategies
   - Expected challenges

4. **Coordinate Agents:** Specify which agents should handle each step

Format your response as a single JSON object. Do not include any text before or after the JSON.
{{
    "intent": "data_extraction|form_submission|navigation",
    "complexity_level": "simple|moderate|complex",
    "workflow_steps": [
        {{
            "step": 1,
            "action": "description",
            "agent": "vision|executor|research",
            "success_criteria": "how to know step succeeded",
            "fallback": "alternative if step fails"
        }}
    ],
    "expected_challenges": ["challenge1", "challenge2"],
    "success_metrics": {{"data_extracted": true, "forms_filled": 0}}
}}
""",

    "vision": """
You are the Vision Agent in a universal web scraper system. You analyze screenshots to understand webpage layout and identify interactive elements.

**Current Task:** {current_task}
**Screenshot Analysis:** {screenshot_data}
**Page URL:** {url}

Your capabilities:
1. **Visual Element Detection:** Identify buttons, forms, links, text areas
2. **Layout Understanding:** Comprehend page structure and navigation flow  
3. **State Recognition:** Detect loading, errors, popups, dynamic content
4. **Element Mapping:** Map visual elements to interaction coordinates

Instructions:
- If elements are unclear or ambiguous, request additional screenshots
- Identify the best interaction points for the current task
- Detect any obstacles (popups, captchas, loading states)
- Provide confidence levels for element identification

Analyze the current screenshot and identify:
1. All interactive elements relevant to the task
2. Current page state (loading, ready, error)
3. Recommended next actions based on visual analysis
4. Any obstacles or challenges present

Format response as a single JSON object. Do not include any text before or after the JSON.
{{
    "page_state": "ready|loading|error|blocked",
    "interactive_elements": [
        {{
            "type": "button|form|link|input",
            "description": "what this element does",
            "coordinates": {{"x": 100, "y": 200}},
            "confidence": 0.95,
            "text_content": "visible text"
        }}
    ],
    "obstacles": ["popup", "captcha", "loading"],
    "recommended_action": "specific next step",
    "screenshot_quality": "good|needs_scroll|needs_zoom"
}}
""",

    "executor": """
You are the Execution Agent in a universal web scraper system. You perform browser interactions based on visual analysis and planning instructions.

**Current Step:** {current_step}
**Action Required:** {action_required}
**Visual Context:** {visual_context}
**Available Actions:** click, fill, scroll, navigate, screenshot, wait

Your responsibilities:
1. **Execute Actions:** Perform browser interactions precisely
2. **Verify Results:** Take screenshots after actions to confirm success
3. **Handle Timing:** Implement appropriate waits and delays
4. **Error Detection:** Identify when actions fail and report issues

Instructions:
- Always take a screenshot after significant actions
- Use progressive waits - start short, increase if needed
- If an action fails, try alternative approaches
- Report both successful actions and failures clearly

Current browser state: {browser_state}
Previous actions: {previous_actions}

Choose the most appropriate action and execute it. If you encounter issues:
1. Try alternative selectors or coordinates
2. Adjust timing and wait strategies  
3. Check for dynamic content or overlays
4. Request guidance from other agents if needed

Response format (as a single JSON object only):
{{
    "action_type": "click|fill|scroll|navigate|wait",
    "parameters": {{"target": "element", "value": "text_if_filling"}},
    "reasoning": "why this action was chosen",
    "expected_outcome": "what should happen next",
    "verification_needed": true
}}
""",

    "validator": """
You are the Validation Agent in a universal web scraper system. You verify task completion and ensure quality control.

**Task Objective:** {objective}
**Current Results:** {current_results}
**Expected Outcome:** {expected_outcome}
**Validation Criteria:** {validation_criteria}

Your responsibilities:
1. **Progress Verification:** Confirm each step was completed successfully
2. **Data Quality:** Validate extracted data meets requirements
3. **Error Detection:** Identify failures and incomplete actions
4. **Success Metrics:** Measure overall task completion

Validation checks:
- Are all required data fields extracted?
- Do the results match the user's objective?
- Are there any obvious errors or inconsistencies?
- Is additional processing needed?

Current validation status: {validation_status}

Provide validation assessment as a single JSON object:
{{
    "overall_success": true/false,
    "completion_percentage": 85,
    "validated_data": {{"extracted_fields": {}}},
    "quality_issues": ["issue1", "issue2"],
    "missing_requirements": ["requirement1"],
    "recommendations": ["next_step1", "next_step2"]
}}
""",

    "research": """
You are the Research Agent in a universal web scraper system. You provide contextual information and navigation guidance through web search.

**Current Challenge:** {challenge}
**Website Context:** {website_info}
**Objective:** {objective}

Your capabilities:
1. **Web Search:** Find guidance for navigation and interaction patterns
2. **Pattern Recognition:** Identify common solutions for similar websites  
3. **Context Building:** Provide background information about website types
4. **Strategy Suggestion:** Recommend approaches based on research

When other agents need guidance:
- Search for specific navigation instructions
- Find documentation about website features
- Research common interaction patterns
- Provide alternative strategies

Current research request: {research_request}

Perform web search and provide actionable insights as a single JSON object:
{{
    "search_queries_used": ["query1", "query2"],
    "key_findings": ["finding1", "finding2"],
    "navigation_guidance": "step by step instructions",
    "alternative_approaches": ["approach1", "approach2"],
    "confidence_level": 0.8
}}
"""
}

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Finds and parses the first valid JSON object from a string.
    Handles markdown code blocks (```json ... ```) and raw JSON.
    """
    # Pattern to find JSON within ```json ... ``` markdown block
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Fallback: find the first occurrence of a JSON object
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return None
        json_str = match.group(0)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Attempt to fix common issues like trailing commas
        try:
            # A more robust parser might be needed for complex errors,
            # but for now, we return None if basic parsing fails.
            return json.loads(re.sub(r',\s*([\}\]])', r'\1', json_str))
        except json.JSONDecodeError:
            return None

def get_agent_prompt(agent_type: AgentType, **kwargs) -> str:
    """Get formatted prompt for specific agent type"""
    if agent_type.value not in AGENT_PROMPTS:
        raise ValueError(f"Unknown agent type: {agent_type}")

    return AGENT_PROMPTS[agent_type.value].format(**kwargs)

def call_llm_with_agent_context(
    provider: LLMProvider,
    agent_type: AgentType,
    context: Dict[str, Any],
    model_params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Enhanced LLM call with agent-specific context and robust JSON parsing.
    """
    try:
        prompt = get_agent_prompt(agent_type, **context)

        if model_params is None:
            model_params = {"temperature": 0.1, "max_tokens": 2000}

        if provider == LLMProvider.ANTHROPIC:
            if not anthropic_client:
                raise ConnectionError("Anthropic client not initialized. Check API key.")
            response = anthropic_client.messages.create(
                model=ANTHROPIC_MODEL,
                messages=[{"role": "user", "content": prompt}],
                **model_params
            )
            content = response.content[0].text

        elif provider == LLMProvider.GROQ:
            if not groq_client:
                raise ConnectionError("Groq client not initialized. Check API key.")
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                **model_params
            )
            content = response.choices[0].message.content

        elif provider == LLMProvider.OPENAI:
            if not openai_client:
                raise ConnectionError("OpenAI client not initialized. Check API key.")
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                **model_params
            )
            content = response.choices[0].message.content

        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # Use robust JSON extraction
        parsed_response = _extract_json(content)

        if parsed_response:
            return {
                "success": True,
                "content": content,
                "parsed": parsed_response,
                "agent_type": agent_type.value
            }
        else:
            return {
                "success": False, # Changed to False to indicate parsing failure
                "content": content,
                "parsed": None,
                "agent_type": agent_type.value,
                "error": "Failed to parse JSON response from LLM"
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "agent_type": agent_type.value
        }

def web_search(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """
    Enhanced web search with better result processing
    """
    try:
        # Use DuckDuckGo HTML search (no API key required)
        search_url = f"https://html.duckduckgo.com/html/?q={query}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(search_url, headers=headers, timeout=10)
        soup = bs(response.content, 'html.parser')

        results = []
        for result in soup.find_all('div', class_='result')[:num_results]:
            title_elem = result.find('a', class_='result__a')
            snippet_elem = result.find('div', class_='result__snippet')

            if title_elem:
                results.append({
                    'title': title_elem.get_text(strip=True),
                    'url': title_elem.get('href', ''),
                    'snippet': snippet_elem.get_text(strip=True) if snippet_elem else ''
                })

        return results

    except Exception as e:
        print(f"Web search error: {e}")
        return []

class MultiAgentCoordinator:
    """
    Coordinates multiple agents for complex web scraping tasks
    """

    def __init__(self, provider: LLMProvider = LLMProvider.ANTHROPIC):
        self.provider = provider
        self.agents = {}
        self.conversation_history = []
        self.knowledge_base = {}

    def plan_workflow(self, query: str, url: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Use planner agent to create workflow strategy
        """
        planning_context = {
            'query': query,
            'url': url,
            'website_context': context or {}
        }

        result = call_llm_with_agent_context(
            self.provider,
            AgentType.PLANNER,
            planning_context
        )

        if result.get('success') and result.get('parsed'):
            self.knowledge_base['current_plan'] = result['parsed']

        return result

    def analyze_screenshot(self, screenshot_data: str, task: str, url: str) -> Dict[str, Any]:
        """
        Use vision agent to analyze screenshot and identify elements
        """
        vision_context = {
            'current_task': task,
            'screenshot_data': "Screenshot data provided.", # Don't send the full base64 string in the prompt
            'url': url
        }

        result = call_llm_with_agent_context(
            self.provider,
            AgentType.VISION,
            vision_context
        )

        return result

    def execute_action(self, step: str, action: str, visual_context: Dict, browser_state: Dict) -> Dict[str, Any]:
        """
        Use executor agent to perform browser actions
        """
        executor_context = {
            'current_step': step,
            'action_required': action,
            'visual_context': visual_context,
            'browser_state': browser_state,
            'previous_actions': self.conversation_history[-5:] if self.conversation_history else []
        }

        result = call_llm_with_agent_context(
            self.provider,
            AgentType.EXECUTOR,
            executor_context
        )

        return result

    def validate_progress(self, objective: str, current_results: Any, expected: Any) -> Dict[str, Any]:
        """
        Use validator agent to check task completion
        """
        validation_context = {
            'objective': objective,
            'current_results': current_results,
            'expected_outcome': expected,
            'validation_criteria': self.knowledge_base.get('success_metrics', {}),
            'validation_status': 'in_progress'
        }

        result = call_llm_with_agent_context(
            self.provider,
            AgentType.VALIDATOR,
            validation_context
        )

        return result

    def research_guidance(self, challenge: str, website_info: Dict, objective: str) -> Dict[str, Any]:
        """
        Use research agent to find navigation guidance
        """
        research_context = {
            'challenge': challenge,
            'website_info': website_info,
            'objective': objective,
            'research_request': challenge
        }

        result = call_llm_with_agent_context(
            self.provider,
            AgentType.RESEARCH,
            research_context
        )

        # Perform actual web search based on agent's recommendations
        if result.get('success') and result.get('parsed'):
            search_queries = result['parsed'].get('search_queries_used', [challenge])
            search_results = []

            for query in search_queries[:3]:  # Limit searches
                search_results.extend(web_search(query, 3))

            result['search_results'] = search_results

        return result

# Enhanced action execution with visual feedback
class ActionExecutor:
    """
    Executes browser actions with visual verification
    """

    def __init__(self, coordinator: MultiAgentCoordinator):
        self.coordinator = coordinator
        self.action_history = []

    def execute_with_visual_feedback(self, action_spec: Dict, browser, page) -> Dict[str, Any]:
        """
        Execute action and verify with screenshot analysis
        """
        # Take before screenshot
        before_screenshot = page.screenshot()
        before_analysis = self.coordinator.analyze_screenshot(
            base64.b64encode(before_screenshot).decode(),
            action_spec.get('description', 'action'),
            page.url
        )

        # Execute the action
        execution_result = self._perform_browser_action(action_spec, page)

        # Take after screenshot
        after_screenshot = page.screenshot()
        after_analysis = self.coordinator.analyze_screenshot(
            base64.b64encode(after_screenshot).decode(),
            'verify_action_result',
            page.url
        )

        # Validate action success
        validation = self.coordinator.validate_progress(
            action_spec.get('objective', ''),
            after_analysis,
            action_spec.get('expected_outcome', '')
        )

        result = {
            'action_executed': execution_result,
            'before_state': before_analysis,
            'after_state': after_analysis,
            'validation': validation,
            'success': execution_result.get('success', False) and validation.get('parsed', {}).get('overall_success', False)
        }

        self.action_history.append(result)
        return result

    def _perform_browser_action(self, action_spec: Dict, page) -> Dict[str, Any]:
        """
        Perform the actual browser action
        """
        try:
            action_type = action_spec.get('action_type', '')
            params = action_spec.get('parameters', {})

            if action_type == 'click':
                if 'coordinates' in params:
                    page.mouse.click(params['coordinates']['x'], params['coordinates']['y'])
                else:
                    page.click(params.get('selector', ''))

            elif action_type == 'fill':
                page.fill(params.get('selector', ''), params.get('value', ''))

            elif action_type == 'scroll':
                direction = params.get('direction', 'down')
                amount = params.get('amount', 500)
                if direction == 'down':
                    page.evaluate(f'window.scrollBy(0, {amount})')
                else:
                    page.evaluate(f'window.scrollBy(0, -{amount})')

            elif action_type == 'wait':
                wait_time = params.get('duration', 2000)
                page.wait_for_timeout(wait_time)

            elif action_type == 'navigate':
                page.goto(params.get('url', ''))

            return {'success': True, 'action': action_type, 'parameters': params}

        except Exception as e:
            return {'success': False, 'error': str(e), 'action': action_type}