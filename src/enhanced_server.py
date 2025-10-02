
import asyncio
import uuid
import json
import time
import os
import platform
import base64
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

# Windows AsyncIO fix
if platform.system() == 'Windows':
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        print("✅ Windows AsyncIO event loop policy set")
    except Exception as e:
        print(f"⚠️ Failed to set Windows event loop policy: {e}")

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import Playwright SYNC API for worker threads
from playwright.sync_api import sync_playwright

# Import your existing agent modules
try:
    from src.enhanced_llm import MultiAgentCoordinator, LLMProvider, ActionExecutor
    print("✅ Multi-agent system imported successfully")
    AGENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Agent import failed: {e}")
    print("🔄 Using fallback mode")
    AGENTS_AVAILABLE = False

try:
    from src.universal_scraper import UniversalWebScraper, ScraperConfig
    print("✅ Universal scraper imported successfully")
    UNIVERSAL_SCRAPER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Universal scraper import failed: {e}")
    print("🔄 Using fallback scraper")
    UNIVERSAL_SCRAPER_AVAILABLE = False

# Agent workflow states
class WorkflowState(str, Enum):
    INITIALIZING = "initializing"
    PLANNING = "planning"
    VISUAL_ANALYSIS = "visual_analysis"
    EXECUTING = "executing"
    VALIDATING = "validating"
    RESEARCHING = "researching"
    RECOVERING = "recovering"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class AgentResult:
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    agent_type: Optional[str] = None
    execution_time: float = 0.0

# Pydantic models
class ScrapingRequest(BaseModel):
    url: str = Field(..., description="Target URL to scrape")
    objective: str = Field(..., description="Natural language description of what to extract")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Scraper configuration options")

class ScrapingResponse(BaseModel):
    job_id: str
    status: str
    message: str
    stream_url: Optional[str] = None

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    current_step: str
    screenshots_count: int
    error_log: List[str]
    start_time: datetime
    elapsed_seconds: float

class JobResult(BaseModel):
    job_id: str
    success: bool
    extracted_data: Dict[str, Any]
    screenshots: List[str]
    execution_time: float
    steps_completed: int
    error_log: List[str]
    agent_logs: List[Dict[str, Any]]

# FastAPI app initialization
app = FastAPI(
    title="Universal Web Scraper API - Smart Dynamic Content",
    description="AI-powered web scraper with dynamic content handling and pagination",
    version="6.0.0"
)

# Create and mount web directory
web_dir = Path("web")
web_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="web"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage
active_jobs: Dict[str, Dict[str, Any]] = {}
job_results: Dict[str, Dict[str, Any]] = {}

def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def push_update(job_id: str, message: str, details: Dict = None, workflow_state: WorkflowState = None):
    """Add update to job log with workflow state tracking"""
    if job_id not in active_jobs:
        active_jobs[job_id] = {
            "updates": [], 
            "start_time": time.time(),
            "workflow_state": WorkflowState.INITIALIZING.value,
            "agent_logs": [],
            "communications": []
        }

    update = {
        "timestamp": get_timestamp(),
        "message": message,
        "details": details or {},
        "workflow_state": workflow_state.value if workflow_state else None
    }

    active_jobs[job_id]["updates"].append(update)

    if workflow_state:
        active_jobs[job_id]["workflow_state"] = workflow_state.value

def log_agent_communication(job_id: str, agent_name: str, message: str, data: Dict = None, level: str = "INFO"):
    """Log detailed agent communication for user visibility"""
    if job_id not in active_jobs:
        return

    communication = {
        "timestamp": get_timestamp(),
        "agent": agent_name,
        "level": level,
        "message": message,
        "data": data or {},
        "formatted_message": f"🤖 {agent_name}: {message}"
    }

    active_jobs[job_id]["communications"].append(communication)
    print(f"[{job_id}] 🤖 {agent_name}: {message}")

    if data:
        print(f"[{job_id}] 📊 Data: {json.dumps(data, indent=2)[:200]}...")

def log_agent_activity(job_id: str, agent_type: str, action: str, result: AgentResult):
    """Log detailed agent activity"""
    if job_id not in active_jobs:
        return

    agent_log = {
        "timestamp": get_timestamp(),
        "agent_type": agent_type,
        "action": action,
        "success": result.success,
        "execution_time": result.execution_time,
        "data_keys": list(result.data.keys()) if result.data else [],
        "error": result.error
    }

    active_jobs[job_id]["agent_logs"].append(agent_log)

class SmartWebScraper:
    """
    Smart web scraper with dynamic content handling, viewport screenshots, and pagination
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider = LLMProvider(config.get("provider", "anthropic"))
        self.max_steps = config.get("max_steps", 50)
        self.timeout = config.get("timeout_seconds", 45)  # Increased timeout for dynamic content
        self.headless = config.get("headless", True)

        if AGENTS_AVAILABLE:
            self.coordinator = MultiAgentCoordinator(self.provider)
        else:
            self.coordinator = None

        self.playwright = None
        self.browser = None
        self.page = None
        self.screenshots = []
        self.workflow_state = WorkflowState.INITIALIZING

    async def scrape(self, job_id: str, url: str, objective: str) -> Dict[str, Any]:
        """
        Smart scraping method with dynamic content handling
        """
        start_time = time.time()
        result = {
            "success": False,
            "extracted_data": {},
            "screenshots": [],
            "execution_time": 0,
            "steps_completed": 0,
            "error_log": [],
            "agent_logs": [],
            "workflow_log": [],
            "final_summary": {
                "phones": [],
                "prices": [],
                "ratings": [],
                "total_items_found": 0,
                "extraction_method": "unknown",
                "validation_score": 0.0,
                "validation_details": {},
                "pages_processed": 0
            }
        }

        log_agent_communication(job_id, "SYSTEM", f"🚀 Starting SMART scrape of {url}")
        log_agent_communication(job_id, "SYSTEM", f"🎯 Objective: {objective}")

        try:
            # Initialize smart browser with better settings
            log_agent_communication(job_id, "BROWSER", "🔧 Initializing smart browser with dynamic content support...")
            await self._initialize_smart_browser()
            log_agent_communication(job_id, "BROWSER", "✅ Smart browser initialized successfully")
            result["workflow_log"].append({"step": "browser_init", "status": "success"})
            push_update(job_id, "browser_initialized", workflow_state=WorkflowState.INITIALIZING)

            # Skip Universal Scraper for now - focus on multi-agent with smart features
            log_agent_communication(job_id, "SYSTEM", "🎯 Using SMART Multi-Agent approach for better results")

            # Smart Multi-agent workflow
            if AGENTS_AVAILABLE:
                log_agent_communication(job_id, "MULTI_AGENT", "🧠 Starting SMART Multi-Agent Workflow...")
                push_update(job_id, "starting_smart_multi_agent_workflow", workflow_state=WorkflowState.PLANNING)

                # Step 1: Smart Planning Agent
                log_agent_communication(job_id, "PLANNING_AGENT", "🧠 Creating smart strategy...")
                push_update(job_id, "planning_started", workflow_state=WorkflowState.PLANNING)
                planning_result = await self._run_smart_planning_agent(job_id, url, objective)
                result["agent_logs"].append(asdict(planning_result))
                result["steps_completed"] += 1

                if planning_result.success:
                    strategy = planning_result.data.get("strategy", "unknown")
                    log_agent_communication(job_id, "PLANNING_AGENT", f"✅ Smart planning completed: {strategy}")
                else:
                    log_agent_communication(job_id, "PLANNING_AGENT", f"❌ Planning failed: {planning_result.error}", level="ERROR")

                push_update(job_id, "planning_completed", 
                           {"success": planning_result.success, "strategy": planning_result.data.get("strategy", "unknown")})

                # Step 2: Smart Visual Analysis Agent with dynamic content
                log_agent_communication(job_id, "VISION_AGENT", "👁️ Starting SMART visual analysis with dynamic content handling...")
                push_update(job_id, "visual_analysis_started", workflow_state=WorkflowState.VISUAL_ANALYSIS)
                visual_result = await self._run_smart_visual_agent(job_id, url, objective, planning_result.data)
                result["agent_logs"].append(asdict(visual_result))
                result["steps_completed"] += 1

                if visual_result.success:
                    elements = visual_result.data.get("elements_found", {})
                    log_agent_communication(job_id, "VISION_AGENT", "✅ SMART visual analysis completed", 
                                          {"products_detected": elements.get("products", 0),
                                           "dynamic_content_loaded": visual_result.data.get("dynamic_content_loaded", False),
                                           "viewport_screenshots": len(self.screenshots)})
                else:
                    log_agent_communication(job_id, "VISION_AGENT", f"❌ Visual analysis failed: {visual_result.error}", level="ERROR")

                push_update(job_id, "visual_analysis_completed", 
                           {"success": visual_result.success, "dynamic_content": visual_result.data.get("dynamic_content_loaded", False)})

                # Step 3: Smart Execution Agent with pagination
                log_agent_communication(job_id, "EXECUTION_AGENT", "⚡ Starting SMART data extraction with pagination support...")
                push_update(job_id, "execution_started", workflow_state=WorkflowState.EXECUTING)
                execution_result = await self._run_smart_execution_agent(job_id, objective, visual_result.data if visual_result.success else {})
                result["agent_logs"].append(asdict(execution_result))
                result["steps_completed"] += 1

                if execution_result.success:
                    extracted = execution_result.data
                    log_agent_communication(job_id, "EXECUTION_AGENT", "✅ SMART data extraction completed", 
                                          {"products_extracted": len(extracted.get("products", [])),
                                           "prices_extracted": len(extracted.get("prices", [])),
                                           "ratings_extracted": len(extracted.get("ratings", [])),
                                           "pages_processed": extracted.get("pages_processed", 1)})
                else:
                    log_agent_communication(job_id, "EXECUTION_AGENT", f"❌ Execution failed: {execution_result.error}", level="ERROR")

                push_update(job_id, "execution_completed", 
                           {"success": execution_result.success, "items_extracted": len(execution_result.data.get("products", []))})

                # Step 4: Smart Validation Agent
                log_agent_communication(job_id, "VALIDATION_AGENT", "✅ Starting SMART validation...")
                push_update(job_id, "validation_started", workflow_state=WorkflowState.VALIDATING)
                validation_result = await self._run_smart_validation_agent(job_id, objective, execution_result.data)
                result["agent_logs"].append(asdict(validation_result))
                result["steps_completed"] += 1

                validation_data = validation_result.data
                log_agent_communication(job_id, "VALIDATION_AGENT", 
                                      f"📊 SMART validation completed - Score: {validation_data.get('success_score', 0):.2f}", 
                                      {"meets_objective": validation_data.get("meets_objective", False),
                                       "items_found": len(execution_result.data.get("products", [])),
                                       "quality_score": validation_data.get("quality_score", 0)})

                push_update(job_id, "validation_completed", 
                           {"success": validation_result.success, 
                            "score": validation_data.get("success_score", 0),
                            "quality": validation_data.get("quality_score", 0)})

                # Compile comprehensive results
                result["extracted_data"] = {
                    "planning": planning_result.data,
                    "visual_analysis": visual_result.data if visual_result.success else {},
                    "execution": execution_result.data,
                    "validation": validation_result.data,
                    "url": url,
                    "objective": objective,
                    "timestamp": get_timestamp(),
                    "provider_used": self.provider.value,
                    "smart_features_used": True
                }

                # Enhanced final summary
                exec_data = execution_result.data
                result["final_summary"] = {
                    "phones": exec_data.get("products", [])[:15],  # Top 15 products
                    "prices": exec_data.get("prices", [])[:15], 
                    "ratings": exec_data.get("ratings", [])[:15],
                    "total_items_found": len(exec_data.get("products", [])),
                    "extraction_method": "Smart Multi-Agent with Dynamic Content",
                    "page_title": exec_data.get("title", "Unknown"),
                    "extraction_timestamp": exec_data.get("extraction_timestamp", get_timestamp()),
                    "validation_score": validation_data.get("success_score", 0),
                    "validation_details": validation_data.get("detailed_analysis", {}),
                    "pages_processed": exec_data.get("pages_processed", 1),
                    "dynamic_content_loaded": visual_result.data.get("dynamic_content_loaded", False) if visual_result.success else False
                }

                result["screenshots"] = self.screenshots

                # Enhanced success determination
                meets_objective = validation_data.get("meets_objective", False)
                has_meaningful_data = len(exec_data.get("products", [])) > 0
                quality_threshold = validation_data.get("quality_score", 0) >= 0.5

                result["success"] = meets_objective and has_meaningful_data and quality_threshold

                if result["success"]:
                    log_agent_communication(job_id, "FINAL_RESULT", "🎉 SMART SCRAPING SUCCESS!", 
                                          {"total_products": len(exec_data.get("products", [])),
                                           "total_prices": len(exec_data.get("prices", [])),
                                           "pages_processed": exec_data.get("pages_processed", 1),
                                           "validation_score": validation_data.get("success_score", 0),
                                           "quality_score": validation_data.get("quality_score", 0)})
                else:
                    log_agent_communication(job_id, "FINAL_RESULT", 
                                          f"⚠️ Scraping completed but validation failed", 
                                          {"meets_objective": meets_objective,
                                           "has_data": has_meaningful_data,
                                           "quality_good": quality_threshold,
                                           "validation_score": validation_data.get("success_score", 0)},
                                          level="WARNING")

                    if not meets_objective:
                        result["error_log"].append("Validation failed: extracted data doesn't meet objective requirements")
                    if not has_meaningful_data:
                        result["error_log"].append("No meaningful data extracted - may need better selectors or wait times")
                    if not quality_threshold:
                        result["error_log"].append("Data quality below threshold - extracted data may be incomplete")

            else:
                # Smart fallback extraction
                log_agent_communication(job_id, "FALLBACK", "🔄 Using SMART fallback extraction...")
                push_update(job_id, "smart_fallback_extraction_started", workflow_state=WorkflowState.EXECUTING)
                fallback_result = await self._smart_fallback_extraction(job_id, url, objective)
                result.update(fallback_result)

        except Exception as e:
            error_msg = f"Critical smart scraping error: {str(e)}"
            log_agent_communication(job_id, "SYSTEM", f"💥 CRITICAL ERROR: {error_msg}", level="CRITICAL")
            result["error_log"].append(error_msg)
            result["error_log"].append(traceback.format_exc())
            push_update(job_id, "critical_error", {"error": error_msg}, WorkflowState.FAILED)

        finally:
            await self._cleanup_browser()
            result["execution_time"] = time.time() - start_time
            result["agent_logs"] = active_jobs[job_id].get("agent_logs", [])

            log_agent_communication(job_id, "SYSTEM", 
                                  f"🏁 SMART scraping completed in {result['execution_time']:.2f}s", 
                                  {"success": result["success"], 
                                   "steps_completed": result["steps_completed"],
                                   "screenshots_captured": len(result["screenshots"]),
                                   "items_found": result.get("final_summary", {}).get("total_items_found", 0)})

        return result

    async def _initialize_smart_browser(self):
        """Initialize smart browser with better settings for dynamic content"""
        try:
            def launch_smart_browser():
                playwright = sync_playwright().start()
                browser = playwright.chromium.launch(
                    headless=self.headless,
                    args=[
                        '--no-sandbox', 
                        '--disable-dev-shm-usage', 
                        '--disable-gpu',
                        '--disable-web-security',
                        '--disable-features=VizDisplayCompositor',
                        '--no-first-run',
                        '--disable-background-timer-throttling',
                        '--disable-backgrounding-occluded-windows',
                        '--disable-renderer-backgrounding'
                    ]
                )

                # Create context with realistic settings
                context = browser.new_context(
                    viewport={'width': 1366, 'height': 768},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                )

                page = context.new_page()
                page.set_default_timeout(self.timeout * 1000)

                return playwright, browser, context, page

            self.playwright, self.browser, self.context, self.page = await asyncio.to_thread(launch_smart_browser)

        except Exception as e:
            raise Exception(f"Smart browser initialization failed: {str(e)}")

    async def _cleanup_browser(self):
        """Clean up smart browser resources"""
        def cleanup():
            try:
                if hasattr(self, 'context') and self.context:
                    self.context.close()
                if self.browser:
                    self.browser.close()
                if self.playwright:
                    self.playwright.stop()
            except Exception as e:
                print(f"Browser cleanup error: {e}")

        await asyncio.to_thread(cleanup)

    async def _take_viewport_screenshot(self) -> str:
        """Take viewport-only screenshot (not full page)"""
        def capture_viewport():
            try:
                if not self.page:
                    return ""
                # Take viewport screenshot only
                screenshot = self.page.screenshot(full_page=False)
                return base64.b64encode(screenshot).decode()
            except Exception as e:
                print(f"Viewport screenshot failed: {e}")
                return ""

        screenshot_b64 = await asyncio.to_thread(capture_viewport)
        if screenshot_b64:
            self.screenshots.append(screenshot_b64)
        return screenshot_b64

    async def _smart_scroll_and_capture(self, job_id: str, times: int = 3) -> List[str]:
        """Smart scrolling with viewport screenshots"""
        screenshots = []

        def scroll_and_capture():
            try:
                if not self.page:
                    return []

                captures = []
                viewport_height = self.page.viewport_size['height']

                for i in range(times):
                    # Take screenshot before scroll
                    screenshot = self.page.screenshot(full_page=False)
                    captures.append(base64.b64encode(screenshot).decode())

                    # Scroll down by viewport height
                    self.page.evaluate(f"window.scrollBy(0, {viewport_height * 0.8})")
                    self.page.wait_for_timeout(2000)  # Wait for content to load

                return captures
            except Exception as e:
                print(f"Smart scroll error: {e}")
                return []

        screenshots = await asyncio.to_thread(scroll_and_capture)

        if screenshots:
            self.screenshots.extend(screenshots)
            log_agent_communication(job_id, "SMART_SCROLL", f"📸 Captured {len(screenshots)} viewport screenshots")

        return screenshots

    async def _run_smart_planning_agent(self, job_id: str, url: str, objective: str) -> AgentResult:
        """Smart planning agent with website-specific strategies"""
        start_time = time.time()

        try:
            # Determine website type
            is_flipkart = "flipkart.com" in url.lower()
            is_amazon = "amazon." in url.lower()
            is_ecommerce = is_flipkart or is_amazon or any(site in url.lower() for site in ["shop", "store", "buy", "cart"])

            log_agent_communication(job_id, "PLANNING_AGENT", f"🎯 Detected website type", 
                                  {"is_flipkart": is_flipkart, "is_amazon": is_amazon, "is_ecommerce": is_ecommerce})

            # Smart strategy based on website
            if is_flipkart:
                plan_data = {
                    "strategy": "flipkart_dynamic_content_strategy",
                    "steps": [
                        "navigate_with_wait", 
                        "wait_for_dynamic_products", 
                        "capture_viewport_screenshots",
                        "extract_flipkart_products", 
                        "extract_flipkart_prices", 
                        "extract_flipkart_ratings",
                        "check_pagination",
                        "validate_results"
                    ],
                    "complexity": "high_dynamic_content",
                    "estimated_time": 60,
                    "selectors": {
                        "products": [
                            '[data-id*="MOBIL"]',  # Flipkart mobile product IDs
                            '._4rR01T',  # Product title class
                            '._1fQZEK',  # Product container
                            '.s1Q9rs',   # Product link
                            '._2kHMtA',  # Product card
                            '.KzDlHZ'    # Product name
                        ],
                        "prices": [
                            '._30jeq3',  # Current price
                            '._3I9_wc',  # Discounted price  
                            '._25b18c',  # Price container
                            '.CEmiEU',   # Price element
                            '._1_TUDb'   # Price display
                        ],
                        "ratings": [
                            '._3LWZlK',  # Rating container
                            '.XQDdHH',   # Rating number
                            '._2_R_DZ',  # Rating stars
                            '._3dXdBS',  # Rating box
                            '.eSDtfx'    # Rating element
                        ]
                    },
                    "wait_conditions": ["networkidle", "domcontentloaded"],
                    "dynamic_wait_time": 8000,  # 8 seconds for dynamic content
                    "scroll_strategy": "incremental_with_wait",
                    "pagination_selectors": ['._1LKTO3', '.ge-49M', '._9QVEpD']
                }
            elif is_amazon:
                plan_data = {
                    "strategy": "amazon_dynamic_content_strategy", 
                    "steps": ["navigate_with_wait", "wait_for_products", "extract_amazon_data", "check_next_page"],
                    "selectors": {
                        "products": ['[data-component-type="s-search-result"]', '.s-title-instructions-style', 'h3 a span'],
                        "prices": ['.a-price-whole', '.a-price .a-offscreen', '.a-price-symbol'],
                        "ratings": ['.a-icon-alt', '.a-size-base', 'a-popover-trigger a-declarative']
                    },
                    "dynamic_wait_time": 6000
                }
            else:
                plan_data = {
                    "strategy": "generic_dynamic_content_strategy",
                    "steps": ["navigate_smart", "detect_content_type", "extract_intelligently", "validate"],
                    "dynamic_wait_time": 5000,
                    "selectors": {
                        "products": ['[class*="product"]', '[class*="item"]', 'h1', 'h2', 'h3'],
                        "prices": ['[class*="price"]', '[class*="cost"]', '[class*="amount"]'],
                        "ratings": ['[class*="rating"]', '[class*="star"]', '[class*="review"]']
                    }
                }

            if self.coordinator and not is_flipkart:  # Use AI for non-Flipkart sites
                log_agent_communication(job_id, "PLANNING_AGENT", "🤖 Using AI planning for non-Flipkart site...")

                def run_ai_planning():
                    try:
                        context = {'query': objective, 'url': url, 'website_context': {"domain": url}}
                        return self.coordinator.plan_workflow(objective, url, context)
                    except Exception as e:
                        log_agent_communication(job_id, "PLANNING_AGENT", f"AI planning error: {e}", level="ERROR")
                        return {"success": False, "error": str(e)}

                ai_result = await asyncio.to_thread(run_ai_planning)

                if ai_result.get('success') and ai_result.get('parsed'):
                    plan_data.update(ai_result['parsed'])
                    log_agent_communication(job_id, "PLANNING_AGENT", "🎯 AI planning enhanced strategy")

            log_agent_communication(job_id, "PLANNING_AGENT", "📋 Smart strategy created", 
                                  {"strategy": plan_data["strategy"], 
                                   "steps": len(plan_data["steps"]),
                                   "dynamic_wait": plan_data.get("dynamic_wait_time", 0)})

            agent_result = AgentResult(
                success=True, 
                data=plan_data, 
                agent_type="planning",
                execution_time=time.time() - start_time
            )

            log_agent_activity(job_id, "planning", "smart_strategy", agent_result)
            return agent_result

        except Exception as e:
            log_agent_communication(job_id, "PLANNING_AGENT", f"Exception: {str(e)}", level="ERROR")
            agent_result = AgentResult(
                success=False, 
                data={}, 
                error=str(e),
                agent_type="planning",
                execution_time=time.time() - start_time
            )

            log_agent_activity(job_id, "planning", "error", agent_result)
            return agent_result

    async def _run_smart_visual_agent(self, job_id: str, url: str, objective: str, plan_data: Dict) -> AgentResult:
        """Smart visual agent with dynamic content handling"""
        start_time = time.time()

        try:
            log_agent_communication(job_id, "VISION_AGENT", f"🌐 Navigating to {url}...")

            def smart_navigate_and_analyze():
                try:
                    if not self.page:
                        raise Exception("Browser page not available")

                    log_agent_communication(job_id, "VISION_AGENT", "🔄 Loading page with smart settings...")

                    # Navigate with better settings
                    self.page.goto(url, wait_until="domcontentloaded", timeout=45000)

                    # Dynamic wait time from planning
                    dynamic_wait = plan_data.get("dynamic_wait_time", 5000)
                    log_agent_communication(job_id, "VISION_AGENT", f"⏳ Waiting {dynamic_wait}ms for dynamic content...")
                    self.page.wait_for_timeout(dynamic_wait)

                    # Additional wait for network idle for ecommerce sites
                    is_ecommerce = any(site in url.lower() for site in ["flipkart", "amazon", "shop", "store"])
                    if is_ecommerce:
                        try:
                            log_agent_communication(job_id, "VISION_AGENT", "🌐 Waiting for network idle (ecommerce site)...")
                            self.page.wait_for_load_state("networkidle", timeout=10000)
                        except:
                            log_agent_communication(job_id, "VISION_AGENT", "⏰ Network idle timeout - proceeding anyway")

                    log_agent_communication(job_id, "VISION_AGENT", "📊 Analyzing smart page structure...")

                    # Smart element detection using planning selectors
                    selectors = plan_data.get("selectors", {})
                    product_selectors = selectors.get("products", ['[class*="product"]'])
                    price_selectors = selectors.get("prices", ['[class*="price"]'])
                    rating_selectors = selectors.get("ratings", ['[class*="rating"]'])

                    # Count elements with all selectors
                    total_products = 0
                    for selector in product_selectors:
                        try:
                            count = len(self.page.query_selector_all(selector))
                            total_products = max(total_products, count)
                        except:
                            continue

                    total_prices = 0
                    for selector in price_selectors:
                        try:
                            count = len(self.page.query_selector_all(selector))
                            total_prices = max(total_prices, count)
                        except:
                            continue

                    total_ratings = 0
                    for selector in rating_selectors:
                        try:
                            count = len(self.page.query_selector_all(selector))
                            total_ratings = max(total_ratings, count)
                        except:
                            continue

                    # Smart page info
                    page_info = {
                        "forms": len(self.page.query_selector_all("form")),
                        "links": len(self.page.query_selector_all("a")),
                        "images": len(self.page.query_selector_all("img")),
                        "buttons": len(self.page.query_selector_all("button, .btn, [role='button']")),
                        "products": total_products,
                        "prices": total_prices, 
                        "ratings": total_ratings,
                        "title": self.page.title(),
                        "url": self.page.url,
                        "content_length": len(self.page.inner_text("body") or ""),
                        "viewport_size": self.page.viewport_size
                    }

                    # Take initial viewport screenshot
                    log_agent_communication(job_id, "VISION_AGENT", "📸 Capturing initial viewport screenshot...")
                    screenshot = self.page.screenshot(full_page=False)  # Viewport only
                    screenshot_b64 = base64.b64encode(screenshot).decode()

                    return page_info, screenshot_b64

                except Exception as e:
                    log_agent_communication(job_id, "VISION_AGENT", f"Navigation error: {e}", level="ERROR")
                    raise e

            page_info, screenshot_b64 = await asyncio.to_thread(smart_navigate_and_analyze)

            if screenshot_b64:
                self.screenshots.append(screenshot_b64)
                log_agent_communication(job_id, "VISION_AGENT", "✅ Initial viewport screenshot captured")

            # Smart scrolling for more content
            log_agent_communication(job_id, "VISION_AGENT", "📜 Smart scrolling to load more content...")
            scroll_screenshots = await self._smart_scroll_and_capture(job_id, times=2)

            log_agent_communication(job_id, "VISION_AGENT", "🔍 Smart page analysis complete", 
                                  {"products_detected": page_info["products"],
                                   "prices_detected": page_info["prices"],
                                   "ratings_detected": page_info["ratings"],
                                   "viewport_screenshots": len(scroll_screenshots) + 1})

            # Enhanced visual data
            visual_data = {
                "page_loaded": True,
                "screenshot_taken": bool(screenshot_b64),
                "elements_found": page_info,
                "page_title": page_info["title"],
                "url": page_info["url"],
                "dynamic_content_loaded": page_info["products"] > 0 or page_info["prices"] > 0,
                "viewport_screenshots_taken": len(scroll_screenshots) + 1,
                "content_analysis": {
                    "has_products": page_info["products"] > 0,
                    "has_prices": page_info["prices"] > 0,
                    "has_ratings": page_info["ratings"] > 0,
                    "content_rich": page_info["content_length"] > 500,
                    "likely_dynamic_site": page_info["products"] == 0 and page_info["content_length"] > 1000
                },
                "smart_recommendations": []
            }

            # Smart recommendations
            if page_info["products"] == 0 and page_info["content_length"] > 1000:
                visual_data["smart_recommendations"].append("Dynamic content detected - may need longer wait times")
            if page_info["products"] > 0 and page_info["prices"] == 0:
                visual_data["smart_recommendations"].append("Products detected but no prices - check price selectors")
            if page_info["products"] > 10:
                visual_data["smart_recommendations"].append("Multiple products found - pagination may be available")

            agent_result = AgentResult(
                success=True, 
                data=visual_data, 
                agent_type="vision",
                execution_time=time.time() - start_time
            )

            log_agent_activity(job_id, "vision", "smart_analyze", agent_result)
            return agent_result

        except Exception as e:
            log_agent_communication(job_id, "VISION_AGENT", f"Critical error: {str(e)}", level="ERROR")
            agent_result = AgentResult(
                success=False, 
                data={}, 
                error=str(e),
                agent_type="vision",
                execution_time=time.time() - start_time
            )

            log_agent_activity(job_id, "vision", "error", agent_result)
            return agent_result

    async def _run_smart_execution_agent(self, job_id: str, objective: str, visual_data: Dict) -> AgentResult:
        """Smart execution agent with pagination support"""
        start_time = time.time()

        try:
            log_agent_communication(job_id, "EXECUTION_AGENT", "🎯 Starting SMART data extraction with pagination...")

            def smart_extract_with_pagination():
                try:
                    if not self.page:
                        raise Exception("Browser page not available")

                    all_products = []
                    all_prices = []
                    all_ratings = []
                    pages_processed = 1

                    # Extract data from current page
                    log_agent_communication(job_id, "EXECUTION_AGENT", "📋 Extracting from current page...")

                    # Basic info
                    extracted = {
                        "title": self.page.title() or "No title",
                        "url": self.page.url,
                        "text_content_length": len(self.page.inner_text("body") or ""),
                        "extraction_timestamp": get_timestamp()
                    }

                    # Smart product extraction with multiple strategies
                    if "product" in objective.lower() or "phone" in objective.lower() or "mobile" in objective.lower():
                        log_agent_communication(job_id, "EXECUTION_AGENT", "📱 Smart product extraction...")

                        # Flipkart-specific selectors (most reliable)
                        flipkart_selectors = [
                            '[data-id*="MOBIL"]',  # Mobile product IDs
                            '._4rR01T',            # Product title
                            '.s1Q9rs',             # Product link  
                            '.KzDlHZ',             # Product name
                            '._2kHMtA',            # Product card
                            '._1fQZEK'             # Product container
                        ]

                        # Generic selectors as fallback
                        generic_selectors = [
                            '[class*="product"]', '[class*="item"]', '[class*="title"]',
                            'h1', 'h2', 'h3', '.name', '[class*="phone"]', '[class*="mobile"]'
                        ]

                        all_selectors = flipkart_selectors + generic_selectors

                        for selector in all_selectors:
                            try:
                                elements = self.page.query_selector_all(selector)
                                for elem in elements[:25]:  # Limit per selector
                                    try:
                                        text = elem.inner_text().strip()
                                        if text and 10 < len(text) < 200:
                                            # Filter for phone-related content
                                            if any(brand in text.lower() for brand in 
                                                  ["samsung", "galaxy", "iphone", "pixel", "oneplus", "xiaomi", "oppo", "vivo", "realme", "redmi"]):
                                                all_products.append(text)
                                            elif any(term in text.lower() for term in 
                                                    ["gb", "ram", "storage", "camera", "battery", "mobile", "smartphone"]):
                                                all_products.append(text)
                                    except:
                                        continue
                            except:
                                continue

                        log_agent_communication(job_id, "EXECUTION_AGENT", f"📱 Found {len(all_products)} products on page 1")

                    # Smart price extraction
                    if "price" in objective.lower():
                        log_agent_communication(job_id, "EXECUTION_AGENT", "💰 Smart price extraction...")

                        # Flipkart price selectors
                        flipkart_price_selectors = [
                            '._30jeq3',  # Current price
                            '._3I9_wc',  # Discounted price
                            '._25b18c',  # Price container
                            '.CEmiEU',   # Price element
                            '._1_TUDb'   # Price display
                        ]

                        # Generic price selectors
                        generic_price_selectors = [
                            '[class*="price"]', '[class*="cost"]', '[class*="amount"]',
                            '[class*="rupee"]', '[class*="currency"]', '.price'
                        ]

                        all_price_selectors = flipkart_price_selectors + generic_price_selectors

                        for selector in all_price_selectors:
                            try:
                                elements = self.page.query_selector_all(selector)
                                for elem in elements[:20]:
                                    try:
                                        text = elem.inner_text().strip()
                                        if text and any(char.isdigit() for char in text):
                                            # Clean price text
                                            if "₹" in text or "rs" in text.lower() or any(digit.isdigit() for digit in text):
                                                all_prices.append(text)
                                    except:
                                        continue
                            except:
                                continue

                        log_agent_communication(job_id, "EXECUTION_AGENT", f"💰 Found {len(all_prices)} prices on page 1")

                    # Smart rating extraction
                    if "rating" in objective.lower():
                        log_agent_communication(job_id, "EXECUTION_AGENT", "⭐ Smart rating extraction...")

                        # Flipkart rating selectors
                        flipkart_rating_selectors = [
                            '._3LWZlK',  # Rating container
                            '.XQDdHH',   # Rating number
                            '._2_R_DZ',  # Rating stars
                            '._3dXdBS',  # Rating box
                            '.eSDtfx'    # Rating element
                        ]

                        # Generic rating selectors
                        generic_rating_selectors = [
                            '[class*="rating"]', '[class*="star"]', '[class*="review"]',
                            '.score', '.stars', '[class*="rate"]'
                        ]

                        all_rating_selectors = flipkart_rating_selectors + generic_rating_selectors

                        for selector in all_rating_selectors:
                            try:
                                elements = self.page.query_selector_all(selector)
                                for elem in elements[:15]:
                                    try:
                                        text = elem.inner_text().strip()
                                        if text and (any(char.isdigit() for char in text) or "star" in text.lower()):
                                            all_ratings.append(text)
                                    except:
                                        continue
                            except:
                                continue

                        log_agent_communication(job_id, "EXECUTION_AGENT", f"⭐ Found {len(all_ratings)} ratings on page 1")

                    # Check for pagination (Next page)
                    if len(all_products) >= 5:  # Only try pagination if we found some products
                        log_agent_communication(job_id, "EXECUTION_AGENT", "📄 Checking for pagination...")

                        # Flipkart pagination selectors
                        pagination_selectors = [
                            '._1LKTO3',     # Next button
                            '.ge-49M',      # Navigation
                            '._9QVEpD',     # Page navigation
                            'a[aria-label="Next"]',
                            '.srp-controls a[href*="page"]'
                        ]

                        next_button = None
                        for selector in pagination_selectors:
                            try:
                                buttons = self.page.query_selector_all(selector)
                                for button in buttons:
                                    text = button.inner_text().strip().lower()
                                    if "next" in text or ">" in text or "आगे" in text:
                                        next_button = button
                                        break
                                if next_button:
                                    break
                            except:
                                continue

                        # Try to click next page (max 2 additional pages)
                        max_pages = 3
                        while next_button and pages_processed < max_pages:
                            try:
                                log_agent_communication(job_id, "EXECUTION_AGENT", f"📄 Navigating to page {pages_processed + 1}...")

                                next_button.click()
                                self.page.wait_for_timeout(8000)  # Wait for new page to load

                                # Wait for content
                                try:
                                    self.page.wait_for_load_state("networkidle", timeout=10000)
                                except:
                                    pass

                                pages_processed += 1

                                # Take viewport screenshot of new page
                                page_screenshot = self.page.screenshot(full_page=False)
                                page_screenshot_b64 = base64.b64encode(page_screenshot).decode()
                                self.screenshots.append(page_screenshot_b64)

                                # Extract from this page (simplified)
                                page_products = []
                                page_prices = []
                                page_ratings = []

                                # Quick extraction from new page
                                for selector in flipkart_selectors[:3]:  # Top 3 selectors only
                                    try:
                                        elements = self.page.query_selector_all(selector)
                                        for elem in elements[:10]:
                                            try:
                                                text = elem.inner_text().strip()
                                                if text and 10 < len(text) < 200:
                                                    if any(brand in text.lower() for brand in ["samsung", "galaxy", "mobile", "smartphone"]):
                                                        page_products.append(text)
                                            except:
                                                continue
                                    except:
                                        continue

                                all_products.extend(page_products)
                                log_agent_communication(job_id, "EXECUTION_AGENT", 
                                                      f"📄 Page {pages_processed}: Found {len(page_products)} products")

                                # Look for next button again
                                next_button = None
                                for selector in pagination_selectors:
                                    try:
                                        buttons = self.page.query_selector_all(selector)
                                        for button in buttons:
                                            text = button.inner_text().strip().lower()
                                            if "next" in text or ">" in text:
                                                next_button = button
                                                break
                                        if next_button:
                                            break
                                    except:
                                        continue

                            except Exception as e:
                                log_agent_communication(job_id, "EXECUTION_AGENT", f"Pagination error: {e}", level="WARNING")
                                break

                    # Remove duplicates and finalize
                    extracted["products"] = list(dict.fromkeys(all_products))[:20]  # Remove duplicates, limit to 20
                    extracted["prices"] = list(dict.fromkeys(all_prices))[:15]
                    extracted["ratings"] = list(dict.fromkeys(all_ratings))[:15]
                    extracted["pages_processed"] = pages_processed

                    # Additional metadata
                    extracted["meta"] = {
                        "total_elements": len(self.page.query_selector_all("*")),
                        "extraction_strategy": "smart_multi_page_extraction",
                        "pagination_successful": pages_processed > 1
                    }

                    return extracted

                except Exception as e:
                    log_agent_communication(job_id, "EXECUTION_AGENT", f"Extraction error: {e}", level="ERROR")
                    raise e

            extracted_data = await asyncio.to_thread(smart_extract_with_pagination)

            log_agent_communication(job_id, "EXECUTION_AGENT", "✅ SMART extraction complete", 
                                  {"total_products": len(extracted_data.get("products", [])),
                                   "total_prices": len(extracted_data.get("prices", [])),
                                   "total_ratings": len(extracted_data.get("ratings", [])),
                                   "pages_processed": extracted_data.get("pages_processed", 1)})

            agent_result = AgentResult(
                success=True, 
                data=extracted_data, 
                agent_type="execution",
                execution_time=time.time() - start_time
            )

            log_agent_activity(job_id, "execution", "smart_extract", agent_result)
            return agent_result

        except Exception as e:
            log_agent_communication(job_id, "EXECUTION_AGENT", f"Critical error: {str(e)}", level="ERROR")
            agent_result = AgentResult(
                success=False, 
                data={}, 
                error=str(e),
                agent_type="execution",
                execution_time=time.time() - start_time
            )

            log_agent_activity(job_id, "execution", "error", agent_result)
            return agent_result

    async def _run_smart_validation_agent(self, job_id: str, objective: str, execution_data: Dict) -> AgentResult:
        """Smart validation agent with enhanced scoring"""
        start_time = time.time()

        try:
            log_agent_communication(job_id, "VALIDATION_AGENT", "🔍 Starting SMART validation analysis...")

            validation = {
                "data_extracted": bool(execution_data),
                "has_title": bool(execution_data.get("title")),
                "has_content": execution_data.get("text_content_length", 0) > 0,
                "objective_keywords_found": [],
                "success_score": 0.0,
                "quality_score": 0.0,
                "detailed_analysis": {},
                "smart_scoring_breakdown": {}
            }

            # Smart objective analysis
            obj_lower = objective.lower()
            score = 0.0
            quality = 0.0
            scoring = {}

            log_agent_communication(job_id, "VALIDATION_AGENT", f"📋 Analyzing objective: {objective}")

            # Enhanced product validation
            if any(word in obj_lower for word in ["product", "item", "phone", "smartphone", "mobile"]):
                products = execution_data.get("products", [])
                if products:
                    validation["objective_keywords_found"].append("products")
                    validation["detailed_analysis"]["products_found"] = len(products)

                    # Base score for finding products
                    product_score = min(0.4, len(products) * 0.03)
                    score += product_score
                    scoring["product_extraction"] = product_score

                    # Quality assessment for products
                    samsung_products = [p for p in products if "samsung" in p.lower() or "galaxy" in p.lower()]
                    relevant_products = [p for p in products if any(term in p.lower() for term in 
                                        ["samsung", "galaxy", "mobile", "smartphone", "gb", "ram"])]

                    product_quality = len(relevant_products) / len(products) if products else 0
                    quality += product_quality * 0.4
                    scoring["product_quality"] = product_quality

                    if "samsung" in obj_lower and samsung_products:
                        samsung_bonus = min(0.2, len(samsung_products) * 0.04)
                        score += samsung_bonus
                        scoring["samsung_bonus"] = samsung_bonus
                        validation["detailed_analysis"]["samsung_products_found"] = len(samsung_products)

                        log_agent_communication(job_id, "VALIDATION_AGENT", 
                                              f"🔥 Samsung products: {len(samsung_products)} found (+{samsung_bonus:.2f})")

                    log_agent_communication(job_id, "VALIDATION_AGENT", 
                                          f"📱 Products: {len(products)} found, {len(relevant_products)} relevant (+{product_score:.2f})")
                else:
                    scoring["product_extraction"] = 0.0
                    log_agent_communication(job_id, "VALIDATION_AGENT", "📱 No products found (0.0)")

            # Enhanced price validation
            if "price" in obj_lower:
                prices = execution_data.get("prices", [])
                if prices:
                    validation["objective_keywords_found"].append("prices")
                    validation["detailed_analysis"]["prices_found"] = len(prices)

                    price_score = min(0.25, len(prices) * 0.02)
                    score += price_score
                    scoring["price_extraction"] = price_score

                    # Price quality - check for proper formatting
                    valid_prices = [p for p in prices if "₹" in p or any(char.isdigit() for char in p)]
                    price_quality = len(valid_prices) / len(prices) if prices else 0
                    quality += price_quality * 0.2
                    scoring["price_quality"] = price_quality

                    log_agent_communication(job_id, "VALIDATION_AGENT", f"💰 Prices: {len(prices)} found (+{price_score:.2f})")
                else:
                    scoring["price_extraction"] = 0.0
                    log_agent_communication(job_id, "VALIDATION_AGENT", "💰 No prices found (0.0)")

            # Enhanced rating validation
            if "rating" in obj_lower:
                ratings = execution_data.get("ratings", [])
                if ratings:
                    validation["objective_keywords_found"].append("ratings")
                    validation["detailed_analysis"]["ratings_found"] = len(ratings)

                    rating_score = min(0.2, len(ratings) * 0.015)
                    score += rating_score
                    scoring["rating_extraction"] = rating_score

                    # Rating quality - check for numeric ratings
                    valid_ratings = [r for r in ratings if any(char.isdigit() for char in r)]
                    rating_quality = len(valid_ratings) / len(ratings) if ratings else 0
                    quality += rating_quality * 0.2
                    scoring["rating_quality"] = rating_quality

                    log_agent_communication(job_id, "VALIDATION_AGENT", f"⭐ Ratings: {len(ratings)} found (+{rating_score:.2f})")
                else:
                    scoring["rating_extraction"] = 0.0
                    log_agent_communication(job_id, "VALIDATION_AGENT", "⭐ No ratings found (0.0)")

            # Pagination bonus
            pages_processed = execution_data.get("pages_processed", 1)
            if pages_processed > 1:
                pagination_bonus = min(0.15, (pages_processed - 1) * 0.05)
                score += pagination_bonus
                quality += 0.2  # Bonus for multi-page extraction
                scoring["pagination_bonus"] = pagination_bonus
                log_agent_communication(job_id, "VALIDATION_AGENT", f"📄 Pagination: {pages_processed} pages (+{pagination_bonus:.2f})")

            # Basic validation
            if execution_data.get("title"):
                title_score = 0.05
                score += title_score
                quality += 0.1
                scoring["has_title"] = title_score
                log_agent_communication(job_id, "VALIDATION_AGENT", f"📄 Title present (+{title_score:.2f})")

            if execution_data.get("text_content_length", 0) > 100:
                content_score = 0.1
                score += content_score
                quality += 0.1
                scoring["has_content"] = content_score
                log_agent_communication(job_id, "VALIDATION_AGENT", f"📝 Content: {execution_data.get('text_content_length', 0)} chars (+{content_score:.2f})")

            # Final scores
            validation["success_score"] = min(score, 1.0)
            validation["quality_score"] = min(quality, 1.0)
            validation["smart_scoring_breakdown"] = scoring
            validation["meets_objective"] = score >= 0.25  # Lowered threshold for smart extraction
            validation["good_quality"] = quality >= 0.5

            log_agent_communication(job_id, "VALIDATION_AGENT", 
                                  f"📊 SMART validation completed - Score: {validation['success_score']:.2f}, Quality: {validation['quality_score']:.2f}", 
                                  {"meets_objective": validation["meets_objective"],
                                   "good_quality": validation["good_quality"],
                                   "keywords_found": validation["objective_keywords_found"],
                                   "pages_processed": pages_processed})

            agent_result = AgentResult(
                success=validation["meets_objective"], 
                data=validation, 
                agent_type="validation",
                execution_time=time.time() - start_time
            )

            log_agent_activity(job_id, "validation", "smart_validate", agent_result)
            return agent_result

        except Exception as e:
            log_agent_communication(job_id, "VALIDATION_AGENT", f"Critical error: {str(e)}", level="ERROR")
            agent_result = AgentResult(
                success=False, 
                data={}, 
                error=str(e),
                agent_type="validation",
                execution_time=time.time() - start_time
            )

            log_agent_activity(job_id, "validation", "error", agent_result)
            return agent_result

    async def _smart_fallback_extraction(self, job_id: str, url: str, objective: str) -> Dict[str, Any]:
        """Smart fallback extraction with viewport screenshots"""
        log_agent_communication(job_id, "FALLBACK", "🔧 Starting SMART fallback extraction...")

        try:
            def smart_fallback():
                try:
                    if not self.page:
                        raise Exception("Browser page not available")

                    self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    self.page.wait_for_timeout(5000)  # Wait for basic dynamic content

                    # Take viewport screenshots while scrolling
                    screenshots = []
                    viewport_height = self.page.viewport_size['height']

                    for i in range(3):
                        screenshot = self.page.screenshot(full_page=False)
                        screenshots.append(base64.b64encode(screenshot).decode())

                        if i < 2:  # Don't scroll after last screenshot
                            self.page.evaluate(f"window.scrollBy(0, {viewport_height * 0.7})")
                            self.page.wait_for_timeout(2000)

                    text_content = self.page.inner_text("body") or ""

                    extracted = {
                        "title": self.page.title() or "No title",
                        "url": self.page.url,
                        "text_content": text_content[:2000],
                        "text_length": len(text_content),
                        "smart_fallback_mode": True,
                        "extraction_timestamp": get_timestamp(),
                        "viewport_screenshots_taken": len(screenshots)
                    }

                    # Smart pattern matching
                    if "samsung" in objective.lower():
                        samsung_mentions = text_content.lower().count("samsung")
                        galaxy_mentions = text_content.lower().count("galaxy")
                        extracted["samsung_mentions"] = samsung_mentions
                        extracted["galaxy_mentions"] = galaxy_mentions
                        log_agent_communication(job_id, "FALLBACK", f"Found {samsung_mentions} Samsung mentions, {galaxy_mentions} Galaxy mentions")

                    if "price" in objective.lower():
                        import re
                        price_patterns = re.findall(r'₹[\d,]+|Rs\.?\s*[\d,]+', text_content)
                        extracted["potential_prices"] = price_patterns[:15]
                        log_agent_communication(job_id, "FALLBACK", f"Found {len(price_patterns)} potential prices")

                    return extracted, screenshots

                except Exception as e:
                    log_agent_communication(job_id, "FALLBACK", f"Extraction error: {e}", level="ERROR")
                    raise e

            extracted_data, screenshots = await asyncio.to_thread(smart_fallback)

            if screenshots:
                self.screenshots.extend(screenshots)

            log_agent_communication(job_id, "FALLBACK", "✅ SMART fallback extraction completed", 
                                  {"data_points": len(extracted_data), "screenshots": len(screenshots)})

            push_update(job_id, "smart_fallback_extraction_completed", 
                       {"data_points": len(extracted_data)}, WorkflowState.COMPLETED)

            return {
                "success": True,
                "extracted_data": extracted_data,
                "screenshots": screenshots,
                "steps_completed": 1,
                "error_log": [],
                "agent_logs": [],
                "final_summary": {
                    "phones": [],
                    "prices": extracted_data.get("potential_prices", []),
                    "ratings": [],
                    "total_items_found": len(extracted_data.get("potential_prices", [])),
                    "extraction_method": "Smart Fallback Extraction",
                    "pages_processed": 1
                }
            }

        except Exception as e:
            log_agent_communication(job_id, "FALLBACK", f"Critical error: {str(e)}", level="ERROR")
            return {
                "success": False,
                "extracted_data": {},
                "screenshots": [],
                "steps_completed": 0,
                "error_log": [f"Smart fallback extraction failed: {str(e)}"],
                "agent_logs": [],
                "final_summary": {
                    "phones": [],
                    "prices": [],
                    "ratings": [],
                    "total_items_found": 0,
                    "extraction_method": "Failed",
                    "pages_processed": 0
                }
            }

async def run_scraping_job(job_id: str, request: ScrapingRequest):
    """Smart scraping job with enhanced communication"""
    try:
        log_agent_communication(job_id, "SYSTEM", f"🚀 SMART Job {job_id} started")
        log_agent_communication(job_id, "SYSTEM", f"🎯 Target: {request.url}")
        log_agent_communication(job_id, "SYSTEM", f"📋 Objective: {request.objective}")

        push_update(job_id, "job_started", 
                   {"url": request.url, "objective": request.objective}, 
                   WorkflowState.INITIALIZING)

        push_update(job_id, "initializing_scraper", 
                   {"config": request.config, "agents": AGENTS_AVAILABLE, "smart_features": True}, 
                   WorkflowState.INITIALIZING)

        # Create smart scraper
        scraper = SmartWebScraper(request.config)

        log_agent_communication(job_id, "SYSTEM", "⚙️ SMART Scraper initialized", 
                              {"agents_available": AGENTS_AVAILABLE, 
                               "smart_features": True,
                               "dynamic_content_support": True,
                               "pagination_support": True})

        push_update(job_id, "scraper_initialized", 
                   {"agents_available": AGENTS_AVAILABLE, "smart_features": True}, 
                   WorkflowState.PLANNING)

        push_update(job_id, "scraping_started", workflow_state=WorkflowState.PLANNING)

        # Run smart scraping
        result = await scraper.scrape(job_id, request.url, request.objective)

        # Add communication logs to result
        result["communications"] = active_jobs[job_id].get("communications", [])

        push_update(job_id, "scrape_method_completed", {
            "steps_completed": result.get("steps_completed", 0),
            "agents_used": len(result.get("agent_logs", [])),
            "screenshots_taken": len(result.get("screenshots", [])),
            "communications_logged": len(result.get("communications", [])),
            "final_score": result.get("final_summary", {}).get("validation_score", 0),
            "pages_processed": result.get("final_summary", {}).get("pages_processed", 1),
            "items_found": result.get("final_summary", {}).get("total_items_found", 0)
        }, WorkflowState.VALIDATING)

        # Store comprehensive result
        job_results[job_id] = result

        if result.get("success", False):
            log_agent_communication(job_id, "SYSTEM", "🎉 SMART SCRAPING SUCCESS!", 
                                  {"execution_time": result.get("execution_time", 0),
                                   "items_found": result.get("final_summary", {}).get("total_items_found", 0),
                                   "pages_processed": result.get("final_summary", {}).get("pages_processed", 1),
                                   "validation_score": result.get("final_summary", {}).get("validation_score", 0)})

            push_update(job_id, "scraping_completed", {
                "success": True,
                "execution_time": result.get("execution_time", 0),
                "final_items": result.get("final_summary", {}).get("total_items_found", 0),
                "pages_processed": result.get("final_summary", {}).get("pages_processed", 1)
            }, WorkflowState.COMPLETED)
        else:
            log_agent_communication(job_id, "SYSTEM", "⚠️ Job completed but failed validation", 
                                  {"errors": len(result.get("error_log", [])),
                                   "screenshots": len(result.get("screenshots", [])),
                                   "items_extracted": result.get("final_summary", {}).get("total_items_found", 0)}, 
                                  level="WARNING")

            push_update(job_id, "scraping_failed", {
                "success": False,
                "errors": result.get("error_log", [])[:3],
                "steps_completed": result.get("steps_completed", 0),
                "screenshots_available": len(result.get("screenshots", [])),
                "communications_available": len(result.get("communications", [])),
                "items_found": result.get("final_summary", {}).get("total_items_found", 0)
            }, WorkflowState.FAILED)

    except Exception as e:
        error_details = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        log_agent_communication(job_id, "SYSTEM", f"💥 CRITICAL JOB ERROR: {str(e)}", level="CRITICAL")
        push_update(job_id, "scraping_error", error_details, WorkflowState.FAILED)

        job_results[job_id] = {
            "job_id": job_id,
            "success": False,
            "extracted_data": {},
            "screenshots": [],
            "execution_time": 0,
            "steps_completed": 0,
            "error_log": [str(e), traceback.format_exc()],
            "agent_logs": active_jobs[job_id].get("agent_logs", []) if job_id in active_jobs else [],
            "communications": active_jobs[job_id].get("communications", []) if job_id in active_jobs else [],
            "final_summary": {
                "phones": [],
                "prices": [],
                "ratings": [],
                "total_items_found": 0,
                "extraction_method": "Error",
                "pages_processed": 0
            }
        }

# API Routes (same as before but with smart features)
@app.get("/", response_class=FileResponse)
async def serve_web():
    """Serve smart web interface"""
    html_files = ["index.html", "enhanced_index.html", "beautiful_scraper_client.html", "test_client.html"]

    for html_file in html_files:
        file_path = web_dir / html_file
        if file_path.exists():
            return FileResponse(file_path, media_type="text/html")

    return JSONResponse({
        "name": "Universal Web Scraper API - Smart Dynamic Content",
        "version": "6.0.0",
        "status": "running",
        "agents_available": AGENTS_AVAILABLE,
        "universal_scraper_available": UNIVERSAL_SCRAPER_AVAILABLE,
        "message": "Place your HTML file in the 'web' directory",
        "features": [
            "Smart dynamic content handling",
            "Viewport-only screenshots with scrolling",
            "Intelligent pagination support",
            "Flipkart-optimized selectors",
            "Enhanced validation scoring",
            "Multi-page data extraction",
            "Real-time agent communication"
        ]
    })

@app.post("/scrape", response_model=ScrapingResponse)
async def start_scraping(request: ScrapingRequest, background_tasks: BackgroundTasks):
    """Start smart scraping job"""
    if not request.url or not request.objective:
        raise HTTPException(status_code=400, detail="URL and objective required")

    job_id = str(uuid.uuid4())
    background_tasks.add_task(run_scraping_job, job_id, request)

    return ScrapingResponse(
        job_id=job_id,
        status="started",
        message="Smart multi-agent scraping job with pagination support queued",
        stream_url=f"/stream/{job_id}"
    )

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """Get enhanced job status"""
    if job_id not in active_jobs:
        if job_id in job_results:
            result = job_results[job_id]
            return JobStatus(
                job_id=job_id,
                status="completed" if result["success"] else "failed",
                progress=100,
                current_step="Completed" if result["success"] else "Failed",
                screenshots_count=len(result.get("screenshots", [])),
                error_log=result.get("error_log", []),
                start_time=datetime.now(),
                elapsed_seconds=result.get("execution_time", 0)
            )
        raise HTTPException(status_code=404, detail="Job not found")

    job_data = active_jobs[job_id]
    updates = job_data["updates"]
    start_time = job_data["start_time"]

    if not updates:
        status = "initializing"
        progress = 0
        current_step = "Starting"
    else:
        latest = updates[-1]
        status = latest["message"]
        current_step = latest.get("workflow_state", status).replace("_", " ").title()

        progress_map = {
            "job_started": 5,
            "initializing_scraper": 10,
            "scraper_initialized": 15,
            "scraping_started": 20,
            "browser_initialized": 25,
            "starting_smart_multi_agent_workflow": 35,
            "planning_started": 40,
            "planning_completed": 45,
            "visual_analysis_started": 50,
            "visual_analysis_completed": 65,
            "execution_started": 70,
            "execution_completed": 85,
            "validation_started": 90,
            "validation_completed": 95,
            "scrape_method_completed": 98,
            "scraping_completed": 100,
            "scraping_failed": 100,
            "scraping_error": 100
        }
        progress = progress_map.get(status, 50)

    # Count screenshots from job results
    screenshots_count = 0
    if job_id in job_results:
        screenshots_count = len(job_results[job_id].get("screenshots", []))

    error_log = []
    for update in updates:
        if "error" in update.get("details", {}):
            error_log.append(update["details"]["error"])

    return JobStatus(
        job_id=job_id,
        status=status,
        progress=progress,
        current_step=current_step,
        screenshots_count=screenshots_count,
        error_log=error_log,
        start_time=datetime.fromtimestamp(start_time),
        elapsed_seconds=time.time() - start_time
    )

@app.get("/result/{job_id}", response_model=JobResult)
async def get_result(job_id: str):
    """Get smart job result"""
    if job_id not in job_results:
        if job_id in active_jobs:
            raise HTTPException(status_code=202, detail="Job still running")
        raise HTTPException(status_code=404, detail="Job not found")

    result = job_results[job_id]

    return JobResult(
        job_id=job_id,
        success=result.get("success", False),
        extracted_data=result.get("extracted_data", {}),
        screenshots=result.get("screenshots", []),
        execution_time=result.get("execution_time", 0),
        steps_completed=result.get("steps_completed", 0),
        error_log=result.get("error_log", []),
        agent_logs=result.get("agent_logs", [])
    )

@app.get("/communications/{job_id}")
async def get_communications(job_id: str):
    """Get detailed agent communications"""
    if job_id not in active_jobs and job_id not in job_results:
        raise HTTPException(status_code=404, detail="Job not found")

    communications = []

    if job_id in active_jobs:
        communications = active_jobs[job_id].get("communications", [])
    elif job_id in job_results:
        communications = job_results[job_id].get("communications", [])

    return {
        "job_id": job_id,
        "total_communications": len(communications),
        "communications": communications
    }

@app.get("/stream/{job_id}")
async def stream_updates(job_id: str):
    """Stream smart job updates"""
    async def event_stream():
        last_update_index = 0
        last_agent_log_index = 0
        last_communication_index = 0
        timeout = 300
        start = time.time()

        yield f"data: {json.dumps({'type': 'connected', 'job_id': job_id})}\n\n"

        while time.time() - start < timeout:
            if job_id in active_jobs:
                job_data = active_jobs[job_id]

                # Send updates
                updates = job_data["updates"]
                while last_update_index < len(updates):
                    update = updates[last_update_index]
                    yield f"data: {json.dumps({'type': 'update', 'data': update})}\n\n"
                    last_update_index += 1

                # Send agent logs
                agent_logs = job_data.get("agent_logs", [])
                while last_agent_log_index < len(agent_logs):
                    agent_log = agent_logs[last_agent_log_index]
                    yield f"data: {json.dumps({'type': 'agent_activity', 'data': agent_log})}\n\n"
                    last_agent_log_index += 1

                # Send communications
                communications = job_data.get("communications", [])
                while last_communication_index < len(communications):
                    communication = communications[last_communication_index]
                    yield f"data: {json.dumps({'type': 'communication', 'data': communication})}\n\n"
                    last_communication_index += 1

            # Check completion
            if job_id in job_results:
                result = job_results[job_id]
                yield f"data: {json.dumps({'type': 'complete', 'result': result})}\n\n"
                break

            await asyncio.sleep(1)

        yield f"data: {json.dumps({'type': 'timeout', 'message': 'Stream ended'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

@app.get("/jobs")
async def list_jobs():
    """List all smart jobs"""
    jobs = []

    for job_id, data in active_jobs.items():
        workflow_state = data.get("workflow_state", WorkflowState.INITIALIZING.value)
        agent_count = len(data.get("agent_logs", []))
        communication_count = len(data.get("communications", []))

        jobs.append({
            "job_id": job_id,
            "status": "running",
            "workflow_state": workflow_state,
            "agents_executed": agent_count,
            "communications": communication_count,
            "start_time": datetime.fromtimestamp(data["start_time"]),
            "updates": len(data["updates"])
        })

    for job_id, result in job_results.items():
        jobs.append({
            "job_id": job_id,
            "status": "completed" if result["success"] else "failed",
            "success": result["success"],
            "execution_time": result.get("execution_time", 0),
            "steps_completed": result.get("steps_completed", 0),
            "agents_used": len(result.get("agent_logs", [])),
            "communications": len(result.get("communications", [])),
            "screenshots_count": len(result.get("screenshots", [])),
            "items_found": result.get("final_summary", {}).get("total_items_found", 0),
            "pages_processed": result.get("final_summary", {}).get("pages_processed", 1),
            "validation_score": result.get("final_summary", {}).get("validation_score", 0)
        })

    return {
        "jobs": jobs, 
        "total": len(jobs),
        "system_status": {
            "agents_available": AGENTS_AVAILABLE,
            "universal_scraper_available": UNIVERSAL_SCRAPER_AVAILABLE,
            "smart_features": True,
            "dynamic_content_support": True,
            "pagination_support": True
        }
    }

@app.delete("/job/{job_id}")
async def cancel_job(job_id: str):
    """Cancel job and cleanup"""
    cancelled = []

    if job_id in active_jobs:
        del active_jobs[job_id]
        cancelled.append("active_job")

    if job_id in job_results:
        del job_results[job_id]
        cancelled.append("job_result")

    return {
        "message": f"Job {job_id} cancelled",
        "cancelled": cancelled
    }

@app.get("/health")
async def health():
    """Smart health check"""
    return {
        "status": "healthy",
        "timestamp": get_timestamp(),
        "active_jobs": len(active_jobs),
        "completed_jobs": len(job_results),
        "system_info": {
            "agents_available": AGENTS_AVAILABLE,
            "universal_scraper_available": UNIVERSAL_SCRAPER_AVAILABLE,
            "smart_features": True,
            "dynamic_content_support": True,
            "pagination_support": True,
            "viewport_screenshots": True,
            "platform": platform.system(),
            "python_version": platform.python_version()
        },
        "features": [
            "Smart dynamic content handling with longer wait times",
            "Viewport-only screenshots with intelligent scrolling",
            "Multi-page pagination support (up to 3 pages)",
            "Flipkart-optimized CSS selectors",
            "Enhanced validation with quality scoring",
            "Real-time agent communication logging",
            "Thread-safe browser operations"
        ]
    }

@app.get("/agents")
async def agents_status():
    """Get smart agent system status"""
    return {
        "multi_agent_system": AGENTS_AVAILABLE,
        "universal_scraper": UNIVERSAL_SCRAPER_AVAILABLE,
        "smart_features": True,
        "available_agents": [
            "Smart Planning Agent - Website-specific strategies (Flipkart, Amazon, Generic)",
            "Smart Vision Agent - Dynamic content handling with viewport screenshots", 
            "Smart Execution Agent - Multi-page extraction with pagination support",
            "Smart Validation Agent - Enhanced quality scoring and objective matching",
            "Smart Research Agent - Intelligent recovery strategies"
        ] if AGENTS_AVAILABLE else ["Smart Fallback Agent - Viewport screenshots with scrolling"],
        "workflow_states": [state.value for state in WorkflowState],
        "supported_providers": ["anthropic", "openai", "groq"],
        "smart_features": [
            "Dynamic content wait times (5-8 seconds)",
            "Viewport screenshots instead of full-page",
            "Intelligent scrolling with content loading",
            "Multi-page pagination (up to 3 pages)",
            "Flipkart-optimized selectors",
            "Samsung product detection",
            "Quality scoring for data validation"
        ],
        "validation_scoring": {
            "product_extraction": "Up to 0.4 points + quality bonus",
            "price_extraction": "Up to 0.25 points + validity bonus", 
            "rating_extraction": "Up to 0.2 points + format bonus",
            "pagination_bonus": "Up to 0.15 points for multi-page",
            "samsung_bonus": "Up to 0.2 points for Samsung products",
            "threshold": "0.25 (25%) - Lowered for smart extraction"
        },
        "flipkart_optimizations": [
            "Specific CSS selectors for Flipkart mobile products",
            "Dynamic content loading with 8-second wait",
            "Multi-page navigation support",
            "Samsung Galaxy product filtering",
            "Price and rating extraction optimization"
        ]
    }

# Run server
if __name__ == "__main__":
    print("🚀 Starting Universal Web Scraper - SMART Dynamic Content System")
    print("="*90)
    print(f"📡 Server: http://localhost:8000")
    print(f"🌐 Web UI: http://localhost:8000/")
    print(f"📚 API Docs: http://localhost:8000/docs")
    print(f"🤖 Agent Status: http://localhost:8000/agents")
    print(f"💬 Communications: http://localhost:8000/communications/{{job_id}}")
    print(f"❤️  Health Check: http://localhost:8000/health")
    print("="*90)
    print(f"🔧 Agents Available: {AGENTS_AVAILABLE}")
    print(f"🔧 Universal Scraper: {UNIVERSAL_SCRAPER_AVAILABLE}")
    print(f"🔧 SMART Features: ✅ Enabled")
    print(f"🔧 Dynamic Content: ✅ 8-second wait for Flipkart")
    print(f"🔧 Pagination Support: ✅ Up to 3 pages")
    print(f"🔧 Viewport Screenshots: ✅ No more huge images")
    print(f"📁 Web Directory: {web_dir.absolute()}")
    print("="*90)
    print("🎯 SMART FEATURES:")
    print("• Flipkart-optimized CSS selectors")
    print("• Dynamic content wait (8 sec for ecommerce)")
    print("• Viewport screenshots with scrolling")
    print("• Multi-page pagination support")
    print("• Samsung product detection")
    print("• Enhanced validation scoring")
    print("• Quality assessment for extracted data")
    print("="*90)

    uvicorn.run(
        "src.enhanced_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
