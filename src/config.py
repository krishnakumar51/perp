
import os
from pathlib import Path
from dotenv import load_dotenv
import anthropic
import openai
from groq import Groq

# Load environment variables
load_dotenv()

# Project directories
BASE_DIR = Path(__file__).parent
SCREENSHOTS_DIR = BASE_DIR / "screenshots"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
SCREENSHOTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# API Configuration - Anthropic as default
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model configurations
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"  # Fast and cost-effective for scraping
OPENAI_MODEL = "gpt-4o-mini"  # Good balance of capability and cost
GROQ_MODEL = "openai/gpt-oss-120b"  # Fast inference

# Default LLM provider (Anthropic as recommended)
DEFAULT_LLM_PROVIDER = "anthropic"

# Initialize API clients
anthropic_client = None
openai_client = None
groq_client = None

# Initialize Anthropic client (primary)
if ANTHROPIC_API_KEY:
    try:
        anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        print("✅ Anthropic client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Anthropic client: {e}")

# Initialize OpenAI client (fallback)
if OPENAI_API_KEY:
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        print("✅ OpenAI client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize OpenAI client: {e}")

# Initialize Groq client (fast inference option)
if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Groq client: {e}")

# Browser settings
BROWSER_CONFIG = {
    "headless": True,
    "viewport": {"width": 1920, "height": 1080},
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "timeout": 30000,  # 30 seconds
    "navigation_timeout": 60000,  # 60 seconds for navigation
}

# Scraping settings
SCRAPING_CONFIG = {
    "max_retries": 3,
    "retry_delay": 2.0,  # seconds
    "screenshot_quality": 80,  # JPEG quality 0-100
    "max_screenshots": 50,  # limit per job
    "element_wait_timeout": 10000,  # 10 seconds
    "page_load_timeout": 30000,  # 30 seconds
}

# Agent settings
AGENT_CONFIG = {
    "max_steps": 50,
    "planning_timeout": 60,  # seconds
    "vision_analysis_timeout": 30,  # seconds
    "execution_timeout": 45,  # seconds
    "validation_timeout": 20,  # seconds
    "research_timeout": 30,  # seconds
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "scraper.log",
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
}

# Rate limiting
RATE_LIMITS = {
    "requests_per_minute": 60,
    "concurrent_jobs": 5,
    "screenshots_per_minute": 120,
    "llm_calls_per_minute": 100,
}

# Feature flags
FEATURES = {
    "enable_caching": True,
    "enable_learning": True,
    "enable_monitoring": True,
    "enable_debug_screenshots": False,
    "enable_error_screenshots": True,
}

# Validation
def validate_config():
    """Validate configuration and warn about missing settings"""
    warnings = []

    if not anthropic_client and not openai_client and not groq_client:
        warnings.append("⚠️  No LLM clients available - set at least one API key")

    if not ANTHROPIC_API_KEY:
        warnings.append("⚠️  ANTHROPIC_API_KEY not set - Anthropic (recommended) unavailable")

    if not SCREENSHOTS_DIR.exists():
        warnings.append("⚠️  Screenshots directory not accessible")

    if warnings:
        print("\n".join(warnings))
        return False

    print("✅ Configuration validated successfully")
    return True

# Auto-validate on import
validate_config()
