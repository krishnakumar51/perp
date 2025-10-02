
#!/usr/bin/env python3
"""
Universal Web Scraper - Main Entry Point

This file serves as the primary interface for the universal web scraper.
It can be used as both a CLI tool and programmatic interface.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.universal_scraper import UniversalWebScraper, ScraperConfig
from src.enhanced_llm import LLMProvider
from src.config import validate_config

def create_scraper_config(args) -> ScraperConfig:
    """Create scraper configuration from CLI arguments"""
    config = ScraperConfig()

    # Set provider
    if args.provider:
        config.provider = LLMProvider(args.provider)

    # Set other options
    config.max_steps = args.max_steps
    config.max_retries = args.max_retries
    config.timeout_seconds = args.timeout
    config.headless = not args.show_browser
    config.enable_learning = args.enable_learning

    return config

async def run_scraper_cli(url: str, objective: str, config: ScraperConfig) -> Dict[str, Any]:
    """Run scraper in CLI mode with progress updates"""
    print(f"🚀 Starting Universal Web Scraper")
    print(f"📱 Target URL: {url}")
    print(f"🎯 Objective: {objective}")
    print(f"🤖 Provider: {config.provider.value}")
    print(f"⚙️  Max Steps: {config.max_steps}")
    print("-" * 60)

    try:
        scraper = UniversalWebScraper(config)
        result = scraper.scrape(url, objective)

        if result.get("success", False):
            print("✅ Scraping completed successfully!")
            print(f"⏱️  Execution time: {result.get('execution_time', 0):.2f} seconds")
            print(f"📊 Steps completed: {result.get('steps_completed', 0)}")
            print(f"📸 Screenshots taken: {len(result.get('screenshots', []))}")

            # Display extracted data
            if result.get("extracted_data"):
                print("\n📋 Extracted Data:")
                print(json.dumps(result["extracted_data"], indent=2))

        else:
            print("❌ Scraping failed!")
            print(f"Error: {result.get('error', 'Unknown error')}")

            if result.get("error_log"):
                print("\nError Log:")
                for error in result["error_log"]:
                    print(f"  - {error}")

        return result

    except KeyboardInterrupt:
        print("\n⚠️  Scraping interrupted by user")
        return {"success": False, "error": "Interrupted by user"}
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return {"success": False, "error": str(e)}

def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the FastAPI server"""
    import uvicorn
    print(f"🌐 Starting Universal Web Scraper Server")
    print(f"📡 Server will be available at: http://{host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"🔄 Auto-reload: {'Enabled' if reload else 'Disabled'}")

    uvicorn.run(
        "enhanced_server_fixed:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

def main():
    """Main entry point with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description="Universal Web Scraper - AI-powered scraping for any website",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape with CLI
  python main.py scrape "https://example.com" "Find contact information"

  # Start server
  python main.py server --port 8000

  # Use different LLM provider
  python main.py scrape "https://shop.com" "Find top products" --provider openai

  # Show browser (non-headless)
  python main.py scrape "https://site.com" "Extract data" --show-browser
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Scrape command
    scrape_parser = subparsers.add_parser('scrape', help='Run scraper on a URL')
    scrape_parser.add_argument('url', help='Target URL to scrape')
    scrape_parser.add_argument('objective', help='Description of what to extract')
    scrape_parser.add_argument('--provider', choices=['anthropic', 'openai', 'groq'], 
                              default='anthropic', help='LLM provider to use')
    scrape_parser.add_argument('--max-steps', type=int, default=50, 
                              help='Maximum steps to execute')
    scrape_parser.add_argument('--max-retries', type=int, default=3,
                              help='Maximum retry attempts')
    scrape_parser.add_argument('--timeout', type=int, default=30,
                              help='Timeout in seconds')
    scrape_parser.add_argument('--show-browser', action='store_true',
                              help='Show browser window (non-headless)')
    scrape_parser.add_argument('--enable-learning', action='store_true', default=True,
                              help='Enable learning from interactions')
    scrape_parser.add_argument('--output', type=str,
                              help='Output file for results (JSON format)')

    # Server command
    server_parser = subparsers.add_parser('server', help='Start the FastAPI server')
    server_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    server_parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    server_parser.add_argument('--reload', action='store_true', 
                              help='Enable auto-reload for development')

    # Config command
    config_parser = subparsers.add_parser('config', help='Show configuration status')

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Validate configuration
    if not validate_config():
        print("❌ Configuration validation failed!")
        sys.exit(1)

    if args.command == 'scrape':
        # Run scraper
        config = create_scraper_config(args)
        result = asyncio.run(run_scraper_cli(args.url, args.objective, config))

        # Save output if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"📁 Results saved to: {args.output}")

        # Exit with appropriate code
        sys.exit(0 if result.get("success", False) else 1)

    elif args.command == 'server':
        # Start server
        start_server(args.host, args.port, args.reload)

    elif args.command == 'config':
        # Show configuration status
        print("📋 Configuration Status:")
        print("-" * 40)

        from config import (
            ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY,
            anthropic_client, openai_client, groq_client,
            SCREENSHOTS_DIR, LOGS_DIR
        )

        print(f"🔧 Anthropic: {'✅ Available' if anthropic_client else '❌ Not configured'}")
        print(f"🔧 OpenAI: {'✅ Available' if openai_client else '❌ Not configured'}")
        print(f"🔧 Groq: {'✅ Available' if groq_client else '❌ Not configured'}")
        print(f"📂 Screenshots: {SCREENSHOTS_DIR}")
        print(f"📋 Logs: {LOGS_DIR}")

        if not any([anthropic_client, openai_client, groq_client]):
            print("\n⚠️  No LLM providers configured!")
            print("Set at least one API key in environment:")
            print("  export ANTHROPIC_API_KEY=your_key")
            print("  export OPENAI_API_KEY=your_key") 
            print("  export GROQ_API_KEY=your_key")

if __name__ == "__main__":
    main()
