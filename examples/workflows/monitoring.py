"""
Example: Monitoring Workflow

Automates website and service monitoring with alerting.
Demonstrates health checks, uptime monitoring, and anomaly detection.

Prerequisites:
- pip install flybrowser
- export OPENAI_API_KEY="sk-..."
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from flybrowser import FlyBrowser


async def check_site_health(browser: FlyBrowser, url: str) -> dict:
    """
    Check health of a website.
    
    Args:
        browser: FlyBrowser instance
        url: URL to check
        
    Returns:
        Health check results
    """
    start_time = datetime.now()
    
    try:
        await browser.goto(url)
        load_time = (datetime.now() - start_time).total_seconds()
        
        # Check for common error indicators
        status = await browser.extract(
            "Is this page loading correctly? "
            "Look for: error messages, 404/500 errors, maintenance notices, "
            "or blank content. Return status and any issues found.",
            schema={
                "type": "object",
                "properties": {
                    "is_healthy": {"type": "boolean"},
                    "status_code": {"type": "string"},
                    "issues": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
        )
        
        return {
            "url": url,
            "timestamp": datetime.now().isoformat(),
            "reachable": True,
            "load_time_seconds": round(load_time, 2),
            "is_healthy": status.data.get("is_healthy", False) if status.success else False,
            "issues": status.data.get("issues", []) if status.success else ["Could not analyze page"]
        }
        
    except Exception as e:
        return {
            "url": url,
            "timestamp": datetime.now().isoformat(),
            "reachable": False,
            "load_time_seconds": None,
            "is_healthy": False,
            "issues": [str(e)]
        }


async def check_element_presence(browser: FlyBrowser, url: str, elements: list[str]) -> dict:
    """
    Verify specific elements are present on a page.
    
    Args:
        browser: FlyBrowser instance
        url: URL to check
        elements: List of element descriptions to verify
        
    Returns:
        Element check results
    """
    await browser.goto(url)
    
    results = {
        "url": url,
        "timestamp": datetime.now().isoformat(),
        "elements": []
    }
    
    for element in elements:
        check = await browser.observe(f"Find: {element}")
        results["elements"].append({
            "description": element,
            "found": check.success and bool(check.elements)
        })
    
    results["all_present"] = all(e["found"] for e in results["elements"])
    return results


async def check_form_functionality(browser: FlyBrowser, url: str, form_test: dict) -> dict:
    """
    Test form submission functionality.
    
    Args:
        browser: FlyBrowser instance
        url: URL of the form page
        form_test: Test configuration with field values
        
    Returns:
        Form test results
    """
    await browser.goto(url)
    
    results = {
        "url": url,
        "timestamp": datetime.now().isoformat(),
        "form_found": False,
        "submission_successful": False,
        "errors": []
    }
    
    # Find form
    form = await browser.observe(form_test.get("form_selector", "Find the main form on this page"))
    results["form_found"] = form.success and bool(form.elements)
    
    if not results["form_found"]:
        results["errors"].append("Form not found")
        return results
    
    # Fill form with test data
    for field_name, value in form_test.get("fields", {}).items():
        try:
            await browser.act(f"Fill the {field_name} field with '{value}'")
        except Exception as e:
            results["errors"].append(f"Failed to fill {field_name}: {str(e)}")
    
    # Submit form
    try:
        await browser.act("Submit the form")
        await asyncio.sleep(2)
        
        # Check for success or error
        outcome = await browser.extract(
            "Was the form submitted successfully? "
            "Look for success messages, error messages, or validation errors.",
            schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"}
                }
            }
        )
        
        results["submission_successful"] = outcome.data.get("success", False) if outcome.success else False
        if outcome.data.get("message"):
            results["response_message"] = outcome.data["message"]
            
    except Exception as e:
        results["errors"].append(f"Submission failed: {str(e)}")
    
    return results


async def monitor_price_changes(browser: FlyBrowser, url: str, product_selector: str) -> dict:
    """
    Monitor price changes on a product page.
    
    Args:
        browser: FlyBrowser instance
        url: Product page URL
        product_selector: Selector or description for the product
        
    Returns:
        Price monitoring results
    """
    await browser.goto(url)
    
    price_info = await browser.extract(
        f"What is the current price of {product_selector}? "
        "Include regular price, sale price if applicable, and availability.",
        schema={
            "type": "object",
            "properties": {
                "product_name": {"type": "string"},
                "current_price": {"type": "string"},
                "original_price": {"type": "string"},
                "is_on_sale": {"type": "boolean"},
                "in_stock": {"type": "boolean"},
                "currency": {"type": "string"}
            }
        }
    )
    
    return {
        "url": url,
        "timestamp": datetime.now().isoformat(),
        "price_data": price_info.data if price_info.success else None
    }


async def run_monitoring_suite(sites: list[dict], output_dir: str = "monitoring"):
    """
    Run comprehensive monitoring on multiple sites.
    
    Args:
        sites: List of site configurations
        output_dir: Directory for results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        "run_timestamp": datetime.now().isoformat(),
        "sites": []
    }
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        for site in sites:
            print(f"Checking: {site['url']}")
            
            site_result = {
                "url": site["url"],
                "name": site.get("name", site["url"]),
                "checks": {}
            }
            
            # Health check
            if site.get("health_check", True):
                site_result["checks"]["health"] = await check_site_health(browser, site["url"])
            
            # Element presence
            if site.get("required_elements"):
                site_result["checks"]["elements"] = await check_element_presence(
                    browser, site["url"], site["required_elements"]
                )
            
            # Form test
            if site.get("form_test"):
                site_result["checks"]["form"] = await check_form_functionality(
                    browser, site["url"], site["form_test"]
                )
            
            results["sites"].append(site_result)
    
    # Save results
    results_file = output_path / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate summary
    print("\n=== Monitoring Summary ===")
    for site in results["sites"]:
        health = site["checks"].get("health", {})
        status = "HEALTHY" if health.get("is_healthy") else "UNHEALTHY"
        print(f"  [{status}] {site['name']}")
        if health.get("load_time_seconds"):
            print(f"           Load time: {health['load_time_seconds']}s")
        if health.get("issues"):
            for issue in health["issues"]:
                print(f"           Issue: {issue}")
    
    return results


async def continuous_monitoring(
    sites: list[dict],
    interval_seconds: int = 300,
    duration_hours: int = 24
):
    """
    Run continuous monitoring with periodic checks.
    
    Args:
        sites: List of site configurations
        interval_seconds: Time between checks
        duration_hours: Total monitoring duration
    """
    end_time = datetime.now().timestamp() + (duration_hours * 3600)
    check_count = 0
    
    print(f"Starting continuous monitoring for {duration_hours} hours")
    print(f"Checking {len(sites)} sites every {interval_seconds} seconds")
    
    while datetime.now().timestamp() < end_time:
        check_count += 1
        print(f"\n--- Check #{check_count} at {datetime.now().isoformat()} ---")
        
        results = await run_monitoring_suite(sites, f"monitoring/run_{check_count}")
        
        # Check for alerts
        for site in results["sites"]:
            health = site["checks"].get("health", {})
            if not health.get("is_healthy"):
                print(f"ALERT: {site['name']} is unhealthy!")
                # Here you could add notification logic (email, Slack, etc.)
        
        await asyncio.sleep(interval_seconds)


async def check_api_endpoint(url: str, expected_status: int = 200) -> dict:
    """
    Check API endpoint health.
    
    Args:
        url: API endpoint URL
        expected_status: Expected HTTP status code
        
    Returns:
        Check results
    """
    import aiohttp
    
    start = datetime.now()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=30) as response:
                response_time = (datetime.now() - start).total_seconds()
                
                return {
                    "url": url,
                    "timestamp": datetime.now().isoformat(),
                    "status_code": response.status,
                    "expected_status": expected_status,
                    "is_healthy": response.status == expected_status,
                    "response_time_seconds": round(response_time, 3)
                }
    except Exception as e:
        return {
            "url": url,
            "timestamp": datetime.now().isoformat(),
            "status_code": None,
            "expected_status": expected_status,
            "is_healthy": False,
            "error": str(e)
        }


async def visual_regression_check(browser: FlyBrowser, url: str, baseline_path: str) -> dict:
    """
    Perform visual regression check against baseline.
    
    Args:
        browser: FlyBrowser instance
        url: URL to check
        baseline_path: Path to baseline screenshot
        
    Returns:
        Visual check results
    """
    await browser.goto(url)
    await asyncio.sleep(2)  # Wait for page to stabilize
    
    # Capture current screenshot
    screenshot = await browser.screenshot()
    
    # Compare using vision
    comparison = await browser.extract(
        f"Compare the current page visually. "
        f"Are there any significant visual differences or issues? "
        f"Look for broken layouts, missing images, text alignment issues, "
        f"or any visual anomalies.",
        use_vision=True,
        schema={
            "type": "object",
            "properties": {
                "has_issues": {"type": "boolean"},
                "issues_found": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "overall_status": {"type": "string"}
            }
        }
    )
    
    return {
        "url": url,
        "timestamp": datetime.now().isoformat(),
        "screenshot_captured": True,
        "visual_issues": comparison.data.get("has_issues", False) if comparison.success else None,
        "issues": comparison.data.get("issues_found", []) if comparison.success else []
    }


async def main():
    """Main entry point for monitoring examples."""
    print("=" * 60)
    print("Website Monitoring Workflow Examples")
    print("=" * 60)
    
    # Example monitoring configuration
    sites_to_monitor = [
        {
            "name": "Hacker News",
            "url": "https://news.ycombinator.com",
            "health_check": True,
            "required_elements": [
                "top navigation",
                "list of stories",
                "points/score for stories"
            ]
        },
        {
            "name": "Example.com",
            "url": "https://example.com",
            "health_check": True,
            "required_elements": [
                "main heading",
                "paragraph text",
                "link to more information"
            ]
        }
    ]
    
    # Run monitoring suite
    print("\n--- Running Monitoring Suite ---")
    results = await run_monitoring_suite(sites_to_monitor)
    
    # Print detailed results
    print("\n--- Detailed Results ---")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
