"""
Example: Report Generation Workflow

Automates report generation by collecting data from multiple sources.
Demonstrates data aggregation, transformation, and output generation.

Prerequisites:
- pip install flybrowser
- export OPENAI_API_KEY="sk-..."
"""

import asyncio
import csv
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from flybrowser import FlyBrowser


async def collect_analytics_data(browser: FlyBrowser, analytics_url: str) -> dict:
    """
    Collect analytics data from dashboard.
    
    Args:
        browser: FlyBrowser instance
        analytics_url: URL of analytics dashboard
        
    Returns:
        Analytics metrics
    """
    await browser.goto(analytics_url)
    
    # Extract key metrics
    metrics = await browser.extract(
        "Extract all key metrics visible on this analytics dashboard",
        schema={
            "type": "object",
            "properties": {
                "total_visits": {"type": "number"},
                "unique_visitors": {"type": "number"},
                "page_views": {"type": "number"},
                "bounce_rate": {"type": "string"},
                "avg_session_duration": {"type": "string"},
                "top_pages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "page": {"type": "string"},
                            "views": {"type": "number"}
                        }
                    }
                },
                "traffic_sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "percentage": {"type": "string"}
                        }
                    }
                }
            }
        }
    )
    
    return metrics.data if metrics.success else {}


async def collect_sales_data(browser: FlyBrowser, sales_url: str) -> dict:
    """
    Collect sales data from dashboard.
    
    Args:
        browser: FlyBrowser instance
        sales_url: URL of sales dashboard
        
    Returns:
        Sales metrics
    """
    await browser.goto(sales_url)
    
    # Set date range if possible
    await browser.act("Set the date range to last 30 days if a date picker is available")
    await asyncio.sleep(1)
    
    # Extract sales data
    sales = await browser.extract(
        "Extract all sales metrics and data from this dashboard",
        schema={
            "type": "object",
            "properties": {
                "total_revenue": {"type": "string"},
                "total_orders": {"type": "number"},
                "average_order_value": {"type": "string"},
                "conversion_rate": {"type": "string"},
                "top_products": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "product": {"type": "string"},
                            "units_sold": {"type": "number"},
                            "revenue": {"type": "string"}
                        }
                    }
                },
                "sales_by_region": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "region": {"type": "string"},
                            "revenue": {"type": "string"},
                            "orders": {"type": "number"}
                        }
                    }
                }
            }
        }
    )
    
    return sales.data if sales.success else {}


async def collect_support_data(browser: FlyBrowser, support_url: str) -> dict:
    """
    Collect customer support metrics.
    
    Args:
        browser: FlyBrowser instance
        support_url: URL of support dashboard
        
    Returns:
        Support metrics
    """
    await browser.goto(support_url)
    
    support = await browser.extract(
        "Extract customer support metrics from this dashboard",
        schema={
            "type": "object",
            "properties": {
                "total_tickets": {"type": "number"},
                "open_tickets": {"type": "number"},
                "resolved_tickets": {"type": "number"},
                "avg_resolution_time": {"type": "string"},
                "customer_satisfaction": {"type": "string"},
                "tickets_by_category": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string"},
                            "count": {"type": "number"}
                        }
                    }
                }
            }
        }
    )
    
    return support.data if support.success else {}


def generate_markdown_report(data: dict, output_path: Path):
    """
    Generate markdown report from collected data.
    
    Args:
        data: Aggregated data from all sources
        output_path: Path to save the report
    """
    report = f"""# Business Report
Generated: {data['generated_at']}
Report Period: {data['period']}

## Executive Summary
This report provides an overview of key business metrics across analytics, 
sales, and customer support.

## Website Analytics
"""
    
    analytics = data.get('analytics', {})
    if analytics:
        report += f"""
- **Total Visits**: {analytics.get('total_visits', 'N/A')}
- **Unique Visitors**: {analytics.get('unique_visitors', 'N/A')}
- **Page Views**: {analytics.get('page_views', 'N/A')}
- **Bounce Rate**: {analytics.get('bounce_rate', 'N/A')}
- **Avg Session Duration**: {analytics.get('avg_session_duration', 'N/A')}

### Top Pages
"""
        for page in analytics.get('top_pages', [])[:5]:
            report += f"- {page.get('page', 'Unknown')}: {page.get('views', 0)} views\n"
    
    report += "\n## Sales Performance\n"
    sales = data.get('sales', {})
    if sales:
        report += f"""
- **Total Revenue**: {sales.get('total_revenue', 'N/A')}
- **Total Orders**: {sales.get('total_orders', 'N/A')}
- **Average Order Value**: {sales.get('average_order_value', 'N/A')}
- **Conversion Rate**: {sales.get('conversion_rate', 'N/A')}

### Top Products
"""
        for product in sales.get('top_products', [])[:5]:
            report += f"- {product.get('product', 'Unknown')}: {product.get('units_sold', 0)} units ({product.get('revenue', 'N/A')})\n"
    
    report += "\n## Customer Support\n"
    support = data.get('support', {})
    if support:
        report += f"""
- **Total Tickets**: {support.get('total_tickets', 'N/A')}
- **Open Tickets**: {support.get('open_tickets', 'N/A')}
- **Resolved Tickets**: {support.get('resolved_tickets', 'N/A')}
- **Avg Resolution Time**: {support.get('avg_resolution_time', 'N/A')}
- **Customer Satisfaction**: {support.get('customer_satisfaction', 'N/A')}
"""
    
    # Write report
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Markdown report saved to: {output_path}")


def generate_csv_export(data: dict, output_dir: Path):
    """
    Export data to CSV files.
    
    Args:
        data: Aggregated data
        output_dir: Directory for CSV files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export top pages
    if 'analytics' in data and data['analytics'].get('top_pages'):
        with open(output_dir / 'top_pages.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['page', 'views'])
            writer.writeheader()
            writer.writerows(data['analytics']['top_pages'])
    
    # Export top products
    if 'sales' in data and data['sales'].get('top_products'):
        with open(output_dir / 'top_products.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['product', 'units_sold', 'revenue'])
            writer.writeheader()
            writer.writerows(data['sales']['top_products'])
    
    # Export support categories
    if 'support' in data and data['support'].get('tickets_by_category'):
        with open(output_dir / 'support_categories.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['category', 'count'])
            writer.writeheader()
            writer.writerows(data['support']['tickets_by_category'])
    
    print(f"CSV files exported to: {output_dir}")


async def generate_business_report(
    analytics_url: str,
    sales_url: str,
    support_url: str,
    output_dir: str = "reports"
) -> dict:
    """
    Generate comprehensive business report from multiple sources.
    
    Args:
        analytics_url: URL of analytics dashboard
        sales_url: URL of sales dashboard
        support_url: URL of support dashboard
        output_dir: Directory for output files
        
    Returns:
        Aggregated report data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_data = {
        "generated_at": datetime.now().isoformat(),
        "period": f"{(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}",
        "analytics": {},
        "sales": {},
        "support": {}
    }
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        # Collect analytics
        print("Collecting analytics data...")
        report_data["analytics"] = await collect_analytics_data(browser, analytics_url)
        
        # Collect sales
        print("Collecting sales data...")
        report_data["sales"] = await collect_sales_data(browser, sales_url)
        
        # Collect support metrics
        print("Collecting support data...")
        report_data["support"] = await collect_support_data(browser, support_url)
    
    # Generate outputs
    print("\nGenerating reports...")
    
    # JSON export
    json_path = output_path / f"report_{datetime.now().strftime('%Y%m%d')}.json"
    with open(json_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    print(f"JSON report: {json_path}")
    
    # Markdown report
    md_path = output_path / f"report_{datetime.now().strftime('%Y%m%d')}.md"
    generate_markdown_report(report_data, md_path)
    
    # CSV exports
    csv_dir = output_path / "csv"
    generate_csv_export(report_data, csv_dir)
    
    return report_data


async def scheduled_report(
    sources: dict,
    output_dir: str,
    schedule_name: str = "daily"
):
    """
    Generate scheduled report (for use with task schedulers).
    
    Args:
        sources: Dictionary of source URLs
        output_dir: Output directory
        schedule_name: Name of the schedule
    """
    print(f"Starting {schedule_name} report generation...")
    start_time = datetime.now()
    
    try:
        report = await generate_business_report(
            analytics_url=sources.get("analytics", ""),
            sales_url=sources.get("sales", ""),
            support_url=sources.get("support", ""),
            output_dir=f"{output_dir}/{schedule_name}"
        )
        
        # Log success
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\n{schedule_name.title()} report completed in {duration:.1f}s")
        
        return {
            "status": "success",
            "schedule": schedule_name,
            "duration_seconds": duration,
            "output_dir": f"{output_dir}/{schedule_name}"
        }
        
    except Exception as e:
        print(f"Report generation failed: {e}")
        return {
            "status": "error",
            "schedule": schedule_name,
            "error": str(e)
        }


async def compare_reports(report1_path: str, report2_path: str) -> dict:
    """
    Compare two reports to identify changes.
    
    Args:
        report1_path: Path to first report JSON
        report2_path: Path to second report JSON
        
    Returns:
        Comparison results
    """
    with open(report1_path) as f:
        report1 = json.load(f)
    with open(report2_path) as f:
        report2 = json.load(f)
    
    comparison = {
        "period1": report1.get("period"),
        "period2": report2.get("period"),
        "changes": {}
    }
    
    # Compare analytics
    if 'analytics' in report1 and 'analytics' in report2:
        a1, a2 = report1['analytics'], report2['analytics']
        comparison['changes']['analytics'] = {
            "visits_change": (a2.get('total_visits', 0) or 0) - (a1.get('total_visits', 0) or 0),
            "visitors_change": (a2.get('unique_visitors', 0) or 0) - (a1.get('unique_visitors', 0) or 0),
        }
    
    # Compare sales
    if 'sales' in report1 and 'sales' in report2:
        s1, s2 = report1['sales'], report2['sales']
        comparison['changes']['sales'] = {
            "orders_change": (s2.get('total_orders', 0) or 0) - (s1.get('total_orders', 0) or 0),
        }
    
    # Compare support
    if 'support' in report1 and 'support' in report2:
        sup1, sup2 = report1['support'], report2['support']
        comparison['changes']['support'] = {
            "tickets_change": (sup2.get('total_tickets', 0) or 0) - (sup1.get('total_tickets', 0) or 0),
            "open_change": (sup2.get('open_tickets', 0) or 0) - (sup1.get('open_tickets', 0) or 0),
        }
    
    return comparison


async def main():
    """Main entry point for report generation."""
    print("=" * 60)
    print("Report Generation Workflow Examples")
    print("=" * 60)
    
    # Example URLs (replace with real dashboards)
    sources = {
        "analytics": os.getenv("ANALYTICS_URL", "https://analytics.example.com"),
        "sales": os.getenv("SALES_URL", "https://sales.example.com"),
        "support": os.getenv("SUPPORT_URL", "https://support.example.com")
    }
    
    # Generate comprehensive report
    print("\n--- Generating Business Report ---")
    report = await generate_business_report(
        analytics_url=sources["analytics"],
        sales_url=sources["sales"],
        support_url=sources["support"],
        output_dir="reports"
    )
    
    print("\n--- Report Summary ---")
    print(f"Analytics collected: {bool(report.get('analytics'))}")
    print(f"Sales collected: {bool(report.get('sales'))}")
    print(f"Support collected: {bool(report.get('support'))}")


if __name__ == "__main__":
    asyncio.run(main())
