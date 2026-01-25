"""
Example: Competitor Research & Data Gathering

Demonstrates autonomous research and data extraction for competitive analysis.
This matches the "Research & Data Gathering" example from the README Autonomous Mode section.

Prerequisites:
- pip install flybrowser
- export OPENAI_API_KEY="sk-..."
"""

import asyncio
import json
import os
from datetime import datetime
from flybrowser import FlyBrowser


async def research_competitors(
    industry: str,
    focus_area: str,
    num_competitors: int = 5
):
    """
    Research competitors and gather their pricing information.
    
    Args:
        industry: Industry to research
        focus_area: Specific focus area
        num_competitors: Number of competitors to research
        
    Returns:
        Research results with competitor data
    """
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
    ) as browser:
        await browser.goto("https://google.com")
        
        # Use agent for complex research task
        result = await browser.agent(
            task=f"Research the top {num_competitors} competitors in the {industry} space and gather their pricing info",
            context={
                "industry": industry,
                "focus": focus_area,
                "competitors_to_find": num_competitors,
                "data_to_collect": [
                    "company_name",
                    "pricing_tiers",
                    "features",
                    "target_market",
                    "pricing_model"
                ]
            },
            max_iterations=50,
            max_time_seconds=600
        )
        
        if result.success:
            print(f"Research complete: {result.data}")
            result.pprint()  # Shows execution summary, LLM usage, costs
            
            # Save research results
            output = {
                "research_date": datetime.now().isoformat(),
                "industry": industry,
                "focus": focus_area,
                "competitors": result.data,
                "llm_usage": {
                    "tokens": result.llm_usage.total_tokens,
                    "cost_usd": result.llm_usage.cost_usd
                }
            }
            
            filename = f"competitor_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(output, f, indent=2)
            
            print(f"\nResearch saved to: {filename}")
        else:
            print(f"Research failed: {result.error}")
        
        return result


async def gather_product_information(product_category: str, num_products: int = 10):
    """Research products in a specific category."""
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        log_verbosity="verbose",
    ) as browser:
        await browser.goto("https://google.com")
        
        result = await browser.agent(
            task=f"Find and analyze the top {num_products} {product_category} products",
            context={
                "category": product_category,
                "num_products": num_products,
                "data_points": [
                    "product_name",
                    "manufacturer",
                    "price",
                    "key_features",
                    "user_rating",
                    "review_count",
                    "availability"
                ]
            },
            max_iterations=60
        )
        
        if result.success:
            print(f"\n=== Product Research Complete ===")
            print(f"Found {len(result.data)} products")
            print(f"\nExecution:")
            print(f"  Iterations: {result.execution.iterations}")
            print(f"  Duration: {result.execution.duration_seconds:.1f}s")
            print(f"\nLLM Usage:")
            print(f"  Tokens: {result.llm_usage.total_tokens:,}")
            print(f"  Cost: ${result.llm_usage.cost_usd:.4f}")
        
        return result


async def market_analysis(market_segment: str):
    """Perform comprehensive market analysis."""
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
    ) as browser:
        await browser.goto("https://google.com")
        
        result = await browser.agent(
            task=f"Conduct market analysis for {market_segment}",
            context={
                "market": market_segment,
                "analysis_areas": [
                    "market_size",
                    "growth_rate",
                    "key_players",
                    "market_share",
                    "trends",
                    "challenges",
                    "opportunities"
                ],
                "sources_to_check": [
                    "industry_reports",
                    "company_websites",
                    "news_articles",
                    "market_research_firms"
                ]
            },
            max_iterations=70,
            max_time_seconds=900
        )
        
        return result


async def compare_saas_pricing():
    """Compare SaaS pricing models (specific example)."""
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
    ) as browser:
        # Research specific SaaS companies
        companies = [
            "Salesforce",
            "HubSpot",
            "Zoho CRM",
            "Pipedrive",
            "Freshsales"
        ]
        
        all_pricing = []
        
        for company in companies:
            print(f"\nResearching {company} pricing...")
            
            # Search for pricing page
            await browser.goto("https://google.com")
            await browser.act(f"Search for '{company} pricing'")
            await asyncio.sleep(2)
            
            # Click on official pricing page
            await browser.act(f"Click on the official {company} pricing page link")
            await asyncio.sleep(2)
            
            # Extract pricing information
            pricing = await browser.extract(
                f"Extract all pricing tiers for {company} with plan names, prices, and key features",
                schema={
                    "type": "object",
                    "properties": {
                        "company": {"type": "string"},
                        "pricing_tiers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "plan_name": {"type": "string"},
                                    "price": {"type": "string"},
                                    "billing_period": {"type": "string"},
                                    "key_features": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            )
            
            if pricing.success:
                all_pricing.append(pricing.data)
                print(f"  Found {len(pricing.data.get('pricing_tiers', []))} pricing tiers")
        
        # Save comparison
        output = {
            "research_date": datetime.now().isoformat(),
            "companies_researched": len(companies),
            "pricing_data": all_pricing
        }
        
        filename = f"saas_pricing_comparison_{datetime.now().strftime('%Y%m%d')}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n\nPricing comparison saved to: {filename}")
        return output


async def main():
    """Main entry point for research examples."""
    print("=" * 60)
    print("Competitor Research & Data Gathering Examples")
    print("=" * 60)
    
    # Example 1: CRM competitor research (matching README)
    print("\n--- Example 1: CRM Competitor Research ---")
    result = await research_competitors(
        industry="CRM software",
        focus_area="small business",
        num_competitors=5
    )
    
    # Example 2: Product research
    print("\n--- Example 2: Product Category Research ---")
    # await gather_product_information("wireless headphones", num_products=10)
    
    # Example 3: SaaS pricing comparison
    print("\n--- Example 3: SaaS Pricing Comparison ---")
    # await compare_saas_pricing()
    
    # Print summary
    if result and result.success:
        print("\n=== Research Summary ===")
        print(f"Success: {result.success}")
        print(f"Competitors found: {len(result.data) if isinstance(result.data, list) else 1}")
        print(f"Duration: {result.execution.duration_seconds:.1f}s")
        print(f"Cost: ${result.llm_usage.cost_usd:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
