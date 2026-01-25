"""
Example: Invoice Processing Workflow

Automates invoice download, extraction, and organization.
Demonstrates multi-step workflow with file handling and data extraction.

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


async def login_to_portal(browser: FlyBrowser, credentials: dict):
    """
    Log in to vendor/invoice portal.
    
    Args:
        browser: FlyBrowser instance
        credentials: Login credentials
    """
    # Store credentials securely
    await browser.store_credential("portal_user", credentials["username"])
    await browser.store_credential("portal_pass", credentials["password"])
    
    # Fill login form
    await browser.secure_fill(
        "username_field",
        "portal_user",
        selector="input[name='username'], input[name='email'], #username"
    )
    await browser.secure_fill(
        "password_field",
        "portal_pass",
        selector="input[type='password'], #password"
    )
    
    # Submit login
    await browser.act("Click the login or sign in button")
    await asyncio.sleep(2)
    
    # Verify login success
    login_check = await browser.extract(
        "Am I logged in? Look for account name, logout button, or dashboard content."
    )
    return login_check.success and login_check.data


async def navigate_to_invoices(browser: FlyBrowser):
    """Navigate to the invoices section."""
    # Use agent for flexible navigation
    result = await browser.agent(
        task="Navigate to the invoices or billing section",
        context="""
        Look for links or menu items like:
        - Invoices
        - Billing
        - Statements
        - Account > Invoices
        - Billing History
        
        Click to navigate there.
        """
    )
    return result.success


async def extract_invoice_list(browser: FlyBrowser) -> list[dict]:
    """
    Extract list of available invoices.
    
    Returns:
        List of invoice metadata
    """
    invoices = await browser.extract(
        "What invoices are listed on this page? "
        "For each invoice, get: invoice number, date, amount, and status.",
        schema={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "invoice_number": {"type": "string"},
                    "date": {"type": "string"},
                    "amount": {"type": "string"},
                    "status": {"type": "string"},
                    "download_available": {"type": "boolean"}
                }
            }
        }
    )
    
    if invoices.success and invoices.data:
        return invoices.data
    return []


async def download_invoice(browser: FlyBrowser, invoice_number: str, output_dir: Path):
    """
    Download a specific invoice.
    
    Args:
        browser: FlyBrowser instance
        invoice_number: Invoice to download
        output_dir: Directory to save downloads
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Click download for specific invoice
    result = await browser.act(
        f"Click the download or PDF button for invoice {invoice_number}"
    )
    
    if result.success:
        await asyncio.sleep(2)  # Wait for download
        print(f"Downloaded invoice: {invoice_number}")
        return True
    return False


async def process_invoice_batch(
    portal_url: str,
    credentials: dict,
    output_dir: str = "invoices"
) -> dict:
    """
    Process batch of invoices from a portal.
    
    Args:
        portal_url: URL of the invoice portal
        credentials: Login credentials
        output_dir: Directory for downloads
        
    Returns:
        Processing summary
    """
    results = {
        "processed_at": datetime.now().isoformat(),
        "portal": portal_url,
        "invoices_found": 0,
        "invoices_downloaded": 0,
        "invoice_data": [],
        "errors": []
    }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        await browser.goto(portal_url)
        
        # Step 1: Login
        print("Logging in...")
        if not await login_to_portal(browser, credentials):
            results["errors"].append("Login failed")
            return results
        
        # Step 2: Navigate to invoices
        print("Navigating to invoices...")
        if not await navigate_to_invoices(browser):
            results["errors"].append("Could not find invoices section")
            return results
        
        # Step 3: Extract invoice list
        print("Extracting invoice list...")
        invoices = await extract_invoice_list(browser)
        results["invoices_found"] = len(invoices)
        results["invoice_data"] = invoices
        
        # Step 4: Download each invoice
        print(f"Found {len(invoices)} invoices")
        for invoice in invoices:
            inv_num = invoice.get("invoice_number", "unknown")
            print(f"  Processing: {inv_num}")
            
            try:
                if await download_invoice(browser, inv_num, output_path):
                    results["invoices_downloaded"] += 1
            except Exception as e:
                results["errors"].append(f"Failed to download {inv_num}: {str(e)}")
    
    # Save summary
    summary_path = output_path / "processing_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nProcessing complete:")
    print(f"  Found: {results['invoices_found']} invoices")
    print(f"  Downloaded: {results['invoices_downloaded']} invoices")
    print(f"  Errors: {len(results['errors'])}")
    
    return results


async def extract_invoice_details(browser: FlyBrowser, invoice_url: str) -> dict:
    """
    Extract detailed information from an invoice page.
    
    Args:
        browser: FlyBrowser instance
        invoice_url: URL of the invoice detail page
        
    Returns:
        Extracted invoice details
    """
    await browser.goto(invoice_url)
    
    details = await browser.extract(
        "Extract all details from this invoice",
        schema={
            "type": "object",
            "properties": {
                "invoice_number": {"type": "string"},
                "invoice_date": {"type": "string"},
                "due_date": {"type": "string"},
                "vendor": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {"type": "string"},
                        "contact": {"type": "string"}
                    }
                },
                "bill_to": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {"type": "string"}
                    }
                },
                "line_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "quantity": {"type": "number"},
                            "unit_price": {"type": "number"},
                            "amount": {"type": "number"}
                        }
                    }
                },
                "subtotal": {"type": "number"},
                "tax": {"type": "number"},
                "total": {"type": "number"},
                "payment_terms": {"type": "string"},
                "notes": {"type": "string"}
            }
        }
    )
    
    return details.data if details.success else {}


async def reconcile_invoices(
    portal_url: str,
    credentials: dict,
    expected_invoices: list[dict]
) -> dict:
    """
    Reconcile portal invoices against expected list.
    
    Args:
        portal_url: URL of the invoice portal
        credentials: Login credentials
        expected_invoices: List of expected invoices with amounts
        
    Returns:
        Reconciliation report
    """
    report = {
        "matched": [],
        "missing": [],
        "unexpected": [],
        "amount_mismatches": []
    }
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        await browser.goto(portal_url)
        await login_to_portal(browser, credentials)
        await navigate_to_invoices(browser)
        
        portal_invoices = await extract_invoice_list(browser)
        
        # Create lookup by invoice number
        portal_lookup = {
            inv.get("invoice_number"): inv
            for inv in portal_invoices
        }
        expected_lookup = {
            inv.get("invoice_number"): inv
            for inv in expected_invoices
        }
        
        # Check each expected invoice
        for inv_num, expected in expected_lookup.items():
            if inv_num in portal_lookup:
                portal_inv = portal_lookup[inv_num]
                
                # Compare amounts
                if portal_inv.get("amount") == expected.get("amount"):
                    report["matched"].append(inv_num)
                else:
                    report["amount_mismatches"].append({
                        "invoice_number": inv_num,
                        "expected": expected.get("amount"),
                        "actual": portal_inv.get("amount")
                    })
            else:
                report["missing"].append(inv_num)
        
        # Check for unexpected invoices
        for inv_num in portal_lookup:
            if inv_num not in expected_lookup:
                report["unexpected"].append(inv_num)
    
    return report


async def main():
    """Main entry point for invoice processing."""
    print("=" * 60)
    print("Invoice Processing Workflow Examples")
    print("=" * 60)
    
    # Example configuration (replace with real values)
    portal_url = "https://example-invoice-portal.com"
    credentials = {
        "username": os.getenv("PORTAL_USERNAME", "demo@example.com"),
        "password": os.getenv("PORTAL_PASSWORD", "demo123")
    }
    
    # Process invoices
    print("\n--- Processing Invoice Batch ---")
    results = await process_invoice_batch(
        portal_url=portal_url,
        credentials=credentials,
        output_dir="downloaded_invoices"
    )
    
    print(f"\nResults: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
