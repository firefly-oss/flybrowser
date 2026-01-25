"""
Example: Data Synchronization Workflow

Automates data synchronization between web applications.
Demonstrates reading from source, transforming, and updating target systems.

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


async def login_to_system(browser: FlyBrowser, url: str, credentials: dict) -> bool:
    """
    Log in to a web application.
    
    Args:
        browser: FlyBrowser instance
        url: Login page URL
        credentials: Username and password
        
    Returns:
        True if login successful
    """
    await browser.goto(url)
    
    # Use agent for flexible login handling
    result = await browser.agent(
        task="Log in to this application",
        context=f"""
        Credentials:
        - Username/Email: {credentials['username']}
        - Password: {credentials['password']}
        
        Find the login form, enter credentials, and submit.
        Handle any 2FA prompts if they appear.
        """
    )
    
    return result.success


async def extract_records(browser: FlyBrowser, list_url: str, schema: dict) -> list[dict]:
    """
    Extract records from a listing page.
    
    Args:
        browser: FlyBrowser instance
        list_url: URL of the listing page
        schema: JSON schema for extraction
        
    Returns:
        List of extracted records
    """
    await browser.goto(list_url)
    
    all_records = []
    page = 1
    
    while True:
        print(f"  Extracting page {page}...")
        
        # Extract records from current page
        result = await browser.extract(
            "Extract all records/items from this listing",
            schema={
                "type": "array",
                "items": schema
            }
        )
        
        if result.success and result.data:
            all_records.extend(result.data)
        else:
            break
        
        # Check for next page
        next_btn = await browser.observe("Find a 'Next' or pagination button to go to the next page")
        if next_btn.success and next_btn.elements:
            await browser.act("Click the next page button")
            await asyncio.sleep(1)
            page += 1
            
            # Safety limit
            if page > 10:
                print("  Reached page limit")
                break
        else:
            break
    
    return all_records


async def update_record(browser: FlyBrowser, record: dict, field_mapping: dict) -> bool:
    """
    Update a single record in target system.
    
    Args:
        browser: FlyBrowser instance
        record: Record data to update
        field_mapping: Map of record fields to form fields
        
    Returns:
        True if update successful
    """
    # Build update instructions
    update_context = "Fill the form with these values:\n"
    for record_field, form_field in field_mapping.items():
        if record_field in record:
            update_context += f"- {form_field}: {record[record_field]}\n"
    
    result = await browser.agent(
        task="Update this record with the provided data",
        context=update_context + "\nThen save/submit the changes."
    )
    
    return result.success


async def sync_contacts(
    source_url: str,
    source_creds: dict,
    target_url: str,
    target_creds: dict
) -> dict:
    """
    Sync contacts between two CRM systems.
    
    Args:
        source_url: Source CRM URL
        source_creds: Source credentials
        target_url: Target CRM URL
        target_creds: Target credentials
        
    Returns:
        Sync results
    """
    results = {
        "started_at": datetime.now().isoformat(),
        "source_records": 0,
        "created": 0,
        "updated": 0,
        "skipped": 0,
        "errors": []
    }
    
    contact_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "email": {"type": "string"},
            "phone": {"type": "string"},
            "company": {"type": "string"},
            "title": {"type": "string"},
            "last_modified": {"type": "string"}
        }
    }
    
    # Extract from source
    print("Connecting to source system...")
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as source_browser:
        if not await login_to_system(source_browser, source_url, source_creds):
            results["errors"].append("Failed to login to source system")
            return results
        
        print("Extracting contacts from source...")
        source_contacts = await extract_records(
            source_browser,
            f"{source_url}/contacts",
            contact_schema
        )
        results["source_records"] = len(source_contacts)
    
    print(f"Found {len(source_contacts)} contacts in source")
    
    # Import to target
    print("\nConnecting to target system...")
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as target_browser:
        if not await login_to_system(target_browser, target_url, target_creds):
            results["errors"].append("Failed to login to target system")
            return results
        
        print("Syncing contacts to target...")
        for i, contact in enumerate(source_contacts):
            print(f"  Processing {i+1}/{len(source_contacts)}: {contact.get('name', 'Unknown')}")
            
            try:
                # Search for existing contact
                await target_browser.goto(f"{target_url}/contacts")
                await target_browser.act(f"Search for contact with email {contact.get('email', '')}")
                
                existing = await target_browser.extract(
                    f"Is there a contact with email {contact.get('email')}?",
                    schema={"type": "object", "properties": {"found": {"type": "boolean"}}}
                )
                
                if existing.data and existing.data.get("found"):
                    # Update existing
                    await target_browser.act(f"Click on the contact with email {contact.get('email')}")
                    await target_browser.act("Click edit button")
                    
                    if await update_record(target_browser, contact, {
                        "name": "Name field",
                        "phone": "Phone field",
                        "company": "Company field",
                        "title": "Title field"
                    }):
                        results["updated"] += 1
                    else:
                        results["errors"].append(f"Failed to update: {contact.get('email')}")
                else:
                    # Create new
                    await target_browser.act("Click create new contact button")
                    
                    if await update_record(target_browser, contact, {
                        "name": "Name field",
                        "email": "Email field",
                        "phone": "Phone field",
                        "company": "Company field",
                        "title": "Title field"
                    }):
                        results["created"] += 1
                    else:
                        results["errors"].append(f"Failed to create: {contact.get('email')}")
                        
            except Exception as e:
                results["errors"].append(f"Error with {contact.get('email')}: {str(e)}")
    
    results["completed_at"] = datetime.now().isoformat()
    return results


async def sync_inventory(
    source_url: str,
    source_creds: dict,
    target_url: str,
    target_creds: dict
) -> dict:
    """
    Sync inventory data between systems.
    
    Args:
        source_url: Source inventory system URL
        source_creds: Source credentials
        target_url: Target system URL
        target_creds: Target credentials
        
    Returns:
        Sync results
    """
    results = {
        "started_at": datetime.now().isoformat(),
        "items_synced": 0,
        "items_failed": 0,
        "errors": []
    }
    
    inventory_schema = {
        "type": "object",
        "properties": {
            "sku": {"type": "string"},
            "name": {"type": "string"},
            "quantity": {"type": "number"},
            "location": {"type": "string"},
            "last_updated": {"type": "string"}
        }
    }
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        # Login to source and extract
        print("Extracting from source...")
        await login_to_system(browser, source_url, source_creds)
        
        inventory = await extract_records(
            browser,
            f"{source_url}/inventory",
            inventory_schema
        )
        
        print(f"Found {len(inventory)} inventory items")
        
        # Login to target and update
        print("\nUpdating target system...")
        await login_to_system(browser, target_url, target_creds)
        
        for item in inventory:
            try:
                await browser.goto(f"{target_url}/inventory")
                await browser.act(f"Search for SKU {item.get('sku')}")
                
                # Update quantity
                await browser.act(f"Click on item with SKU {item.get('sku')}")
                await browser.act("Click edit")
                await browser.act(f"Set quantity to {item.get('quantity')}")
                await browser.act("Save changes")
                
                results["items_synced"] += 1
                
            except Exception as e:
                results["items_failed"] += 1
                results["errors"].append(f"Failed {item.get('sku')}: {str(e)}")
    
    results["completed_at"] = datetime.now().isoformat()
    return results


async def bidirectional_sync(
    system_a_url: str,
    system_a_creds: dict,
    system_b_url: str,
    system_b_creds: dict,
    entity_type: str,
    schema: dict,
    key_field: str
) -> dict:
    """
    Perform bidirectional sync based on last modified time.
    
    Args:
        system_a_url: System A URL
        system_a_creds: System A credentials
        system_b_url: System B URL
        system_b_creds: System B credentials
        entity_type: Type of entity to sync (e.g., "contacts")
        schema: JSON schema for entities
        key_field: Field to use as unique key
        
    Returns:
        Sync results
    """
    results = {
        "a_to_b": 0,
        "b_to_a": 0,
        "conflicts": [],
        "errors": []
    }
    
    # Extract from both systems
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        # Get records from A
        await login_to_system(browser, system_a_url, system_a_creds)
        records_a = await extract_records(browser, f"{system_a_url}/{entity_type}", schema)
        
        # Get records from B
        await login_to_system(browser, system_b_url, system_b_creds)
        records_b = await extract_records(browser, f"{system_b_url}/{entity_type}", schema)
    
    # Build lookup tables
    lookup_a = {r.get(key_field): r for r in records_a if r.get(key_field)}
    lookup_b = {r.get(key_field): r for r in records_b if r.get(key_field)}
    
    # Find records to sync
    all_keys = set(lookup_a.keys()) | set(lookup_b.keys())
    
    for key in all_keys:
        record_a = lookup_a.get(key)
        record_b = lookup_b.get(key)
        
        if record_a and not record_b:
            # Only in A, sync to B
            results["a_to_b"] += 1
        elif record_b and not record_a:
            # Only in B, sync to A
            results["b_to_a"] += 1
        elif record_a and record_b:
            # In both, compare timestamps
            mod_a = record_a.get("last_modified", "")
            mod_b = record_b.get("last_modified", "")
            
            if mod_a != mod_b:
                results["conflicts"].append({
                    "key": key,
                    "a_modified": mod_a,
                    "b_modified": mod_b
                })
    
    return results


async def export_to_csv(browser: FlyBrowser, list_url: str, output_path: str):
    """
    Export data from a web app to CSV using built-in export.
    
    Args:
        browser: FlyBrowser instance
        list_url: URL of the listing page
        output_path: Path for the CSV file
    """
    await browser.goto(list_url)
    
    # Try to find and click export button
    result = await browser.agent(
        task="Export this data to CSV",
        context="""
        Look for an export button, download button, or 'Export to CSV' option.
        This might be in a menu, toolbar, or actions dropdown.
        Click it to start the export.
        """
    )
    
    if result.success:
        await asyncio.sleep(3)  # Wait for download
        print(f"Export initiated to: {output_path}")
    else:
        print("Could not find export option")


async def main():
    """Main entry point for data sync examples."""
    print("=" * 60)
    print("Data Synchronization Workflow Examples")
    print("=" * 60)
    
    # Example configuration
    source_config = {
        "url": os.getenv("SOURCE_URL", "https://source-crm.example.com"),
        "credentials": {
            "username": os.getenv("SOURCE_USER", "admin"),
            "password": os.getenv("SOURCE_PASS", "password")
        }
    }
    
    target_config = {
        "url": os.getenv("TARGET_URL", "https://target-crm.example.com"),
        "credentials": {
            "username": os.getenv("TARGET_USER", "admin"),
            "password": os.getenv("TARGET_PASS", "password")
        }
    }
    
    # Run contact sync
    print("\n--- Syncing Contacts ---")
    results = await sync_contacts(
        source_url=source_config["url"],
        source_creds=source_config["credentials"],
        target_url=target_config["url"],
        target_creds=target_config["credentials"]
    )
    
    print(f"\nSync Results:")
    print(f"  Source records: {results['source_records']}")
    print(f"  Created: {results['created']}")
    print(f"  Updated: {results['updated']}")
    print(f"  Errors: {len(results['errors'])}")


if __name__ == "__main__":
    asyncio.run(main())
