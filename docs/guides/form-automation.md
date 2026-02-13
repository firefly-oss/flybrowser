# Form Automation Guide

This guide covers techniques for filling out and submitting forms with FlyBrowser. You will learn how to handle various form elements, validate inputs, and build robust form automation workflows.

## Basic Form Filling

### Text Fields

Fill text inputs using natural language:

```python
import asyncio
from flybrowser import FlyBrowser

async def main():
    async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
        await browser.goto("https://example.com/signup")
        
        # Fill by field label
        await browser.act("type 'John Doe' in the Name field")
        
        # Fill by placeholder
        await browser.act("enter 'john@example.com' in the email input")
        
        # Fill by field type
        await browser.act("type 'secretpassword123' in the password field")

asyncio.run(main())
```

### Identifying Fields

FlyBrowser finds form fields using:

1. Associated `<label>` elements
2. Placeholder text
3. `name` and `id` attributes
4. ARIA labels and descriptions
5. Visual proximity to labels
6. Field type (`type="email"`, `type="password"`, etc.)

### Clearing Fields First

When fields have existing content:

```python
# Clear before typing
await browser.act("clear the search field and type 'new search term'")

# Or explicitly
await browser.act("select all text in the name field")
await browser.act("type 'New Name'")
```

## Complex Form Elements

### Dropdown Menus

Standard HTML `<select>` elements:

```python
# Select by visible text
await browser.act("select 'California' from the State dropdown")

# Select by description
await browser.act("choose the third option in the country selector")

# Select with partial matching
await browser.act("select the option containing 'United States'")
```

### Custom Dropdowns

Many modern sites use custom dropdown implementations:

```python
# Open the dropdown first
await browser.act("click the country selector")

# Wait for options to appear
await asyncio.sleep(0.5)

# Then select the option
await browser.act("click 'United States' in the dropdown list")
```

### Checkboxes

```python
# Check a checkbox
await browser.act("check the 'I agree to terms' checkbox")

# Uncheck
await browser.act("uncheck the 'Subscribe to newsletter' checkbox")

# Toggle
await browser.act("toggle the 'Remember me' checkbox")

# Check if currently checked
result = await browser.extract("is the newsletter checkbox checked?")
```

### Radio Buttons

```python
# Select a radio option
await browser.act("select 'Express Shipping' from the shipping options")

# Select by position
await browser.act("choose the second payment method option")

# Select by description
await browser.act("select the free shipping radio button")
```

### Date Pickers

Date pickers vary widely in implementation:

```python
# Simple date inputs
await browser.act("enter '2026-02-13' in the date field")

# Calendar pickers - open first
await browser.act("click the date picker")
await browser.act("select February 13, 2026 from the calendar")

# Complex date pickers
await browser.act("set the check-in date to February 13, 2026")
```

### File Uploads

```python
# Standard file input
await browser.act("upload '/path/to/document.pdf' to the file input")

# Named uploads
await browser.act("upload 'resume.pdf' to the Resume upload field")
```

### Rich Text Editors

For WYSIWYG editors:

```python
# Click into the editor first
await browser.act("click in the message editor")

# Then type
await browser.act("type 'Hello, this is my message...'")

# Some editors need special handling
await browser.act("enter the following text in the rich text editor: 'My formatted content'")
```

## Multi-Step Forms

### Wizard-Style Forms

```python
async def fill_wizard_form(browser):
    await browser.goto("https://example.com/signup")
    
    # Step 1: Personal Info
    await browser.act("type 'John Doe' in the full name field")
    await browser.act("type 'john@example.com' in the email field")
    await browser.act("click Next or Continue")
    
    # Step 2: Address
    await browser.act("type '123 Main St' in the address field")
    await browser.act("type 'New York' in the city field")
    await browser.act("select 'NY' from the state dropdown")
    await browser.act("type '10001' in the zip code field")
    await browser.act("click Next or Continue")
    
    # Step 3: Payment
    await browser.act("type '4111111111111111' in the card number field")
    await browser.act("type '12/25' in the expiration field")
    await browser.act("type '123' in the CVV field")
    await browser.act("click Submit or Complete")
```

### Handling Form Progress

```python
async def fill_form_with_progress(browser, url, form_data):
    await browser.goto(url)
    
    for field_name, value in form_data.items():
        result = await browser.act(f"type '{value}' in the {field_name} field")
        
        if not result.success:
            print(f"Failed to fill {field_name}: {result.error}")
            # Try alternative approach
            await browser.act(f"find the field labeled {field_name} and enter '{value}'")
```

## Form Validation

### Client-Side Validation

```python
async def fill_with_validation(browser):
    await browser.goto("https://example.com/contact")
    
    # Fill email field
    await browser.act("type 'not-an-email' in the email field")
    
    # Check for validation error
    result = await browser.extract("is there an error message about the email?")
    
    if result.data:
        # Fix the error
        await browser.act("clear the email field and type 'valid@email.com'")
```

### Form Submission Errors

```python
async def submit_with_error_handling(browser):
    await browser.goto("https://example.com/login")
    
    await browser.act("type 'user@example.com' in the email field")
    await browser.act("type 'wrongpassword' in the password field")
    await browser.act("click the Login button")
    
    # Wait for response
    await asyncio.sleep(2)
    
    # Check for errors
    result = await browser.extract("is there a login error message?")
    
    if result.data:
        print(f"Login failed: {result.data}")
        return False
    
    return True
```

## Secure Form Handling

### Using Credentials Securely

FlyBrowser provides secure credential handling:

```python
async with FlyBrowser(...) as browser:
    await browser.goto("https://bank.example.com/login")
    
    # Store credential securely
    await browser.store_credential(
        name="bank_password",
        value="my-secret-password",
        pii_type="password"
    )
    
    # Use secure fill (credential never appears in logs)
    await browser.act("type 'username' in the username field")
    await browser.secure_fill("#password", "bank_password")
    
    await browser.act("click Login")
```

### PII Masking

Mask sensitive data in screenshots and logs:

```python
async with FlyBrowser(pii_masking_enabled=True, ...) as browser:
    await browser.goto("https://example.com/checkout")
    
    # PII will be masked in logs and screenshots
    await browser.act("type '4111111111111111' in the card number field")
    
    # Take screenshot with masked data
    screenshot = await browser.screenshot(mask_pii=True)
```

## Common Form Patterns

### Registration Form

```python
async def complete_registration(browser, user_data):
    """Complete a typical registration form."""
    await browser.goto("https://example.com/register")
    
    # Personal info
    await browser.act(f"type '{user_data['first_name']}' in the First Name field")
    await browser.act(f"type '{user_data['last_name']}' in the Last Name field")
    await browser.act(f"type '{user_data['email']}' in the Email field")
    
    # Password
    await browser.act(f"type '{user_data['password']}' in the Password field")
    await browser.act(f"type '{user_data['password']}' in the Confirm Password field")
    
    # Optional fields
    if user_data.get('phone'):
        await browser.act(f"type '{user_data['phone']}' in the Phone field")
    
    # Terms acceptance
    await browser.act("check the 'I agree to terms' checkbox")
    
    # Submit
    result = await browser.act("click the Register or Sign Up button")
    
    return result.success
```

### Contact Form

```python
async def submit_contact_form(browser, message_data):
    """Submit a contact form."""
    await browser.goto("https://example.com/contact")
    
    await browser.act(f"type '{message_data['name']}' in the Name field")
    await browser.act(f"type '{message_data['email']}' in the Email field")
    
    if message_data.get('subject'):
        await browser.act(f"type '{message_data['subject']}' in the Subject field")
    
    await browser.act(f"type '{message_data['message']}' in the Message field")
    
    # Handle captcha if present
    captcha_result = await browser.extract("is there a captcha on this page?")
    if captcha_result.data:
        print("Warning: Captcha detected, may need human intervention")
    
    await browser.act("click Send or Submit")
```

### Checkout Form

```python
async def complete_checkout(browser, checkout_data):
    """Complete an e-commerce checkout."""
    
    # Shipping information
    await browser.act(f"type '{checkout_data['shipping']['name']}' in the full name field")
    await browser.act(f"type '{checkout_data['shipping']['address']}' in the address field")
    await browser.act(f"type '{checkout_data['shipping']['city']}' in the city field")
    await browser.act(f"select '{checkout_data['shipping']['state']}' from the state dropdown")
    await browser.act(f"type '{checkout_data['shipping']['zip']}' in the zip code field")
    
    # Same billing address?
    if checkout_data.get('same_billing', True):
        await browser.act("check the 'Same as shipping' checkbox")
    else:
        # Fill billing address
        await browser.act("click to expand billing address section")
        # ... fill billing fields
    
    # Payment
    await browser.act(f"type '{checkout_data['payment']['card_number']}' in the card number field")
    await browser.act(f"type '{checkout_data['payment']['expiry']}' in the expiration field")
    await browser.act(f"type '{checkout_data['payment']['cvv']}' in the CVV field")
    
    # Review and submit
    await browser.act("click Place Order or Complete Purchase")
    
    # Verify success
    await asyncio.sleep(3)
    result = await browser.extract("was the order placed successfully?")
    return result.data
```

### Search Form with Filters

```python
async def search_with_filters(browser, search_params):
    """Perform search with multiple filters."""
    await browser.goto("https://example.com/search")
    
    # Main search
    await browser.act(f"type '{search_params['query']}' in the search box")
    
    # Apply filters
    if search_params.get('category'):
        await browser.act(f"select '{search_params['category']}' from the Category dropdown")
    
    if search_params.get('price_min'):
        await browser.act(f"type '{search_params['price_min']}' in the minimum price field")
    
    if search_params.get('price_max'):
        await browser.act(f"type '{search_params['price_max']}' in the maximum price field")
    
    if search_params.get('in_stock_only'):
        await browser.act("check the 'In Stock Only' filter")
    
    # Submit search
    await browser.act("click Search or Apply Filters")
```

## Handling Difficult Forms

### Dynamic Forms

Forms that change based on selections:

```python
async def fill_dynamic_form(browser):
    await browser.goto("https://example.com/application")
    
    # Select type - this might reveal new fields
    await browser.act("select 'Business Account' from the account type dropdown")
    
    # Wait for new fields to appear
    await asyncio.sleep(1)
    
    # Now fill the business-specific fields
    await browser.act("type 'Acme Corp' in the Company Name field")
    await browser.act("type '12-3456789' in the Tax ID field")
```

### Forms with Auto-Complete

```python
async def fill_autocomplete_form(browser):
    await browser.goto("https://example.com/address")
    
    # Type to trigger autocomplete
    await browser.act("type '123 Main' in the address field")
    
    # Wait for suggestions
    await asyncio.sleep(1)
    
    # Select from suggestions
    await browser.act("click the first autocomplete suggestion")
```

### Forms in Modals

```python
async def fill_modal_form(browser):
    await browser.goto("https://example.com")
    
    # Open the modal
    await browser.act("click Sign Up")
    
    # Wait for modal to appear
    await asyncio.sleep(0.5)
    
    # Fill form in modal
    await browser.act("in the signup modal, type 'email@example.com' in the email field")
    await browser.act("in the signup modal, click Submit")
```

### Forms with AJAX Validation

```python
async def fill_ajax_validated_form(browser):
    await browser.goto("https://example.com/register")
    
    # Fill email - triggers AJAX validation
    await browser.act("type 'user@example.com' in the email field")
    
    # Tab away to trigger validation
    await browser.act("press Tab")
    
    # Wait for validation response
    await asyncio.sleep(1)
    
    # Check validation result
    result = await browser.extract("is the email available?")
    
    if not result.data or "taken" in str(result.data).lower():
        # Try different email
        await browser.act("clear email and type 'user2@example.com'")
```

## Best Practices

### Use Configuration Objects

```python
# Define form data structure
CONTACT_FORM = {
    "fields": [
        {"name": "name", "value": "John Doe", "type": "text"},
        {"name": "email", "value": "john@example.com", "type": "email"},
        {"name": "message", "value": "Hello...", "type": "textarea"},
    ],
    "submit_button": "Send Message"
}

async def fill_from_config(browser, url, form_config):
    await browser.goto(url)
    
    for field in form_config["fields"]:
        await browser.act(f"type '{field['value']}' in the {field['name']} field")
    
    await browser.act(f"click {form_config['submit_button']}")
```

### Add Delays for Reliability

```python
async def fill_form_reliably(browser, fields):
    for field_name, value in fields.items():
        await browser.act(f"type '{value}' in the {field_name} field")
        await asyncio.sleep(0.3)  # Small delay between fields
```

### Verify Before Submit

```python
async def verify_before_submit(browser):
    # Fill form
    await browser.act("type 'John Doe' in the name field")
    await browser.act("type 'john@example.com' in the email field")
    
    # Verify fields before submitting
    result = await browser.extract(
        "what are the current values in the name and email fields?",
        schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"}
            }
        }
    )
    
    if result.data["name"] == "John Doe" and result.data["email"] == "john@example.com":
        await browser.act("click Submit")
    else:
        print("Form values don't match expected, not submitting")
```

## Error Recovery

### Retry Failed Fields

```python
async def fill_with_retry(browser, field_name, value, max_retries=3):
    for attempt in range(max_retries):
        result = await browser.act(f"type '{value}' in the {field_name} field")
        
        if result.success:
            return True
        
        print(f"Attempt {attempt + 1} failed for {field_name}")
        
        # Try alternative approaches
        if attempt == 1:
            # Try with vision
            result = await browser.act(
                f"find the {field_name} field and enter '{value}'",
                use_vision=True
            )
        elif attempt == 2:
            # Try clicking first
            await browser.act(f"click on the {field_name} field")
            await asyncio.sleep(0.5)
            result = await browser.act(f"type '{value}'")
        
        if result.success:
            return True
    
    return False
```

### Handle Form Submission Errors

```python
async def submit_with_error_recovery(browser):
    result = await browser.act("click Submit")
    
    # Wait for response
    await asyncio.sleep(2)
    
    # Check for errors
    error_result = await browser.extract(
        "are there any error messages on the form?",
        schema={
            "type": "array",
            "items": {"type": "string"}
        }
    )
    
    if error_result.data:
        for error in error_result.data:
            print(f"Form error: {error}")
            
            # Try to fix based on error
            if "email" in error.lower():
                await browser.act("fix the email field")
            elif "required" in error.lower():
                # Find and fill missing required fields
                await browser.act("fill any empty required fields with placeholder values")
        
        # Retry submission
        await browser.act("click Submit again")
```

## Next Steps

- [Multi-Page Workflows](multi-page-workflows.md) - Complex form sequences
- [Authentication Guide](authentication.md) - Login and session handling
- [Error Handling Guide](error-handling.md) - Robust error recovery
