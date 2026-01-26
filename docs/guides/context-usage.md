# Context Usage Guide

This guide provides practical examples of using FlyBrowser's context system for common automation scenarios.

## Quick Start

The context system lets you pass structured data to browser actions instead of encoding everything in natural language.

```python
from flybrowser import FlyBrowser
from flybrowser.agents.context import ContextBuilder

async with FlyBrowser(llm_provider="openai", api_key="...") as browser:
    # Create context with form data
    context = ContextBuilder()\
        .with_form_data({"#email": "user@example.com"})\
        .build()
    
    await browser.act("Fill the email field", context=context)
```

## Form Filling

### Login Forms

```python
# Simple login
context = ContextBuilder()\
    .with_form_data({
        "input[name=email]": "user@example.com",
        "input[name=password]": "secure_password",
        "input[type=checkbox]": True  # Remember me
    })\
    .build()

await browser.goto("https://example.com/login")
await browser.act("Fill and submit the login form", context=context)
```

### Registration Forms

```python
context = ContextBuilder()\
    .with_form_data({
        "#firstName": "John",
        "#lastName": "Doe",
        "input[name=email]": "john.doe@example.com",
        "input[name=phone]": "+1-555-123-4567",
        "#password": "SecurePass123!",
        "#confirmPassword": "SecurePass123!",
        "select[name=country]": "US",
        "#agreeTerms": True
    })\
    .build()

await browser.goto("https://example.com/register")
await browser.act("Complete the registration form", context=context)
```

### Dynamic Field Matching

The TypeTool uses smart matching - selectors can be CSS selectors, field names, or IDs:

```python
context = ContextBuilder()\
    .with_form_data({
        # CSS selector
        "input[name=username]": "john_doe",
        # ID
        "#email": "john@example.com",
        # Name only (will match input[name=password])
        "password": "secret123"
    })\
    .build()
```

## File Uploads

### Single File Upload

```python
context = ContextBuilder()\
    .with_file("resume", "/path/to/resume.pdf", "application/pdf")\
    .build()

await browser.act("Upload resume to the application", context=context)
```

### Multiple File Uploads

```python
context = ContextBuilder()\
    .with_file("cv", "/docs/resume.pdf", "application/pdf")\
    .with_file("cover_letter", "/docs/cover.docx")\
    .with_file("photo", "/images/headshot.jpg", "image/jpeg")\
    .build()

await browser.act("Upload all documents to the job application", context=context)
```

### File Upload with Form Data

```python
context = ContextBuilder()\
    .with_form_data({
        "#applicantName": "John Doe",
        "#position": "Senior Developer",
        "#yearsExperience": "5"
    })\
    .with_file("cv", "/docs/resume.pdf", "application/pdf")\
    .with_file("portfolio", "/docs/portfolio.zip", "application/zip")\
    .build()

await browser.agent(
    "Fill out the job application form and upload documents",
    context=context
)
```

## Data Extraction with Filters

### Filtered Product Extraction

```python
context = ContextBuilder()\
    .with_filters({
        "price_max": 500,
        "category": "electronics",
        "brand": "Apple"
    })\
    .with_preferences({
        "sort_by": "price",
        "limit": 10
    })\
    .build()

products = await browser.extract(
    "Get product listings",
    context=context,
    schema={
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "price": {"type": "number"},
                "rating": {"type": "number"}
            }
        }
    }
)
```

### Content Extraction with Preferences

```python
context = ContextBuilder()\
    .with_preferences({
        "max_paragraphs": 5,
        "include_images": False,
        "max_headings": 10
    })\
    .build()

content = await browser.extract(
    "Get the main article content",
    context=context
)
```

## Search Operations

### Web Search with Filters

```python
context = ContextBuilder()\
    .with_filters({
        "site": "python.org",
        "filetype": "pdf"
    })\
    .with_preferences({
        "max_results": 5,
        "safe_search": True
    })\
    .build()

await browser.agent(
    "Search for Python tutorials and download the best one",
    context=context
)
```

### Domain-Specific Search

```python
context = ContextBuilder()\
    .with_filters({
        "site": "github.com",
        "language": "python"
    })\
    .with_preferences({
        "max_results": 10,
        "sort_by": "stars"
    })\
    .build()

await browser.agent(
    "Find the best Python web frameworks",
    context=context
)
```

## Multi-Step Agent Tasks

### E-commerce Purchase

```python
context = ContextBuilder()\
    .with_form_data({
        "#searchQuery": "wireless headphones"
    })\
    .with_filters({
        "price_max": 200,
        "brand": "Sony",
        "rating_min": 4.0
    })\
    .with_preferences({
        "sort_by": "price"
    })\
    .with_constraints({
        "timeout_seconds": 120
    })\
    .build()

result = await browser.agent(
    "Search for headphones, find the best deal, and add to cart",
    context=context,
    max_iterations=30
)
```

### Form Submission with File Upload

```python
context = ContextBuilder()\
    .with_form_data({
        "input[name=company]": "Acme Corporation",
        "textarea[name=description]": "Leading provider of enterprise solutions",
        "select[name=industry]": "Technology",
        "#employees": "500-1000"
    })\
    .with_file("logo", "/assets/company-logo.png", "image/png")\
    .with_file("brochure", "/docs/company-brochure.pdf")\
    .build()

await browser.agent(
    "Complete the company registration and upload materials",
    context=context
)
```

### Research Task

```python
context = ContextBuilder()\
    .with_filters({
        "date_after": "2024-01-01",
        "source_type": "academic"
    })\
    .with_preferences({
        "max_results": 20,
        "include_citations": True
    })\
    .with_metadata({
        "research_topic": "machine learning",
        "output_format": "structured"
    })\
    .build()

result = await browser.agent(
    "Research recent advances in machine learning and summarize key findings",
    context=context,
    max_iterations=50
)
```

## Validation

### Automatic Validation

By default, context is validated when you call `.build()`:

```python
try:
    # This will fail if file doesn't exist
    context = ContextBuilder()\
        .with_file("resume", "/nonexistent/file.pdf")\
        .build()
except ValueError as e:
    print(f"Validation error: {e}")
```

### Manual Validation

```python
from flybrowser.agents.context import ContextValidator

context = ContextBuilder()\
    .with_form_data({"#email": "test@example.com"})\
    .build(validate=False)  # Skip auto-validation

# Validate manually
is_valid, errors = ContextValidator.validate(context)
if not is_valid:
    print(f"Validation errors: {errors}")
```

### Validate for Specific Tool

```python
from flybrowser.agents.context import ContextValidator, ContextType

context = ContextBuilder()\
    .with_form_data({"#email": "test@example.com"})\
    .with_filters({"price_max": 100})\
    .build()

# Check if context is valid for TypeTool (expects form_data)
is_valid, errors = ContextValidator.validate_for_tool(
    context,
    expected_types=[ContextType.FORM_DATA]
)
```

## Convenience Functions

For simple cases, use the convenience functions:

```python
from flybrowser.agents.context import (
    create_form_context,
    create_upload_context,
    create_filter_context
)

# Quick form context
context = create_form_context({
    "email": "user@example.com",
    "password": "secret"
})

# Quick file upload context
context = create_upload_context([
    {"field": "resume", "path": "/path/to/resume.pdf"}
])

# Quick filter context
context = create_filter_context(
    filters={"price_max": 100},
    preferences={"sort_by": "price"}
)
```

## API Usage

When using the REST API, pass context as a JSON object:

```bash
# Form filling via API
curl -X POST "http://localhost:8000/sessions/${SESSION_ID}/act" \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Fill and submit the login form",
    "context": {
      "form_data": {
        "input[name=email]": "user@example.com",
        "input[name=password]": "secret123"
      }
    }
  }'

# File upload via API
curl -X POST "http://localhost:8000/sessions/${SESSION_ID}/act" \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Upload the resume",
    "context": {
      "files": [
        {
          "field": "resume",
          "path": "/path/to/resume.pdf",
          "mime_type": "application/pdf"
        }
      ]
    }
  }'

# Agent task with full context
curl -X POST "http://localhost:8000/sessions/${SESSION_ID}/agent" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Complete the application and upload documents",
    "context": {
      "form_data": {
        "#name": "John Doe",
        "#email": "john@example.com"
      },
      "files": [
        {"field": "cv", "path": "/docs/cv.pdf"}
      ],
      "preferences": {
        "timeout_seconds": 120
      }
    },
    "max_iterations": 30
  }'
```

## Best Practices

### 1. Use Specific Selectors

```python
# Good - specific selectors
context = ContextBuilder().with_form_data({
    "input[name=email]": "user@example.com",
    "#password-field": "secret"
}).build()

# Avoid - ambiguous selectors
context = ContextBuilder().with_form_data({
    "input": "user@example.com"  # Which input?
}).build()
```

### 2. Combine Context Types Logically

```python
# Good - related context types together
context = ContextBuilder()\
    .with_filters({"category": "books"})\
    .with_preferences({"sort_by": "rating", "limit": 10})\
    .build()

# Avoid - mixing unrelated context
context = ContextBuilder()\
    .with_form_data({"#search": "books"})\
    .with_constraints({"category": "books"})  # Wrong type
    .build()
```

### 3. Handle File Validation

```python
# Good - explicit file existence check
import os

file_path = "/path/to/file.pdf"
if os.path.exists(file_path):
    context = ContextBuilder()\
        .with_file("resume", file_path)\
        .build()
else:
    print(f"File not found: {file_path}")
```

### 4. Let LLM Handle Simple Cases

```python
# Simple single-field typing - no context needed
await browser.act("Type 'hello' in the search box")

# Complex multi-field form - use context
context = ContextBuilder().with_form_data({...}).build()
await browser.act("Fill the registration form", context=context)
```

### 5. Use Constraints Wisely

```python
# Good - reasonable constraints
context = ContextBuilder()\
    .with_constraints({"timeout_seconds": 60, "max_retries": 3})\
    .build()

# Avoid - too restrictive
context = ContextBuilder()\
    .with_constraints({"timeout_seconds": 1, "max_retries": 0})\
    .build()
```

## Error Handling

```python
from flybrowser import FlyBrowser
from flybrowser.agents.context import ContextBuilder

async with FlyBrowser(llm_provider="openai", api_key="...") as browser:
    try:
        # Build context with validation
        context = ContextBuilder()\
            .with_file("resume", "/path/to/resume.pdf")\
            .build()
        
        result = await browser.act(
            "Upload the resume",
            context=context
        )
        
        if not result.success:
            print(f"Action failed: {result.error}")
            
    except ValueError as e:
        print(f"Context validation error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
```

## See Also

- [Context System Architecture](../architecture/context.md) - Technical details
- [SDK Reference](../reference/sdk.md) - SDK method signatures
- [REST API Reference](../reference/rest-api.md) - API endpoints
