# Context System Architecture

FlyBrowser's context system provides a standardized, type-safe way to pass structured data to agent actions, enabling powerful features like automated form filling, file uploads, filtered extraction, and conditional navigation.

## Overview

The context system bridges the gap between high-level user intent and low-level tool execution. Instead of embedding all parameters in natural language instructions, users can provide structured context that tools understand directly.

### Key Benefits

- **Type Safety**: Builder pattern with runtime validation
- **Standardization**: Consistent schemas across all tools
- **Composability**: Multiple context types in one action
- **Tool Discovery**: LLM knows what context each tool accepts
- **Separation of Concerns**: Intent vs. data

## Architecture

```
User Code
    │
    ├─> ContextBuilder (Type-safe construction)
    │       │
    │       └─> ActionContext (Validated)
    │
    └─> SDK Method (act, extract, observe, navigate, agent)
            │
            ├─> Validation (ContextValidator)
            │
            ├─> Convert to dict
            │
            └─> Store in AgentMemory.working.scratch["user_context"]
                    │
                    └─> Tool.get_user_context()
                            │
                            └─> Tool.execute() uses context
```

## Context Types

### FORM_DATA
**Purpose**: Automated form filling  
**Schema**: `{field_selector: value}`  
**Tools**: TypeTool  
**Example**:
```python
context = ContextBuilder()\
    .with_form_data({
        "input[name=email]": "user@example.com",
        "input[name=password]": "secure_password",
        "#remember": True
    })\
    .build()
await browser.act("Fill and submit login form", context=context)
```

### FILES
**Purpose**: File uploads with metadata  
**Schema**: `[{field, path, mime_type?, name?}]`  
**Tools**: UploadFileTool  
**Example**:
```python
context = ContextBuilder()\
    .with_file("resume", "/path/to/resume.pdf", "application/pdf")\
    .with_file("cover_letter", "/path/to/letter.docx")\
    .build()
await browser.act("Upload application documents", context=context)
```

### FILTERS
**Purpose**: Data filtering criteria  
**Schema**: `{filter_name: filter_value}`  
**Tools**: ExtractTextTool, SearchAPITool  
**Example**:
```python
context = ContextBuilder()\
    .with_filters({
        "price_max": 100,
        "category": "electronics",
        "site": "amazon.com",  # For search
        "filetype": "pdf"      # For search
    })\
    .build()
result = await browser.extract("Get product listings", context=context)
```

### PREFERENCES
**Purpose**: User preferences for behavior  
**Schema**: `{pref_name: pref_value}`  
**Tools**: ExtractTextTool, SearchAPITool  
**Example**:
```python
context = ContextBuilder()\
    .with_preferences({
        "max_results": 20,
        "sort_by": "price",
        "max_headings": 10,
        "safe_search": True
    })\
    .build()
```

### CONDITIONS
**Purpose**: Conditional navigation/actions  
**Schema**: `{condition_name: expected_value}`  
**Tools**: NavigateTool (future enhancement)  
**Example**:
```python
context = ContextBuilder()\
    .with_conditions({
        "requires_login": False,
        "max_redirects": 3
    })\
    .build()
```

### CONSTRAINTS
**Purpose**: General constraints/limits  
**Schema**: `{constraint_name: value}`  
**Tools**: Universal  
**Example**:
```python
context = ContextBuilder()\
    .with_constraints({
        "timeout_seconds": 30,
        "max_retries": 3
    })\
    .build()
```

### METADATA
**Purpose**: Tool-specific metadata  
**Schema**: `{key: value}`  
**Tools**: Universal  
**Example**:
```python
context = ContextBuilder()\
    .with_metadata({
        "request_id": "abc123",
        "user_agent": "custom"
    })\
    .build()
```

## Tool Support Matrix

| Tool | form_data | files | filters | preferences | conditions | constraints | metadata |
|------|-----------|-------|---------|-------------|------------|-------------|----------|
| TypeTool | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| UploadFileTool | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| ExtractTextTool | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| SearchAPITool | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| NavigateTool | ❌ | ❌ | ❌ | ❌ | 🔄 | ❌ | ❌ |

✅ = Supported | ❌ = Not applicable | 🔄 = Planned

## Classes and APIs

### ActionContext

The main context container with all context types.

```python
@dataclass
class ActionContext:
    form_data: Dict[str, Any]
    files: List[FileUploadSpec]
    filters: Dict[str, Any]
    preferences: Dict[str, Any]
    conditions: Dict[str, Any]
    constraints: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]
    def is_empty(self) -> bool
    def has_type(self, context_type: ContextType) -> bool
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ActionContext
```

### ContextBuilder

Fluent builder for type-safe context construction.

```python
class ContextBuilder:
    def with_form_data(self, form_data: Dict[str, Any]) -> ContextBuilder
    def with_form_field(self, field: str, value: Any) -> ContextBuilder
    
    def with_file(self, field: str, path: str, mime_type: Optional[str] = None) -> ContextBuilder
    def with_files(self, files: List[Dict[str, Any]]) -> ContextBuilder
    
    def with_filters(self, filters: Dict[str, Any]) -> ContextBuilder
    def with_filter(self, name: str, value: Any) -> ContextBuilder
    
    def with_preferences(self, preferences: Dict[str, Any]) -> ContextBuilder
    def with_preference(self, name: str, value: Any) -> ContextBuilder
    
    def with_conditions(self, conditions: Dict[str, Any]) -> ContextBuilder
    def with_condition(self, name: str, value: Any) -> ContextBuilder
    
    def with_constraints(self, constraints: Dict[str, Any]) -> ContextBuilder
    def with_constraint(self, name: str, value: Any) -> ContextBuilder
    
    def with_metadata(self, metadata: Dict[str, Any]) -> ContextBuilder
    
    def build(self, validate: bool = True) -> ActionContext
```

### ContextValidator

Runtime validation with detailed error messages.

```python
class ContextValidator:
    @staticmethod
    def validate(context: ActionContext) -> tuple[bool, List[str]]
    
    @staticmethod
    def validate_for_tool(
        context: ActionContext,
        expected_types: List[ContextType]
    ) -> tuple[bool, List[str]]
```

### FileUploadSpec

Dedicated file upload specification.

```python
@dataclass
class FileUploadSpec:
    field: str  # Form field name or selector
    path: str   # File path
    mime_type: Optional[str] = None
    name: Optional[str] = None
    verify_exists: bool = True
    
    def to_dict(self) -> Dict[str, Any]
    def validate(self) -> tuple[bool, Optional[str]]
```

## Usage Patterns

### Form Filling

```python
from flybrowser import FlyBrowser
from flybrowser.agents.context import ContextBuilder

async with FlyBrowser(llm_provider="openai", api_key="...") as browser:
    await browser.goto("https://example.com/login")
    
    context = ContextBuilder()\
        .with_form_data({
            "input[name=username]": "john.doe",
            "input[name=password]": "secret123",
            "input[type=checkbox]": True
        })\
        .build()
    
    await browser.act("Fill and submit the login form", context=context)
```

### File Upload

```python
context = ContextBuilder()\
    .with_file("cv", "~/Documents/resume.pdf", "application/pdf")\
    .with_file("photo", "~/Pictures/headshot.jpg", "image/jpeg")\
    .build()

await browser.act("Upload CV and photo to the application form", context=context)
```

### Filtered Extraction

```python
context = ContextBuilder()\
    .with_filters({"price_max": 500, "brand": "Apple"})\
    .with_preferences({"sort_by": "price", "limit": 10})\
    .build()

products = await browser.extract(
    "Extract product listings with prices",
    context=context,
    schema={
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "price": {"type": "number"}
            }
        }
    }
)
```

### Search with Filters

```python
context = ContextBuilder()\
    .with_filters({"site": "python.org", "filetype": "pdf"})\
    .with_preferences({"max_results": 5})\
    .build()

await browser.agent(
    "Search for Python tutorials and open the best one",
    context=context
)
```

### Combined Context

```python
context = ContextBuilder()\
    .with_form_data({"search": "laptop"})\
    .with_filters({"price_max": 1000})\
    .with_preferences({"sort_by": "rating"})\
    .with_constraints({"timeout_seconds": 60})\
    .build()

await browser.agent(
    "Search for laptops and buy the best one under budget",
    context=context
)
```

## Validation

### Automatic Validation

Context is automatically validated when:
1. `ContextBuilder.build(validate=True)` is called (default)
2. SDK methods receive an ActionContext instance

```python
# This will raise ValueError if validation fails
context = ContextBuilder()\
    .with_file("resume", "/nonexistent/file.pdf")\
    .build()  # Raises: File not found
```

### Validation Rules

- **form_data**: Must be dict with string keys
- **files**: Each FileUploadSpec must have field and path; file must exist if verify_exists=True
- **filters**: Must be dict
- **preferences**: Must be dict
- **conditions**: Must be dict
- **constraints**: Must be dict
- **metadata**: Must be dict

### Disabling Validation

```python
# Build without validation (use with caution)
context = ContextBuilder()\
    .with_file("resume", "/path/to/file.pdf", verify_exists=False)\
    .build(validate=False)
```

## Tool Integration

### Accessing Context in Tools

Tools access context via `BaseTool.get_user_context()`:

```python
class MyTool(BaseTool):
    async def execute(self, **kwargs) -> ToolResult:
        from flybrowser.agents.context import ActionContext
        
        # Get all context
        user_context = self.get_user_context()
        
        # Handle both dict and ActionContext
        if isinstance(user_context, dict):
            filters = user_context.get("filters", {})
        elif isinstance(user_context, ActionContext):
            filters = user_context.filters
        
        # Use filters in tool logic
        max_price = filters.get("price_max", float('inf'))
        ...
```

### Declaring Expected Context Types

Tools declare expected context types in metadata:

```python
@property
def metadata(self) -> ToolMetadata:
    return ToolMetadata(
        name="search",
        description="Search with filters support",
        parameters=[...],
        expected_context_types=["filters", "preferences"],
    )
```

This helps the LLM understand what context the tool can use.

## SDK Integration

All SDK methods support the `context` parameter:

```python
# act() - Actions with context
await browser.act(instruction, context=context)

# extract() - Extraction with filters
await browser.extract(query, context=context)

# observe() - Element search with filters
await browser.observe(query, context=context)

# navigate() - Navigation with conditions
await browser.navigate(instruction, context=context)

# agent() - Multi-step with context
await browser.agent(task, context=context)
```

### Context Flow

1. User creates context with ContextBuilder
2. SDK method validates context (if ActionContext)
3. Context converted to dict and stored in agent memory
4. Tools retrieve context via get_user_context()
5. Tools use context in their logic

## API Integration

All API endpoints accept context in request bodies:

```json
POST /sessions/{id}/act
{
  "instruction": "Fill login form",
  "context": {
    "form_data": {
      "email": "user@example.com",
      "password": "***"
    }
  }
}
```

Request models (ActionRequest, ExtractRequest, etc.) include `context` field.

## Best Practices

### 1. Use ContextBuilder for Type Safety

```python
# Good
context = ContextBuilder().with_form_data({...}).build()

# Avoid (no validation)
context = {"form_data": {...}}
```

### 2. Validate File Paths

```python
# Good - validation enabled by default
context = ContextBuilder().with_file("file", "path.pdf").build()

# Risky - could fail at runtime
context = ContextBuilder()\
    .with_file("file", "path.pdf", verify_exists=False)\
    .build(validate=False)
```

### 3. Use Appropriate Context Types

```python
# Good - filters for data filtering
context = ContextBuilder().with_filters({"price_max": 100}).build()

# Avoid - constraints for UI preferences
context = ContextBuilder().with_constraints({"price_max": 100}).build()
```

### 4. Combine Context Types When Needed

```python
# Good - multiple related contexts
context = ContextBuilder()\
    .with_filters({"category": "books"})\
    .with_preferences({"sort_by": "rating", "limit": 10})\
    .build()
```

### 5. Let LLM Handle Simple Cases

```python
# Good for simple single-field forms
await browser.act("Type 'hello' in the search box")

# Better for complex multi-field forms
context = ContextBuilder().with_form_data({...}).build()
await browser.act("Fill the entire registration form", context=context)
```

## Convenience Functions

Quick context creation for common patterns:

```python
from flybrowser.agents.context import (
    create_form_context,
    create_upload_context,
    create_filter_context
)

# Form filling
context = create_form_context({"email": "user@example.com"})

# File upload
context = create_upload_context([
    {"field": "resume", "path": "resume.pdf"}
])

# Filtered extraction
context = create_filter_context(
    filters={"price_max": 100},
    preferences={"sort_by": "price"}
)
```

## Error Handling

### Validation Errors

```python
try:
    context = ContextBuilder()\
        .with_file("resume", "/bad/path.pdf")\
        .build()
except ValueError as e:
    print(f"Validation failed: {e}")
    # Output: Context validation failed: files[0]: File not found: /bad/path.pdf
```

### Missing Context

Tools gracefully handle missing context:

```python
# If no context provided, tools use default behavior
await browser.act("Fill form")  # Works without context
```

## Future Enhancements

- **CONDITIONS** support in NavigateTool for conditional navigation
- **CONSTRAINTS** enforcement at framework level
- Context inheritance across agent steps
- Context templates for common scenarios
- Context validation against tool schemas
- Context merging strategies

## See Also

- [SDK Reference](../reference/sdk.md) - SDK method signatures with context
- [API Reference](../reference/api.md) - API endpoints with context
- [Context Usage Guide](../guides/context-usage.md) - Practical examples
- [Tool Development](./tools.md) - Creating context-aware tools
