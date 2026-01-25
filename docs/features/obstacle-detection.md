# Obstacle Detection

FlyBrowser includes a state-of-the-art obstacle detection system that automatically identifies and dismisses page obstacles like cookie banners, modal dialogs, newsletter popups, and other overlays that block user interaction.

## Overview

Modern websites frequently use overlays that can block browser automation:
- Cookie consent banners (GDPR, CCPA compliance)
- Newsletter signup popups
- Promotional modals
- Age verification gates
- Login walls

FlyBrowser's obstacle detector handles these automatically—including obstacles that appear **dynamically via JavaScript** after initial page load.

## Two-Phase Detection Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: Quick DOM Analysis (~10ms, no LLM call)               │
│  ├── Multi-point viewport sampling (5 positions)                │
│  ├── ARIA role detection (dialog, alertdialog)                  │
│  ├── Framework modal detection                                  │
│  ├── Newsletter tool detection                                  │
│  ├── Consent tool detection                                     │
│  └── Weighted confidence scoring (0.0-1.0)                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼ (only if confidence > 0.3)
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: Full VLM Analysis + Dismissal (~2-5s)                 │
│  ├── Screenshot capture                                         │
│  ├── AI-driven obstacle classification                          │
│  ├── Multi-strategy dismissal selection                         │
│  ├── Action execution with verification                         │
│  └── Cooldown period (3s) to prevent re-detection loops         │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 1: Quick DOM Check

The fast first phase performs multi-point sampling without any LLM calls:

**5-Point Viewport Sampling:**
- Center of viewport
- Top-left quadrant
- Top-right quadrant
- Bottom-left quadrant
- Bottom-right quadrant

**Detection Signals:**

| Signal Type | Examples | Weight |
|-------------|----------|--------|
| ARIA Roles | `role="dialog"`, `role="alertdialog"` | High |
| Framework Classes | Bootstrap `.modal`, MUI `MuiModal`, React-Modal | High |
| Newsletter Tools | MailPoet, Mailchimp, HubSpot, Klaviyo | High |
| Consent Tools | OneTrust, CookieBot, Quantcast, Termly | High |
| Overlay Patterns | Fixed positioning, high z-index, backdrop | Medium |
| Button Text | "Accept", "Subscribe", "Close", "Dismiss" | Low |

### Phase 2: VLM Analysis

When Phase 1 confidence exceeds the threshold (default: 0.3), the system captures a screenshot and uses AI to:

1. **Classify the obstacle type** (cookie banner, newsletter popup, etc.)
2. **Select dismissal strategy** (prioritized order):
   - Text-based button click ("Accept", "Close", etc.)
   - ARIA-labeled element interaction
   - Coordinate-based click (last resort)
3. **Execute and verify** dismissal was successful

## Integration Points

### Before Screenshot Capture

Dynamic obstacles are checked before every screenshot in the ReAct loop:

```python
async def _generate_with_optional_vision(self, prompts):
    if self._should_use_vision(iteration):
        # Handle dynamic obstacles before capturing screenshot
        await self._check_and_handle_dynamic_obstacles()
        
        # Now capture clean screenshot
        screenshot = await self.page.screenshot()
```

### Auto-Recovery on Click Failures

When a click fails because another element intercepts it:

```python
# In _execute_action():
if action.tool_name == "click" and "intercept" in error_msg:
    logger.info("[AutoRecovery] Click intercepted - checking for obstacles...")
    obstacle_handled = await self._check_and_handle_dynamic_obstacles()
    if obstacle_handled:
        # Obstacle dismissed - agent should retry click
        result.metadata['recovery_action'] = "Obstacle dismissed. Retry the click."
```

### Cooldown System

After successfully dismissing obstacles, a 3-second cooldown prevents re-detection loops:

```python
result = await detector.detect_and_handle_if_needed(
    cooldown_seconds=3.0,  # Prevent re-detection after dismissal
    min_confidence=0.3,    # Only trigger VLM if confident
)
```

## Supported Frameworks & Tools

### Modal Frameworks

| Framework | Detection Method |
|-----------|------------------|
| Bootstrap | `.modal`, `.modal-dialog`, `data-bs-dismiss` |
| Material-UI (MUI) | `.MuiModal-root`, `.MuiDialog-root` |
| React-Modal | `.ReactModal__Overlay`, `.ReactModal__Content` |
| Ant Design | `.ant-modal`, `.ant-modal-mask` |
| Tailwind UI | `x-data`, `x-show` (Alpine.js patterns) |
| Custom | ARIA roles, fixed positioning, z-index > 1000 |

### Newsletter/Marketing Tools

| Tool | Detection Method |
|------|------------------|
| MailPoet | `mailpoet`, form classes |
| Mailchimp | `mc-embedded`, `mc_embed` |
| HubSpot | `hs-form`, `hbspt` |
| Klaviyo | `klaviyo`, `klav-` prefixes |
| OptinMonster | `om-`, `optinmonster` |
| Sumo | `sumo-`, `sumome` |

### Consent Management Platforms

| Platform | Detection Method |
|----------|------------------|
| OneTrust | `onetrust`, `ot-sdk` |
| CookieBot | `CybotCookiebot`, `cookiebot` |
| Quantcast | `qc-cmp`, `quantcast` |
| Termly | `termly`, `t-` prefixes |
| TrustArc | `truste`, `consent-banner` |
| Osano | `osano`, `cookie-consent` |

## Usage Examples

### Automatic Handling (Default)

Obstacle detection works transparently:

```python
async with FlyBrowser(llm_provider="openai", api_key="sk-...") as browser:
    await browser.goto("https://shop.example.com")
    
    # Obstacles handled automatically
    result = await browser.extract("Get all product prices")
    # Cookie banners and popups are dismissed automatically
```

### During Agent Tasks

```python
result = await browser.agent(
    task="Add the first product to cart and proceed to checkout"
)
# Newsletter popups that appear mid-task are automatically dismissed
```

### Explicit Control

For fine-grained control, use `act()`:

```python
# If automatic detection doesn't catch a specific popup
await browser.act("Click the X button to close the newsletter popup")
```

## Debugging

### Enable Debug Logging

```python
import logging

# Enable obstacle detector logging
logging.getLogger("flybrowser.agents.obstacle_detector").setLevel(logging.DEBUG)
logging.getLogger("flybrowser.agents.react_agent").setLevel(logging.DEBUG)
```

### Log Message Reference

| Log Message | Meaning |
|-------------|---------|
| `[QuickCheck] confidence=X.XX` | Phase 1 confidence score |
| `[DynamicObstacle] ✓ Handled N obstacle(s)` | Obstacles successfully dismissed |
| `[DynamicObstacle] Check failed (non-critical)` | Detection error (continues execution) |
| `[VISION] Skipped: page is blank` | Screenshot skipped for about:blank |

### Troubleshooting

**Popup not being dismissed:**
1. Check if confidence threshold is met (default: 0.3)
2. Verify the popup framework is supported
3. Try lowering `min_confidence` or using explicit `act()` instruction

**Re-detection loops:**
- The 3-second cooldown should prevent this
- If occurring, check for rapidly changing page content

**Performance concerns:**
- Phase 1 is ~10ms (no LLM call)
- Phase 2 only runs when confidence > 0.3
- Cooldown prevents repeated expensive checks

## Configuration

Obstacle detection is enabled by default. Configuration options:

```python
from flybrowser.agents.config import ObstacleDetectorConfig

config = ObstacleDetectorConfig(
    enabled=True,                    # Enable/disable obstacle detection
    min_confidence_threshold=0.3,    # Phase 1 confidence to trigger Phase 2
    cooldown_seconds=3.0,            # Cooldown after successful dismissal
    max_retries=2,                   # Retries for dismissal strategies
)
```

## Architecture Notes

### PageController vs Playwright Page

The obstacle detector needs to access the current URL synchronously. Understanding the architecture:

```python
# self.page is PageController (wrapper class)
# self.page.page is the underlying Playwright Page object

# CORRECT: Access Playwright Page's sync .url property
if hasattr(self.page, 'page') and hasattr(self.page.page, 'url'):
    current_url = self.page.page.url

# PageController has async get_url() method (not sync .url property)
# Use .page.url for synchronous URL access in non-async contexts
```

## See Also

- [Agent Documentation](agent.md) - Autonomous task execution with obstacle handling
- [ReAct Architecture](../architecture/react.md) - ReAct loop and vision integration
- [Troubleshooting](../advanced/troubleshooting.md) - Common issues and solutions
