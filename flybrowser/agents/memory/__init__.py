"""Browser-specific memory extensions for fireflyframework-genai."""

# Re-export everything from the legacy memory module so that existing
# ``from flybrowser.agents.memory import AgentMemory`` style imports
# continue to work after the flat module was converted to a package.
from flybrowser.agents.memory._legacy_memory import (  # noqa: F401
    AgentMemory,
    ContextStore,
    LearnedPattern,
    LongTermMemory,
    MemoryEntry,
    ShortTermMemory,
    StateSnapshot,
    WorkingMemory,
)

# New browser-specific memory manager
from flybrowser.agents.memory.browser_memory import BrowserMemoryManager, PageSnapshot  # noqa: F401

__all__ = [
    # Legacy
    "AgentMemory",
    "ContextStore",
    "LearnedPattern",
    "LongTermMemory",
    "MemoryEntry",
    "ShortTermMemory",
    "StateSnapshot",
    "WorkingMemory",
    # New
    "BrowserMemoryManager",
    "PageSnapshot",
]
