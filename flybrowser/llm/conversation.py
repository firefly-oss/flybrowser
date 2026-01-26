# Copyright 2026 Firefly Software Solutions Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Conversation Manager for Multi-Turn LLM Interactions.

This module provides the ConversationManager class which handles:
- Multi-turn conversation history tracking
- Large content splitting across multiple turns
- Token budget management to prevent overflow
- Structured output preservation across turns
- Context window optimization
- Vision/VLM support with screenshots and images

The ConversationManager acts as the PRIMARY interface between the ReAct agent
and LLM providers, ensuring conversations stay within token limits while
preserving all necessary context. ALL LLM interactions should go through
this manager to ensure consistent token tracking and conversation management.

Usage:
    >>> manager = ConversationManager(llm_provider)
    >>> manager.set_system_prompt("You are a browser automation agent.")
    >>> 
    >>> # Text-only structured response
    >>> response = await manager.send_structured(
    ...     "Navigate to google.com",
    ...     schema=REACT_RESPONSE_SCHEMA
    ... )
    >>> 
    >>> # Vision-enabled structured response
    >>> response = await manager.send_structured_with_vision(
    ...     "Click the search button",
    ...     image_data=screenshot_bytes,
    ...     schema=REACT_RESPONSE_SCHEMA
    ... )
"""

from __future__ import annotations

import base64
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Union

from flybrowser.llm.token_budget import (
    TokenEstimator, TokenBudgetManager, ContentType, BudgetAllocation
)
from flybrowser.llm.chunking import (
    Chunk, ChunkingStrategy, SmartChunker, get_chunker
)

# Create module-specific logger
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from flybrowser.llm.base import BaseLLMProvider, ModelInfo, ModelCapability


class MessageRole(str, Enum):
    """Roles for conversation messages."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ConversationMessage:
    """
    A message in a conversation.
    
    Attributes:
        role: Message role (system, user, assistant)
        content: Message content (text or structured)
        tokens: Estimated token count
        timestamp: When message was created
        metadata: Additional message metadata
    """
    role: MessageRole
    content: Union[str, Dict[str, Any], List[Dict[str, Any]]]
    tokens: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_api_format(self) -> Dict[str, Any]:
        """Convert to API message format."""
        return {
            "role": self.role.value,
            "content": self.content,
        }
    
    @classmethod
    def system(cls, content: str) -> "ConversationMessage":
        """Create a system message."""
        tokens = TokenEstimator.estimate(content).tokens
        return cls(role=MessageRole.SYSTEM, content=content, tokens=tokens)
    
    @classmethod
    def user(cls, content: Union[str, List[Dict[str, Any]]]) -> "ConversationMessage":
        """Create a user message."""
        if isinstance(content, str):
            tokens = TokenEstimator.estimate(content).tokens
        else:
            tokens = TokenEstimator.estimate_messages([{"content": content}])
        return cls(role=MessageRole.USER, content=content, tokens=tokens)
    
    @classmethod
    def assistant(cls, content: str) -> "ConversationMessage":
        """Create an assistant message."""
        tokens = TokenEstimator.estimate(content).tokens
        return cls(role=MessageRole.ASSISTANT, content=content, tokens=tokens)


@dataclass
class ConversationHistory:
    """
    Tracks conversation history with token management.
    
    Maintains a list of messages and provides methods for
    pruning history to stay within token budgets.
    """
    messages: List[ConversationMessage] = field(default_factory=list)
    system_message: Optional[ConversationMessage] = None
    
    @property
    def total_tokens(self) -> int:
        """Total tokens in conversation history."""
        total = sum(m.tokens for m in self.messages)
        if self.system_message:
            total += self.system_message.tokens
        return total
    
    @property
    def message_count(self) -> int:
        """Number of messages (excluding system)."""
        return len(self.messages)
    
    def add(self, message: ConversationMessage) -> None:
        """Add a message to history."""
        if message.role == MessageRole.SYSTEM:
            self.system_message = message
        else:
            self.messages.append(message)
    
    def get_messages_for_api(self) -> List[Dict[str, Any]]:
        """Get messages in API format."""
        result = []
        if self.system_message:
            result.append(self.system_message.to_api_format())
        result.extend(m.to_api_format() for m in self.messages)
        return result
    
    def prune_to_fit(self, max_tokens: int, keep_recent: int = 2) -> int:
        """
        Prune history to fit within token budget.
        
        Removes oldest messages first, but always keeps the most recent
        `keep_recent` messages.
        
        Args:
            max_tokens: Maximum total tokens allowed
            keep_recent: Minimum recent messages to keep
            
        Returns:
            Number of messages removed
        """
        removed = 0
        
        while self.total_tokens > max_tokens and len(self.messages) > keep_recent:
            # Remove oldest non-system message
            self.messages.pop(0)
            removed += 1
        
        return removed
    
    def clear(self) -> None:
        """Clear conversation history (keeps system message)."""
        self.messages.clear()
    
    def reset(self) -> None:
        """Reset everything including system message."""
        self.messages.clear()
        self.system_message = None


class AccumulationPhase(str, Enum):
    """Phases for multi-turn accumulation protocol."""
    SINGLE = "single"          # No accumulation needed
    ACCUMULATING = "accumulating"  # Processing chunks
    SYNTHESIZING = "synthesizing"  # Final synthesis


@dataclass
class AccumulationContext:
    """Context for multi-turn content accumulation."""
    phase: AccumulationPhase = AccumulationPhase.SINGLE
    total_chunks: int = 0
    processed_chunks: int = 0
    chunk_summaries: List[str] = field(default_factory=list)
    original_instruction: str = ""
    
    @property
    def is_complete(self) -> bool:
        """Check if accumulation is complete."""
        return self.phase == AccumulationPhase.SINGLE or \
               (self.phase == AccumulationPhase.SYNTHESIZING and 
                self.processed_chunks >= self.total_chunks)
    
    @property
    def progress(self) -> float:
        """Get accumulation progress (0-1)."""
        if self.total_chunks == 0:
            return 1.0
        return self.processed_chunks / self.total_chunks


class ConversationManager:
    """
    Manages multi-turn conversations with LLM providers.
    
    This is the PRIMARY interface for all LLM interactions in the ReAct agent.
    All requests (text and vision) should go through this manager to ensure:
    - Consistent token tracking and budget management
    - Conversation history preservation across turns
    - Automatic handling of large content via chunking
    - Proper logging and statistics
    
    Handles:
    - Conversation history tracking
    - Token budget management
    - Large content chunking and multi-turn processing
    - Structured output across turns
    - Vision/VLM support with screenshots
    
    Example:
        >>> manager = ConversationManager(llm_provider)
        >>> manager.set_system_prompt("You are a helpful assistant.")
        >>> 
        >>> # Text-only structured response
        >>> response = await manager.send_structured(
        ...     "What is the capital of France?",
        ...     schema={"type": "object", "properties": {"answer": {"type": "string"}}}
        ... )
        >>> 
        >>> # Vision-enabled structured response
        >>> response = await manager.send_structured_with_vision(
        ...     "Click the search button",
        ...     image_data=screenshot_bytes,
        ...     schema=REACT_RESPONSE_SCHEMA
        ... )
        >>> 
        >>> # Large content with automatic chunking
        >>> response = await manager.send_with_large_content(
        ...     large_html_content,
        ...     instruction="Extract all product names",
        ...     schema=product_schema
        ... )
    """
    
    # Image token estimation constants (approximate)
    # Based on OpenAI/Anthropic image token calculations
    IMAGE_BASE_TOKENS = 85  # Base tokens for any image
    IMAGE_TOKENS_PER_TILE = 170  # Tokens per 512x512 tile
    
    def __init__(
        self,
        llm_provider: "BaseLLMProvider",
        model_info: Optional["ModelInfo"] = None,
        max_history_tokens: Optional[int] = None,
        chunking_strategy: Optional[ChunkingStrategy] = None,
        max_request_tokens: Optional[int] = None,
    ) -> None:
        """
        Initialize the ConversationManager.
        
        Args:
            llm_provider: LLM provider instance
            model_info: Model information (fetched from provider if not given)
            max_history_tokens: Maximum tokens for history (default: 50% of context)
            chunking_strategy: Strategy for splitting large content
            max_request_tokens: Hard limit on tokens per single request. This is critical
                for staying within API rate limits (e.g., OpenAI TPM limits). If not set,
                uses DEFAULT_MAX_REQUEST_TOKENS (25000).
        """
        self.llm = llm_provider
        
        # Get model info
        if model_info is None:
            self.model_info = llm_provider.get_model_info()
        else:
            self.model_info = model_info
        
        # Check for vision capability
        from flybrowser.llm.base import ModelCapability
        self._has_vision = ModelCapability.VISION in self.model_info.capabilities
        
        # Initialize budget manager
        self.budget = TokenBudgetManager(
            context_window=self.model_info.context_window,
            max_output_tokens=self.model_info.max_output_tokens,
            safety_margin=0.1,
        )
        
        # History token budget (default 50% of available)
        if max_history_tokens is None:
            max_history_tokens = int(self.budget.available_for_input * 0.5)
        self.max_history_tokens = max_history_tokens
        
        # Hard limit per request (for API rate limit compliance)
        # This is configurable to support different org TPM limits (e.g., 30K for some OpenAI orgs)
        self._max_request_tokens = max_request_tokens or self.DEFAULT_MAX_REQUEST_TOKENS
        
        # Initialize history
        self.history = ConversationHistory()
        
        # Chunking strategy
        self.chunker = chunking_strategy or SmartChunker()
        
        # Accumulation context for multi-turn processing
        self._accumulation: Optional[AccumulationContext] = None
        
        # Statistics
        self._total_requests = 0
        self._total_tokens_used = 0
        self._multi_turn_requests = 0
        self._vision_requests = 0
        
        vision_status = "enabled" if self._has_vision else "disabled"
        logger.info(
            f"[ConversationManager] Initialized: model={self.model_info.name}, "
            f"context_window={self.model_info.context_window:,}, "
            f"max_output={self.model_info.max_output_tokens:,}, "
            f"max_history={max_history_tokens:,}, max_request={self._max_request_tokens:,}, "
            f"vision={vision_status}"
        )
    
    def set_system_prompt(self, content: str) -> None:
        """Set the system prompt for the conversation."""
        self.history.add(ConversationMessage.system(content))
        logger.debug(f"System prompt set ({TokenEstimator.estimate(content).tokens} tokens)")
    
    def add_user_message(self, content: Union[str, List[Dict[str, Any]]]) -> None:
        """Add a user message to history."""
        self.history.add(ConversationMessage.user(content))
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to history."""
        self.history.add(ConversationMessage.assistant(content))
    
    # Default hard limit for single request (can be overridden via config)
    # This is a safety net for API rate limits (e.g., OpenAI 30K TPM orgs)
    DEFAULT_MAX_REQUEST_TOKENS = 25000
    
    async def send_structured(
        self,
        content: str,
        schema: Dict[str, Any],
        temperature: float = 0.7,
        add_to_history: bool = True,
        max_tokens: Optional[int] = None,
        max_request_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send a text message and get structured response.
        
        This is the primary method for text-only LLM interactions.
        For vision/image requests, use send_structured_with_vision().
        
        Automatically handles:
        - Token budget checking
        - History pruning if needed
        - Hard limit enforcement for API rate limits
        - Response validation
        
        Args:
            content: User message content
            schema: JSON schema for response
            temperature: Sampling temperature
            add_to_history: Whether to add to conversation history
            max_tokens: Optional max output tokens override
            max_request_tokens: Hard limit on total input tokens per request
                               (helps stay within API rate limits)
            **kwargs: Additional LLM parameters
            
        Returns:
            Structured response matching schema
        """
        request_id = self._total_requests + 1
        
        # Calculate effective hard limit for this request
        # Priority: method arg > instance config > class default
        hard_limit = max_request_tokens or self._max_request_tokens
        
        # Check if content fits within hard limit and available budget
        content_tokens = TokenEstimator.estimate(content).tokens
        system_tokens = self.history.system_message.tokens if self.history.system_message else 0
        history_tokens = self.history.total_tokens - system_tokens  # Exclude system from history count
        total_request_tokens = content_tokens + system_tokens + history_tokens
        available = self.budget.available_for_input - self.history.total_tokens
        
        logger.debug(
            f"[ConversationManager] Request #{request_id}: text-only, "
            f"content={content_tokens:,} tokens, history={history_tokens:,}, "
            f"total={total_request_tokens:,}, hard_limit={hard_limit:,}, "
            f"history_msgs={self.history.message_count}"
        )
        
        # HARD LIMIT CHECK: Enforce maximum request size for API rate limits
        if total_request_tokens > hard_limit:
            excess = total_request_tokens - hard_limit
            logger.warning(
                f"[ConversationManager] Request exceeds hard limit! "
                f"({total_request_tokens:,} > {hard_limit:,}, excess={excess:,}). "
                f"Aggressively pruning history."
            )
            
            # Calculate how much we need to free from history
            # Target: content + system + remaining_history <= hard_limit * 0.9 (10% buffer)
            target_total = int(hard_limit * 0.9)
            target_history = max(0, target_total - content_tokens - system_tokens)
            
            # Prune history to fit within target
            if target_history < history_tokens:
                removed = self.history.prune_to_fit(target_history, keep_recent=2)
                if removed > 0:
                    logger.info(
                        f"[ConversationManager] Hard limit cleanup: pruned {removed} messages, "
                        f"history now {self.history.total_tokens - system_tokens:,} tokens"
                    )
            
            # Recalculate after pruning
            history_tokens = self.history.total_tokens - system_tokens
            total_request_tokens = content_tokens + system_tokens + history_tokens
            
            # If STILL over limit after pruning, we need to truncate the content itself
            if total_request_tokens > hard_limit:
                logger.warning(
                    f"[ConversationManager] Still over limit after pruning ({total_request_tokens:,}). "
                    f"Content must be truncated externally."
                )
        
        # Standard available budget check (softer than hard limit)
        if content_tokens > available:
            logger.warning(
                f"[ConversationManager] Content too large ({content_tokens:,} > {available:,} tokens). "
                f"Pruning history to make room."
            )
            removed = self.history.prune_to_fit(available - content_tokens)
            if removed > 0:
                logger.info(f"[ConversationManager] Pruned {removed} messages from history")
        
        # Build messages
        messages = self.history.get_messages_for_api()
        messages.append({"role": "user", "content": content})
        
        # Get system prompt from history if present
        system_prompt = None
        if self.history.system_message:
            system_prompt = self.history.system_message.content
            # Remove system from messages (will be passed separately)
            messages = [m for m in messages if m["role"] != "system"]
        
        # Call LLM
        self._total_requests += 1
        
        try:
            # Format messages as prompt for generate_structured
            formatted_prompt = self._format_messages_as_prompt(messages)
            
            # Final safety check on formatted prompt
            formatted_tokens = TokenEstimator.estimate(formatted_prompt).tokens
            if system_prompt:
                formatted_tokens += TokenEstimator.estimate(system_prompt).tokens
            
            if formatted_tokens > hard_limit:
                logger.error(
                    f"[ConversationManager] CRITICAL: Formatted prompt still exceeds hard limit "
                    f"({formatted_tokens:,} > {hard_limit:,}). Request may fail."
                )
            
            logger.debug(
                f"[ConversationManager] Calling LLM: prompt={len(formatted_prompt):,} chars "
                f"(~{formatted_tokens:,} tokens), temp={temperature}"
            )
            
            response = await self.llm.generate_structured(
                prompt=formatted_prompt,
                schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs,
            )
            
            # Estimate token usage since generate_structured returns Dict, not LLMResponse
            input_tokens = TokenEstimator.estimate(formatted_prompt).tokens
            if system_prompt:
                input_tokens += TokenEstimator.estimate(system_prompt).tokens
            output_tokens = TokenEstimator.estimate(json.dumps(response)).tokens
            self._total_tokens_used += input_tokens + output_tokens
            
            logger.debug(
                f"[ConversationManager] Response received: ~{output_tokens:,} output tokens, "
                f"total_used={self._total_tokens_used:,}"
            )
            
            # Add to history
            if add_to_history:
                self.add_user_message(content)
                self.add_assistant_message(json.dumps(response))
            
            return response
            
        except Exception as e:
            logger.error(
                f"[ConversationManager] Request #{request_id} failed: {e}",
                exc_info=True
            )
            raise
    
    async def send_structured_with_vision(
        self,
        content: str,
        image_data: bytes,
        schema: Dict[str, Any],
        temperature: float = 0.7,
        add_to_history: bool = True,
        max_tokens: Optional[int] = None,
        image_detail: str = "auto",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send a message with image and get structured response.
        
        This is the primary method for vision/VLM interactions.
        Automatically handles token estimation for images and budget management.
        
        Args:
            content: User message content (text prompt)
            image_data: Image bytes (screenshot, etc.)
            schema: JSON schema for response
            temperature: Sampling temperature
            add_to_history: Whether to add to conversation history
            max_tokens: Optional max output tokens override
            image_detail: Image detail level ("low", "high", "auto")
            **kwargs: Additional LLM parameters
            
        Returns:
            Structured response matching schema
            
        Raises:
            ValueError: If model doesn't support vision
        """
        request_id = self._total_requests + 1
        
        # Check vision capability
        if not self._has_vision:
            raise ValueError(
                f"Model {self.model_info.name} does not support vision. "
                f"Use send_structured() for text-only requests."
            )
        
        # Estimate tokens
        content_tokens = TokenEstimator.estimate(content).tokens
        image_tokens = self._estimate_image_tokens(len(image_data), image_detail)
        total_input_tokens = content_tokens + image_tokens
        available = self.budget.available_for_input - self.history.total_tokens
        
        logger.debug(
            f"[ConversationManager] Request #{request_id}: vision, "
            f"text={content_tokens:,} tokens, image={image_tokens:,} tokens (~{len(image_data)//1024}KB), "
            f"total={total_input_tokens:,}, available={available:,}"
        )
        
        if total_input_tokens > available:
            logger.warning(
                f"[ConversationManager] Vision request may exceed budget "
                f"({total_input_tokens:,} > {available:,} tokens). Pruning history."
            )
            removed = self.history.prune_to_fit(available - total_input_tokens)
            if removed > 0:
                logger.info(f"[ConversationManager] Pruned {removed} messages from history")
        
        # Get system prompt from history if present
        system_prompt = None
        if self.history.system_message:
            system_prompt = self.history.system_message.content
        
        # Call LLM with vision
        self._total_requests += 1
        self._vision_requests += 1
        
        try:
            logger.debug(
                f"[ConversationManager] Calling VLM: image={len(image_data)//1024}KB, "
                f"detail={image_detail}, temp={temperature}"
            )
            
            response = await self.llm.generate_structured_with_vision(
                prompt=content,
                image_data=image_data,
                schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            
            # Track token usage
            output_tokens = TokenEstimator.estimate(json.dumps(response)).tokens
            self._total_tokens_used += total_input_tokens + output_tokens
            
            logger.debug(
                f"[ConversationManager] Vision response received: ~{output_tokens:,} output tokens, "
                f"total_used={self._total_tokens_used:,}"
            )
            
            # Add to history (text only, images are not stored)
            if add_to_history:
                # Store a placeholder for the image in history
                self.add_user_message(f"{content}\n[Image: {len(image_data)//1024}KB screenshot]")
                self.add_assistant_message(json.dumps(response))
            
            return response
            
        except Exception as e:
            logger.error(
                f"[ConversationManager] Vision request #{request_id} failed: {e}",
                exc_info=True
            )
            raise
    
    def _estimate_image_tokens(self, image_size_bytes: int, detail: str = "auto") -> int:
        """
        Estimate tokens for an image based on size and detail level.
        
        Based on OpenAI/Anthropic image token calculations:
        - Low detail: Fixed ~85 tokens
        - High detail: 85 base + 170 per 512x512 tile
        - Auto: Estimate based on image size
        
        Args:
            image_size_bytes: Size of image in bytes
            detail: Detail level ("low", "high", "auto")
            
        Returns:
            Estimated token count for the image
        """
        if detail == "low":
            return self.IMAGE_BASE_TOKENS
        
        # Estimate dimensions from file size (rough approximation)
        # Assumes ~3 bytes per pixel for PNG, ~1 byte for compressed JPEG
        estimated_pixels = image_size_bytes // 2  # Middle ground estimate
        estimated_dimension = int(estimated_pixels ** 0.5)
        
        # Calculate tiles (512x512 each)
        tiles_wide = max(1, (estimated_dimension + 511) // 512)
        tiles_high = max(1, (estimated_dimension + 511) // 512)
        total_tiles = tiles_wide * tiles_high
        
        # Cap at reasonable maximum (e.g., 2048x2048 = 16 tiles)
        total_tiles = min(total_tiles, 16)
        
        return self.IMAGE_BASE_TOKENS + (self.IMAGE_TOKENS_PER_TILE * total_tiles)
    
    @property
    def has_vision(self) -> bool:
        """Check if the model supports vision."""
        return self._has_vision
    
    async def send_with_large_content(
        self,
        content: str,
        instruction: str,
        schema: Dict[str, Any],
        temperature: float = 0.7,
        content_type: Optional[ContentType] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send large content that may need to be split across multiple turns.
        
        Implements the accumulation protocol:
        1. If content fits, send as single turn
        2. If too large, chunk and process incrementally
        3. Final synthesis turn produces structured output
        
        Args:
            content: Large content to process
            instruction: What to do with the content
            schema: JSON schema for final response
            temperature: Sampling temperature
            content_type: Optional content type hint
            **kwargs: Additional LLM parameters
            
        Returns:
            Structured response matching schema
        """
        # Estimate content size
        content_estimate = TokenEstimator.estimate(content, content_type)
        
        # Calculate available space (leaving room for instruction and response)
        instruction_tokens = TokenEstimator.estimate(instruction).tokens
        available = self.budget.available_for_input - self.history.total_tokens - instruction_tokens - 500
        
        logger.info(
            f"Processing large content: {content_estimate.tokens} tokens, "
            f"available: {available}, type: {content_estimate.content_type.value}"
        )
        
        # If content fits, send as single turn
        if content_estimate.tokens <= available:
            combined = f"{instruction}\n\n---\nContent to process:\n{content}"
            return await self.send_structured(combined, schema, temperature, **kwargs)
        
        # Need multi-turn processing
        self._multi_turn_requests += 1
        
        # Get appropriate chunker
        chunker = get_chunker(content_type or content_estimate.content_type)
        
        # Calculate chunk size (leave room for accumulation overhead)
        chunk_budget = int(available * 0.7)  # 70% for content, 30% for overhead
        chunks = chunker.chunk(content, chunk_budget)
        
        logger.info(f"Split content into {len(chunks)} chunks for multi-turn processing")
        
        # Initialize accumulation context
        self._accumulation = AccumulationContext(
            phase=AccumulationPhase.ACCUMULATING,
            total_chunks=len(chunks),
            processed_chunks=0,
            original_instruction=instruction,
        )
        
        # Process chunks
        for chunk in chunks:
            await self._process_accumulation_chunk(chunk)
        
        # Synthesis phase
        return await self._synthesize_accumulated(schema, temperature, **kwargs)
    
    async def _process_accumulation_chunk(self, chunk: Chunk) -> None:
        """Process a single chunk during accumulation phase."""
        if not self._accumulation:
            raise ValueError("No accumulation context")
        
        # Build accumulation prompt
        prompt = self._build_accumulation_prompt(chunk)
        
        # Simple acknowledgment schema
        ack_schema = {
            "type": "object",
            "properties": {
                "acknowledged": {"type": "boolean"},
                "key_points": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key points extracted from this chunk"
                }
            },
            "required": ["acknowledged", "key_points"]
        }
        
        try:
            response = await self.send_structured(
                prompt,
                schema=ack_schema,
                temperature=0.3,  # Low temp for consistent accumulation
                add_to_history=False,  # Don't bloat history with chunks
            )
            
            # Store key points for synthesis
            if response.get("key_points"):
                self._accumulation.chunk_summaries.extend(response["key_points"])
            
            self._accumulation.processed_chunks += 1
            
            logger.debug(
                f"Processed chunk {chunk.index + 1}/{chunk.total_chunks}, "
                f"extracted {len(response.get('key_points', []))} key points"
            )
            
        except Exception as e:
            logger.warning(f"Chunk processing failed: {e}, continuing...")
            self._accumulation.processed_chunks += 1
    
    async def _synthesize_accumulated(
        self,
        schema: Dict[str, Any],
        temperature: float,
        **kwargs,
    ) -> Dict[str, Any]:
        """Synthesize final response from accumulated chunks."""
        if not self._accumulation:
            raise ValueError("No accumulation context")
        
        self._accumulation.phase = AccumulationPhase.SYNTHESIZING
        
        # Build synthesis prompt with accumulated key points
        synthesis_prompt = self._build_synthesis_prompt()
        
        logger.info(
            f"Synthesizing from {len(self._accumulation.chunk_summaries)} key points"
        )
        
        try:
            response = await self.send_structured(
                synthesis_prompt,
                schema=schema,
                temperature=temperature,
                **kwargs,
            )
            
            return response
            
        finally:
            # Clear accumulation context
            self._accumulation = None
    
    def _build_accumulation_prompt(self, chunk: Chunk) -> str:
        """Build prompt for accumulation phase."""
        header = chunk.format_header()
        instruction = self._accumulation.original_instruction if self._accumulation else ""
        
        return f"""I'm processing a large document in multiple parts. This is {header}.

INSTRUCTION: {instruction}

For now, just acknowledge receiving this chunk and extract the key points relevant to the instruction.
Do NOT produce the final answer yet - that will come after all chunks are processed.

---
CHUNK CONTENT:
{chunk.content}
---

Extract key points from this chunk that are relevant to the instruction."""
    
    def _build_synthesis_prompt(self) -> str:
        """Build prompt for synthesis phase."""
        if not self._accumulation:
            return ""
        
        key_points = self._accumulation.chunk_summaries
        instruction = self._accumulation.original_instruction
        
        key_points_text = "\n".join(f"- {point}" for point in key_points)
        
        return f"""You have processed a large document in {self._accumulation.total_chunks} chunks.
Here are the key points extracted from all chunks:

{key_points_text}

Now, based on these key points, fulfill the original instruction:
INSTRUCTION: {instruction}

Provide your final, complete response based on all the information gathered."""
    
    def _format_messages_as_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages array as a single prompt string."""
        # For providers that don't support messages array natively
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if isinstance(content, list):
                # Multi-part content (text + images)
                text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                content = "\n".join(text_parts)
            
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        
        return "\n\n".join(parts)
    
    def get_available_tokens(self) -> int:
        """Get tokens available for next message."""
        return self.budget.available_for_input - self.history.total_tokens
    
    def would_exceed_budget(self, content: str) -> tuple[bool, int]:
        """Check if content would exceed budget."""
        content_tokens = TokenEstimator.estimate(content).tokens
        available = self.get_available_tokens()
        overflow = content_tokens - available
        return overflow > 0, max(0, overflow)
    
    def ensure_budget_available(
        self,
        required_tokens: int,
        min_free_tokens: int = 5000,
        aggressive: bool = False,
    ) -> int:
        """
        Ensure sufficient token budget is available by pruning history if needed.
        
        This method should be called before any LLM request to proactively
        clean up history when budget is running low. It implements a tiered
        cleanup strategy:
        
        1. If budget is sufficient, do nothing
        2. If budget is low, prune oldest messages keeping recent context
        3. If still insufficient and aggressive=True, clear most history
        
        Args:
            required_tokens: Tokens needed for the upcoming request
            min_free_tokens: Minimum free tokens to maintain (default 5000)
            aggressive: If True, more aggressively clear history
            
        Returns:
            Number of messages pruned
        """
        available = self.get_available_tokens()
        target_free = max(required_tokens, min_free_tokens)
        
        # Check if we have enough budget
        if available >= target_free:
            return 0
        
        total_pruned = 0
        
        # Phase 1: Standard pruning - remove oldest messages, keep recent 4
        shortfall = target_free - available
        logger.warning(
            f"[ConversationManager] Budget low: {available:,} available, "
            f"need {target_free:,} ({shortfall:,} shortfall). Pruning history."
        )
        
        pruned = self.history.prune_to_fit(
            self.max_history_tokens - shortfall,
            keep_recent=4
        )
        total_pruned += pruned
        
        if pruned > 0:
            logger.info(
                f"[ConversationManager] Pruned {pruned} messages. "
                f"Now have {self.get_available_tokens():,} available."
            )
        
        # Check if still insufficient
        available = self.get_available_tokens()
        if available >= target_free:
            return total_pruned
        
        # Phase 2: Aggressive pruning if enabled - keep only last 2 messages
        if aggressive and self.history.message_count > 2:
            logger.warning(
                f"[ConversationManager] Aggressive cleanup: still need {target_free - available:,} more tokens"
            )
            
            # More aggressive: keep only 2 most recent
            pruned = self.history.prune_to_fit(
                self.max_history_tokens // 4,  # 25% of normal budget
                keep_recent=2
            )
            total_pruned += pruned
            
            if pruned > 0:
                logger.info(
                    f"[ConversationManager] Aggressive pruning removed {pruned} more messages. "
                    f"Now have {self.get_available_tokens():,} available."
                )
        
        return total_pruned
    
    def cleanup_if_needed(self, threshold_percent: float = 0.15) -> int:
        """
        Cleanup history if budget usage exceeds threshold.
        
        This is a convenience method for periodic maintenance cleanup.
        Call this at natural boundaries (e.g., after each ReAct iteration)
        to prevent budget exhaustion.
        
        Args:
            threshold_percent: Cleanup if free budget drops below this % of total
                              (default 15% = cleanup when 85% used)
                              
        Returns:
            Number of messages pruned (0 if no cleanup needed)
        """
        available = self.get_available_tokens()
        total_available = self.budget.available_for_input
        free_percent = available / total_available if total_available > 0 else 1.0
        
        if free_percent >= threshold_percent:
            return 0
        
        logger.debug(
            f"[ConversationManager] Budget at {(1-free_percent)*100:.1f}% used, "
            f"triggering cleanup (threshold: {(1-threshold_percent)*100:.1f}%)"
        )
        
        # Aim to restore to 30% free
        target_free = int(total_available * 0.30)
        return self.ensure_budget_available(target_free, min_free_tokens=target_free)
    
    async def compress_history_if_needed(
        self,
        llm_provider: Optional["BaseLLMProvider"] = None,
        message_threshold: int = 10,
        keep_recent: int = 4,
    ) -> bool:
        """
        Compress older conversation history into a summary message.
        
        Instead of just deleting old messages (losing context), this method:
        1. Takes older messages (beyond keep_recent)
        2. Uses LLM to compress them into a summary
        3. Replaces them with a single summary message
        
        This preserves important context while reducing token count.
        
        Args:
            llm_provider: LLM provider for compression (uses self.llm if not provided)
            message_threshold: Only compress if history exceeds this many messages
            keep_recent: Number of recent messages to exclude from compression
            
        Returns:
            True if compression was performed, False otherwise
        """
        # Check if compression is needed
        if self.history.message_count <= message_threshold:
            return False
        
        # Get messages to compress (excluding system and recent)
        messages_to_compress = self.history.messages[:-keep_recent] if keep_recent > 0 else self.history.messages.copy()
        
        if len(messages_to_compress) < 3:
            # Not enough messages to warrant compression
            return False
        
        provider = llm_provider or self.llm
        
        try:
            from flybrowser.llm.context_compressor import ContextCompressor
            
            # Use higher output tokens for browser automation - we need to preserve
            # URLs, selectors, and action results that the agent depends on
            compressor = ContextCompressor(
                llm_provider=provider,
                compression_temperature=0.1,
                max_output_tokens=1500,  # Higher for browser automation context
            )
            
            # Format messages for compression
            messages_for_compression = [
                {"role": m.role.value, "content": m.content if isinstance(m.content, str) else str(m.content)}
                for m in messages_to_compress
            ]
            
            # Compress
            compressed = await compressor.compress_history(
                messages=messages_for_compression,
                keep_recent=0,  # We've already filtered recent
            )
            
            if compressed.turns_compressed == 0:
                return False
            
            # Create a summary message to replace the compressed ones
            summary_content = compressed.format_as_message()
            summary_msg = ConversationMessage(
                role=MessageRole.USER,
                content=summary_content,
                tokens=TokenEstimator.estimate(summary_content).tokens,
                metadata={"is_compressed_summary": True, "turns_compressed": compressed.turns_compressed}
            )
            
            # Remove old messages and insert summary
            recent_messages = self.history.messages[-keep_recent:] if keep_recent > 0 else []
            self.history.messages.clear()
            self.history.messages.append(summary_msg)
            self.history.messages.extend(recent_messages)
            
            logger.info(
                f"[ConversationManager] Compressed {compressed.turns_compressed} history turns: "
                f"{compressed.original_tokens:,} → {compressed.compressed_tokens:,} tokens"
            )
            
            return True
            
        except Exception as e:
            logger.warning(f"[ConversationManager] History compression failed: {e}")
            return False
    
    def reset(self) -> None:
        """Reset conversation (clear history, keep system prompt)."""
        self.history.clear()
        self._accumulation = None
        self.budget.reset()
        logger.debug("[ConversationManager] Conversation reset")
    
    def full_reset(self) -> None:
        """Full reset including system prompt."""
        self.history.reset()
        self._accumulation = None
        self.budget.reset()
        logger.debug("Full conversation reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        return {
            "total_requests": self._total_requests,
            "vision_requests": self._vision_requests,
            "multi_turn_requests": self._multi_turn_requests,
            "total_tokens_used": self._total_tokens_used,
            "history_messages": self.history.message_count,
            "history_tokens": self.history.total_tokens,
            "available_tokens": self.get_available_tokens(),
            "has_vision": self._has_vision,
            "model": self.model_info.name,
            "budget_stats": self.budget.get_stats(),
        }
