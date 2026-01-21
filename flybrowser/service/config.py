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
Service configuration management for FlyBrowser.

This module provides centralized configuration for the FlyBrowser service,
including browser pool settings, cluster mode configuration, and deployment options.
Configuration can be loaded from environment variables, config files, or programmatically.

Example:
    >>> from flybrowser.service.config import ServiceConfig
    >>> config = ServiceConfig()  # Loads from environment
    >>> print(config.pool.max_size)
    10
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DeploymentMode(str, Enum):
    """Deployment mode for FlyBrowser service."""
    
    STANDALONE = "standalone"  # Single node deployment
    CLUSTER = "cluster"  # Multi-node cluster deployment


class BrowserPoolConfig(BaseModel):
    """Configuration for the browser pool.
    
    Attributes:
        min_size: Minimum number of browser instances to maintain
        max_size: Maximum number of browser instances allowed
        idle_timeout_seconds: Time before idle browsers are closed
        max_session_age_seconds: Maximum lifetime of a browser session
        startup_timeout_seconds: Timeout for browser startup
        shutdown_timeout_seconds: Timeout for browser shutdown
        health_check_interval_seconds: Interval between health checks
        headless: Whether to run browsers in headless mode
        browser_type: Default browser type (chromium, firefox, webkit)
    """
    
    min_size: int = Field(default=1, ge=0, le=100, description="Minimum pool size")
    max_size: int = Field(default=10, ge=1, le=100, description="Maximum pool size")
    idle_timeout_seconds: float = Field(default=300.0, ge=30.0, description="Idle timeout")
    max_session_age_seconds: float = Field(default=3600.0, ge=60.0, description="Max session age")
    startup_timeout_seconds: float = Field(default=30.0, ge=5.0, description="Startup timeout")
    shutdown_timeout_seconds: float = Field(default=10.0, ge=1.0, description="Shutdown timeout")
    health_check_interval_seconds: float = Field(default=60.0, ge=10.0, description="Health check interval")
    headless: bool = Field(default=True, description="Run browsers in headless mode")
    browser_type: str = Field(default="chromium", description="Default browser type")


class ClusterNodeConfig(BaseModel):
    """Configuration for a cluster node.
    
    Attributes:
        node_id: Unique identifier for this node
        host: Host address for this node
        port: Port for inter-node communication
        role: Node role (coordinator or worker)
        max_browsers: Maximum browsers this node can handle
    """
    
    node_id: str = Field(default="", description="Unique node identifier")
    host: str = Field(default="0.0.0.0", description="Node host address")
    port: int = Field(default=8001, ge=1024, le=65535, description="Node communication port")
    role: str = Field(default="worker", description="Node role: coordinator or worker")
    max_browsers: int = Field(default=10, ge=1, description="Max browsers for this node")


class RaftConfig(BaseModel):
    """Configuration for Raft consensus in HA cluster mode.

    Attributes:
        bind_host: Host to bind Raft RPC server
        bind_port: Port to bind Raft RPC server
        election_timeout_min_ms: Minimum election timeout
        election_timeout_max_ms: Maximum election timeout
        heartbeat_interval_ms: Leader heartbeat interval
        data_dir: Directory for Raft persistent data
    """

    bind_host: str = Field(default="0.0.0.0", description="Raft bind host")
    bind_port: int = Field(default=4321, ge=1024, le=65535, description="Raft bind port")
    election_timeout_min_ms: int = Field(default=300, ge=100, description="Min election timeout")
    election_timeout_max_ms: int = Field(default=500, ge=200, description="Max election timeout")
    heartbeat_interval_ms: int = Field(default=100, ge=50, description="Heartbeat interval")
    data_dir: str = Field(default="./data/raft", description="Raft data directory")


class ClusterConfig(BaseModel):
    """Configuration for cluster mode deployment.

    Supports both legacy coordinator/worker mode and new HA Raft mode.

    Attributes:
        enabled: Whether cluster mode is enabled
        mode: Cluster mode (legacy or ha)
        peers: List of peer nodes (host:raft_port:api_port format)
        raft: Raft consensus configuration
        node: Configuration for this node
        discovery_method: How nodes discover each other (static, dns, kubernetes)
        heartbeat_interval_seconds: Interval between heartbeat messages
        node_timeout_seconds: Time before a node is considered dead
        load_balancing_strategy: Load balancing strategy (least_load, round_robin, etc.)
    """

    enabled: bool = Field(default=False, description="Enable cluster mode")
    mode: str = Field(default="ha", description="Cluster mode: ha (Raft) or legacy")
    peers: List[str] = Field(default_factory=list, description="Peer nodes (host:raft_port:api_port)")
    raft: RaftConfig = Field(default_factory=RaftConfig)
    node: ClusterNodeConfig = Field(default_factory=ClusterNodeConfig)
    discovery_method: str = Field(default="static", description="Node discovery method")
    heartbeat_interval_seconds: float = Field(default=5.0, description="Heartbeat interval")
    node_timeout_seconds: float = Field(default=30.0, description="Node timeout")
    load_balancing_strategy: str = Field(default="least_load", description="Load balancing strategy")


class ServiceConfig(BaseSettings):
    """Main service configuration loaded from environment variables.
    
    Environment variables are prefixed with FLYBROWSER_ and use uppercase.
    Nested configs use double underscore as separator.
    
    Example:
        FLYBROWSER_HOST=0.0.0.0
        FLYBROWSER_PORT=8000
        FLYBROWSER_POOL__MAX_SIZE=20
        FLYBROWSER_CLUSTER__ENABLED=true
    """
    
    # Service settings
    host: str = Field(default="0.0.0.0", description="Service host")
    port: int = Field(default=8000, ge=1024, le=65535, description="Service port")
    workers: int = Field(default=1, ge=1, le=32, description="Number of workers")
    env: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Session settings
    max_sessions: int = Field(default=100, ge=1, description="Maximum concurrent sessions")
    session_timeout: int = Field(default=3600, ge=60, description="Session timeout in seconds")
    
    # Deployment mode
    deployment_mode: DeploymentMode = Field(
        default=DeploymentMode.STANDALONE,
        description="Deployment mode: standalone or cluster"
    )
    
    # Browser pool configuration
    pool: BrowserPoolConfig = Field(default_factory=BrowserPoolConfig)
    
    # Cluster configuration
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    
    # API settings
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    cors_origins: List[str] = Field(default_factory=lambda: ["*"], description="CORS origins")
    
    # Recording settings
    recording_enabled: bool = Field(default=True, description="Enable recording features")
    recording_output_dir: str = Field(default="./recordings", description="Recording output directory")
    recording_storage: str = Field(default="local", description="Storage backend: local, s3, or shared")
    recording_retention_days: int = Field(default=7, ge=1, description="Recording retention in days")
    
    # FFmpeg settings
    ffmpeg_path: Optional[str] = Field(default=None, description="Path to ffmpeg binary")
    ffmpeg_enable_hw_accel: bool = Field(default=True, description="Enable hardware acceleration")
    
    # Streaming settings
    streaming_enabled: bool = Field(default=True, description="Enable streaming features")
    streaming_base_url: Optional[str] = Field(default=None, description="Base URL for streaming (HLS/DASH)")
    
    # S3 storage settings (for cluster mode)
    s3_bucket: Optional[str] = Field(default=None, description="S3 bucket name for recordings")
    s3_region: str = Field(default="us-east-1", description="S3 region")
    s3_endpoint_url: Optional[str] = Field(default=None, description="S3 endpoint URL (for MinIO)")
    s3_access_key: Optional[str] = Field(default=None, description="S3 access key")
    s3_secret_key: Optional[str] = Field(default=None, description="S3 secret key")
    s3_prefix: str = Field(default="recordings/", description="S3 key prefix for recordings")
    
    # PII settings
    pii_masking_enabled: bool = Field(default=True, description="Enable PII masking")
    
    model_config = {
        "env_prefix": "FLYBROWSER_",
        "env_nested_delimiter": "__",
        "case_sensitive": False,
    }


# Global configuration instance
_config: Optional[ServiceConfig] = None


def get_config() -> ServiceConfig:
    """Get the global service configuration.
    
    Returns:
        ServiceConfig instance loaded from environment
    """
    global _config
    if _config is None:
        _config = ServiceConfig()
    return _config


def reload_config() -> ServiceConfig:
    """Reload configuration from environment.

    Returns:
        Fresh ServiceConfig instance
    """
    global _config
    _config = ServiceConfig()
    return _config


def load_config_from_file(path: str) -> ServiceConfig:
    """Load configuration from a YAML or JSON file.

    Args:
        path: Path to configuration file

    Returns:
        ServiceConfig instance

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    import json
    import yaml

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            data = yaml.safe_load(f)
        elif path.endswith(".json"):
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path}")

    global _config
    _config = ServiceConfig(**data)
    return _config


def save_config_to_file(config: ServiceConfig, path: str) -> None:
    """Save configuration to a YAML or JSON file.

    Args:
        config: Configuration to save
        path: Path to save to
    """
    import json
    import yaml

    data = config.model_dump()

    # Create directory if needed
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w") as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            yaml.dump(data, f, default_flow_style=False)
        elif path.endswith(".json"):
            json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {path}")


def create_ha_node_config(service_config: ServiceConfig) -> "HANodeConfig":
    """Create HANodeConfig from ServiceConfig.

    Args:
        service_config: Service configuration

    Returns:
        HANodeConfig for HA cluster mode
    """
    from flybrowser.service.cluster.ha_node import HANodeConfig
    from flybrowser.service.cluster.load_balancer import LoadBalancingStrategy

    # Map strategy string to enum
    strategy_map = {
        "least_load": LoadBalancingStrategy.LEAST_LOAD,
        "round_robin": LoadBalancingStrategy.ROUND_ROBIN,
        "least_connections": LoadBalancingStrategy.LEAST_CONNECTIONS,
        "random": LoadBalancingStrategy.RANDOM,
        "weighted": LoadBalancingStrategy.WEIGHTED,
    }

    return HANodeConfig(
        node_id=service_config.cluster.node.node_id,
        api_host=service_config.host,
        api_port=service_config.port,
        raft_host=service_config.cluster.raft.bind_host,
        raft_port=service_config.cluster.raft.bind_port,
        peers=service_config.cluster.peers,
        data_dir=service_config.cluster.raft.data_dir,
        max_sessions=service_config.max_sessions,
        lb_strategy=strategy_map.get(
            service_config.cluster.load_balancing_strategy,
            LoadBalancingStrategy.LEAST_LOAD
        ),
    )


def create_storage_backend(service_config: Optional[ServiceConfig] = None) -> "RecordingStorage":
    """Create recording storage backend from configuration.
    
    Args:
        service_config: Service configuration (uses global if not provided)
        
    Returns:
        RecordingStorage instance
    """
    from flybrowser.service.cluster.storage import StorageBackend, create_storage_backend as _create
    
    if service_config is None:
        service_config = get_config()
    
    backend = StorageBackend(service_config.recording_storage)
    
    if backend == StorageBackend.LOCAL:
        return _create(backend, base_dir=service_config.recording_output_dir)
    
    elif backend == StorageBackend.S3:
        if not service_config.s3_bucket:
            raise ValueError("S3 storage requires s3_bucket configuration")
        
        return _create(
            backend,
            bucket=service_config.s3_bucket,
            region=service_config.s3_region,
            endpoint_url=service_config.s3_endpoint_url,
            access_key=service_config.s3_access_key or os.environ.get("AWS_ACCESS_KEY_ID"),
            secret_key=service_config.s3_secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY"),
            prefix=service_config.s3_prefix,
        )
    
    elif backend == StorageBackend.SHARED:
        return _create(backend, base_dir=service_config.recording_output_dir)
    
    else:
        raise ValueError(f"Unknown storage backend: {backend}")
