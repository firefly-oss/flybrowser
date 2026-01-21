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
FlyBrowser service layer for REST API and cluster deployment.

This module provides:
- REST API service (FastAPI-based)
- Session management
- Authentication
- Cluster mode support (coordinator and worker nodes)
- Configuration management

Deployment Modes:
    Standalone (default):
        Single node deployment, suitable for development and small workloads.

    Cluster:
        Multi-node deployment with coordinator and worker nodes.
        Enables horizontal scaling across multiple machines.

Example:
    Standalone mode:
    >>> uvicorn flybrowser.service.app:app --host 0.0.0.0 --port 8000

    Cluster coordinator:
    >>> FLYBROWSER_DEPLOYMENT_MODE=cluster FLYBROWSER_CLUSTER__NODE__ROLE=coordinator \\
    ...     uvicorn flybrowser.service.app:app --port 8001

    Cluster worker:
    >>> FLYBROWSER_DEPLOYMENT_MODE=cluster FLYBROWSER_CLUSTER__COORDINATOR_HOST=coordinator \\
    ...     uvicorn flybrowser.service.app:app --port 8000
"""

from flybrowser.service.config import (
    ServiceConfig,
    DeploymentMode,
    BrowserPoolConfig,
    ClusterConfig,
    get_config,
)
from flybrowser.service.session_manager import SessionManager

__all__ = [
    "ServiceConfig",
    "DeploymentMode",
    "BrowserPoolConfig",
    "ClusterConfig",
    "SessionManager",
    "get_config",
]

