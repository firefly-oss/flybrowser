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

"""FlyBrowser CLI tools.

This module provides command-line tools for:
- Unified CLI (flybrowser command)
- Interactive REPL for browser automation
- Project setup and configuration
- Service management
- Cluster deployment
- Administrative commands

CLI Entry Points:
    flybrowser           Unified CLI (recommended)
    flybrowser-setup     Setup and configuration
    flybrowser-serve     API server
    flybrowser-cluster   Cluster management
    flybrowser-admin     Administrative tasks
"""

from typing import TYPE_CHECKING

# Use lazy imports to avoid RuntimeWarning when running modules directly
# (e.g., python -m flybrowser.cli.setup)
if TYPE_CHECKING:
    from flybrowser.cli.setup import setup_wizard, generate_config

__all__ = [
    "setup_wizard",
    "generate_config",
]


def __getattr__(name: str):
    """Lazy import attributes to avoid import cycles and RuntimeWarnings."""
    if name == "setup_wizard":
        from flybrowser.cli.setup import setup_wizard
        return setup_wizard
    elif name == "generate_config":
        from flybrowser.cli.setup import generate_config
        return generate_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

