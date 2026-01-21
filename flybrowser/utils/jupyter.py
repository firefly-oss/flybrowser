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
Jupyter Notebook Support Utilities

This module provides utilities for using FlyBrowser in Jupyter notebooks.
Jupyter notebooks run their own event loop, which can conflict with asyncio.run().
This module provides tools to detect and configure the environment for Jupyter.
"""

import sys
from typing import Optional


def is_jupyter_environment() -> bool:
    """
    Detect if code is running in a Jupyter notebook environment.
    
    Returns:
        True if running in Jupyter/IPython, False otherwise
    """
    try:
        # Check for IPython
        from IPython import get_ipython
        ipython = get_ipython()
        
        if ipython is None:
            return False
        
        # Check if it's a ZMQInteractiveShell (Jupyter) or TerminalInteractiveShell
        ipython_type = type(ipython).__name__
        return ipython_type in ['ZMQInteractiveShell', 'InteractiveShell']
    except (ImportError, NameError):
        return False


def setup_jupyter(force: bool = False) -> bool:
    """
    Configure the environment for Jupyter notebook usage.
    
    This function applies nest_asyncio to allow nested event loops,
    which is necessary when using asyncio.run() in Jupyter notebooks.
    
    Args:
        force: If True, apply nest_asyncio even if not in Jupyter environment
        
    Returns:
        True if configuration was applied, False otherwise
        
    Raises:
        ImportError: If nest_asyncio is not installed
        
    Example:
        >>> from flybrowser.utils.jupyter import setup_jupyter
        >>> setup_jupyter()
        True
        >>> # Now you can use asyncio.run() in your notebook
        >>> import asyncio
        >>> asyncio.run(main())
    """
    if not force and not is_jupyter_environment():
        return False
    
    try:
        import nest_asyncio
    except ImportError:
        raise ImportError(
            "nest_asyncio is required for Jupyter notebook support. "
            "Install it with: pip install flybrowser[jupyter]"
        )
    
    nest_asyncio.apply()
    return True


def auto_setup_jupyter() -> Optional[str]:
    """
    Automatically configure Jupyter if detected.
    
    This is a convenience function that detects Jupyter and applies
    the necessary configuration without requiring explicit setup.
    
    Returns:
        A message string if configuration was applied, None otherwise
        
    Example:
        >>> from flybrowser.utils.jupyter import auto_setup_jupyter
        >>> auto_setup_jupyter()
        'Jupyter environment detected. Applied nest_asyncio for compatibility.'
    """
    if is_jupyter_environment():
        try:
            setup_jupyter()
            return "Jupyter environment detected. Applied nest_asyncio for compatibility."
        except ImportError:
            return (
                "Jupyter environment detected but nest_asyncio is not installed. "
                "Install with: pip install flybrowser[jupyter] or use 'await' directly instead of asyncio.run()"
            )
    return None


def get_usage_tip() -> str:
    """
    Get usage tips for the current environment.
    
    Returns:
        Usage tip string appropriate for the environment
    """
    if is_jupyter_environment():
        return (
            "Jupyter notebook detected! You have two options:\n"
            "1. Use 'await' directly (recommended):\n"
            "   async with FlyBrowser(...) as browser:\n"
            "       await browser.goto('https://example.com')\n"
            "\n"
            "2. Use asyncio.run() with nest_asyncio:\n"
            "   from flybrowser.utils.jupyter import setup_jupyter\n"
            "   setup_jupyter()\n"
            "   asyncio.run(main())"
        )
    else:
        return (
            "Use asyncio.run() in standard Python scripts:\n"
            "   asyncio.run(main())"
        )
