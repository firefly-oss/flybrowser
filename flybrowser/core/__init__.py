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

"""Core browser automation components."""

from flybrowser.core.browser import BrowserManager
from flybrowser.core.browser_pool import (
    BrowserPool,
    BrowserSession,
    BrowserSessionState,
    Job,
    JobQueue,
    JobState,
    PoolConfig,
)
from flybrowser.core.page import PageController
from flybrowser.core.recording import (
    RecordingConfig,
    RecordingManager,
    RecordingState,
    Screenshot,
    ScreenshotCapture,
    ScreenshotFormat,
    VideoRecorder,
    VideoRecording,
)

__all__ = [
    "BrowserManager",
    "BrowserPool",
    "BrowserSession",
    "BrowserSessionState",
    "Job",
    "JobQueue",
    "JobState",
    "PageController",
    "PoolConfig",
    "RecordingConfig",
    "RecordingManager",
    "RecordingState",
    "Screenshot",
    "ScreenshotCapture",
    "ScreenshotFormat",
    "VideoRecorder",
    "VideoRecording",
]

