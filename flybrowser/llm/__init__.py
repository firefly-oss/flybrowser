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

"""LLM integration layer for FlyBrowser.

Remaining modules (base, factory, provider_status, context_compressor) are
kept for backward compatibility but have transitive dependencies on files
removed in Task 16. They will be lazily importable once Task 17 completes
the SDK switch.
"""

__all__: list[str] = []
