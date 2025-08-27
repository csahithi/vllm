# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# This conftest.py is now simplified since we use package-scoped servers
# in each test group (basic_tests, embedding_tests, multimodal_tests, advanced_tests)
# 
# Each test group has its own conftest.py with appropriate server configurations:
# - basic_tests: Simple server for core API functionality
# - embedding_tests: Pooling runner for embedding operations  
# - multimodal_tests: Specialized servers for vision/audio/video
# - advanced_tests: Factory pattern for custom server needs
#
# This approach prevents OOM errors by ensuring only one server per test group
# and allows parallel execution of different test groups.

