#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# re-validate login information
mkdir -p ./.auth
uv run python ./visualwebarena/browser_env/auto_login.py