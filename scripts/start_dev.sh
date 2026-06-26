#!/usr/bin/env bash
# Copyright 2026 Matt Dyer / Dyer-Tech
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
set -euo pipefail

echo "Starting AI Engineer dev environment..."

# Start infrastructure
docker compose up -d postgres redis qdrant neo4j minio mlflow

echo "Waiting for services to be ready..."
sleep 10

# Run migrations
python scripts/init_db.py

# Seed initial memory
python scripts/seed_memory.py || echo "Skipping seed"

# Start API in background
uvicorn ai_engineer.api.server:app --reload --host 0.0.0.0 --port 8080 &
API_PID=$!

# Start UI
streamlit run src/ai_engineer/ui/app.py --server.port 8501 --server.address 0.0.0.0 &
UI_PID=$!

trap "kill $API_PID $UI_PID" EXIT
wait
