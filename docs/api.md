# API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/system/health` | GET | Liveness probe |
| `/api/system/config` | GET | Current configuration |
| `/api/tasks` | POST | Submit a new engineering task |
| `/api/tasks` | GET | List all tasks |
| `/api/tasks/{id}` | GET | Get a specific task |
| `/api/tasks/{id}/events` | GET | Stream of events for a task |
| `/api/projects` | GET/POST | Manage projects |
| `/api/datasets` | GET | List datasets |
| `/api/models` | GET | List models |
| `/api/training/start` | POST | Start a training run |
| `/api/evaluation/run` | POST | Run an evaluation |
| `/api/deployment/serve` | POST | Deploy a model for serving |
| `/ws/tasks/{id}` | WS | Real-time event stream |
