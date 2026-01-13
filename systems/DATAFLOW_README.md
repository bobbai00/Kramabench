# DataflowSystem for KramaBench

This document describes how to run KramaBench using the Texera DataflowAgent.

## Prerequisites

1. **Texera Services Running**: Ensure the following services are running:
   - Texera Backend API (default: `http://localhost:8080`)
   - Texera Computing Unit (default: `http://localhost:8888`)
   - Texera Agent Service (default: `http://localhost:3001`)

2. **Python Environment**: Set up a virtual environment with dependencies:
   ```bash
   cd KramaBench
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Environment Variables** (if using other baseline systems):
   ```bash
   export OPENAI_API_KEY=sk-your-key      # For GPT-based systems
   export ANTHROPIC_API_KEY=your-key      # For Claude-based systems
   ```

## Running the Benchmark

### Basic Usage

```bash
# Activate virtual environment
source .venv/bin/activate

# Run with DataflowSystem on a small test workload
python evaluate.py \
  --sut DataflowSystem \
  --workload legal-tiny \
  --no_pipeline_eval \
  --verbose
```

### Full Domain Evaluation

```bash
# Legal domain (30 tasks)
python evaluate.py --sut DataflowSystem --workload legal --no_pipeline_eval --verbose

# Other domains
python evaluate.py --sut DataflowSystem --workload astronomy --no_pipeline_eval --verbose
python evaluate.py --sut DataflowSystem --workload environment --no_pipeline_eval --verbose
python evaluate.py --sut DataflowSystem --workload biomedical --no_pipeline_eval --verbose
python evaluate.py --sut DataflowSystem --workload archeology --no_pipeline_eval --verbose
python evaluate.py --sut DataflowSystem --workload wildfire --no_pipeline_eval --verbose
```

### Available System Variants

| System Class | Model | Description |
|--------------|-------|-------------|
| `DataflowSystem` | claude-sonnet-4-5 | Default configuration |
| `DataflowSystemHaiku` | claude-haiku-4.5 | Faster, lower cost |
| `DataflowSystemSonnet` | claude-sonnet-4-5 | Same as default |
| `DataflowSystemGPT` | gpt-5-mini | GPT-based variant |

```bash
# Use a specific variant
python evaluate.py --sut DataflowSystemHaiku --workload legal-tiny --no_pipeline_eval --verbose
```

## Command-Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--sut` | Required | System class name (e.g., `DataflowSystem`) |
| `--workload` | Required | Domain name (e.g., `legal`, `legal-tiny`) |
| `--no_pipeline_eval` | False | Skip LLM-based code evaluation (recommended for DataflowSystem) |
| `--verbose` | False | Enable detailed logging |
| `--use_system_cache` | False | Reuse previous system outputs |
| `--use_evaluation_cache` | False | Reuse previous evaluation results |
| `--run_subtasks` | False | Also evaluate subtasks |

## Output Files

Results are saved to:

```
results/
├── DataflowSystem/
│   ├── legal-tiny_measures_YYYYMMDD_HHMMSS.csv   # Per-task metrics
│   └── response_cache/                            # Cached responses
└── aggregated_results.csv                         # Summary across domains

system_scratch/
└── DataflowSystem/
    └── {task_id}/
        ├── prompt.txt       # Prompt sent to agent
        ├── response.txt     # Agent's response
        └── messages.json    # Full conversation history
```

## Configuration

To modify DataflowAgent settings, edit `systems/dataflow_system.py` or create a new variant:

```python
from systems.dataflow_system import DataflowSystem

class MyCustomDataflowSystem(DataflowSystem):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(
            model_type="claude-sonnet-4-5",  # LLM model
            max_steps=100,                    # Max agent steps per query
            name="MyCustomDataflowSystem",
            verbose=verbose,
            *args, **kwargs
        )
```

You can also modify the Texera endpoints in `dataflow_agent.py`:

```python
TEXERA_API_ENDPOINT = "http://localhost:8080"
TEXERA_COMPUTING_UNIT_ENDPOINT = "http://localhost:8888"
TEXERA_AGENT_SERVICE_ENDPOINT = "http://localhost:3001"
```

## Why `--no_pipeline_eval`?

KramaBench has two evaluation modes:

1. **Answer Evaluation** (always runs): Compares final answers using metrics like F1, RAE, exact match
2. **Pipeline Evaluation** (optional): Uses GPT-4o-mini to check if generated Python code implements required functionalities

Since DataflowSystem generates **dataflow workflows** (not Python code), the pipeline evaluation is not applicable. Use `--no_pipeline_eval` to skip it.

## Troubleshooting

### Agent returns empty response
- Increase `max_steps` in `DataflowSystem.__init__()` (default: 100)
- Check if Texera services are running: `curl http://localhost:3001/api/agents/`

### Import errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Set a dummy OpenAI key if not using OpenAI: `export OPENAI_API_KEY=sk-dummy`

### Connection refused
- Verify Texera services are running on the expected ports
- Check firewall settings
