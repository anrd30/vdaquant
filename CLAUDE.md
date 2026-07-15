# CLAUDE.md - vdaquant Multi-Model Orchestration & Workflow

## 1. Team Architecture & Routing
- **Lead Architect (Orchestrator)**: Claude Fable 5. 
  - Handles high-level system architecture, mathematical verification, and final PR reviews.
  - *Prohibition*: Never write long, routine implementation blocks. Delegate to Sonnet.
- **Implementation Worker**: Claude 3.5 Sonnet.
  - Handles Python scripts, data processing, loop writing, and writing unit tests.
  - Automatically triggered via subagent panel or custom script delegation.

## 2. Technical Stack & Focus Areas
- **Domain**: Training-free extreme vector/lattice quantization for Video Vision Transformers.
- **Base Architecture**: Video-Depth-Anything (VDA).
- **Core Modules**:
  - `research/` - Hadamard transforms, matrix surgeries, scale/shift quantizers.
  - `notebooks/` - Integration testing on small validation datasets.
  - `scripts/` - Production benchmarking code.

## 3. Strict Development Workflow (4-Step Loop)
1. **Plan (Fable)**: Fable analyzes the repository state and writes strict mathematical boundary specs into `docs/optimization_ledger.md`.
2. **Build (Sonnet)**: Sonnet is invoked to implement the math inside `research/` or `scripts/`, generating matching unit tests.
3. **Validate (VS Code + Colab)**: User or agent executes the notebook in VS Code using the cloud Colab remote kernel. Outputs are written directly to `results/colab_test_log.json`.
4. **Audit & Merge (Fable)**: Fable reads the JSON log diff. If tolerances pass, it approves merging to the `main` branch.

## 4. Coding & Tensor Conventions
- **Precision**: Enforce strict dimension, shape, and dtype verification (`torch.float32` vs `torch.float16`) at the start of any matrix surgery.
- **Safety Assertions**: Every custom Hadamard transformation layer must include shape-matching `assert` statements.
- **Caching**: Ensure dataloader batches clear device caches (`torch.cuda.empty_cache()`) to prevent Colab and GPU server runtime out-of-memory (OOM) errors.

## 5. Automation & Tool Commands
- **Local Testing command**: `pytest tests/`
- **Headless Notebook testing**: `papermill notebooks/colab_research.ipynb results/output_log.ipynb`
- **GPU Deployment pull**: `git checkout main && git pull origin main`
