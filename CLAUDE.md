# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

tinygrad is a minimal, end-to-end deep learning framework. It provides a tensor library with autograd, an IR-based compiler, JIT execution, and neural network modules. The philosophy is intentional minimalism: every line must earn its keep, prefer readability over cleverness.

## Common Commands

### Testing
```bash
python3 -m pytest test/                    # Full test suite
python3 test/test_ops.py                   # Operator tests
python3 test/test_tiny.py                  # Core tests (must always pass)
python3 -m pytest test/test_ops.py -k "test_add"  # Run specific test

# Pre-commit subset (fast)
OMP_NUM_THREADS=1 python3 -m pytest -n=6 test/test_ops.py test/test_dtype.py test/test_schedule.py test/test_assign.py
```

### Linting
```bash
python3 -m ruff check .                    # Code style
python3 -m mypy tinygrad/ --strict-equality  # Type checking
pre-commit run --all-files                 # All checks
```

### Installation
```bash
python3 -m pip install -e .                # Basic
python3 -m pip install -e '.[testing]'     # With test deps
python3 -m pip install -e '.[linting]'     # With lint deps
```

## Code Style

- **2-space indentation** (not 4)
- **150 character max line length**
- Match existing style throughout
- No code golf - avoid overly clever/compact code
- Never mix functionality changes with whitespace changes

## Architecture

```
User Code (Tensor API)
    ↓
tinygrad.tensor.Tensor (lazy evaluation, builds computation graph)
    ↓
UOp (micro-operations, intermediate representation in tinygrad/uop/)
    ↓
Scheduler (tinygrad/schedule/) → fuses operations into kernels
    ↓
Codegen (tinygrad/codegen/) → generates device-specific code
    ↓
Runtime Backend (tinygrad/runtime/ops_*.py) → executes on hardware
```

Key directories:
- `tinygrad/tensor.py` - Main Tensor class (~6000 lines), the primary abstraction
- `tinygrad/uop/` - UOp intermediate representation
- `tinygrad/engine/` - JIT, scheduling, memory management
- `tinygrad/runtime/` - Device backends (CPU, CUDA, Metal, AMD, etc.)
- `tinygrad/nn/` - Neural network modules, optimizers, ONNX import
- `extra/` - Non-core utilities (avoid changing, not well tested)

## Debugging Environment Variables

- `DEBUG={1-5}` - Verbosity (3+ shows fused kernels, 4+ shows generated code)
- `CPU=1` - Force CPU backend
- `BEAM=N` - Search over N kernel schedules

## Testing Patterns

Use `helper_test_op()` from test/helpers.py for tensor operation tests - it compares against PyTorch for correctness and tests gradients.

For refactors with no behavior change, include `[pr]` in PR title to enable process replay testing.

## Contributing Guidelines

PRs should be small and focused. What gets PRs closed:
- Code golf (clever but unreadable)
- Pure whitespace changes
- Unsubstantiated performance claims (must be benchmarked)
- Large/complex PRs - break into smaller wins

What's valued:
- Bug fixes with regression tests
- Features matching PyTorch/NumPy API
- Clear refactors improving readability
- New tests and fuzzers
- Dead code removal from core
