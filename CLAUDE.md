# CLAUDE.md

This repository is a fork of [llama.cpp](https://github.com/ggerganov/llama.cpp). Custom work is developed on the `ubp` branch.

## Repository Overview

- **Build system**: CMake (primary), Makefile (legacy)
- **Language**: C/C++ with CUDA, Metal, Vulkan, and other backend extensions
- **Testing**: CTest-based tests in the `tests/` directory

## Branch Strategy

- **`master`**: tracks upstream `ggerganov/llama.cpp` (fast-forward only, no custom commits)
- **`ubp`**: active development branch — all custom work goes here
- Never push directly to `master`. Push only to the current working branch (e.g., `ubp`).

## Syncing with Upstream

```bash
git fetch upstream
git checkout master
git merge --ff-only upstream/master
git checkout ubp
git rebase master
```

## Commit Guidelines

- Commit when a meaningful unit of work is complete.
- Run relevant tests/build before committing and do not commit in a state that breaks previously passing tests.
- Use **Conventional Commits** style for all commit messages:

```
<type>(<scope>): <short summary>

[optional body]

[optional footer]
```

Common types: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`, `build`

Examples:
```
feat(ubp): add unified batch processing scheduler
fix(cuda): correct memory alignment in MMVQ kernel
perf(ggml): reduce allocation overhead in metal backend
```

## Build

```bash
cmake -B build -DLLAMA_CUDA=ON   # or relevant backend flags
cmake --build build --parallel
```

## Running Tests

```bash
cd build && ctest --output-on-failure
```

Run specific test suites before committing changes to related components:
```bash
ctest -R <test-pattern> --output-on-failure
```
