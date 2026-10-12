# Changelog — vLLM PagedAttention Studio

All notable changes to this project are documented in this file.

## [1.2.0] — 2026-10-12
### Added
- Complete 9+ DevSecOps Governance Suite (.gitignore, .env.example, pre-commit, dependabot, docs suite).
- Cleaned git tracking to exclude `__pycache__` and temporary build artifacts.
- Synchronized architecture references with 341-studio developer ecosystem.

## [1.1.0] — 2026-06-25
### Added
- Paged cache memory engine (`paged_cache_engine.py`) and continuous batcher (`continuous_batcher.ts`).
- Kubernetes H100 deployment manifest (`k8s-vllm-h100.yaml`).
- SRE architecture flow diagram (`docs/vllm_architecture_flow.png`).

## [1.0.0] — 2026-06-25
### Added
- Initial workspace release.
