# Per-model environment overrides for DeepSeek Proxy se0-e256-h4096.
# Sourced after train_env.sh, before CLI _env_ overrides.
#
# At 48N (production scale) with dp=6, RCCL requires more memory for
# communication buffers across the larger data-parallel group.
# .85 leaves enough headroom for RCCL allocations.
export XLA_PYTHON_CLIENT_MEM_FRACTION=.85
