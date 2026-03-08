# Per-model environment overrides for Grok-2.
# Sourced after train_env.sh, before CLI _env_ overrides.
#
# Grok-2 requires router_logits_soft_cap which is not yet in upstream MaxText.
export MAXTEXT_PATCH_BRANCH="yihuang/grok2-support"
