#!/bin/bash

# Thin wrapper — delegates to tgs_tagger.py while preserving the `tag_tgs`
# function name for interactive use (e.g. sourced in .bashrc).

_TAG_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

tag_tgs() {
  python3 "$_TAG_SCRIPT_DIR/tgs_tagger.py" "$@"
}

tag_tgs "$@"
