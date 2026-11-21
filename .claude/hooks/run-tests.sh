#!/usr/bin/env bash
# Claude Code Stop hook: run the test suite when Claude finishes a turn that
# changed code (src/, tests/, Cargo.toml, Cargo.lock).
#
# Skipping:
#   - Slow tests carry #[ignore = "slow"]; plain `cargo test` (what this hook
#     runs) skips them. Run them with `cargo test -- --include-ignored`.
#   - MUSICRS_SKIP_HOOK_TESTS=1 in the environment Claude Code was started
#     from disables the hook entirely.
#
# Exit 2 with the failure output on stderr makes Claude keep working on it.
set -uo pipefail

input=$(cat)
[[ "${MUSICRS_SKIP_HOOK_TESTS:-0}" == "1" ]] && exit 0

cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}" || exit 0

stamp_dir=target/.claude-hook
mkdir -p "$stamp_dir"
paths=(src tests Cargo.toml Cargo.lock)

# Cheap fingerprint of the code: name, size and mtime of every tracked or
# untracked (non-ignored) file under the code paths.
fingerprint=$(git ls-files -co --exclude-standard -- "${paths[@]}" \
  | xargs -r -d '\n' stat -c '%n %s %Y' 2>/dev/null | sha256sum | cut -d' ' -f1)

# Nothing changed since the last passing run.
[[ -f "$stamp_dir/passed" && "$(cat "$stamp_dir/passed")" == "$fingerprint" ]] && exit 0

# First run on a clean tree: assume the committed code is tested.
if [[ ! -f "$stamp_dir/passed" && -z "$(git status --porcelain -- "${paths[@]}")" ]]; then
  echo "$fingerprint" > "$stamp_dir/passed"
  exit 0
fi

# Claude already saw these failures and stopped without changing code: let it
# stop instead of looping.
stop_hook_active=$(jq -r '.stop_hook_active // false' <<<"$input" 2>/dev/null)
if [[ "$stop_hook_active" == "true" && -f "$stamp_dir/failed" \
      && "$(cat "$stamp_dir/failed")" == "$fingerprint" ]]; then
  exit 0
fi

log="$stamp_dir/last-run.log"
if cargo test --quiet >"$log" 2>&1; then
  echo "$fingerprint" > "$stamp_dir/passed"
  rm -f "$stamp_dir/failed"
  exit 0
fi

echo "$fingerprint" > "$stamp_dir/failed"
{
  echo "Stop hook: \`cargo test\` failed (slow #[ignore] tests skipped). Fix the failures before finishing. Last 80 lines of $log:"
  tail -n 80 "$log"
} >&2
exit 2
