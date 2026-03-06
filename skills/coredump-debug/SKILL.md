---
name: coredump-debug
description: Debug segfaults and crashes in JAX/XLA/ROCm training workloads using coredump analysis. Use when the user has a coredump file, SIGSEGV, segfault, crash dump, or core file to analyze. Covers GDB backtrace extraction, identifying the crash cause from registers and disassembly, finding and cloning the correct source code versions, and reading the relevant code to determine the root cause.
---

# Coredump Debugging for JAX/XLA/ROCm

Systematic workflow for analyzing coredumps from GPU training workloads. Produces: crash call chain, root cause hypothesis, and the exact source lines responsible.

**Important**: Coredumps capture the crash symptom, not necessarily the root cause. A common pattern in GPU workloads: a **data race** silently corrupts memory during normal operation, and the corruption only manifests as a crash much later (e.g., during exit cleanup). If the crash is **non-deterministic** (e.g., ~2% repro rate), suspect a data race or thread-safety bug — the coredump shows where the corrupted data was *read*, but the *write* that caused corruption happened earlier. ASAN/TSAN may be needed to find the actual root cause.

## Step 0: Determine if the Crash is Deterministic

Before diving into GDB, establish the repro rate:
- **100% repro** → likely a logic bug, NULL deref, or missing check. Coredump analysis alone is usually sufficient.
- **Low repro (1-10%)** → likely a data race, use-after-free, or shutdown ordering issue. The coredump identifies the symptom; sanitizers (ASAN/TSAN) identify the cause.
- **One-off** → could be hardware (GPU memory error, network), OOM, or cosmic ray. Check `dmesg`, `rocm-smi`, and system logs first.

## Prerequisites

- GDB installed (`apt install gdb`)
- The coredump file and the matching Python binary (e.g. `/opt/venv/bin/python3`)
- Software version info from the container (see Phase 3 for how to collect)

## Phase 1: Extract the Crash Backtrace

### 1.1 Full backtrace with locals

```bash
gdb -batch \
  -ex "set pagination off" \
  -ex "set print frame-arguments all" \
  -ex "bt full" \
  -ex "info threads" \
  -ex "thread apply all bt" \
  /opt/venv/bin/python3 <corefile> 2>&1 | head -2000 > /tmp/gdb_bt.txt
```

Coredumps from GPU workloads are large (50-100GB). GDB load time can be 1-3 minutes; allow sufficient time before assuming it is stuck.

### 1.2 Identify the crash signal and thread

Search the GDB output for:

```
Program terminated with signal SIGSEGV, Segmentation fault.
#0  <function> at <file>:<line>
```

If frame #0 is `syscall()` and frame #1 is `SignalHandler`, the crash was caught by LLVM's signal handler which re-raised it. The **real crash** is at frame #2 (`<signal handler called>`) and above.

### 1.3 Get the crashing instruction

```bash
gdb -batch \
  -ex "frame 2" -ex "info registers" \
  -ex "frame 3" -ex "x/10i \$rip-20" -ex "x/5i \$rip" \
  /opt/venv/bin/python3 <corefile> 2>&1 | tail -60
```

Key things to look for:
- **`rip`**: the instruction pointer at crash time
- **Misaligned pointers**: values like `0x73a330f14c3e` (not 8-byte aligned) indicate use-after-free or corruption
- **NULL dereference**: `rdi=0x0` or similar in a `mov` instruction operand
- **`=> 0xADDR: mov offset(%rax),%rdi`**: the exact memory access that faulted

### 1.4 Get detailed frame info for key frames

```bash
gdb -batch \
  -ex "frame N" -ex "info locals" -ex "info args" \
  /opt/venv/bin/python3 <corefile>
```

Repeat for each interesting frame in the crash chain. Focus on:
- Frames in `rocprofiler-sdk`, CLR (`libamdhip64`), HSA runtime, RCCL, hipBLASLt
- The frame where the crash transitions from known library code to `??()` (stripped symbols)

## Phase 2: Read the Crash Call Chain

### 2.1 Build the crash chain table

From the backtrace, construct a table mapping each frame to its component. Example (actual frames will vary):

| Frame | Function | Source File | Component |
|---|---|---|---|
| #0 | `syscall()` | libc | Signal delivery |
| #1 | `SignalHandler(Sig=11)` | LLVM | Signal re-raise |
| #2 | `<signal handler called>` | - | - |
| #3 | `??()` from `libsomething.so` | unknown | **Crash origin** |
| ... | ... | ... | ... |

The specific frames depend on the crash. The goal is to identify which **component** owns each frame.

### 2.2 Identify the crash context

**Exit-time crashes** — look for frames like:
- `__run_exit_handlers` / `__GI_exit` → static destructor / atexit handler
- Destructor names (`~ClassName`) → object cleanup
- `hipModuleUnload` / `hsa_executable_destroy` → GPU resource teardown

Exit-time crashes in GPU workloads are often **caused by data races during runtime** that corrupt state silently. The corruption only crashes when cleanup code traverses the corrupted data.

**Runtime crashes** — look for frames like:
- Module loading functions (`executable_freeze`, `hipModuleLoad`)
- Worker thread frames (Eigen, XLA thread pool)
- Multiple threads in the same function → potential race

**Key question**: Is this crash the **cause** or the **symptom**? If non-deterministic, the coredump likely shows the symptom; the cause is an earlier unsynchronized write.

### 2.3 Check for thread-safety clues

Run `info threads` in GDB and check:
- How many threads are active (200+ is normal for GPU workloads)
- Whether multiple threads are in the same function (potential race)
- Whether RCCL/NCCL threads are still running during exit

## Phase 3: Find and Clone the Correct Source Code

### 3.1 Inventory the container environment

Collect the git hash and version for every component in the crash chain so you can clone the exact code that produced the crashing binary.

Key components to inventory (examples — discover the full list from the container):
- **Python packages**: `jax`, `jaxlib`, `jax-rocm*-plugin`, `jax-rocm*-pjrt`, `maxtext`, `transformer-engine`, etc. Check `pip list`, `pip show`, and embedded `version.py` / `commit_info.py` files for git hashes.
- **ROCm system/library debs**: `rocprofiler-sdk`, `hip-runtime-amd`, `hsa-rocr`, `rccl`, `hipblaslt`, `rocblas`, `miopen-hip`, etc. Use `dpkg -l` and `/opt/rocm*/.info/version`.
- **Source repos in the container**: scan common locations (`/opt/`, `/workspace/`) for `.git` directories and record each repo's `HEAD` commit and branch.

Record which repos are present (can read source directly) vs. absent (need cloning).

### 3.2 Identify which repos are needed

Map each frame in the crash chain to a source repo. The "Typical Path" column shows example paths from the standard maxtext-slurm container; adjust to match your environment.

| Library in backtrace | Source Repo | Typical Path (example) |
|---|---|---|
| `librocprofiler-sdk.so` | `ROCm/rocm-systems` | `/workspace/rocm-systems/projects/rocprofiler-sdk` |
| `libamdhip64.so` (CLR) | `ROCm/rocm-systems` | `/workspace/rocm-systems/projects/clr` |
| `libhsa-runtime64.so` | `ROCm/rocm-systems` | `/workspace/rocm-systems/projects/rocr-runtime` |
| `librccl.so` | `ROCm/rccl` | `/workspace/rccl` |
| `libhipblaslt.so` | `ROCm/rocm-libraries` | `/workspace/rocm-libraries/projects/hipblaslt` |
| `xla_rocm_plugin.so` | `ROCm/xla` | `/opt/xla` |

### 3.3 Find the matching version for system packages

For libraries installed via dpkg (not from a git repo in the container):

```bash
# Get the package version
dpkg -l <package-name> 2>/dev/null | grep ^ii

# The build number (e.g., -43~24.04) identifies the monorepo build.
# All packages with the same build number come from the same commit.
```

For ROCm system packages, the release branch follows the pattern `release/rocm-rel-X.Y`. Replace the version below with the one matching your container (from Step 3.1):

```bash
# Example: ROCm 7.2 (substitute your actual version)
git clone --branch release/rocm-rel-7.2 --depth 1 \
  https://github.com/ROCm/rocm-systems.git /workspace/rocm-systems
```

### 3.4 Clone missing repos

Clone to the **expected path** so GDB source file references match:

```bash
# Check what paths appear in the backtrace
# e.g., /workspace/rocm-systems/projects/clr/... means clone to /workspace/rocm-systems

# For rocm-systems (CLR, HSA, rocprofiler-sdk) — substitute your ROCm version:
git clone --branch release/rocm-rel-7.2 --depth 1 \
  https://github.com/ROCm/rocm-systems.git /workspace/rocm-systems

# For rocm-libraries (hipBLASLt, rocBLAS, MIOpen):
# Check HIPBLASLT_BRANCH env var for the exact commit
git clone https://github.com/ROCm/rocm-libraries.git /workspace/rocm-libraries
cd /workspace/rocm-libraries && git checkout $HIPBLASLT_BRANCH
```

### 3.5 Handle long build-path prefixes

ROCm packages are often built with paths like:
```
/longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/rocm-systems/projects/clr/...
```

The real source path maps as: strip the long prefix up to `rocm-systems/` or `rocm-libraries/`, then the rest matches the git repo layout.

### 3.6 Verify upstream fixes

After identifying the bug, check if it's already fixed upstream:

```bash
cd /workspace/rocm-systems
# Try to fetch the specific fix commit
git fetch origin <commit-hash>

# Check if it's in your branch
git merge-base --is-ancestor <commit-hash> release/rocm-rel-X.Y \
  && echo "YES - included" || echo "NO - not included"
```

## Phase 4: Read Source Code and Determine Root Cause

### 4.1 Read the crashing function

Using the file paths from the backtrace, read the source at the crash line:

```bash
# Frame shows: code_object.cpp:892
# Read context around line 892
```

Focus on:
- What data structure is being accessed
- Whether the access is under a lock (and what kind -- read vs write)
- Whether the data could be modified concurrently by another thread

### 4.2 Check for common crash patterns

| Pattern | What to look for in coredump | Typical repro rate |
|---|---|---|
| **Data race** (most common for non-deterministic) | Misaligned/garbage pointers, corrupted container internals, `rlock` guarding write operations | 1-10% |
| **Use-after-free** | Pointer to freed region, `fd` bytes in ASAN | 1-50% |
| **Shutdown ordering** | Crash in destructor during `__run_exit_handlers` | 1-20% |
| **NULL deref** | `rdi=0x0`, `rax=0x0` before a `mov` through pointer | ~100% |
| **Stack overflow** | Thousands of recursive frames, `rsp` near stack limit | ~100% |

**Data race red flags in the coredump:**
- Pointer values that are valid addresses but **misaligned** (e.g., `0x73a330f14c3e` — not 8-byte aligned for a pointer field)
- Hash map / vector internal pointers that look like small integers (e.g., `__end = 0x226`)
- The crash is in a container's internal function (`_M_insert_bucket_begin`, `_M_deallocate_node`)
- Multiple threads were executing the same code path (check `thread apply all bt`)

### 4.3 Trace the data lifecycle

For corruption bugs, trace the corrupted data backward through the source:
1. **Where is it read?** (the crash point — you know this from the coredump)
2. **Where is it written?** (search for all writes to the field — the corruption happened here)
3. **Where is it created?** (constructor / allocation)
4. **Where is it destroyed?** (destructor / free)
5. **Are all accesses properly synchronized?** (locks, atomics, thread-safe containers)

For data races specifically: look for a `rlock` (read lock) protecting code that actually writes (e.g., `unordered_map::operator[]` inserts if the key is absent). Also check if the same data is accessed from different threads without any lock.

### 4.4 Check for existing fixes upstream

```bash
# Search the upstream repo for fixes related to the crash
git log --all --oneline --grep="data race" --grep="thread safety" \
  -- path/to/crashing/file.cpp

# Or search by file
git log --all --oneline -- path/to/crashing/file.cpp | head -20
```

## Output Format

Report findings in this structure:

```
## Coredump Analysis: <corefile_path>

**Crash signal:** <SIGSEGV / SIGABRT / SIGBUS / etc.>
**Crashing thread:** Thread <N> (<component context, e.g., "RCCL polling thread", "XLA worker", "exit handler">)
**Crash context:** <runtime | exit-time cleanup | module load>
**Determinism:** <deterministic (100% repro) | non-deterministic (~N% repro) | unknown (single occurrence)>

### Crash call chain

| Frame | Function | Library / Source | Component |
|-------|----------|-----------------|-----------|
| #N | ... | ... | ... |

### Crashing instruction

<Register state and disassembly at the fault point. Key pointer values, alignment, NULL indicators.>

### Container environment

| Component | Version | Git Hash |
|-----------|---------|----------|
| ... | ... | ... |

### Root cause

<Plain-English explanation: what data structure was corrupted, what access pattern
caused the fault, and — for non-deterministic crashes — whether the coredump shows
the symptom or the cause.>

### Evidence

<Key GDB output: register values, local variables, disassembly — quoted verbatim.
For data races: misaligned pointers, corrupted container internals, lock analysis.>

### Upstream fix status

<"Fixed in <commit> on <branch>" / "Not fixed — filed as <issue>" / "Not checked">

### Recommended next steps

<Numbered list. Examples:
- For deterministic: code fix or workaround
- For non-deterministic: ASAN/TSAN build to find the write-side race
- For upstream fix available: cherry-pick or upgrade path>
```

## Quick Reference: GDB Commands for Coredumps

| Command | Purpose |
|---|---|
| `bt full` | Full backtrace with local variables |
| `frame N` | Switch to frame N |
| `info locals` | Show local variables in current frame |
| `info registers` | Show CPU registers |
| `info threads` | List all threads |
| `thread N` | Switch to thread N |
| `thread apply all bt` | Backtrace for every thread |
| `x/10i $rip` | Disassemble 10 instructions at crash point |
| `x/s ADDR` | Print string at address |
| `p expr` | Evaluate expression |
| `info proc mappings` | Show library load addresses |

## Quick Reference: Crash Exit Codes

| Exit Code | Signal | Meaning |
|---|---|---|
| 139 | SIGSEGV (11) | Segmentation fault (invalid memory access) |
| 134 | SIGABRT (6) | Abort (assertion failure, double free) |
| 136 | SIGFPE (8) | Floating point exception (division by zero) |
| 137 | SIGKILL (9) | Killed (OOM killer, timeout) |
| 138 | SIGBUS (7) | Bus error (misaligned access, bad mmap) |
