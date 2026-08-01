# OIST HPC skill pack — deigo + saion

Self-contained operational reference for running compute on OIST's clusters.
Written to be inherited by a fresh repository: nothing here depends on any
previous codebase. Every path, partition name, and limit below is either
quoted from official OIST docs, verified by a dated production run, or
explicitly marked unverified.

---

## 0. Provenance legend

Facts are labelled because the three sources disagree in places and age at
different rates. Never silently upgrade a `[VERIFIED]` or `[UNVERIFIED]` fact
to `[OFFICIAL]`.

| Tag | Meaning |
|---|---|
| `[OFFICIAL]` | From OIST SCDA documentation (scda-guide.oist.jp: deigo, large-memory-partitions, saion-system, saion-open, saion-restrict). Authoritative but can be stale relative to the live Slurm config. |
| `[VERIFIED <date>]` | Observed in production by a real cross-cluster pipeline (the AntsArray / Reiter-unit deigo↔saion runbook). Ground truth *for that account, on that date*. Association limits are per-account — re-verify. |
| `[PROJECT]` | Convention of the medaka multi-camera tracking project (StephensU). Not a cluster fact; a choice. Portable as a pattern, not as a path. |
| `[UNVERIFIED]` | Not established by any source read here. Do not assert it. |
| `[CONFLICT]` | Two sources disagree; both recorded, resolution requires a live check. |

**Support contact** `[OFFICIAL]`: ASK-SCDA@OIST.JP, ext. 12047.
Open Hours (for restricted-partition applications): room B648, 15:30–17:30 daily.

---

## 1. TL;DR — which cluster for what

| Need | Cluster | Partition |
|---|---|---|
| CPU-parallel work, ffmpeg, transcoding, chunking, aggregation | **deigo** | `compute` (≤4 d) / `short` (≤2 h) |
| >500 GB RAM per process | **deigo** | `largemem` / `bigmem` (application required) |
| Thousands of cores in one job | **deigo** | `largejob` (unit-leader request) |
| GPU inference / training | **saion** | `gpu-a100` (was `largegpu`), `gpu-v100`/`gpu-p100` (was `gpu`) |
| Free GPU smoke test, no application | **saion** | `test-gpu` (2 GPU, 18 cores, 8 h, preemptible) |
| Orchestration, submission, data movement, copy-to-`/bucket` | **login nodes** of either cluster | n/a |
| Bulk data staging between clusters | *neither* — use the read-only cross-mounts (§6) | |

- **deigo = CPU + large memory + orchestration.** No GPUs. 663 nodes / 66,608 cores / 339 TB RAM `[OFFICIAL]`.
- **saion = GPUs.** No general CPU partition `[VERIFIED 2026-05]` — CPU-only postprocessing belongs on deigo.
- **`/bucket` is the only backed-up storage and it is read-only from every compute node on both clusters** `[OFFICIAL]`. This single fact drives most of the architecture in §5–§7.

---

## 2. ⚠️ Partition rename (read this before copying any old script)

`[OFFICIAL]` saion partition names were changed. **Old names still work for
existing users, but new accounts are granted the new names only.** Any
inherited script referencing the legacy name will fail with
`Invalid partition name specified` on a fresh account.

| Legacy name | Current name | What happened |
|---|---|---|
| `largegpu` | **`gpu-a100`** | straight rename |
| `gpu` | **`gpu-v100`** *and* **`gpu-p100`** | one partition split by GPU generation |

`[OFFICIAL]` Legacy `gpu` users can still target a generation inside the old
partition with `--gres=gpu:v100:1` / `--gres=gpu:p100:1`.

`[OFFICIAL]` OIST explicitly calls out shared software installs (Relion,
CryoSparc, and by extension any lab pipeline) as needing their partition
strings updated.

**Migration rule for inherited code:** make the partition a single variable with
the new name as default, e.g. `PARTITION="${PARTITION:-gpu-a100}"`, and let the
caller override to `largegpu` if `sinfo` shows only the legacy name for that
account. Never hardcode.

`[VERIFIED 2026-05]` A `short-a100` partition also existed for the Reiter-unit
account (same physical A100 nodes as `largegpu`, low priority / preemptible).
It does **not** appear in the official docs read here — treat its existence and
its limits as account-specific and `[UNVERIFIED]` for a new account until
`sinfo` says otherwise.

---

## 3. Access & login

```bash
ssh <oist-id>@deigo.oist.jp     # CPU cluster
ssh <oist-id>@saion.oist.jp     # GPU cluster
```

`[OFFICIAL]`
- Login nodes are for **submitting jobs, moving data, preparing work** — not for
  running computation.
- saion has **two login nodes and two admin nodes** for redundancy; deigo login
  nodes are **Intel** (matters for §9 build-architecture rules).
- **saion requires a login shell.** Node configuration means `bash` must be
  started with `-l`:
  - batch scripts: `#!/bin/bash -l` (not plain `#!/bin/bash`)
  - interactive: `srun ... --pty bash -l`
  Omitting `-l` gives an environment where `module` and the unit modulefiles
  are not set up. deigo does not require this.
- **Start jobs from your home directory**, not from `/bucket` or `/flash` —
  Slurm writes its `.out`/`.err` relative to the submit directory and will fail
  on a read-only or unusual CWD. Alternatively pass an explicit
  `-o /absolute/path/%x_%A_%a.out`.

Useful `~/.ssh/config` aliases (both clusters, and used by compute-node
upload paths in §5):

```
Host deigo
    HostName deigo.oist.jp
    User <oist-id>
Host saion
    HostName saion.oist.jp
    User <oist-id>
```

`[VERIFIED 2026-05]` Compute nodes can `ssh` back to their own cluster's login
alias; this is how they write to `/bucket`. Confirm the alias resolves to a
login node with `/bucket` **write** access before relying on it.

---

## 4. Storage map

### 4.1 The table

| Mount | Cluster | Tech / size | Per-unit quota | RW from login | RW from compute | Backed up | Retention |
|---|---|---|---|---|---|---|---|
| `/bucket` | both | GPFS, **60 PB** `[OFFICIAL]` | per-unit allocation `[UNVERIFIED]` | **yes** | **no — read-only** | **yes** (weekly snapshots `[VERIFIED 2026-05]`) | permanent; this is the archive |
| `/flash` | deigo | SSD Lustre, **500 TB** `[OFFICIAL]` | **10 TB / unit**, fixed `[OFFICIAL]` | yes | **yes** | **no** | scratch — you must delete after the job `[OFFICIAL]` |
| `/work` | saion | SSD, **200 TB** `[OFFICIAL]` | **10 TB / unit**, fixed `[OFFICIAL]` | yes | **yes** | **no** | scratch — temporary/job data only |
| `/hpacquire` | saion | fast import staging `[OFFICIAL]` | `[UNVERIFIED]` | read | **readable from compute, read-only cluster-side** | `[UNVERIFIED]` | landing zone for instrument/external data |
| `/home` | both | slow | **50 GB / user** (saion) `[OFFICIAL]`; deigo quota `[UNVERIFIED]` | yes | yes (writable) `[VERIFIED 2026-06]` | `[UNVERIFIED]` | code + configs only; **no research data** |
| `/apps/unit/<Unit>` | both | shared unit software tree | `[UNVERIFIED]` | yes (if group-writable) | read | `[UNVERIFIED]` | permanent; venvs + modulefiles live here (§9) |
| `/deigo_flash` | **saion** | cross-mount of deigo `/flash` | — | **read-only** | **read-only** | n/a | mirrors deigo scratch lifetime |
| `/saion_work` | **deigo** | cross-mount of saion `/work` | — | **read-only** | **read-only** | n/a | mirrors saion scratch lifetime |
| `Datashare` (login nodes, deigo) | deigo | quick user↔user sharing `[OFFICIAL]` | `[UNVERIFIED]` | yes | `[UNVERIFIED]` | `[UNVERIFIED]` | ad-hoc |

### 4.2 Windows / UNC equivalents

The same bucket tree, three ways `[PROJECT]`:

```
/bucket/StephensU/medaka/...                    # POSIX, on deigo + saion
\\bucket.oist.jp\bucket\StephensU\medaka\...    # UNC, from Windows
Z:\StephensU\medaka\...                         # mapped drive (per-machine letter)
```

`[PROJECT]` The drive letter is **per-machine** and must never be hardcoded in
code — resolve it from a gitignored local config. On Windows, a mapped `Z:` path
may resolve back to the `\\bucket.oist.jp\...` UNC form; use `pathlib.Path` /
`Path.resolve()` rather than hand-editing separators.

### 4.3 The canonical data-flow pattern `[OFFICIAL]`

1. **Read inputs directly from `/bucket`.** Do not pre-copy the whole dataset to
   scratch "to be fast" — the docs explicitly advise against it.
2. **Write all job output to scratch** (`/flash` on deigo, `/work` on saion),
   ideally into a unique per-job directory: `TMPD=$(mktemp -d /flash/<Unit>/$USER/XXXXXX)`.
3. **Copy the results you want to keep back to `/bucket`** — routed through the
   **login node**, because `/bucket` is read-only from compute (§5.5).
4. **Delete the scratch directory.**

`[OFFICIAL]` For multi-step pipelines whose intermediates exceed the 10 TB flash
quota, use a temporary directory on bucket created/removed from the login node:
`ssh deigo mktemp -d /bucket/<Unit>/tmp/XXXX` … `ssh deigo rm -r <dir>`.

`[VERIFIED 2026-08]` **Treat every staging directory as a disposable cache.**
Videos and model artifacts copied onto scratch are re-derivable from `/bucket`;
only delete a staged copy once it is byte-verified against a surviving original,
and archive anything that is not.

### 4.4 OIST Bucket Backup archives `[VERIFIED 2026-05]`

The unit `Backup` area takes zip archives plus a sidecar description file:

```
/bucket/<Unit>/Backup/<collection>/<date>_<block>_raw_videos.zip
/bucket/<Unit>/Backup/<collection>/<date>_<block>_raw_videos.txt   # must contain `name:` and `project:` lines
```

Re-running the same block **updates the same archive** with `zip -0 -FS`
(store-only, filesystem-sync) instead of creating timestamped duplicates; OIST's
weekly snapshots keep the older versions remotely.

---

## 5. Partitions

### 5.1 deigo

| Partition | Access | Nodes / hardware | Per-user caps | MaxWall | Notes |
|---|---|---|---|---|---|
| `compute` | **open** `[OFFICIAL]` | AMD Epyc 7702 (2×64c, 512 GB) + Intel Xeon 6230 (2×20c, 512 GB) | 2000 cores, 7500 G mem `[OFFICIAL]`; GrpSubmit **2016** `[VERIFIED 2026-05-20]` | **4 days** `[OFFICIAL]` | default partition |
| `short` | **open** `[OFFICIAL]` | same pool | 4000 cores, 6500 G mem `[OFFICIAL]`; GrpSubmit **4016** `[VERIFIED 2026-05-20]` | **2 h** `[OFFICIAL]` | low priority; **jobs can be suspended** to make room for compute/largemem/largejob |
| `largemem` | **application + Open Hours** `[OFFICIAL]` | 44 nodes @500 GB allocatable + 14 nodes @750 GB | **max 5 nodes at once**; ≥500 GB and ≥1 week per job or you don't belong here | effectively unlimited; **recommended ≤3 weeks** | Slurm auto-routes >500 GB requests to the scarcer 750 GB tier |
| `bigmem` | **application + Open Hours, sparingly granted** `[OFFICIAL]` | 2 nodes: ~2987 GB and 1500 GB | **max 8 cores**; must prove a 750 GB largemem node already failed | unlimited; recommended ≤3 weeks | actively monitored; over-allocating jobs get killed |
| `largejob` | **unit-leader request → Scientific Computing Committee** `[OFFICIAL]` | — | multi-thousand-core jobs | `[UNVERIFIED]` | |
| `datacp` | `[VERIFIED 2026-05-20]` — **not in the official summary** | 4 data-mover nodes `[VERIFIED]` | cpu **4**, mem **19 G**, **GrpSubmit = 20** | none observed | dedicated to data movement. **The 20-job submit cap is trivial to blow — never submit a per-chunk array here.** |

`[OFFICIAL]` **Default memory is 4 GB per core.** Always specify `--mem`
explicitly.

`[OFFICIAL]` deigo users also get access to saion's *open* partitions
(including `test-gpu`).

### 5.2 saion

| Partition | Current name | Legacy name | Access | Hardware | Per-user caps | MaxWall |
|---|---|---|---|---|---|---|
| A100 | **`gpu-a100`** | `largegpu` | **restricted**: must show P100/V100 are insufficient, then Open Hours `[OFFICIAL]` | 4 nodes × **8× A100-SXM4 80 GB**, 128 AMD cores, 2 TB RAM `[OFFICIAL]` | **no default allocation — tailored per request** `[OFFICIAL]`. One real grant: cpu 128, **gpu 8**, mem 1 T `[VERIFIED 2026-05-20]` | 12 h for that grant `[VERIFIED 2026-05-20]` |
| V100 | **`gpu-v100`** | part of `gpu` | **restricted** (application + Open Hours) `[OFFICIAL]` | 8 nodes × 4× V100 16 GB, 36 cores, 512 GB `[OFFICIAL]` | default **combined** across v100+p100: 72 cores + 8 GPUs (4 per partition) `[OFFICIAL]` | **7 days** `[OFFICIAL]` |
| P100 | **`gpu-p100`** | part of `gpu` | **restricted** `[OFFICIAL]` | 8 nodes × 4× P100 16 GB, 36 cores, 512 GB `[OFFICIAL]` | (same combined pool) | 7 days `[OFFICIAL]` |
| — swap option | | | | | request **8 GPU + 72 cores on a single partition** instead `[OFFICIAL]` | **2 days** `[OFFICIAL]` |
| legacy `gpu` | | | | mixed V100/P100 | cpu 72, gpu 8 `[VERIFIED 2026-05-20]` | 2 days `[VERIFIED 2026-05-20]` |
| test | **`test-gpu`** | — | **open to any OIST account** `[OFFICIAL]` | 6 nodes × 4× P100, 36 cores, 512 GB `[OFFICIAL]` | **18 cores, 2 GPUs** `[OFFICIAL]` | **8 h** `[OFFICIAL]` |
| CPU test | **`intel`** | — | **open** `[OFFICIAL]` | 4 Xeon nodes, 40 cores | `[CONFLICT]` overview says *128 GB/user*; open-partitions page says *nodes have 512 GB with a 120 GB per-job cap* | `[UNVERIFIED]` |
| low-prio A100 | `short-a100` | — | `[VERIFIED 2026-05]` account-specific; **not in official docs** | same 4 A100 nodes as `gpu-a100` | cpu 256, **gpu 32**, mem 2048 G | **2 h (1 h non-preemptible)**; `PreemptMode=REQUEUE` |

`[OFFICIAL]` **`test-gpu` is preemptible** — it shares hardware with
high-priority work and jobs can be stopped at any time. Fine for smoke tests,
never for a long run. Its environment is deliberately older (older CUDA).

`[OFFICIAL]` **`intel` is gang-scheduled** — jobs start immediately but share
CPU in 10-second slices. No queue wait, useless for interactive work, good for a
quick CPU smoke test.

### 5.3 Sizing arithmetic that actually binds

`[VERIFIED 2026-05]` Work out which of {cpu, gpu, mem} saturates first, then set
the array's `%N` concurrency to that number. Two worked examples:

- A100 at 8-GPU cap, `-c 16 --mem=128G --gres=gpu:1` per task, 8 concurrent →
  128 cpu / 8 gpu / 1024 G. All three caps saturate simultaneously. `%8` is the
  ceiling; a higher `%N` only produces `AssocGrpGPULimit` pending reasons.
- deigo `compute` CPU array at `-c 16`: `2000 / 16 ≈ 125` concurrent tasks.
  Running at `%100` (~1600 cpu) leaves headroom for the other jobs you have on
  the same partition.

`[VERIFIED 2026-05]` GPU inference is often **CPU-bound in postprocessing**
(peak finding, instance assembly), not GPU-bound (~30 % average A100
utilisation, bursty). Give each task the maximum `-c` your gpu-cap arithmetic
allows rather than packing more GPU tasks.

---

## 6. Association limits — verify, don't assume

Per-user limits are **account/association properties**, not cluster constants.
A new account in a different unit will have different numbers. Run these on
**each cluster** before trusting any table above:

```bash
# 1. Your associations and their caps (run on deigo, then on saion)
sacctmgr show assoc user=$USER \
  format=Cluster,Account,User,Partition,QOS,GrpJobs,GrpSubmit,GrpTRES,MaxJobs,MaxSubmit,MaxWall \
  -p | column -t -s'|'

# 2. Which partitions actually exist for you, and their hard limits
sinfo -o '%20P %5a %10l %6D %N'
for p in compute short datacp largemem bigmem largejob; do
  scontrol show partition=$p 2>/dev/null | grep -E 'PartitionName|MaxTime|QoS|TRES|State'
done
# saion (try BOTH new and legacy names — whichever errors tells you your era)
for p in gpu-a100 gpu-v100 gpu-p100 test-gpu intel largegpu gpu short-a100; do
  scontrol show partition=$p 2>/dev/null | grep -E 'PartitionName|MaxTime|QoS|TRES|State'
done

# 3. Which unit account(s) you can charge to
sacctmgr show user $USER withassoc format=User,Account,Cluster,Partition -p

# 4. Physical GPUs on a node (see §11 for why nvidia-smi lies)
scontrol show node <node> | grep -E 'Gres|CfgTRES'
```

### The unit / account question `[CONFLICT]` — resolve before first submit

- The medaka project belongs to unit **StephensU**, group **`stephensuni`** `[PROJECT]`.
- The GPU modules it consumed (`sleap-nn/*`, `detectron.2/0.6`) live under
  **ReiterU**: `/apps/unit/ReiterU/.modulefiles` `[PROJECT]`.
- The AntsArray runbook labels its limit tables *"Reiter unit, `stephensuni`
  account"* `[VERIFIED 2026-05]` — which mixes the two and cannot be taken at
  face value.

**Do not assert an account↔unit mapping.** Establish it empirically with
`sacctmgr show user $USER withassoc` and `id -Gn` on the new account, and note
that *reading another unit's modulefiles* is independent of *which account your
jobs charge to*.

---

## 7. Submission patterns

### 7.1 Array job + `chunks.txt` + `sed -n` (the core idiom)

Split a long input into frame/time ranges on the login node, write one range per
line, and let each array task read **its own line**. Keeps the sbatch script
free of any per-task logic and makes re-running a single failed task trivial.

Submitter (login node):

```bash
# chunks.txt: one "START END" pair per line
write_chunks() {
  local total="$1" n="$2" out="$3"
  local per=$(( (total + n - 1) / n ))
  : > "$out"
  for i in $(seq 0 $((n-1))); do
    local s=$((i*per)) e=$(( (i+1)*per - 1 ))
    (( e >= total )) && e=$((total-1))
    (( s > e )) && continue
    echo "$s $e" >> "$out"
  done
}
write_chunks "$TOTAL_FRAMES" "$NUM_CHUNKS" "$WORKDIR/chunks.txt"
NTASK=$(wc -l < "$WORKDIR/chunks.txt")

sbatch --parsable \
  --partition="$PARTITION" --time="$TIME_LIMIT" \
  --array="0-$((NTASK-1))%${CONCURRENCY}" \
  --output="$WORKDIR/slurm_%A_%a.out" --error="$WORKDIR/slurm_%A_%a.err" \
  --export=ALL,CHUNK_FILE="$WORKDIR/chunks.txt",OUTPUT_DIR="$WORKDIR",VIEW="$VIEW" \
  worker.slurm
```

Worker (`worker.slurm`, note `#!/bin/bash -l` on saion):

```bash
CHUNK_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CHUNK_FILE")
START=$(awk '{print $1}' <<<"$CHUNK_LINE")
END=$(awk   '{print $2}' <<<"$CHUNK_LINE")
OUT="${OUTPUT_DIR}/${VIEW}.frames_${START}-${END}.slp"
```

Notes:
- `--array=0-N%K` — the `%K` is the concurrency throttle; derive `K` from §5.3.
- `sbatch --parsable` prints only the job id; capture it for dependencies.
- `--export=ALL,VAR=...` passes config without templating the script.
- Get the exact item count from an authoritative sidecar (a recorder's
  `metadata.csv`, a manifest) rather than probing the media, when one exists —
  probing needs an ffmpeg/ffprobe module on the node and can disagree with the
  true count `[PROJECT]`.

### 7.2 Idempotent skip (makes every job safely re-runnable)

```bash
if [ -s "$OUT" ] && [ "${FORCE:-0}" != "1" ]; then
    # Cheap existence check is not enough — validate the artifact opens.
    if python -c 'import sleap_io as sio,sys; l=sio.load_slp(sys.argv[1]); assert l.videos and l.skeletons' "$OUT"; then
        echo "Skip: $OUT already exists and is loadable (FORCE=1 to redo)."; exit 0
    fi
    echo "ERROR: existing output is not loadable: $OUT" >&2; exit 1
fi
```

`[PROJECT]` The **validate-don't-just-stat** rule matters: a truncated output
from a walltime kill has non-zero size and will silently poison downstream
stages. `[VERIFIED 2026-05]` The same skip is what makes preemptible partitions
(`PreemptMode=REQUEUE`) usable — a requeued task resumes instead of redoing
finished chunks.

### 7.3 Staging sentinel (N tasks, one copy)

When all tasks of an array need the same large input on local scratch, have
task 0 copy and the rest spin on a sentinel file:

```bash
STAGING_DONE="${OUTPUT_DIR}/.staging_${VIEW}.done"
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    [ -f "$STAGING_DONE" ] || { mkdir -p "$(dirname "$STAGED")"; cp "$SRC" "$STAGED"; touch "$STAGING_DONE"; }
else
    WAITED=0
    while [ ! -f "$STAGING_DONE" ]; do
        sleep 5; WAITED=$((WAITED+5))
        [ "$WAITED" -gt 1800 ] && { echo "ERROR: staging wait >30 min"; exit 1; }
    done
fi
```

Caveats: task 0 must actually be scheduled (don't `--array=1-N`), and the
timeout must exceed a realistic copy of the largest input. `[PROJECT]`

### 7.4 `DRY_RUN`

Every submitter should support printing the exact `sbatch` command line without
submitting. `[PROJECT]` uses a three-level convention:

| `DRY_RUN` | Behaviour |
|---|---|
| `0` | submit |
| `1` | print `sbatch` commands only |
| `2` | also skip the input probe (use a placeholder count) — lets you dry-run without touching the data |

### 7.5 Copy-back from the login node

`/bucket` is read-only on compute, so results must be moved by something running
on a login node. Three mechanisms, in increasing order of robustness:

**(a) `nohup` watcher on the login node** — submitter starts a background poller
that waits for the arrays and then rsyncs.

```bash
nohup bash "$WORKDIR/merge_and_copy.sh" >> "$WORKDIR/merge_and_copy.log" 2>&1 &
```

Poll **`sacct`, not `squeue`.** `squeue` only shows *pending/running* jobs, so a
job that finishes between two polls is indistinguishable from a job that never
existed — and `squeue` tells you nothing about *why* it left. `sacct` is
authoritative, retains terminal state, and lets you refuse the copy when tasks
failed:

```bash
sacct -n -X -P -j "$JID" --format=JobID,State,ExitCode
# -X = allocations only (no .batch/.extern steps)   -P = parsable   -n = no header
# Require: row count == expected task count, every State=COMPLETED and ExitCode=0:0.
# Anything in FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED
#   -> refuse the copy and leave scratch intact for diagnosis.
```

Always bound the watcher (`MAX_WAIT`, e.g. 7 days) so it cannot wedge forever.

**(b) Streaming upload from inside each task** `[VERIFIED 2026-05]` — the task
`ssh`es its own cluster login and rsyncs its single output the moment it is
produced. Results appear incrementally instead of in one batch at the end, and a
mid-run failure has already delivered everything completed so far.

```bash
rsync -ah --chmod=Du=rwx,Dg=rwx,Fu=rw,Fg=rw --chown=:"$OUTPUT_GROUP" \
    "$WORKDIR"/*.slp  "deigo:$DATA_DIR/"
```

**(c) End-of-array safety-net job** `[VERIFIED 2026-05]` — one *single*
(not array) idempotent rsync job on `datacp`, catching anything (b) lost to a
transient SSH failure. **Single job, because `datacp` GrpSubmit is 20.**

### 7.6 Shared-group permissions `[VERIFIED 2026-05]`

Outputs meant for a whole unit need explicit group handling; `rsync -a` will
otherwise preserve *your* group onto bucket.

```bash
umask 002
chgrp -R "$GROUP" "$DIR"; chmod -R g+rX "$DIR"
find "$DIR" -type d -exec chmod g+s {} \;     # setgid: new files inherit the group
rsync ... --chown=:"$GROUP"                   # do not let rsync -a carry a source group over
```

Make permission fixes **best-effort** (`|| true`) — they must not abort a run
when a directory contains files owned by another user.

### 7.7 Sentinel-file completion instead of Slurm dependencies

`[VERIFIED 2026-05]` **There is no cross-cluster Slurm dependency graph.** A
deigo job cannot `--dependency=afterok:` a saion job. Use the *durable artifact*
as the sentinel: poll `/bucket` for the expected output set, and only then
proceed / free scratch.

```bash
# cleanup job: refuse to delete scratch until every expected result exists on bucket
expected=$(wc -l < worklist.txt)
have=$(ls "$DATA_DIR"/*_result.h5 2>/dev/null | wc -l)
[ "$have" -ge "$expected" ] || { echo "incomplete ($have/$expected) — preserving scratch"; exit 1; }
rm -rf "$SCRATCH"
```

`[VERIFIED 2026-05]` Two hard-won corollaries:
- **Gate on the artifact the next stage actually reads.** If stage N+1 consumes
  `_data.h5` derived from a `.slp`, gating on `.slp` presence lets a silent
  conversion failure fire the next stage on invisible data.
- **Use `afterany:`, not `afterok:`, for cleanup jobs.** An `afterok:` cleanup
  chained behind a leg that dies on a transient SSH error leaves scratch
  allocated forever. Free the scratch regardless; let the *content* check
  (above) decide whether deletion is safe.

---

## 8. Cross-cluster orchestration (deigo ↔ saion)

### 8.1 Use the read-only cross-mounts, not rsync

| Path | Visible from | Backing filesystem |
|---|---|---|
| `/deigo_flash/<Unit>/$USER/…` | saion **login + compute**, read-only | deigo `/flash` |
| `/saion_work/<Unit>/$USER/…` | deigo **login + compute**, read-only | saion `/work` |

`[OFFICIAL]` `/deigo_flash` on saion and `/saion_work` on deigo are documented
cross-mounts; `[VERIFIED 2026-05]` both are visible from compute nodes too.

**Pattern** `[VERIFIED 2026-05]`: deigo produces chunks into `/flash`; the saion
job reads them straight from `/deigo_flash`, `cp`s only what it needs into its
own `/work` for isolation, and runs. **No bulk deigo→saion rsync.** The GPU leg
starts seconds after the handoff instead of hours.

**Monitoring without ssh** `[VERIFIED 2026-05]`:

```bash
# from saion-login: watch deigo's progress
ls   /deigo_flash/<Unit>/$USER/jobs/<exp>/
tail /deigo_flash/<Unit>/$USER/jobs/<exp>/*.out

# from deigo-login: watch saion's outputs land
ls /saion_work/<Unit>/$USER/<exp>/output/
```

### 8.2 The handoff job

A "bridge" job on deigo `ssh`es saion and `sbatch`es the GPU leg.

`[VERIFIED 2026-05]` **The bridge must run on `compute`, not `short`** — if it
does any data movement at all, the 2 h `short` cap will kill it mid-transfer.

### 8.3 `ssh_retry` — mandatory, not optional

`[VERIFIED 2026-05]` Cross-cluster SSH intermittently fails with
`kex_exchange_identification: Connection closed by remote host`. A single bare
`ssh` in a bridge or upload path will eventually wedge an entire pipeline run.
Wrap **every** cross-cluster ssh/rsync:

```bash
ssh_retry() {  # 5 attempts, backoff 10·n seconds
  local n=1
  until ssh -o BatchMode=yes -o ConnectTimeout=30 "$@"; do
    (( n >= 5 )) && { echo "ssh failed after $n attempts: $*" >&2; return 1; }
    sleep $(( 10 * n )); n=$(( n + 1 ))
  done
}
rsync_retry() {
  local n=1
  until rsync -ah --partial "$@"; do
    (( n >= 5 )) && { echo "rsync failed after $n attempts" >&2; return 1; }
    sleep $(( 10 * n )); n=$(( n + 1 ))
  done
}
```

### 8.4 `datacp` discipline

`[VERIFIED 2026-05-20]` deigo `datacp`: **4 cpu, 19 G, GrpSubmit = 20**, 4
physical mover nodes.

- **Never submit a per-chunk array on `datacp`.** Anything multiplicative blows
  the 20-job cap instantly and blocks your other data movement.
- One single idempotent rsync job per pipeline leg.
- If you do use an array there, cap concurrency at 4 (one per mover node).

---

## 9. Log durability (4 layers)

`[VERIFIED 2026-05]` **The problem:** Slurm `.out`/`.err` normally live on
scratch. Scratch cleanup (`rm -rf`) destroys them — so a walltime kill, node
failure, mass `scancel`, or maintenance drain leaves nothing to diagnose. Fix by
shipping logs to `/bucket` in four independent layers:

**Layer 1 — live streaming.** Each task ships its own `.out`/`.err` to bucket
every N seconds (ship-only-on-change, per-task jitter) and on a `TERM`/`EXIT`
trap. Add to the sbatch header:

```bash
#SBATCH --signal=TERM@60     # SIGTERM ~60 s before the walltime SIGKILL
```

That grace window is what lets the final lines plus a `.status` marker
(`reason=signal …`) reach bucket **before** the task dies. Compute nodes can't
write bucket, so each ship is an `rsync` over SSH to the login alias — payloads
are KB–MB of text, so cost is negligible if you keep the interval coarse
(default 300 s), gate on change, and jitter per task.

**Layer 2 — `sacct` post-mortem.** Authoritative
`State/ExitCode/Reason/Elapsed/MaxRSS` from slurmdbd; survives the scratch wipe.
Capture it in the cleanup job, *after* the array is terminal:

```bash
sacct -j "$JID" -P --format=JobID,JobName,State,ExitCode,Reason,Elapsed,MaxRSS,NodeList \
  > "$BUCKET_LOGS/sacct_${JID}.tsv"
```

**Layer 3 — archive-before-delete.** The cleanup job rsyncs all task logs to
bucket *before* any `rm`/scratch reclamation. Idempotent; catches whatever
layer 1 missed.

**Layer 4 — throughput line.** Every task prints a greppable, durable
throughput record:

```bash
echo "[FPS] $chunk frames=$N elapsed=${T}s fps=$F"
```

This is how you calibrate chunk size for a walltime (see §5.3 and the
preemptible-partition case) without re-running anything.

Suggested bucket layout and triage:

```
<exp>/hpc_logs/
  <stage>/   <stage>_<A>_<a>.out|.err|.status, sacct_<stage>_<jid>.tsv
  pipeline/  submitter logs, manifest.csv, pipeline.env
```

```bash
grep -h '^\[FPS\]' <exp>/hpc_logs/<stage>/*.out | sort      # throughput per chunk
cat <exp>/hpc_logs/<stage>/*.status                          # which tasks died and why
column -t -s'|' <exp>/hpc_logs/<stage>/sacct_*.tsv           # TIMEOUT / OOM / NODE_FAIL / CANCELLED
```

---

## 10. Software & environments

### 10.1 Modules

```bash
module avail                 # everything visible
module use /apps/unit/<Unit>/.modulefiles   # add a unit's private modulefiles
module load <name>/<ver>
module purge                 # start clean inside a job — always do this first
```

`[PROJECT]` Unit modulefile trees are readable across units. The medaka project
loaded SLEAP and Detectron2 from **`/apps/unit/ReiterU/.modulefiles`** despite
being a StephensU project (`module use` that path, then
`module load sleap-nn/0.1.2` / `detectron.2/0.6`).

Modules seen in use (existence is account/date-dependent — check `module avail`):

| Module | Cluster | Source |
|---|---|---|
| `sleap-nn/0.1.0`, `0.1.2`, `0.2.0`, `0.2.0-cu128` | saion | `[PROJECT]` / `[VERIFIED 2026-05]` |
| `sleap/1.5.2`, `1.6.2` | saion | `[PROJECT]` |
| `detectron.2/0.6` (built 2026-06-24, torch cu121) | saion, ReiterU | `[PROJECT]` |
| `singularity/4.1.4` | saion | `[PROJECT]` |
| `uv/0.11.0`, `gcc/11.2.1`, `ninja/1.12.0.git`, `cmake/3.31.7`, `python/3.10.10` | saion | `[PROJECT]` |
| `cuda/…` | saion | see §10.6 `[CONFLICT]` |
| `ruse` | deigo | `[OFFICIAL]` |
| `uv` | deigo | `[VERIFIED 2026-05]` |

`[OFFICIAL]` deigo compiler guidance: newest `gcc` + `OpenBLAS.gcc` by default;
only Intel compiler versions `2019_update5` or `2023.0` are recommended.

### 10.2 Build architecture on deigo — AMD vs Intel `[OFFICIAL]`

deigo has **two CPU architectures** and **login nodes are Intel**.

> Software built on an Intel node runs *only* on Intel nodes and crashes
> elsewhere. Software built on an **AMD node runs everywhere on deigo and
> saion**.

**Therefore: build on AMD.**

```bash
srun -t 0-4 -p compute -C epyc -c 16 --mem=16G --pty bash    # AMD Epyc — do this
srun -t 0-4 -p compute -C xeon -c 16 --mem=16G --pty bash    # Intel Xeon — only if you must
```

Building on the login node is building on Intel. That is the trap.

### 10.3 Build on a GPU **compute** node on saion `[OFFICIAL]`

saion **login nodes have no GPU drivers or libraries.** Anything that probes the
GPU at build time — or whose smoke test calls `torch.cuda.is_available()` —
must be built inside an interactive GPU job:

```bash
srun -p test-gpu -t 0-2 -c 8 --mem=32G --gres=gpu:1 --pty bash -l
```

A useful pattern is to make the installer **re-exec itself** onto a GPU node
when it detects it isn't in a job `[PROJECT]`:

```bash
if [[ "${RUN_ON_GPU:-1}" == "1" && -z "${SLURM_JOB_ID:-}" ]]; then
  exec srun -p "$SRUN_PARTITION" -c 8 --mem=32G --gres=gpu:1 --time=02:00:00 \
       --pty bash -lc "RUN_ON_GPU=0 bash '$(readlink -f "$0")'"
fi
```

### 10.4 Compute nodes are offline `[PROJECT]`

Compute nodes have **no internet**. Every download — pip resolution, model
checkpoints, git clones — must happen on the **login node** first. Resolve the
entire dependency graph there, then run the batch job against the already-built
environment. Fragile old stacks (pinned py3.8, `pillow==9.4.0`, `pyside6<6.3.2`,
`kornia`, `timm`, `numba`, `tables`) will fight each other; do that fight
online.

### 10.5 Building a shared unit module (recipe) `[PROJECT]`

The convention that produced `detectron.2/0.6` — reusable for any package:

```
/apps/unit/<Unit>/<pkg>/<ver>/venv        # the environment
/apps/unit/<Unit>/.modulefiles/<pkg>/<ver>  # the modulefile
```

1. `umask 002`; refuse to overwrite an existing prefix unless `REPLACE=1`.
2. `module purge; module load python/3.10.10 gcc/11.2.1`.
3. `python -m venv $PREFIX/venv && source .../activate; pip install -U pip setuptools wheel ninja`.
4. Install torch from the **CUDA wheel index** (`--index-url https://download.pytorch.org/whl/cu121`)
   — the CUDA *runtime* ships inside the wheel, so you do not need a matching
   system CUDA module. Set `TORCH_CUDA_ARCH_LIST="7.0;8.0"` (V100;A100) for any
   from-source extension.
5. `pip install --only-binary=:all: <runtime deps>` — see the HDF5 gotcha (§11).
6. Build the package from source with `--no-build-isolation --prefer-binary`.
7. Smoke test *on the GPU node*: import the package, print
   `torch.cuda.is_available()`.
8. Permissions: `chgrp -R $GROUP $PREFIX; chmod -R g+rX $PREFIX;
   find $PREFIX -type d -exec chmod g+s {} \;`.
9. Emit a `#%Module1.0` modulefile that loads the base python module, sets
   `VIRTUAL_ENV`, prepends `$root/venv/bin` to `PATH`, and sets
   `PYTHONDONTWRITEBYTECODE=1` + `PYTHONPYCACHEPREFIX=/tmp/$USER/pycache`.

Consumers then get either form:

```bash
module use /apps/unit/<Unit>/.modulefiles && module load <pkg>/<ver>
# or bypass modules entirely:
PY=/apps/unit/<Unit>/<pkg>/<ver>/venv/bin/python
```

`[VERIFIED 2026-05]` A venv intended to run **without any `module load`** must
have its base interpreter inside the unit tree too (e.g.
`UV_PYTHON_INSTALL_DIR=/apps/unit/<Unit>/<pkg>/python`), otherwise the venv
silently depends on your `$HOME` or a module's `LD_LIBRARY_PATH`. Also override
`UV_CACHE_DIR` to a path you own — the shared `uv` module's default cache is not
group-writable.

### 10.6 CUDA module ceiling `[CONFLICT]` — check before designing around it

- `[PROJECT, 2026-06]` A saion install script records: *"saion's cuda modules top
  out at cuda/11.3, too old for current torch"* — the reason that install used
  torch CUDA wheels instead of a system CUDA module.
- `[PROJECT, 2026-06-22]` A different saion recipe does `module load cuda/12.2`
  successfully and builds cu121 torch extensions with it.
- `[OFFICIAL]` The `gpu-a100` page states **CUDA 11.0–13.2 available as
  modules** (A100 does not support CUDA 10 or older).

Most likely the ceiling is partition/OS-image dependent (`gpu-a100` runs Rocky
8.8 `[OFFICIAL]`). **Run `module avail cuda` on the actual target partition.**

The robust default either way: **get CUDA from the torch wheel** (`cu121`,
`cu128`, …) and only require a system CUDA module when you genuinely need
`nvcc` at build or run time.

### 10.7 JIT CUDA extensions — pre-warm into a shared cache `[PROJECT]`

Some stacks JIT-compile a custom CUDA op **at runtime**, which needs `nvcc` on
the node. Offline compute nodes plus a missing `nvcc` = a job that dies minutes
in. Compile it **once** on a GPU node into a shared, persistent extension cache,
then point every batch job at the same cache:

```bash
export TORCH_EXTENSIONS_DIR=/bucket/<Unit>/<proj>/torch_ext_cache   # or $HOME/... if bucket RO bites
export TORCH_CUDA_ARCH_LIST=7.0      # must match the target GPU (§12)
export FORCE_CUDA=1
python -c '<import + one forward/backward call on a tiny tensor>'   # gate: must print OK
```

`[PROJECT]` Note the tension: `/bucket` is read-only from compute nodes, so a
bucket-hosted `TORCH_EXTENSIONS_DIR` works only if the `.so` is already built
and never needs a rebuild. A `$HOME`-hosted cache is writable from compute and
therefore safer.

### 10.8 Singularity `[PROJECT]`

Preferred when the stack is fragile *and* needs `nvcc` at runtime: a CUDA
**devel** base image ships `nvcc` inside, so the job needs no `cuda` module and
is reproducible.

```singularity
Bootstrap: docker
From: nvidia/cuda:12.1.1-devel-ubuntu22.04
%post
    export DEBIAN_FRONTEND=noninteractive
    apt-get update && apt-get install -y python3.8 python3.8-venv python3.8-dev \
        python3-pip git ffmpeg libgl1 libglib2.0-0 ninja-build && rm -rf /var/lib/apt/lists/*
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121
%environment
    export TORCH_CUDA_ARCH_LIST=7.0
    export QT_QPA_PLATFORM=offscreen
    export MPLBACKEND=Agg
```

Run with `singularity exec --nv <img>.sif …` (`--nv` = expose the NVIDIA
driver). Build where you have Docker + internet, or
`singularity build --fakeroot` on saion **if fakeroot is permitted** —
`[UNVERIFIED]` whether it is.

### 10.9 Headless GUI-derived code `[PROJECT]`

Research trackers are often Qt applications. To run their internals in a batch
job: `export QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg`, neutralise `sys.argv`
(libraries that call `argparse.parse_args()` at import/call time will choke on
Slurm's argv), flip module-level display flags off, and pass `None` for widget
arguments. Smoke-test on a tiny input first — these failures are all
crash-immediately, so a 200-frame gate costs minutes and saves days.

### 10.10 Measuring memory — `ruse` (deigo) `[OFFICIAL]`

```bash
module load ruse
ruse <your command ...>
```

Samples memory/CPU every 10 s and writes `<program>_<pid>.ruse` at exit with
peak memory, runtime, cores, process count. Negligible overhead — use it
routinely to right-size `--mem` (and it is what a `largemem` application
reviewer will want to see).

---

## 11. GPU / architecture compatibility

### 11.1 The matrix

| GPU | Partition (current / legacy) | Compute capability | VRAM | Max CUDA | Notes |
|---|---|---|---|---|---|
| **P100** | `gpu-p100` / `gpu`, also `test-gpu` | **SM 6.0** | 16 GB | **12.8 (final)** `[OFFICIAL]` | oldest; the portability floor |
| **V100** | `gpu-v100` / `gpu` | **SM 7.0** | 16 GB | **13.2** `[OFFICIAL]` | somewhat faster than P100 |
| **A100-SXM4** | `gpu-a100` / `largegpu` | **SM 8.0** | **80 GB** | 11.0–13.2 as modules `[OFFICIAL]` | ~5× the VRAM, 2–3× perf (3×+ on A100-optimised DL code) |

### 11.2 Portability rules `[OFFICIAL]`

- Code built on **P100 with CUDA ≤ 12.8** runs on P100, V100, and generally
  A100 → **build here for maximum portability**.
- Code built on **P100/V100 with CUDA 11+** generally works on `gpu-a100`.
- Code built on **`gpu-a100` generally will NOT run on the older partitions.**
- A100 hardware does not support CUDA 10 or older.
- For a from-source CUDA extension covering both eras:
  `TORCH_CUDA_ARCH_LIST="7.0;8.0"` (add `6.0` if you need P100).

### 11.3 The SM 7.5 floor — V100 is not viable for modern torch stacks

`[VERIFIED 2026-05-21]` Attempting to extend a GPU pipeline onto the V100s
failed at the stack level, not the code level:

- **PyTorch 2.11 wheels (cu128 and cu130) are compiled for SM ≥ 7.5 only.**
  V100 is SM 7.0 → not covered.
- **cuDNN 9.19 (pulled transitively by torch) also requires SM ≥ 7.5.**

Consequence: `sleap-nn 0.2.0` (torch 2.11 + cuDNN 9.19) **cannot run on V100**
without custom-building PyTorch. The older `sleap-nn 0.1.2` *does* work on V100
but has a different CLI (`sleap-nn-track`; no `sleap-nn predict`, no TensorRT
export), so mixing versions across partitions would require split-architecture
pipeline support — judged not worth ~40 % extra throughput. **Decision: stay
A100-only for that stack.**

`[PROJECT, verified]` Independently, **torch 2.1.1 + cu121 runs fine on V100
(SM 7.0)** — cu121 wheels ship sm_70 kernels. So the SM 7.5 floor is a property
of *recent* torch/cuDNN releases, not of CUDA 12 generally. **When you pick a
torch version, check its supported SM list against your target partition
before designing the pipeline around it.**

### 11.4 Driver versions matter too `[PROJECT, 2026-06]`

- `largegpu`/`gpu-a100` A100 nodes: driver **595** — fine for torch cu121.
- The legacy `gpu` partition is **heterogeneous**: some V100 nodes still carried
  a CUDA-11.1-era driver, too old for torch cu121 → `torch.cuda.is_available()`
  returns **False** with no other symptom. Requesting `--gres=gpu:v100:1` landed
  on a driver-570 node in one test, but that is luck, not a guarantee. Prefer
  `gpu-a100`, or assert the driver at task start:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -c 'import torch; assert torch.cuda.is_available(), "no CUDA"; print(torch.cuda.get_device_capability(0))'
```

---

## 12. Gotchas

Each line: **the trap** — *why it bites you*.

1. **Partition renamed (`largegpu`→`gpu-a100`, `gpu`→`gpu-v100`/`gpu-p100`)** — *inherited scripts fail with `Invalid partition name` on a new account, because old names only survive for pre-existing users.*
2. **`/bucket` is read-only on compute nodes** — *every job that writes its result directly to bucket dies at the last line, after all the compute is done.*
3. **saion needs `bash -l`** — *without `#!/bin/bash -l` (or `--pty bash -l`), `module` isn't initialised and every `module load` fails inside the job.*
4. **Submitting from a `/bucket` or `/flash` CWD** — *Slurm writes `.out`/`.err` relative to the submit dir; on read-only or odd CWDs the job fails before your code runs. Submit from `$HOME` or pass an absolute `-o`.*
5. **deigo login nodes are Intel** — *"just build it on the login node" silently produces a binary that crashes on the 456 AMD nodes. Build with `-C epyc`.*
6. **saion login nodes have no GPU driver** — *builds whose smoke test calls `torch.cuda.is_available()` fail or (worse) silently install a CPU-only wheel. Build in a GPU job.*
7. **Compute nodes have no internet** — *pip/model-checkpoint downloads hang or fail deep inside a long job. Resolve everything on the login node first.*
8. **`nvidia-smi` on a node shows only ONE GPU** `[VERIFIED 2026-05]` — *saion uses `pam_slurm_adopt`, so your `ssh` session is adopted into one of your own jobs' cgroups and `nvidia-smi` is filtered by that job's `CUDA_VISIBLE_DEVICES`. You will conclude the node has 1 GPU and mis-size everything.* Three commands that work instead:
   ```bash
   scontrol show node saion-gpu26 | grep -E 'Gres|CfgTRES'                 # physical GPUs Slurm sees
   ssh saion srun --jobid=<jobid>_<task> nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader
   ssh saion squeue -h -w saion-gpu26 -o '%i %u %T %c %m %G'               # all your jobs on that node
   ```
9. **`HDF5_*` environment leakage** `[PROJECT]` — *the compute node's system HDF5 is ancient (1.8.12) and leaks via `HDF5_DIR`/`HDF5_VERSION`/`HDF5_PLUGIN_PATH`, making `pip install h5py` attempt a source build that fails. Fix: `unset HDF5_DIR HDF5_VERSION HDF5_PLUGIN_PATH` and `pip install --only-binary=:all: h5py` so the manylinux wheel's bundled HDF5 is used.*
10. **JIT-compiled CUDA extensions** `[PROJECT]` — *they need `nvcc` at runtime; offline compute nodes without a CUDA module fail minutes into a long job. Pre-warm into a shared `TORCH_EXTENSIONS_DIR` and gate on a forward+backward smoke test.*
11. **`datacp` GrpSubmit = 20** `[VERIFIED 2026-05-20]` — *one per-chunk array blows the cap and blocks all your other data movement. Single jobs only.*
12. **`short` (deigo, 2 h) for a data-moving bridge** `[VERIFIED 2026-05]` — *cross-cluster transfers can take hours; the 2 h cap kills the bridge mid-copy and wedges the pipeline. Use `compute`.*
13. **`afterok:` on a cleanup job** `[VERIFIED 2026-05]` — *a transient SSH failure in an upstream leg leaves scratch allocated forever. Use `afterany:` and let a content check decide whether deletion is safe.*
14. **`kex_exchange_identification` on cross-cluster ssh** `[VERIFIED 2026-05]` — *intermittent connection resets wedge whole runs. Wrap every ssh/rsync in a retry with backoff.*
15. **Polling `squeue` for completion** — *`squeue` only lists pending/running jobs, so "not in squeue" is indistinguishable from "never submitted", and it tells you nothing about failure. Poll `sacct -X -P` and require `COMPLETED` + `0:0`.*
16. **Scratch cleanup destroys your logs** `[VERIFIED 2026-05]` — *the very failures you need to debug (walltime kill, NODE_FAIL, drain) are the ones that leave nothing behind. Ship logs to bucket (§9).*
17. **Existence check as an idempotency test** — *a truncated output from a walltime kill has non-zero size; a `-s` check will skip it and poison every downstream stage. Validate that the artifact opens.*
18. **`rsync -a` preserves the source group onto bucket** `[VERIFIED 2026-05]` — *unit-shared outputs end up owned by your personal group. Pass `--chown=:$GROUP` and set the setgid bit on directories.*
19. **Default memory is 4 GB/core** `[OFFICIAL]` — *large-memory jobs get silently OOM-killed. Always pass `--mem`.*
20. **`largemem` >500 GB requests route to the scarce 750 GB tier** `[OFFICIAL]` — *asking for 501 GB "just in case" can add days of queue wait. Measure with `ruse` first.*
21. **Over-allocating on `largemem`/`bigmem`** `[OFFICIAL]` — *usage is actively monitored and grossly over-allocating jobs get stopped.*
22. **Runs longer than ~2–3 weeks** `[OFFICIAL]` — *failure risk rises sharply and long jobs are not factored into maintenance-shutdown planning. Checkpoint and split.*
23. **`test-gpu` is preemptible** `[OFFICIAL]` — *jobs can be stopped at any time. Never put real work there.*
24. **`intel` is gang-scheduled in 10 s slices** `[OFFICIAL]` — *feels hung interactively; fine for a quick batch test.*
25. **A pinned dependency you didn't pin** `[PROJECT]` — *an unpinned transitive library (in one real case `sleap-io`) resolved to a newer major in a rebuilt environment and silently corrupted saved data on write. Pin the versions your on-disk formats depend on, and round-trip save→reopen after any environment rebuild.*
26. **Assuming a torch wheel supports your GPU** — *torch 2.11 wheels drop SM < 7.5, so V100 fails; torch 2.1.1+cu121 is fine on V100. Check the wheel's SM list against the partition before committing to a design.*

---

## 13. Verify-on-a-new-account checklist

Run top to bottom on a fresh account before trusting anything above. Everything
here is cheap and read-only except where noted.

**Access**
- [ ] `ssh <id>@deigo.oist.jp` and `ssh <id>@saion.oist.jp` both succeed.
- [ ] `~/.ssh/config` has `deigo` and `saion` aliases; `ssh saion hostname` works **from a deigo compute node** (needed for §7.5b uploads).
- [ ] `ssh saion 'bash -lc "module avail 2>&1 | head"'` — confirms `-l` gives a working module env.

**Identity / accounting**
- [ ] `id -Gn` — which unix groups (e.g. `stephensuni`) you are in.
- [ ] `sacctmgr show user $USER withassoc format=User,Account,Cluster,Partition -p` — **which account your jobs charge to**. Resolve the unit/account question in §6 here; do not assume it matches the unit whose modulefiles you load.
- [ ] `sacctmgr show assoc user=$USER format=Cluster,Account,Partition,QOS,GrpSubmit,GrpTRES,MaxWall -p | column -t -s'|'` on **both** clusters — record the real caps, replace §5's tables.

**Partitions**
- [ ] `sinfo -o '%20P %5a %10l %6D %N'` on both clusters — **which partition names your account actually has** (new vs legacy, §2).
- [ ] `scontrol show partition=gpu-a100` (and `largegpu`) — exactly one should exist for you.
- [ ] Confirm whether `datacp` (deigo) and `short-a100` (saion) exist for you.

**Storage**
- [ ] `df -h /bucket /flash /work /hpacquire 2>/dev/null` and `ls -ld /bucket/<Unit>` — confirm the unit tree and your write access.
- [ ] Write-test from a **compute** node: `srun -p short --pty bash -c 'touch /bucket/<Unit>/x 2>&1; touch /flash/<Unit>/$USER/x && echo flash-ok'` — expect bucket to fail (read-only) and flash to succeed.
- [ ] `ls /deigo_flash/` on saion and `ls /saion_work/` on deigo — confirm cross-mounts are present.
- [ ] Confirm the Windows mapping: `Z:\<Unit>\…` == `\\bucket.oist.jp\bucket\<Unit>\…` == `/bucket/<Unit>/…`.
- [ ] Home quota: `quota -s` or equivalent — saion is 50 GB `[OFFICIAL]`; deigo `[UNVERIFIED]`.

**GPU / build environment**
- [ ] `srun -p test-gpu -t 0-1 -c 4 --mem=16G --gres=gpu:1 --pty bash -l` gets an allocation.
- [ ] Inside it: `nvidia-smi`, then `python -c 'import torch;print(torch.cuda.get_device_capability(0))'` → expect `(6,0)` P100 / `(7,0)` V100 / `(8,0)` A100.
- [ ] `scontrol show node <gpu-node> | grep -E 'Gres|CfgTRES'` — physical GPU count (§12.8).
- [ ] `module avail cuda` **on the target partition** — resolve the §10.6 conflict.
- [ ] `module avail` | grep for the stacks you need; if a unit module is missing, `module use /apps/unit/<Unit>/.modulefiles` first.
- [ ] deigo: `srun -p compute -C epyc --pty bash -c 'lscpu | grep "Model name"'` — confirm `-C epyc` really lands on AMD.

**End-to-end smoke (do this before any long run)**
- [ ] Submit a **1-task array** with a tiny input: confirm it runs, writes to scratch, emits a `[FPS]` line, and ships its log to bucket.
- [ ] Re-submit the same task: confirm the idempotent skip fires.
- [ ] Kill it mid-run (`scancel`) and confirm the `--signal=TERM@60` trap delivered final log lines + a `.status` marker to bucket.
- [ ] Confirm the login-node watcher detects terminal state via `sacct` and refuses the copy on a deliberately failed task.
- [ ] `sacct -j <jid> --format=JobID,State,ExitCode,Elapsed,MaxRSS` — the numbers you will use to size the real run.
```
