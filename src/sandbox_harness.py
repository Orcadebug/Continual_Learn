# ===== STAGE 1 MONITORED SANDBOX HARNESS (Laya continual-learning program) =====
# Engineering wrapper ONLY: this file changes no experiment science. It runs ONE
# lifetime of the frozen Exp 16 arm-A recipe (full-trajectory self-imitation,
# rounds=5, encoder lr 1e-5, protected core, percentile admission, novelty 30%)
# and adds the Stage 1 unattended-run infrastructure:
#   * checkpoint / resume (a single lifetime may span several Colab runtimes)
#   * episode-level eval logging + consistency assert (Exp 16 audit /
#     RESEARCH_LOG.md "REQUIRED INSTRUMENTATION FIXES", Directive sections 4/17)
#   * per-phase battery + regression detection with a protected best checkpoint
#   * REGRESSION ROLLBACK: on a regression flag the live model is rolled back
#     to the best checkpoint automatically (diagnostics -> results JSON)
#   * held-out checkpoint comparison set (seed0=909000) used ONLY to compare
#     frozen policies across checkpoints -- never training/admission/selection
#   * RECOVERY DRILL mode (SANDBOX_MODE=drill or --drill): kill-and-resume
#     integrity verification with a printed PASS/FAIL summary
#   * incremental results flush (notebook-output recovery pattern)
# Transport: same typed kernel-string route as the Exp 13-17 cells (see
# EXP17_LAUNCH_PACKET.md) -- paste the whole file as ONE cell and run it.
# REUSE MAP (verbatim provenance):
#   World, question_for, encode_state, forward_full, TOK proxy : lab_machinery.py
#   cell_passable                                              : exp17_stage1_cell.py
#     (kind-aware version; lab_machinery.py's older cell_passable misclassifies
#      the switch-world door once the decoy key is held -- do not swap back)
#   ego_text + ALL lifetime internals (mem_push, admit, contrib, policy_probs,
#     train_on_memory, purge_old, probe, behavior_action, episode loop)
#                                                              : exp16_segments_cell.py
#   greedy_eval is exp16's, extended ONLY to record per-episode data.
import json, random, statistics, time, copy, os, shutil, sys
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from laya.train import (load_checkpoint, items_from_rows, encode_item, collate_items)

# ------------------------------------------------------------------ CONFIG
RUN_TAG        = "SANDBOX-A-R1"          # arm A only; single long lifetime
TORCH_SEED     = 6000                    # exp16: 6000 + run_idx*101, run_idx=0
RNG_SEED       = 5000                    # exp16: 5000 + run_idx*101, run_idx=0
PHASES         = [("P1_empty", "empty", 80), ("P2_keydoor", "keydoor", 100),
                  ("P3_switch_rulechange", "switch", 100),
                  ("P4_return_empty", "empty", 40)]
# Phase lengths match exp16_segments_cell.py exactly (P3 = 100; 140 was the
# Exp 12 variant and is NOT used here — corrected 2026-10-12).
EVAL_N         = 24                      # unified eval instrument (Exp 15/16)
EVAL_SEED0     = 707000
CHECKPOINT_EPS = 25                      # checkpoint + in-phase eval cadence
MEM_CAP        = 4000                    # exp16 working-memory cap
ROUNDS         = 5                       # exp16 train_on_memory rounds
ENC_LR         = 1e-5                    # exp16 encoder lr (head lr 1e-3)
REGRESSION_DROP = 0.15                   # battery drop vs best -> REGRESSION + ROLLBACK
HELDOUT_N      = 24                      # held-out eval size per world
HELDOUT_SEED0  = 909000                  # held-out seed base (battery uses 707000)
STATE_PATH     = "/content/laya_sandbox_state.pt"
BEST_PATH      = "/content/laya_sandbox_best.pt"
RESULTS_PATH   = "/content/laya_sandbox_results.json"
EPISODES_PATH  = "/content/laya_sandbox_episodes.jsonl"
DRIVE_DIR      = "/content/drive/MyDrive/laya_sandbox"
# Mode: default "run". DRILL mode (acceptance criterion 2) uses fully separate
# state/results paths + Drive folder, so a drill can never contaminate a real
# run's checkpoints. Enable via a prior cell:  import os; os.environ["SANDBOX_MODE"]="drill"
# (or any argv containing --drill).
DRILL_MODE = ("--drill" in sys.argv[1:]) or os.environ.get("SANDBOX_MODE") == "drill"
if DRILL_MODE:
    RUN_TAG = RUN_TAG + "-DRILL"
    STATE_PATH = "/content/laya_sandbox_drill_state.pt"
    BEST_PATH = "/content/laya_sandbox_drill_best.pt"
    RESULTS_PATH = "/content/laya_sandbox_drill_results.json"
    EPISODES_PATH = "/content/laya_sandbox_drill_episodes.jsonl"
    DRIVE_DIR = DRIVE_DIR + "_drill"
RESUME_REPORT = {}   # filled by run_sandbox() whenever it resumes; read by the drill
T0             = time.time()

def log(*a): print(*a, flush=True)
def elapsed(): return round(time.time() - T0, 1)

# ------------------------------------------------------- DRIVE MIRROR (guarded)
DRIVE_OK = False
try:
    from google.colab import drive as _gdrive
    _gdrive.mount("/content/drive")
    os.makedirs(DRIVE_DIR, exist_ok=True)
    DRIVE_OK = True
    log("DRIVE mounted; mirror dir:", DRIVE_DIR)
except Exception as e:
    log("DRIVE unavailable (", type(e).__name__, ") -- /content only; a reclaim loses state")

def mirror(path):
    """Copy a /content file into the Drive sandbox folder. Never raises."""
    if not DRIVE_OK:
        return
    try:
        shutil.copy(path, os.path.join(DRIVE_DIR, os.path.basename(path)))
    except Exception as e:
        log("MIRROR-FAIL", path, type(e).__name__)

def find_state():
    """Resume source: Drive first (survives reclaims), then /content."""
    if DRIVE_OK:
        p = os.path.join(DRIVE_DIR, os.path.basename(STATE_PATH))
        if os.path.exists(p):
            return p
    if os.path.exists(STATE_PATH):
        return STATE_PATH
    return None

def find_best():
    """Best-checkpoint source: /content first, then the Drive mirror."""
    if os.path.exists(BEST_PATH):
        return BEST_PATH
    if DRIVE_OK:
        p = os.path.join(DRIVE_DIR, os.path.basename(BEST_PATH))
        if os.path.exists(p):
            return p
    return None

# -------------------------------------------- MACHINERY (verbatim, lab_machinery.py)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN, HEAD_MAX_LEN = 256, 128
ACTIONS = ["north", "south", "west", "east"]
DESC = {"north": "move one cell up", "south": "move one cell down",
        "west": "move one cell left", "east": "move one cell right"}
STEP_PEN, BUMP_PEN, MAX_STEPS = -0.01, -0.05, 60

def question_for(order):
    return {"act": {"type": "choice",
                    "instructions": "Choose the agent's next action to reach the goal G in as few steps as possible.",
                    "criteria": {a: DESC[a] for a in order}}}

class World:
    def __init__(self, kind, rng):
        self.kind, self.rng = kind, rng
        self.size = 8
        self.reset_layout()
    def reset_layout(self):
        rng, n = self.rng, self.size
        while True:
            walls = {(x, 0) for x in range(n)} | {(x, n - 1) for x in range(n)} | \
                    {(0, y) for y in range(n)} | {(n - 1, y) for y in range(n)}
            if self.kind in ("keydoor", "switch"):
                door_y = rng.randint(1, n - 2)
                for y in range(n):
                    if y != door_y:
                        walls.add((4, y))
                self.door = (4, door_y)
            else:
                self.door = None
                for _ in range(5):
                    walls.add((rng.randint(1, n - 2), rng.randint(1, n - 2)))
            free = [(x, y) for x in range(n) for y in range(n) if (x, y) not in walls and (x, y) != self.door]
            self.agent = rng.choice(free)
            rest = [c for c in free if c != self.agent]
            self.goal = rng.choice(rest); rest.remove(self.goal)
            same_side = [c for c in rest if (c[0] < 4) == (self.agent[0] < 4)] or rest
            self.key = rng.choice(same_side) if self.kind in ("keydoor", "switch") else None
            if self.kind == "switch":
                rest2 = [c for c in same_side if c != self.key] or [c for c in rest if c != self.key]
                self.switch = rng.choice(rest2)
            else:
                self.switch = None
            if self.agent != self.goal:
                break
        self.walls = walls
        self.has_key, self.door_open, self.switch_on = False, False, False
        self.steps = 0
    def step(self, a):
        dx, dy = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}[a]
        nx, ny = self.agent[0] + dx, self.agent[1] + dy
        self.steps += 1
        r = STEP_PEN
        target = (nx, ny)
        blocked = target in self.walls or (self.door and target == self.door and not self.door_open)
        if blocked:
            r += BUMP_PEN
        else:
            self.agent = target
            if self.key and target == self.key and not self.has_key:
                self.has_key = True
                if self.kind == "keydoor":
                    self.door_open = True
            if self.switch and target == self.switch:
                self.switch_on = True; self.door_open = True
        done = self.agent == self.goal
        if done: r += 1.0
        if self.steps >= MAX_STEPS: done = True
        return r, done

TOK_HOLDER = [None]
class _TokProxy:
    @property
    def pad_token_id(self):
        return TOK_HOLDER[0].pad_token_id
TOK = _TokProxy()

def encode_state(tok, text, order=ACTIONS):
    rows = [{"state": text, "questions": question_for(order), "expected": {"act": "north"}}]
    items, skipped = items_from_rows(tok, rows, MAX_LEN, HEAD_MAX_LEN)
    assert items, skipped
    return encode_item(tok, items[0], MAX_LEN, HEAD_MAX_LEN)

def forward_full(model, vhead, encs):
    batch = collate_items([[e for e in encs]], TOK.pad_token_id)
    ids = batch["input_ids"].to(DEVICE); att = batch["attention_mask"].to(DEVICE)
    mpos = batch["marker_pos"].to(DEVICE); mmask = batch["marker_mask"].to(DEVICE)
    qt = batch["qtype"].to(DEVICE)
    with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=(DEVICE.type == "cuda")):
        h = model.encoder(input_ids=ids, attention_mask=att).last_hidden_state
        h = h + model.type_emb(qt)[:, None, :]
        if model.head is not None:
            pad = ~att.bool()
            for layer in model.head.layers:
                h = layer(h, src_key_padding_mask=pad)
    idx = mpos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
    m = torch.gather(h, 1, idx)
    logits = model.scorer(m).squeeze(-1).float()
    logits = logits.masked_fill(~mmask, -1e4)
    value = vhead(h[:, 0].float()).squeeze(-1)
    return logits, value

# cell_passable: verbatim from exp17_stage1_cell.py (kind-aware; see header note)
def cell_passable(world, cell, has_key):
    if cell in world.walls: return False
    if world.door and cell == world.door:
        if world.kind == "keydoor": return has_key
        if world.kind == "switch": return world.switch_on
    return True

# ego_text: verbatim from exp16_segments_cell.py
def ego_text(world):
    ax, ay = world.agent; gx, gy = world.goal
    def rel(tx, ty, name):
        dx, dy = tx - ax, ty - ay
        parts = []
        if dx < 0: parts.append(f"{-dx} west")
        if dx > 0: parts.append(f"{dx} east")
        if dy < 0: parts.append(f"{-dy} north")
        if dy > 0: parts.append(f"{dy} south")
        return f"{name}: " + (", ".join(parts) if parts else "here") + "."
    blocked = []
    for a in ACTIONS:
        dx, dy = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}[a]
        nc = (ax + dx, ay + dy)
        if not (0 <= nc[0] < world.size and 0 <= nc[1] < world.size) or not cell_passable(world, nc, world.has_key):
            blocked.append(a)
    lines = ["You are in a grid room. Reach the goal G.",
             rel(gx, gy, "Goal"),
             "Blocked directions: " + (", ".join(blocked) if blocked else "none") + ".",
             "You carry: " + ("the key." if world.has_key else "nothing.")]
    if world.key is not None and not world.has_key:
        lines.append(rel(world.key[0], world.key[1], "Key"))
    if world.switch is not None and not world.switch_on:
        lines.append(rel(world.switch[0], world.switch[1], "Switch"))
    if world.door:
        if world.door_open:
            door_txt = "open."
        elif world.kind == "keydoor":
            door_txt = "locked. The key opens it."
        else:
            door_txt = "locked."
        lines.append("The door is " + door_txt)
        lines.append(rel(world.door[0], world.door[1], "Door"))
    return "\n".join(lines)

# ============================================================ SANDBOX LIFETIME
# exp16 run_lifetime(tag, run_idx, "A") restructured so ALL loop state lives in
# one progress dict S that is checkpointed at episode boundaries. Every science
# function below is exp16's, verbatim; arm is fixed to "A" as specified.
def run_sandbox(stop_after_checkpoints=None, max_episodes=None):
    state_src = find_state()
    S = None
    if state_src:
        S = torch.load(state_src, map_location="cpu", weights_only=False)
        log(f"RESUMED from {state_src}: phase_idx={S['phase_idx']} "
            f"next_ep={S['ep_idx']} best_battery={S['best_overall']} "
            f"episodes_logged={len(S['episode_log'])}")

    # ---- model / tokenizer / optimizer: exp16 construction, verbatim ----
    if S is None:
        torch.manual_seed(TORCH_SEED)
        rng = random.Random(RNG_SEED)
    model, tok, _c = load_checkpoint("convaiinnovations/laya")
    TOK_HOLDER[0] = tok
    model = model.to(DEVICE)
    d = model.encoder.config.hidden_size
    vhead = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1)).to(DEVICE)
    torch.cuda.empty_cache()
    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    groups = [{"params": [p for n, p in model.named_parameters() if n.startswith("encoder.")], "lr": ENC_LR},
              {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")], "lr": 1e-3}]
    opt = torch.optim.AdamW(groups, weight_decay=0.0)

    if S is None:
        S = {"phase_idx": 0, "ep_idx": 0,              # next episode to run
             "memory": [], "protected": [],             # lists; deque rebuilt below
             "succ_lens": {0: [], 1: [], 2: [], 3: []},
             "visits": {}, "admitted": 0,
             "phase_pairs": [], "recent": [], "door_recent": [],
             "first12": [], "purged": 0,
             "history": [], "episode_log": [],
             "best_overall": None, "batteries": [], "heldout": [],
             "rollbacks": [], "stats": {"opt_steps": 0}}
        rng_state = None
    else:
        RESUME_REPORT.update({
            "phase_idx": S["phase_idx"], "ep_idx": S["ep_idx"],
            "admitted": S["admitted"], "mem": len(S["memory"]),
            "prot": len(S["protected"]),
            "opt_entries": len(S["opt_sd"]["state"]),
            "opt_steps": S.get("stats", {}).get("opt_steps", 0),
            "rng_ok": S.get("torch_rng") is not None and S.get("py_rng") is not None})
        S.setdefault("heldout", []); S.setdefault("rollbacks", [])
        S.setdefault("stats", {"opt_steps": 0})
        model.load_state_dict(S.pop("model_sd"))
        vhead.load_state_dict(S.pop("vhead_sd"))
        opt.load_state_dict(S.pop("opt_sd"))
        torch.set_rng_state(S.pop("torch_rng"))
        if torch.cuda.is_available() and S.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(S.pop("cuda_rng"))
        else:
            S.pop("cuda_rng", None)
        rng = random.Random()
        rng.setstate(S.pop("py_rng"))
        RESUME_REPORT["opt_entries_loaded"] = len(opt.state_dict()["state"])
        log("RESUMED state restored (weights, optimizer, memory, RNG)")

    ck = [0]; eps_run = [0]   # per-invocation counters (drill stop conditions)
    memory = deque(S["memory"])
    protected = S["protected"]
    succ_lens = S["succ_lens"]
    visits = S["visits"]
    admitted_eps = [S["admitted"]]

    def save_checkpoint(where):
        S["memory"] = list(memory)
        S["protected"] = protected
        S["succ_lens"] = succ_lens
        S["visits"] = visits
        S["admitted"] = admitted_eps[0]
        payload = dict(S)
        payload.update({
            "model_sd": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "vhead_sd": {k: v.detach().cpu() for k, v in vhead.state_dict().items()},
            "opt_sd": opt.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
            "py_rng": rng.getstate(),
            "saved_at_elapsed_s": elapsed()})
        ck[0] += 1
        torch.save(payload, where)
        mirror(where)
        # episode JSONL is REWRITTEN from S (never appended): idempotent across
        # a reclaim that lands between an eval and its checkpoint.
        with open(EPISODES_PATH, "w") as f:
            for rec in S["episode_log"]:
                f.write(json.dumps(rec) + "\n")
        mirror(EPISODES_PATH)
        log(f"CHECKPOINT saved -> {where} phase={PHASES[S['phase_idx']][0]} "
            f"next_ep={S['ep_idx']} mem={len(memory)} prot={len(protected)} t={elapsed()}s")

    def flush_results(status):
        out = {"run": RUN_TAG, "status": status, "elapsed_s": elapsed(),
               "history": S["history"], "batteries": S["batteries"],
               "heldout": S["heldout"], "rollbacks": S["rollbacks"],
               "best_overall": S["best_overall"],
               "episodes_logged": len(S["episode_log"])}
        with open(RESULTS_PATH, "w") as f:
            json.dump(out, f, indent=2)
        mirror(RESULTS_PATH)

    # ---- exp16 internals, verbatim (arm fixed to "A") ----
    def mem_push(pairs, ph, times=1):
        for _ in range(times):
            for e, a in pairs:
                memory.append((e, a, ph))
        while len(memory) > MEM_CAP:
            memory.popleft()

    def admit(pairs, ph):
        L = len(pairs)
        succ_lens[ph].append(L)
        prev = sorted(succ_lens[ph][:-1])
        if len(prev) < 4:
            return True
        p25 = prev[len(prev) // 4]
        return L <= max(16, 2.0 * p25)

    def contrib(prs, midx):
        # exp16 arm "A": full trajectory, no extra segment.
        return prs, []

    def policy_probs(text):
        enc = encode_state(tok, text)
        with torch.no_grad():
            logits, _v = forward_full(model, vhead, [enc])
        probs = torch.softmax(logits[0], -1)
        probs = torch.nan_to_num(probs, nan=0.25, posinf=0.0, neginf=0.0).clamp(min=0)
        tot = float(probs.sum())
        probs = probs / tot if tot > 0 else torch.full_like(probs, 0.25)
        return probs, enc

    def train_on_memory(cur_ph, rounds=ROUNDS, bs=32):
        if len(memory) + len(protected) < 64:
            return
        mem = list(memory)
        weights = [3.0 if m[2] == cur_ph else 1.0 for m in mem]
        for _ in range(rounds):
            batch = []
            if len(protected) >= 8:
                batch += rng.choices(protected, k=8)
            k = bs - len(batch)
            if mem:
                batch += rng.choices(mem, weights=weights, k=k)
            if len(batch) < 8:
                return
            encs = [b[0] for b in batch]
            acts = torch.tensor([b[1] for b in batch], device=DEVICE)
            logits, _v = forward_full(model, vhead, encs)
            loss = F.cross_entropy(logits, acts)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            S["stats"]["opt_steps"] += 1

    def purge_old(cur_ph, frac=0.5):
        mem = list(memory)
        old_idx = [i for i, m in enumerate(mem) if m[2] < cur_ph]
        if len(old_idx) < 100:
            return 0
        drop = set(rng.sample(old_idx, int(len(old_idx) * frac)))
        kept = [m for i, m in enumerate(mem) if i not in drop]
        memory.clear()
        memory.extend(kept)
        return len(drop)

    def probe(kind, seed):
        ents, mars = [], []
        prng = random.Random(seed)
        for _ in range(16):
            w = World(kind, prng)
            p, _e = policy_probs(ego_text(w))
            ents.append(float(-(p * p.clamp(min=1e-12).log()).sum()))
            t2 = torch.topk(p, 2).values
            mars.append(float(t2[0] - t2[1]))
        return round(sum(ents) / len(ents), 3), round(sum(mars) / len(mars), 3)

    def state_key(w):
        return (w.agent, hash(ego_text(w)))

    def behavior_action(w, text):
        p, enc = policy_probs(text)
        if rng.random() < 0.30:
            best, best_c = [], None
            for ai, a in enumerate(ACTIONS):
                w2 = copy.deepcopy(w)
                w2.step(a)
                c = visits.get(state_key(w2), 0)
                if best_c is None or c < best_c:
                    best, best_c = [ai], c
                elif c == best_c:
                    best.append(ai)
            return rng.choice(best), enc
        return int(torch.multinomial(p, 1)), enc

    # ---- exp16 greedy_eval, extended ONLY with per-episode records ----
    def greedy_eval_logged(kind, n, seed0, eval_tag, record=True):
        succ, steps = 0, []
        mile_hit, mile_succ = 0, 0
        dr_n, dr_succ, dr_mile_succ = 0, 0, 0
        for i in range(n):
            w = World(kind, random.Random(seed0 + i))
            start = list(w.agent); goal = list(w.goal)
            door_required = bool(kind != "empty" and (start[0] < 4) != (goal[0] < 4))
            done = False
            mile = False
            while not done:
                p, _e = policy_probs(ego_text(w))
                _r, done = w.step(ACTIONS[int(torch.argmax(p))])
                if kind == "keydoor" and w.has_key:
                    mile = True
                if kind == "switch" and w.switch_on:
                    mile = True
            got = int(w.agent == w.goal)
            succ += got
            if kind != "empty":
                mile_hit += int(mile)
                mile_succ += int(mile and got)
            if door_required:
                dr_n += 1; dr_succ += got; dr_mile_succ += int(mile and got)
            steps.append(w.steps)
            if record:
                S["episode_log"].append({
                    "run": RUN_TAG, "eval_tag": eval_tag, "world": kind,
                    "episode_seed": seed0 + i, "door_required": door_required,
                    "agent_start": start, "goal_pos": goal,
                    "milestone_reached": bool(mile), "success": bool(got),
                    "steps": w.steps, "success_via_milestone": bool(got and mile)})
        # Consistency assert (audit fix 3): raw counts must decompose exactly.
        nomile_succ = succ - mile_succ
        assert succ == mile_succ + nomile_succ, (kind, succ, mile_succ, nomile_succ)
        out = {"greedy_eval": round(succ / n, 3),
               "greedy_steps": round(sum(steps) / n, 1),
               "mile_rate": (round(mile_hit / n, 3) if kind != "empty" else None),
               "cond_succ": (round(mile_succ / mile_hit, 3) if mile_hit else None),
               "counts": {"succ": succ, "n": n, "mile_succ": mile_succ,
                          "nomile_succ": nomile_succ, "mile_hit": mile_hit},
               "door_required": {"n": dr_n, "succ": dr_succ,
                                 "mile_succ": dr_mile_succ,
                                 "rate": (round(dr_succ / dr_n, 3) if dr_n else None)}}
        log(f"EVAL {eval_tag} {kind}: succ {succ}/{n} (milestone {mile_succ} + "
            f"no-milestone {nomile_succ}) rate={out['greedy_eval']} "
            f"door_required {dr_succ}/{dr_n}")
        return out

    def run_heldout(tag_label):
        # Held-out checkpoint comparison (acceptance criterion 4): seed base
        # 909000 -- disjoint from the battery (707000) and from every behavior /
        # admission path. Results NEVER feed training, admission, best-checkpoint
        # selection, regression, or rollback; they exist only so frozen-policy
        # learning can be compared across checkpoints independently. Episodes
        # are not written to the episode JSONL (record=False).
        res = {}
        for kind in ["empty", "keydoor", "switch"]:
            res[kind] = greedy_eval_logged(kind, n=HELDOUT_N, seed0=HELDOUT_SEED0,
                                           eval_tag=f"HELDOUT-{tag_label}", record=False)
        overall = round(sum(res[k]["greedy_eval"] for k in res) / 3.0, 3)
        entry = {"tag": tag_label, "overall": overall, "results": res,
                 "elapsed_s": elapsed()}
        S["heldout"].append(entry)
        log(f"HELDOUT-{RUN_TAG} at {tag_label}: overall {overall} "
            f"(empty {res['empty']['counts']['succ']}/{HELDOUT_N}, "
            f"keydoor {res['keydoor']['counts']['succ']}/{HELDOUT_N}, "
            f"switch {res['switch']['counts']['succ']}/{HELDOUT_N}) "
            f"[comparison only -- never used for selection]")
        return entry

    def rollback_to_best(tag_label, live_overall):
        # Acceptance criterion 3: on REGRESSION, restore the best checkpoint's
        # weights + optimizer + RNG into the live run and record diagnostics.
        # Memory, protected core, and progress counters stay LIVE (experience
        # is not discarded); the best file itself is never modified here, and
        # it is only ever replaced by a strictly better battery (run_battery).
        src = find_best()
        if src is None:
            log(f"ROLLBACK-FAILED at {tag_label}: no best checkpoint found")
            S["rollbacks"].append({"tag": tag_label, "ok": False,
                                   "live_battery": live_overall,
                                   "best_battery": S["best_overall"],
                                   "elapsed_s": elapsed()})
            return False
        payload = torch.load(src, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_sd"])
        vhead.load_state_dict(payload["vhead_sd"])
        opt.load_state_dict(payload["opt_sd"])
        torch.set_rng_state(payload["torch_rng"])
        if torch.cuda.is_available() and payload.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        rng.setstate(payload["py_rng"])
        diag = {"tag": tag_label, "ok": True,
                "phase": PHASES[S["phase_idx"]][0],
                "live_battery": live_overall, "best_battery": S["best_overall"],
                "drop": round(S["best_overall"] - live_overall, 3),
                "memory": len(memory), "protected": len(protected),
                "admitted_total": admitted_eps[0],
                "memory_stale_frac": round(sum(1 for m in memory if m[2] < S["phase_idx"]) / max(1, len(memory)), 2),
                "best_checkpoint": src, "elapsed_s": elapsed()}
        S["rollbacks"].append(diag)
        log(f"ROLLBACK event at {tag_label}: restored best checkpoint {src} "
            f"(battery {S['best_overall']}) over live battery {live_overall}; "
            f"memory={len(memory)} protected={len(protected)} kept live; "
            f"diagnostics recorded in results JSON")
        return True

    def run_battery(tag_label):
        battery = {}
        for kind in ["empty", "keydoor", "switch"]:
            battery[kind] = greedy_eval_logged(kind, n=EVAL_N, seed0=EVAL_SEED0,
                                               eval_tag=tag_label)
        overall = round(sum(battery[k]["greedy_eval"] for k in battery) / 3.0, 3)
        entry = {"tag": tag_label, "overall": overall, "results": battery,
                 "elapsed_s": elapsed()}
        S["batteries"].append(entry)
        log(f"BATTERY-{RUN_TAG} after {tag_label}: overall {overall} "
            f"(empty {battery['empty']['counts']['succ']}/{EVAL_N}, "
            f"keydoor {battery['keydoor']['counts']['succ']}/{EVAL_N}, "
            f"switch {battery['switch']['counts']['succ']}/{EVAL_N})")
        if S["best_overall"] is None or overall > S["best_overall"]:
            S["best_overall"] = overall
            save_checkpoint(BEST_PATH)
            log(f"BEST battery now {overall} (best checkpoint preserved)")
        elif overall < S["best_overall"] - REGRESSION_DROP:
            log(f"REGRESSION flag: battery {overall} is more than "
                f"{REGRESSION_DROP} below best {S['best_overall']} "
                f"(after {tag_label}); best checkpoint untouched at {BEST_PATH}")
            rollback_to_best(tag_label, overall)
        return battery

    # ---- phase / episode loop (exp16 body; state in S for resume) ----
    while S["phase_idx"] < len(PHASES):
        ph = S["phase_idx"]
        pname, kind, n_eps = PHASES[ph]
        if S["ep_idx"] == 0 and not S.get("_phase_started") == ph:
            visits.clear()
            S["recent"], S["door_recent"], S["first12"] = [], [], []
            S["purged"] = 0
            S["phase_pairs"] = []
            S["_phase_started"] = ph
        recent = deque(S["recent"], maxlen=25)
        door_recent = deque(S["door_recent"], maxlen=25)
        first12 = S["first12"]
        phase_pairs = S["phase_pairs"]
        purged = S["purged"]
        for ep in range(S["ep_idx"], n_eps):
            w = World(kind, rng)
            k0 = state_key(w)
            visits[k0] = visits.get(k0, 0) + 1
            pairs, positions, done = [], [w.agent], False
            milestone_idx = None
            while not done:
                a, enc = behavior_action(w, ego_text(w))
                _r, done = w.step(ACTIONS[a])
                pairs.append((enc, a))
                positions.append(w.agent)
                if milestone_idx is None:
                    if (kind == "keydoor" and w.has_key) or (kind == "switch" and w.switch_on):
                        milestone_idx = len(pairs) - 1
                kk = state_key(w)
                visits[kk] = visits.get(kk, 0) + 1
            ok = w.agent == w.goal
            door_used = w.door is not None and any(pos == w.door for pos in positions)
            recent.append(int(ok))
            door_recent.append(int(ok and door_used))
            if ep < 12:
                first12.append(int(ok))
            if ok and door_used:
                main_prs, seg_prs = contrib(pairs, milestone_idx)
                mem_push(main_prs, ph, times=3)
                mem_push(main_prs[-20:], ph, times=1)
                if seg_prs:
                    mem_push(seg_prs, ph, times=1)
                phase_pairs.extend(main_prs)
                admitted_eps[0] += 1
            elif ok and admit(pairs, ph):
                main_prs, seg_prs = contrib(pairs, milestone_idx)
                mem_push(main_prs, ph, times=1)
                if seg_prs:
                    mem_push(seg_prs, ph, times=1)
                phase_pairs.extend(main_prs)
                admitted_eps[0] += 1
            if ep == 11 and ph >= 1 and first12 and sum(first12) / len(first12) < 0.45:
                purged = purge_old(ph)
                if purged:
                    log(f"PURGE-{RUN_TAG} phase={pname} dropped={purged}")
            train_on_memory(ph)
            S["ep_idx"] = ep + 1
            S["recent"], S["door_recent"] = list(recent), list(door_recent)
            S["first12"], S["phase_pairs"], S["purged"] = first12, phase_pairs, purged
            eps_run[0] += 1
            if max_episodes is not None and eps_run[0] >= max_episodes:
                save_checkpoint(STATE_PATH)
                flush_results("stopped_episodes")
                return {"run": RUN_TAG, "status": "stopped_episodes",
                        "phase_idx": S["phase_idx"], "ep_idx": S["ep_idx"]}
            if (ep + 1) % CHECKPOINT_EPS == 0 or ep == n_eps - 1:
                ev = greedy_eval_logged(kind, n=EVAL_N, seed0=EVAL_SEED0,
                                        eval_tag=f"{pname}@ep{ep + 1}")
                ent, mar = probe(kind, 9090 + ph)
                stale = round(sum(1 for m in memory if m[2] < ph) / max(1, len(memory)), 2)
                rec = {"phase": pname, "ep": ep + 1,
                       "train_success_last25": round(sum(recent) / len(recent), 2),
                       "door_success_last25": round(sum(door_recent) / len(door_recent), 2),
                       "greedy_eval": ev["greedy_eval"], "greedy_steps": ev["greedy_steps"],
                       "mile_rate": ev["mile_rate"], "cond_succ": ev["cond_succ"],
                       "counts": ev["counts"], "door_required": ev["door_required"],
                       "entropy": ent, "margin": mar,
                       "memory": len(memory), "protected": len(protected),
                       "memory_stale_frac": stale,
                       "purged_total": purged, "admitted_total": admitted_eps[0]}
                S["history"].append(rec)
                log(f"SBX-{RUN_TAG} " + json.dumps(rec))
                log(f"HEARTBEAT phase={pname} ep={ep + 1}/{n_eps} "
                    f"elapsed={elapsed()}s drive={'ok' if DRIVE_OK else 'absent'}")
                save_checkpoint(STATE_PATH)
                if stop_after_checkpoints is not None and ck[0] >= stop_after_checkpoints:
                    flush_results("stopped_checkpoint")
                    return {"run": RUN_TAG, "status": "stopped_checkpoint",
                            "phase_idx": S["phase_idx"], "ep_idx": S["ep_idx"]}
        protected.extend(phase_pairs[-300:])
        log(f"CORE-{RUN_TAG} phase={pname} protected_now={len(protected)}")
        run_battery(pname)
        run_heldout(pname)
        S["phase_idx"] = ph + 1
        S["ep_idx"] = 0
        S.pop("_phase_started", None)
        save_checkpoint(STATE_PATH)
        flush_results(f"after_{pname}")
        log(f"PHASE-DONE {pname} t={elapsed()}s")

    battery = {b["tag"]: b for b in S["batteries"]}
    log(f"FINAL-{RUN_TAG} batteries=" + json.dumps(
        {t: b["overall"] for t, b in battery.items()}) +
        f" best={S['best_overall']}")
    flush_results("complete")
    out = {"run": RUN_TAG, "history": S["history"], "batteries": S["batteries"],
           "heldout": S["heldout"], "rollbacks": S["rollbacks"],
           "best_overall": S["best_overall"]}
    log("RESULTS_JSON:" + json.dumps(out))
    del model, vhead, opt
    torch.cuda.empty_cache()
    return out

def run_drill():
    # Acceptance criterion 2 -- RECOVERY DRILL. Runs entirely on the drill
    # paths (see DRILL_MODE block): a real run's checkpoints are never touched.
    # Stage 1: fresh run to the first checkpoint (ep 25 of P1). Simulated kill:
    # the invocation returns, every in-memory object is dropped, and stage 2
    # restarts from the checkpoint file only, verifies resume integrity against
    # the pre-kill checkpoint, then trains 3 more episodes to prove progress
    # continues. (On a real first use, also do it once across an actual runtime
    # reclaim: kill the Colab runtime between stages instead -- same files.)
    log("DRILL: recovery drill starting (acceptance criterion 2)")
    for p in [STATE_PATH, BEST_PATH, RESULTS_PATH, EPISODES_PATH]:
        try:
            if os.path.exists(p): os.remove(p)
        except Exception: pass
        if DRIVE_OK:
            try:
                dp = os.path.join(DRIVE_DIR, os.path.basename(p))
                if os.path.exists(dp): os.remove(dp)
            except Exception: pass
    log("DRILL stage 1: fresh run to first checkpoint")
    run_sandbox(stop_after_checkpoints=1)
    pre = torch.load(STATE_PATH, map_location="cpu", weights_only=False)
    expect = {"phase_idx": pre["phase_idx"], "ep_idx": pre["ep_idx"],
              "admitted": pre["admitted"], "mem": len(pre["memory"]),
              "prot": len(pre["protected"]),
              "opt_entries": len(pre["opt_sd"]["state"]),
              "opt_steps": pre.get("stats", {}).get("opt_steps", 0),
              "rng_ok": pre.get("torch_rng") is not None and pre.get("py_rng") is not None}
    del pre
    torch.cuda.empty_cache()
    log("DRILL simulated kill: all in-memory state dropped; "
        "restarting from the checkpoint file only")
    RESUME_REPORT.clear()
    run_sandbox(max_episodes=3)
    post = torch.load(STATE_PATH, map_location="cpu", weights_only=False)
    rep = RESUME_REPORT
    checks = []
    def chk(name, ok, detail):
        checks.append(bool(ok))
        log(f"DRILL CHECK {name}: {'PASS' if ok else 'FAIL'} ({detail})")
    chk("resume_found", bool(rep), f"report keys={sorted(rep.keys())}")
    chk("counters_identical",
        rep.get("phase_idx") == expect["phase_idx"] and rep.get("ep_idx") == expect["ep_idx"]
        and rep.get("admitted") == expect["admitted"],
        f"phase/ep/admitted pre={expect['phase_idx']}/{expect['ep_idx']}/{expect['admitted']} "
        f"post={rep.get('phase_idx')}/{rep.get('ep_idx')}/{rep.get('admitted')}")
    chk("memory_core_identical",
        rep.get("mem") == expect["mem"] and rep.get("prot") == expect["prot"],
        f"mem/prot pre={expect['mem']}/{expect['prot']} post={rep.get('mem')}/{rep.get('prot')}")
    chk("optimizer_state_loaded",
        rep.get("opt_entries") == expect["opt_entries"]
        and rep.get("opt_entries_loaded") == expect["opt_entries"]
        and (expect["opt_steps"] == 0 or expect["opt_entries"] > 0),
        f"opt_steps_pre={expect['opt_steps']} entries pre/loaded="
        f"{expect['opt_entries']}/{rep.get('opt_entries_loaded')}")
    chk("rng_states_present", bool(expect["rng_ok"] and rep.get("rng_ok")),
        f"pre={expect['rng_ok']} post={rep.get('rng_ok')}")
    chk("progress_continuous",
        post["phase_idx"] == expect["phase_idx"] and post["ep_idx"] == expect["ep_idx"] + 3,
        f"ep_idx {expect['ep_idx']} -> {post['ep_idx']} in phase {post['phase_idx']}")
    ok_all = all(checks) and len(checks) == 6
    log(f"DRILL RESULT: {'PASS' if ok_all else 'FAIL'} "
        f"({sum(checks)}/{len(checks)} checks) -- a long run counts as trusted "
        f"only after this drill has passed once")
    return ok_all

if __name__ == "__main__" or True:
    if DRILL_MODE:
        run_drill()
        log("DRILL done -- disconnect and delete the runtime (T4 credit discipline)")
    else:
        log(f"SANDBOX start tag={RUN_TAG} device={DEVICE} drive_ok={DRIVE_OK}")
        run_sandbox()
        log("SANDBOX done -- disconnect and delete the runtime (T4 credit discipline)")
