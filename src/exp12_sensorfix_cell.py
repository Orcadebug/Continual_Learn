# ===== EXPERIMENT 12: sensor repair + calibration fixes =====
# Runs after the bootstrap cell (exp7b machinery).
# Vs Exp 11: (1) ego_text overridden — the Exp 8-11 observation never mentioned
# the switch (invisible mechanic) and claimed "The key opens it" in switch
# worlds where it does not; this version adds a relative Switch line and honest
# door text (keydoor text unchanged). (2) Admission cap from the 25th percentile
# of past success lengths instead of the median. (3) Purge 70% -> 50%.
# (4) P3 lengthened to 140 episodes. (5) admitted-episode count logged.
import json, random, statistics, time, copy
from collections import deque
import torch
import torch.nn.functional as F

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

model, tok, _c = load_checkpoint("convaiinnovations/laya")
TOK_HOLDER[0] = tok
model = model.to(DEVICE)
d = model.encoder.config.hidden_size
vhead = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1)).to(DEVICE)
torch.cuda.empty_cache()
if hasattr(model.encoder, "gradient_checkpointing_enable"):
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
groups = [{"params": [p for n, p in model.named_parameters() if n.startswith("encoder.")], "lr": 2e-5},
          {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")], "lr": 1e-3}]
opt = torch.optim.AdamW(groups, weight_decay=0.0)

MEM_CAP = 6000
memory = deque()          # (enc, action_idx, phase_idx)
succ_lens = {0: [], 1: [], 2: [], 3: []}
visits = {}               # reset at each phase start
admitted_eps = [0]

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

def policy_probs(text):
    enc = encode_state(tok, text)
    with torch.no_grad():
        logits, _v = forward_full(model, vhead, [enc])
    probs = torch.softmax(logits[0], -1)
    probs = torch.nan_to_num(probs, nan=0.25, posinf=0.0, neginf=0.0).clamp(min=0)
    tot = float(probs.sum())
    probs = probs / tot if tot > 0 else torch.full_like(probs, 0.25)
    return probs, enc

def train_on_memory(rng, cur_ph, rounds=2, bs=32):
    if len(memory) < 64:
        return
    mem = list(memory)
    weights = [3.0 if m[2] == cur_ph else 1.0 for m in mem]
    for _ in range(rounds):
        batch = rng.choices(mem, weights=weights, k=bs)
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

def purge_old(rng, cur_ph, frac=0.5):
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

def greedy_eval(kind, n=12, seed0=606000):
    succ, steps = 0, []
    for i in range(n):
        w = World(kind, random.Random(seed0 + i))
        done = False
        while not done:
            p, _e = policy_probs(ego_text(w))
            _r, done = w.step(ACTIONS[int(torch.argmax(p))])
        succ += int(w.agent == w.goal)
        steps.append(w.steps)
    return round(succ / n, 3), round(sum(steps) / n, 1)

def state_key(w):
    return (w.agent, hash(ego_text(w)))

def behavior_action(w, text, rng):
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

PHASES12 = [("P1_empty", "empty", 80), ("P2_keydoor", "keydoor", 100),
            ("P3_switch_rulechange", "switch", 140), ("P4_return_empty", "empty", 40)]
rng = random.Random(SEED)
history = []
for ph, (pname, kind, n_eps) in enumerate(PHASES12):
    visits.clear()
    recent = deque(maxlen=25)
    door_recent = deque(maxlen=25)
    first12 = []
    purged = 0
    for ep in range(n_eps):
        w = World(kind, rng)
        k0 = state_key(w)
        visits[k0] = visits.get(k0, 0) + 1
        pairs, positions, done = [], [w.agent], False
        while not done:
            a, enc = behavior_action(w, ego_text(w), rng)
            _r, done = w.step(ACTIONS[a])
            pairs.append((enc, a))
            positions.append(w.agent)
            kk = state_key(w)
            visits[kk] = visits.get(kk, 0) + 1
        ok = w.agent == w.goal
        door_used = w.door is not None and any(pos == w.door for pos in positions)
        recent.append(int(ok))
        door_recent.append(int(ok and door_used))
        if ep < 12:
            first12.append(int(ok))
        if ok and door_used:
            mem_push(pairs, ph, times=3)
            mem_push(pairs[-20:], ph, times=1)
            admitted_eps[0] += 1
        elif ok and admit(pairs, ph):
            mem_push(pairs, ph, times=1)
            admitted_eps[0] += 1
        if ep == 11 and ph >= 1 and first12 and sum(first12) / len(first12) < 0.45:
            purged = purge_old(rng, ph)
            if purged:
                print(f"PURGE phase={pname} dropped={purged}", flush=True)
        train_on_memory(rng, ph)
        if (ep + 1) % 25 == 0 or ep == n_eps - 1:
            gs, gst = greedy_eval(kind)
            ent, mar = probe(kind, 9090 + ph)
            stale = round(sum(1 for m in memory if m[2] < ph) / max(1, len(memory)), 2)
            rec = {"phase": pname, "ep": ep + 1,
                   "train_success_last25": round(sum(recent) / len(recent), 2),
                   "door_success_last25": round(sum(door_recent) / len(door_recent), 2),
                   "greedy_eval": gs, "greedy_steps": gst, "entropy": ent, "margin": mar,
                   "memory": len(memory), "memory_stale_frac": stale,
                   "purged_total": purged, "admitted_total": admitted_eps[0]}
            history.append(rec)
            print("E12 " + json.dumps(rec), flush=True)
            with open("/content/laya_exp12_results.json", "w") as f:
                json.dump(history, f, indent=2)
torch.save({"model": model.state_dict(), "vhead": vhead.state_dict()}, "/content/exp12_policy.pt")
print("policy saved", flush=True)
print("RESULTS_JSON:" + json.dumps(history), flush=True)
