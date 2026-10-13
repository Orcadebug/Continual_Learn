# ===== EXPERIMENT 11: exploration + consolidation fixes =====
# Runs after the bootstrap cell (exp7b machinery).
# Vs Exp 10: (1) visit counts reset each phase, so new-world mechanics stay
# novel; (2) door-using successes BYPASS the efficiency admission filter and are
# stored x3 plus one extra copy of their final 20 pairs (consolidation weight on
# rare mechanic trajectories); (3) NaN hardening on policy probs and training
# loss; (4) partial results flushed to file at every checkpoint.
import json, random, statistics, time, copy
from collections import deque
import torch
import torch.nn.functional as F

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

def mem_push(pairs, ph, times=1):
    for _ in range(times):
        for e, a in pairs:
            memory.append((e, a, ph))
    while len(memory) > MEM_CAP:
        memory.popleft()

def admit(pairs, ph):
    L = len(pairs)
    succ_lens[ph].append(L)
    if len(succ_lens[ph]) <= 3:
        return True
    med = statistics.median(succ_lens[ph][:-1]) if len(succ_lens[ph]) > 1 else L
    return L <= max(14, 1.6 * med)

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

def purge_old(rng, cur_ph, frac=0.7):
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

PHASES11 = [("P1_empty", "empty", 80), ("P2_keydoor", "keydoor", 100),
            ("P3_switch_rulechange", "switch", 80), ("P4_return_empty", "empty", 40)]
rng = random.Random(SEED)
history = []
for ph, (pname, kind, n_eps) in enumerate(PHASES11):
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
        elif ok and admit(pairs, ph):
            mem_push(pairs, ph, times=1)
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
                   "memory": len(memory), "memory_stale_frac": stale, "purged_total": purged}
            history.append(rec)
            print("E11 " + json.dumps(rec), flush=True)
            with open("/content/laya_exp11_results.json", "w") as f:
                json.dump(history, f, indent=2)
torch.save({"model": model.state_dict(), "vhead": vhead.state_dict()}, "/content/exp11_policy.pt")
print("policy saved", flush=True)
print("RESULTS_JSON:" + json.dumps(history), flush=True)
