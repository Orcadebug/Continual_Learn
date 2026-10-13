# ===== EXPERIMENT 16: what should a successful episode contribute? =====
# Controlled 3-arm comparison on the frozen Exp 15 recipe (rounds=5, encoder
# lr 1e-5, protected core, unified eval n=24 seed0=707000). Arms differ ONLY
# in what an admitted successful episode contributes to memory:
#   A = full trajectory (Exp 15 baseline; doubles as its replication)
#   B = post-milestone suffix only (from the milestone-causing action on)
#   C = full trajectory + the post-milestone segment stored once more as
#       its own example (segment gets independent weight; beginning kept)
# Milestone: keydoor -> has_key first true; switch -> switch_on first true;
# empty -> none (all arms identical). Episodes with no milestone contribute
# full pairs in every arm. Admission DECISIONS (door bypass + admit() on the
# full episode length) are identical across arms; only content differs.
# Seeds matched across arms (same rep index -> same seeds). Decomposed eval:
# mile_rate = P(reach milestone), cond_succ = P(goal | milestone reached).
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

def run_lifetime(tag, run_idx, arm):
    torch.manual_seed(6000 + run_idx * 101)
    rng = random.Random(5000 + run_idx * 101)
    model, tok, _c = load_checkpoint("convaiinnovations/laya")
    TOK_HOLDER[0] = tok
    model = model.to(DEVICE)
    d = model.encoder.config.hidden_size
    vhead = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1)).to(DEVICE)
    torch.cuda.empty_cache()
    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    groups = [{"params": [p for n, p in model.named_parameters() if n.startswith("encoder.")], "lr": 1e-5},
              {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")], "lr": 1e-3}]
    opt = torch.optim.AdamW(groups, weight_decay=0.0)

    MEM_CAP = 4000
    memory = deque()
    protected = []
    succ_lens = {0: [], 1: [], 2: [], 3: []}
    visits = {}
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

    def contrib(prs, midx):
        # What this episode contributes, per arm. midx = index of the action
        # that first set the milestone flag, or None.
        if arm == "A" or midx is None:
            return prs, []
        seg = prs[midx:]
        if arm == "B":
            return seg, []
        return prs, seg

    def policy_probs(text):
        enc = encode_state(tok, text)
        with torch.no_grad():
            logits, _v = forward_full(model, vhead, [enc])
        probs = torch.softmax(logits[0], -1)
        probs = torch.nan_to_num(probs, nan=0.25, posinf=0.0, neginf=0.0).clamp(min=0)
        tot = float(probs.sum())
        probs = probs / tot if tot > 0 else torch.full_like(probs, 0.25)
        return probs, enc

    def train_on_memory(cur_ph, rounds=5, bs=32):
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

    def greedy_eval(kind, n=12, seed0=606000):
        succ, steps = 0, []
        mile_hit, mile_succ = 0, 0
        for i in range(n):
            w = World(kind, random.Random(seed0 + i))
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
            steps.append(w.steps)
        return {"greedy_eval": round(succ / n, 3),
                "greedy_steps": round(sum(steps) / n, 1),
                "mile_rate": (round(mile_hit / n, 3) if kind != "empty" else None),
                "cond_succ": (round(mile_succ / mile_hit, 3) if mile_hit else None)}

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

    PHASES = [("P1_empty", "empty", 80), ("P2_keydoor", "keydoor", 100),
              ("P3_switch_rulechange", "switch", 100), ("P4_return_empty", "empty", 40)]
    history = []
    for ph, (pname, kind, n_eps) in enumerate(PHASES):
        visits.clear()
        recent = deque(maxlen=25)
        door_recent = deque(maxlen=25)
        first12 = []
        purged = 0
        phase_pairs = []
        for ep in range(n_eps):
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
                    print(f"PURGE-{tag} phase={pname} dropped={purged}", flush=True)
            train_on_memory(ph)
            if (ep + 1) % 25 == 0 or ep == n_eps - 1:
                ev = greedy_eval(kind, n=24, seed0=707000)
                ent, mar = probe(kind, 9090 + ph)
                stale = round(sum(1 for m in memory if m[2] < ph) / max(1, len(memory)), 2)
                rec = {"phase": pname, "ep": ep + 1,
                       "train_success_last25": round(sum(recent) / len(recent), 2),
                       "door_success_last25": round(sum(door_recent) / len(door_recent), 2),
                       "greedy_eval": ev["greedy_eval"], "greedy_steps": ev["greedy_steps"],
                       "mile_rate": ev["mile_rate"], "cond_succ": ev["cond_succ"],
                       "entropy": ent, "margin": mar,
                       "memory": len(memory), "protected": len(protected),
                       "memory_stale_frac": stale,
                       "purged_total": purged, "admitted_total": admitted_eps[0]}
                history.append(rec)
                print(f"E16-{tag} " + json.dumps(rec), flush=True)
        protected.extend(phase_pairs[-300:])
        print(f"CORE-{tag} phase={pname} protected_now={len(protected)}", flush=True)
    battery = {}
    for kind in ["empty", "keydoor", "switch"]:
        battery[kind] = greedy_eval(kind, n=24, seed0=707000)
    print(f"BATTERY-{tag} " + json.dumps(battery), flush=True)
    history.append({"phase": "BATTERY", "results": battery})
    del model, vhead, opt
    torch.cuda.empty_cache()
    return history

all_runs = []
for rep in range(2):
    for arm in ["A", "B", "C"]:
        tag = f"{arm}-R{rep + 1}"
        print(f"LIFETIME {tag} starting (arm {arm}, rep {rep + 1})", flush=True)
        h = run_lifetime(tag, rep, arm)
        all_runs.append({"run": tag, "arm": arm, "rep": rep + 1, "history": h})
        with open("/content/laya_exp16_results.json", "w") as f:
            json.dump(all_runs, f, indent=2)

def last_greedy(hist, phase_name):
    vals = [r["greedy_eval"] for r in hist if r.get("phase") == phase_name]
    return vals[-1] if vals else None

summary = {}
for arm in ["A", "B", "C"]:
    runs = [r for r in all_runs if r["arm"] == arm]
    if not runs:
        continue
    summary[arm] = {}
    for key, fn in [("P2_end", lambda h: last_greedy(h, "P2_keydoor")),
                    ("P3_end", lambda h: last_greedy(h, "P3_switch_rulechange")),
                    ("P4_end", lambda h: last_greedy(h, "P4_return_empty")),
                    ("battery_empty", lambda h: h[-1]["results"]["empty"]["greedy_eval"]),
                    ("battery_keydoor", lambda h: h[-1]["results"]["keydoor"]["greedy_eval"]),
                    ("battery_switch", lambda h: h[-1]["results"]["switch"]["greedy_eval"])]:
        vals = [fn(r["history"]) for r in runs]
        summary[arm][key] = {"values": vals, "mean": round(sum(vals) / len(vals), 3),
                            "min": min(vals), "max": max(vals)}
    for mk in ["mile_rate", "cond_succ"]:
        for wk in ["keydoor", "switch"]:
            vals = [r["history"][-1]["results"][wk][mk] for r in runs]
            vals = [v for v in vals if v is not None]
            if vals:
                summary[arm][f"battery_{wk}_{mk}"] = {"values": vals,
                                                     "mean": round(sum(vals) / len(vals), 3)}
print("AGGREGATE " + json.dumps(summary), flush=True)
print("RESULTS_JSON:" + json.dumps({"runs": all_runs, "aggregate": summary}), flush=True)
