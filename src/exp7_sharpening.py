# Experiment 6C = production run + Experiment 6B (Sai's Policy Extraction Test).
# PART 1 (production): replicate Exp 6 v2's RL arm (no replay) through P1+P2 with the
#   same seed/hyperparameters, and SAVE the trained policy (v2 saved none - design gap).
# PART 2 (6B): freeze that policy; evaluate it on identical world seed sets under six
#   action-selection strategies, no learning: greedy, sample T=1/0.5/2, greedy with
#   option-order rotation averaging, greedy with a short action history in the state.
#   Diagnostics: success, steps, entropy, top-2 margin, repeat visits, 2-step oscillations.
import json, random, time
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from laya.train import (load_checkpoint, items_from_rows, encode_item, collate_items,
                        train_model, TrainConfig)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN, HEAD_MAX_LEN = 256, 128
SEED = 2026
ACTIONS = ["north", "south", "west", "east"]
DESC = {"north": "move one cell up", "south": "move one cell down",
        "west": "move one cell left", "east": "move one cell right"}
GAMMA, STEP_PEN, BUMP_PEN, MAX_STEPS = 0.95, -0.01, -0.05, 60

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
    def observe(self):
        n = self.size
        grid = []
        for y in range(n):
            row = ""
            for x in range(n):
                c = (x, y)
                if c == self.agent: row += "A"
                elif c == self.goal: row += "G"
                elif self.door and c == self.door: row += ("d" if self.door_open else "D")
                elif self.key and c == self.key and not self.has_key: row += "K"
                elif self.switch and c == self.switch: row += "S"
                elif c in self.walls: row += "#"
                else: row += "."
            grid.append(" ".join(row))
        carry = "key" if self.has_key else "nothing"
        door = "open" if self.door_open else ("locked" if self.door else "none")
        return "Grid world (A=you, G=goal, #=wall, K=key, D=door, S=switch):\n" + "\n".join(grid) + \
               f"\nYou carry: {carry}. Door: {door}."
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

def compute_returns(rewards):
    out, g = [], 0.0
    for r in reversed(rewards):
        g = r + GAMMA * g
        out.append(g)
    return torch.tensor(list(reversed(out)), dtype=torch.float32, device=DEVICE)

def update_on_episode(model, vhead, opt, traj):
    if not traj: return 0.0
    encs = [t[0] for t in traj]
    acts_all = torch.tensor([t[1] for t in traj], device=DEVICE)
    rets_all = compute_returns([t[2] for t in traj])
    T = len(traj); MICRO = 8
    opt.zero_grad(set_to_none=True)
    total = 0.0
    for s0 in range(0, T, MICRO):
        logits, value = forward_full(model, vhead, encs[s0:s0 + MICRO])
        acts, rets = acts_all[s0:s0 + MICRO], rets_all[s0:s0 + MICRO]
        logp_all = F.log_softmax(logits, -1)
        logp = logp_all.gather(1, acts.unsqueeze(1)).squeeze(1)
        adv = (rets - value).detach()
        loss = (-(logp * adv).mean() + 0.5 * ((value - rets) ** 2).mean()
                - 0.01 * (-(logp_all.exp() * logp_all).sum(-1).mean())) * (len(acts) / T)
        loss.backward()
        total += float(loss.detach())
    nn.utils.clip_grad_norm_(list(model.parameters()) + list(vhead.parameters()), 1.0)
    opt.step()
    return total

def train_episode(model, vhead, tok, world, opt):
    traj, done = [], False
    while not done:
        enc = encode_state(tok, world.observe())
        with torch.no_grad():
            logits, _v = forward_full(model, vhead, [enc])
        probs = torch.softmax(logits[0], -1)
        a = int(torch.multinomial(probs, 1))
        r, done = world.step(ACTIONS[a])
        traj.append((enc, a, r))
    update_on_episode(model, vhead, opt, traj)
    return world.agent == world.goal


# ------------------------------------------------------------- Exp 7 machinery
def cell_passable(world, cell, has_key):
    if cell in world.walls: return False
    if world.door and cell == world.door and not (has_key or world.door_open): return False
    return True

def dist_to_goal_map(world):
    """BFS from every (x,y,has_key) state -> distance to goal (small graph, brute force)."""
    n = world.size
    states = [(x, y, hk) for x in range(n) for y in range(n) for hk in (False, True)]
    out = {}
    for s in states:
        if not cell_passable(world, (s[0], s[1]), s[2]) and (s[0], s[1]) != world.agent:
            out[s] = float("inf"); continue
        seen = {s: 0}; dq = deque([s]); d = None
        while dq:
            cur = dq.popleft()
            if (cur[0], cur[1]) == world.goal: d = seen[cur]; break
            for a in ACTIONS:
                dx, dy = {"north": (0,-1), "south": (0,1), "west": (-1,0), "east": (1,0)}[a]
                nc = (cur[0]+dx, cur[1]+dy)
                if not (0 <= nc[0] < n and 0 <= nc[1] < n): continue
                if not cell_passable(world, nc, cur[2]): continue
                nhk = cur[2] or (world.key is not None and nc == world.key)
                ns = (nc[0], nc[1], nhk)
                if ns not in seen:
                    seen[ns] = seen[cur] + 1; dq.append(ns)
        out[s] = d if d is not None else float("inf")
    return out

def reducing_actions(world, dmap):
    cur = (world.agent[0], world.agent[1], world.has_key)
    d0 = dmap.get(cur, float("inf"))
    if d0 == float("inf") or d0 == 0: return []
    res = []
    for a in ACTIONS:
        dx, dy = {"north": (0,-1), "south": (0,1), "west": (-1,0), "east": (1,0)}[a]
        nc = (world.agent[0]+dx, world.agent[1]+dy)
        if not (0 <= nc[0] < world.size and 0 <= nc[1] < world.size): continue
        if not cell_passable(world, nc, world.has_key): continue
        nhk = world.has_key or (world.key is not None and nc == world.key)
        if dmap.get((nc[0], nc[1], nhk), float("inf")) == d0 - 1: res.append(a)
    return res

def ego_text(world):
    ax, ay = world.agent; gx, gy = world.goal
    def rel(tx, ty, name):
        dx, dy = tx-ax, ty-ay
        parts = []
        if dx < 0: parts.append(f"{-dx} west")
        if dx > 0: parts.append(f"{dx} east")
        if dy < 0: parts.append(f"{-dy} north")
        if dy > 0: parts.append(f"{dy} south")
        return f"{name}: " + (", ".join(parts) if parts else "here") + "."
    blocked = []
    for a in ACTIONS:
        dx, dy = {"north": (0,-1), "south": (0,1), "west": (-1,0), "east": (1,0)}[a]
        nc = (ax+dx, ay+dy)
        if not (0 <= nc[0] < world.size and 0 <= nc[1] < world.size) or not cell_passable(world, nc, world.has_key):
            blocked.append(a)
    lines = ["You are in a grid room. Reach the goal G.",
             rel(gx, gy, "Goal"),
             "Blocked directions: " + (", ".join(blocked) if blocked else "none") + ".",
             "You carry: " + ("the key." if world.has_key else "nothing.")]
    if world.key is not None and not world.has_key:
        lines.append(rel(world.key[0], world.key[1], "Key"))
    if world.door:
        lines.append("The door is " + ("open." if world.door_open else "locked. The key opens it."))
        lines.append(rel(world.door[0], world.door[1], "Door"))
    return "\n".join(lines)

def gen_examples(kind, fmt, n, rng):
    ex = []
    while len(ex) < n:
        w = World(kind, rng)
        dmap = dist_to_goal_map(w)
        reds = reducing_actions(w, dmap)
        if not reds: continue
        text = w.observe() if fmt == "ascii" else ego_text(w)
        ex.append((text, reds[0], set(reds)))
    return ex

def supervised_part(part, fmt):
    print(f"===== PART {part} supervised fmt={fmt} =====", flush=True)
    model, tok, _c = load_checkpoint("convaiinnovations/laya")
    TOK_HOLDER[0] = tok
    model = model.to(DEVICE)
    rng = random.Random(SEED + (1 if fmt == "ascii" else 2))
    train_ex, test_ex = [], []
    for kind, ntr, nte in (("empty", 2400, 600), ("keydoor", 2400, 600)):
        train_ex += gen_examples(kind, fmt, ntr, rng)
        test_ex += gen_examples(kind, fmt, nte, rng)
    def to_items(examples):
        rows = [{"state": t, "questions": question_for(ACTIONS), "expected": {"act": lab}}
                for (t, lab, _s) in examples]
        items, skipped = items_from_rows(tok, rows, MAX_LEN, HEAD_MAX_LEN)
        assert items, skipped
        return items
    train_items = to_items(train_ex)
    cfg = TrainConfig(loss="soft-ce", epochs=3, micro_batch=16, grad_accum=1,
                      freeze_encoder=False, shuffle_options=("choice",),
                      seed=SEED, log_every=100000)
    train_model(model, tok, train_items, cfg)
    # accuracy: argmax in reducing set?
    correct = 0
    B = 32
    for i in range(0, len(test_ex), B):
        chunk = test_ex[i:i+B]
        encs = [encode_state(tok, t) for (t, _l, _s) in chunk]
        # manual forward (no value head needed here)
        batch = collate_items([[e for e in encs]], tok.pad_token_id)
        ids = batch["input_ids"].to(DEVICE); att = batch["attention_mask"].to(DEVICE)
        mpos = batch["marker_pos"].to(DEVICE); mmask = batch["marker_mask"].to(DEVICE)
        qt = batch["qtype"].to(DEVICE)
        with torch.no_grad():
            h = model.encoder(input_ids=ids, attention_mask=att).last_hidden_state
            h = h + model.type_emb(qt)[:, None, :]
            if model.head is not None:
                pad = ~att.bool()
                for layer in model.head.layers:
                    h = layer(h, src_key_padding_mask=pad)
            idx = mpos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
            lg = model.scorer(torch.gather(h, 1, idx)).squeeze(-1).float()
            lg = lg.masked_fill(~mmask, -1e4)
        for j, (_t, _l, redset) in enumerate(chunk):
            if ACTIONS[int(torch.argmax(lg[j]))] in redset: correct += 1
    acc = correct / len(test_ex)
    # greedy env rollout in the same format
    succ, steps = 0, []
    for i in range(40):
        w = World("empty" if i % 2 == 0 else "keydoor", random.Random(31337 + i))
        done = False
        while not done:
            text = w.observe() if fmt == "ascii" else ego_text(w)
            enc = encode_state(tok, text)
            batch = collate_items([[enc]], tok.pad_token_id)
            ids = batch["input_ids"].to(DEVICE); att = batch["attention_mask"].to(DEVICE)
            mpos = batch["marker_pos"].to(DEVICE); mmask = batch["marker_mask"].to(DEVICE)
            qt = batch["qtype"].to(DEVICE)
            with torch.no_grad():
                h = model.encoder(input_ids=ids, attention_mask=att).last_hidden_state
                h = h + model.type_emb(qt)[:, None, :]
                if model.head is not None:
                    pad = ~att.bool()
                    for layer in model.head.layers:
                        h = layer(h, src_key_padding_mask=pad)
                idx = mpos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
                lg = model.scorer(torch.gather(h, 1, idx)).squeeze(-1).float()
                lg = lg.masked_fill(~mmask, -1e4)
            _r, done = w.step(ACTIONS[int(torch.argmax(lg[0]))])
        succ += int(w.agent == w.goal); steps.append(w.steps)
    rec = {"part": part, "fmt": fmt, "heldout_accuracy": round(acc, 3),
           "greedy_env_success": round(succ / 40, 3), "greedy_env_steps": round(sum(steps) / 40, 1)}
    print("SUP " + json.dumps(rec), flush=True)
    del model
    torch.cuda.empty_cache()
    return rec

def shaped_rl_part():
    print("===== PART C shaped RL (empty worlds) =====", flush=True)
    model, tok, _c = load_checkpoint("convaiinnovations/laya")
    TOK_HOLDER[0] = tok
    model = model.to(DEVICE)
    d = model.encoder.config.hidden_size
    vhead = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1)).to(DEVICE)
    torch.cuda.empty_cache()
    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    groups = [{"params": [p for n, p in model.named_parameters() if n.startswith("encoder.")], "lr": 1e-5},
              {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")], "lr": 5e-4},
              {"params": list(vhead.parameters()), "lr": 5e-4}]
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    probe_states = []
    prng = random.Random(4242)
    while len(probe_states) < 64:
        w = World("empty", prng); probe_states.append(w.observe())
    def probe():
        ents, mars = [], []
        for i in range(0, 64, 16):
            encs = [encode_state(tok, t) for t in probe_states[i:i+16]]
            with torch.no_grad():
                logits, _v = forward_full(model, vhead, encs)
            p = torch.softmax(logits, -1)
            ents += [float(-(p * p.clamp(min=1e-12).log()).sum(-1).mean())]
            top2 = torch.topk(p, 2, dim=-1).values
            mars += [float((top2[:, 0] - top2[:, 1]).mean())]
        return round(sum(ents) / len(ents), 3), round(sum(mars) / len(mars), 3)
    e0, m0 = probe()
    print(f"SHARP ep 0 entropy={e0} margin={m0}", flush=True)
    rng = random.Random(SEED)
    recent = deque(maxlen=25)
    for ep in range(80):
        w = World("empty", rng)
        traj, done = [], False
        while not done:
            enc = encode_state(tok, w.observe())
            with torch.no_grad():
                logits, _v = forward_full(model, vhead, [enc])
            a = int(torch.multinomial(torch.softmax(logits[0], -1), 1))
            d_prev = abs(w.agent[0] - w.goal[0]) + abs(w.agent[1] - w.goal[1])
            r, done = w.step(ACTIONS[a])
            d_new = abs(w.agent[0] - w.goal[0]) + abs(w.agent[1] - w.goal[1])
            r += 0.1 * (d_prev - d_new)     # diagnostic shaping only
            traj.append((enc, a, r))
        update_on_episode(model, vhead, opt, traj)
        recent.append(int(w.agent == w.goal))
        if (ep + 1) % 20 == 0:
            e, m = probe()
            print(f"SHARP ep {ep+1} entropy={e} margin={m} train_success_last25={sum(recent)/len(recent):.2f}", flush=True)
    out = {}
    for mode in ("greedy", "sample"):
        succ, steps = 0, []
        for i in range(24):
            w = World("empty", random.Random(777000 + i)); done = False
            while not done:
                enc = encode_state(tok, w.observe())
                with torch.no_grad():
                    logits, _v = forward_full(model, vhead, [enc])
                p = torch.softmax(logits[0], -1)
                a = int(torch.argmax(p)) if mode == "greedy" else int(torch.multinomial(p, 1))
                _r, done = w.step(ACTIONS[a])
            succ += int(w.agent == w.goal); steps.append(w.steps)
        out[mode] = {"success": round(succ / 24, 3), "avg_steps": round(sum(steps) / 24, 1)}
    print("SHAPED_EVAL " + json.dumps(out), flush=True)
    return out

def main():
    print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "", flush=True)
    results = {}
    results["A_ascii"] = supervised_part("A", "ascii")
    results["B_ego"] = supervised_part("B", "ego")
    results["C_shaped"] = shaped_rl_part()
    print("\n===== SUMMARY =====", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    print("RESULTS_JSON:" + json.dumps(results), flush=True)
    with open("/content/laya_exp7_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
