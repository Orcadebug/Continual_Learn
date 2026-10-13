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
from laya.train import (load_checkpoint, items_from_rows, encode_item, collate_items)

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

# ------------------------------------------------------------- PART 2: 6B eval
ROTATIONS = [ACTIONS[r:] + ACTIONS[:r] for r in range(4)]

def action_probs(model, vhead, tok, text, strategy, hist_text=""):
    """Return (probs over ACTIONS in canonical order, entropy, top2 margin)."""
    if strategy == "greedy_rotavg":
        encs = [encode_state(tok, text, order) for order in ROTATIONS]
        with torch.no_grad():
            logits, _v = forward_full(model, vhead, encs)
        acc = torch.zeros(4, device=DEVICE)
        for r, order in enumerate(ROTATIONS):
            p = torch.softmax(logits[r], -1)
            for slot, act in enumerate(order):
                acc[ACTIONS.index(act)] += p[slot]
        probs = (acc / 4).to(DEVICE)
    else:
        t = text + (hist_text if strategy == "greedy_history" else "")
        enc = encode_state(tok, t)
        with torch.no_grad():
            logits, _v = forward_full(model, vhead, [enc])
        probs = torch.softmax(logits[0], -1)
    top2 = torch.topk(probs, 2).values
    entropy = float(-(probs * probs.clamp(min=1e-12).log()).sum())
    return probs, entropy, float(top2[0] - top2[1])

def eval_strategy(model, vhead, tok, strategy, kind, n_eps=40):
    succ, steps_l, ents, margins, repeats, osc = 0, [], [], [], [], []
    for i in range(n_eps):
        world = World(kind, random.Random(90000 + i))     # identical worlds per strategy
        act_rng = random.Random(777 + i)                  # identical sampling seeds
        visited = {world.agent: 1}
        positions = [world.agent]
        hist = deque(maxlen=4)
        rep_steps, ep_osc = 0, 0
        done = False
        while not done:
            hist_text = "\nYour recent moves (oldest first): " + \
                        (", ".join(hist) if hist else "none yet") + "."
            temp = {"sample_t1": 1.0, "sample_t05": 0.5, "sample_t2": 2.0}.get(strategy)
            if temp:
                enc = encode_state(tok, world.observe())
                with torch.no_grad():
                    logits, _v = forward_full(model, vhead, [enc])
                probs = torch.softmax(logits[0] / temp, -1)
                top2 = torch.topk(probs, 2).values
                ents.append(float(-(probs * probs.clamp(min=1e-12).log()).sum()))
                margins.append(float(top2[0] - top2[1]))
                gen = torch.Generator().manual_seed(act_rng.randrange(2**31))
                a = int(torch.multinomial(probs.cpu(), 1, generator=gen))
            else:
                probs, ent, mar = action_probs(model, vhead, tok, world.observe(), strategy, hist_text)
                ents.append(ent); margins.append(mar)
                a = int(torch.argmax(probs))
            before, had_key = world.agent, world.has_key
            r, done = world.step(ACTIONS[a])
            if world.agent == before: outcome = "blocked"
            elif world.agent == world.goal: outcome = "reached the goal"
            elif world.has_key and not had_key: outcome = "picked up the key"
            else: outcome = "moved"
            hist.append(f"{ACTIONS[a]} ({outcome})")
            if visited.get(world.agent, 0) > 0 and world.agent != before: rep_steps += 1
            visited[world.agent] = visited.get(world.agent, 0) + 1
            positions.append(world.agent)
            if len(positions) >= 3 and positions[-1] == positions[-3] and positions[-1] != positions[-2]:
                ep_osc += 1
        succ += int(world.agent == world.goal)
        steps_l.append(world.steps)
        repeats.append(rep_steps / max(1, world.steps))
        osc.append(ep_osc)
    return {"strategy": strategy, "world": kind, "episodes": n_eps,
            "success_rate": round(succ / n_eps, 3),
            "avg_steps": round(sum(steps_l) / n_eps, 1),
            "mean_entropy": round(sum(ents) / max(1, len(ents)), 3),
            "mean_top2_margin": round(sum(margins) / max(1, len(margins)), 3),
            "repeat_visit_rate": round(sum(repeats) / n_eps, 3),
            "oscillations_per_episode": round(sum(osc) / n_eps, 2)}

def main():
    print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "", flush=True)
    model, tok, _c = load_checkpoint("convaiinnovations/laya")
    TOK_HOLDER[0] = tok
    model = model.to(DEVICE)
    d = model.encoder.config.hidden_size
    vhead = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1)).to(DEVICE)
    ckpt = torch.load("/content/exp6c_policy.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model"]); vhead.load_state_dict(ckpt["vhead"])
    print("loaded saved policy from /content/exp6c_policy.pt", flush=True)
    model.eval()
    strategies = ["greedy", "sample_t1", "sample_t05", "sample_t2", "greedy_rotavg", "greedy_history"]
    results = []
    for kind in ("empty", "keydoor"):
        for strat in strategies:
            rec = eval_strategy(model, vhead, tok, strat, kind)
            results.append(rec)
            print("EXTRACT " + json.dumps(rec), flush=True)
            with open("/content/laya_exp6c_partial.json", "w") as f:
                json.dump(results, f, indent=2)
    print("\n===== SUMMARY =====", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    print("RESULTS_JSON:" + json.dumps(results), flush=True)
    with open("/content/laya_exp6c_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
