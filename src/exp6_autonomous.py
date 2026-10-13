# Experiment 6 v1: The Autonomous Mastery Test (scoped first version).
# Laya is dropped into an unfamiliar TEXT gridworld with ONE objective (reach the goal,
# few steps). No labeled actions, no task IDs, no warnings. It learns only from reward
# via an actor-critic path built on its own option scorer (policy) + a small value head.
# One continuous lifetime per arm: P1 empty rooms -> P2 key+locked door -> P3 hidden rule
# change (door now needs a SWITCH; the key is a decoy) -> P4 P1 worlds return (retention).
# Arms: FROZEN (no learning, baseline), RL (online actor-critic only),
#       RL+REPLAY (online + trajectory replay from a persistent FIFO experience memory).
import copy, json, random, time
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from laya.train import (load_checkpoint, items_from_rows, encode_item, collate_items)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN, HEAD_MAX_LEN = 256, 128
SEED = 2026
ACTIONS = ["north", "south", "west", "east"]
QUESTION = {"act": {"type": "choice",
                    "instructions": "Choose the agent's next action to reach the goal G in as few steps as possible.",
                    "criteria": {"north": "move one cell up", "south": "move one cell down",
                                 "west": "move one cell left", "east": "move one cell right"}}}
GAMMA, STEP_PEN, BUMP_PEN, MAX_STEPS = 0.95, -0.01, -0.05, 50

# ------------------------------------------------------------------ environment
class World:
    def __init__(self, kind, rng):
        self.kind, self.rng = kind, rng
        self.size = 6
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
                        walls.add((3, y))
                self.door = (3, door_y)
            else:
                self.door = None
                for _ in range(3):
                    walls.add((rng.randint(1, n - 2), rng.randint(1, n - 2)))
            free = [(x, y) for x in range(n) for y in range(n) if (x, y) not in walls and (x, y) != self.door]
            self.agent = rng.choice(free)
            rest = [c for c in free if c != self.agent]
            self.goal = rng.choice(rest); rest.remove(self.goal)
            # key / switch always spawn on the AGENT's side of the door wall, so the
            # episode is always solvable in principle (P2/P3 would otherwise waste
            # episodes on impossible layouts)
            same_side = [c for c in rest if (c[0] < 3) == (self.agent[0] < 3)] or rest
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

# ------------------------------------------------------------------ laya policy
def encode_state(tok, text):
    rows = [{"state": text, "questions": QUESTION, "expected": {"act": "north"}}]
    items, skipped = items_from_rows(tok, rows, MAX_LEN, HEAD_MAX_LEN)
    assert items, skipped
    return encode_item(tok, items[0], MAX_LEN, HEAD_MAX_LEN)

def forward_full(model, vhead, encs):
    batch = collate_items([[e for e in encs]], TOK.pad_token_id)
    ids = batch["input_ids"].to(DEVICE); att = batch["attention_mask"].to(DEVICE)
    mpos = batch["marker_pos"].to(DEVICE); mmask = batch["marker_mask"].to(DEVICE)
    qt = batch["qtype"].to(DEVICE)
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

def run_episode(model, vhead, tok, world, greedy=False, train_ctx=None):
    """Roll one episode. If train_ctx (optimizer) given, collect trajectory for update."""
    traj = []
    total_r, done = 0.0, False
    while not done:
        text = world.observe()
        enc = encode_state(tok, text)
        with torch.no_grad():
            logits, _v = forward_full(model, vhead, [enc])
        probs = torch.softmax(logits[0], -1)
        a = int(torch.argmax(probs)) if greedy else int(torch.multinomial(probs, 1))
        r, done = world.step(ACTIONS[a])
        total_r += r
        if train_ctx is not None:
            traj.append((enc, a, r))
    success = world.agent == world.goal
    return success, world.steps, traj

def compute_returns(rewards):
    out, g = [], 0.0
    for r in reversed(rewards):
        g = r + GAMMA * g
        out.append(g)
    return torch.tensor(list(reversed(out)), dtype=torch.float32, device=DEVICE)

def update_on_episode(model, vhead, opt, traj):
    if not traj: return 0.0
    encs = [t[0] for t in traj]
    acts = torch.tensor([t[1] for t in traj], device=DEVICE)
    rets = compute_returns([t[2] for t in traj])
    logits, value = forward_full(model, vhead, encs)
    logp_all = F.log_softmax(logits, -1)
    logp = logp_all.gather(1, acts.unsqueeze(1)).squeeze(1)
    adv = (rets - value).detach()
    loss_pg = -(logp * adv).mean()
    loss_v = ((value - rets) ** 2).mean()
    probs = logp_all.exp()
    entropy = -(probs * logp_all).sum(-1).mean()
    loss = loss_pg + 0.5 * loss_v - 0.01 * entropy
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(list(model.parameters()) + list(vhead.parameters()), 1.0)
    opt.step()
    return float(loss.detach())

PHASES = [("P1_empty", "empty", 100), ("P2_keydoor", "keydoor", 120),
          ("P3_switch_rulechange", "switch", 100), ("P4_return_empty", "empty", 60)]

def run_arm(arm):
    model, tok, _c = load_checkpoint("convaiinnovations/laya")
    TOK_HOLDER[0] = tok
    model = model.to(DEVICE)
    d = model.encoder.config.hidden_size
    vhead = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1)).to(DEVICE)
    if arm == "FROZEN":
        rng = random.Random(SEED + 99)
        res = {}
        for kind in ("empty", "keydoor"):
            succ, steps = 0, []
            for _ in range(30):
                w = World(kind, rng)
                s, st, _ = run_episode(model, vhead, tok, w, greedy=False)
                succ += int(s); steps.append(st)
            res[kind] = {"success_rate": succ / 30, "avg_steps": sum(steps) / 30}
            print(f"FROZEN {kind}: {res[kind]}", flush=True)
        return res
    groups = [{"params": [p for n, p in model.named_parameters() if n.startswith("encoder.")], "lr": 1e-5},
              {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")], "lr": 5e-4},
              {"params": list(vhead.parameters()), "lr": 5e-4}]
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    memory = deque(maxlen=150)   # FIFO experience memory (trajectories); old eras retire naturally
    rng = random.Random(SEED)
    history, ep_global, recent = [], 0, deque(maxlen=25)
    for pname, kind, n_eps in PHASES:
        for ep in range(n_eps):
            w = World(kind, rng)
            s, st, traj = run_episode(model, vhead, tok, w, greedy=False, train_ctx=True)
            recent.append(int(s)); ep_global += 1
            update_on_episode(model, vhead, opt, traj)
            if arm == "RL+REPLAY" and len(memory) >= 4:
                for old in rng.sample(list(memory), 3):
                    update_on_episode(model, vhead, opt, old)
            memory.append(traj)
            if (ep + 1) % 25 == 0 or ep == n_eps - 1:
                ev_s, ev_st = 0, []
                erng = random.Random(SEED + 5000 + ep_global)
                for _ in range(12):
                    ew = World(kind, erng)
                    es, est, _ = run_episode(model, vhead, tok, ew, greedy=True)
                    ev_s += int(es); ev_st.append(est)
                rec = {"arm": arm, "phase": pname, "episodes_in_phase": ep + 1,
                       "train_success_last25": round(sum(recent) / max(1, len(recent)), 3),
                       "eval_success": round(ev_s / 12, 3), "eval_avg_steps": round(sum(ev_st) / 12, 1)}
                history.append(rec)
                print("EVAL " + json.dumps(rec), flush=True)
    return {"evals": history}

TOK_HOLDER = [None]
def _tok():
    return TOK_HOLDER[0]
TOK = None
class _TokProxy:
    @property
    def pad_token_id(self):
        return TOK_HOLDER[0].pad_token_id
TOK = _TokProxy()

def main():
    print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "", flush=True)
    results = {}
    for arm in ["FROZEN", "RL", "RL+REPLAY"]:
        print(f"=== ARM {arm} ===", flush=True)
        t0 = time.time()
        results[arm] = run_arm(arm)
        print(f"arm {arm} done in {time.time()-t0:.0f}s", flush=True)
        with open("/content/laya_exp6_partial.json", "w") as f:
            json.dump(results, f, indent=2)
    print("\n===== SUMMARY =====", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    print("RESULTS_JSON:" + json.dumps(results), flush=True)
    with open("/content/laya_exp6_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
