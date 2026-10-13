# Experiment 6c: The Coexistence Probe (V3 of the Autonomous Mastery Test).
# Built 2026-10-09 (evening) BEFORE the Exp 6 v2 replay arm finished, to close the
# gap Sai identified at 19:47 EDT and the v2 log already records:
#   v2 never re-probes the switch world after P4, and v2 saves no policies,
#   so "competent at BOTH the new and the still-valid old, simultaneously"
#   is not directly measured there and cannot be probed post-hoc.
# Sai's operative criterion (not peak success): become good, STAY good,
# adapt to an unwarned change, then remain competent at new + old together.
#
# What this script changes vs exp6_autonomous_b.py (everything else identical
# for comparability: same world, phases, rewards, arms, seeds, hyperparams):
#   1. SAVES each trainable arm's policy (model + value head) after every phase
#      to /content/laya_exp6c_<arm>_<phase>.pt  (fp32 state_dicts on CPU).
#   2. Adds a NO-TRAINING coexistence probe after P2, P3 and P4: greedy eval,
#      20 fresh layouts per era, eras INTERLEAVED round-robin (empty, keydoor,
#      switch, empty, ...) so no era is privileged by probe order.
#      Probe after P2 covers empty+keydoor (switch not yet encountered);
#      probes after P3/P4 cover all three eras.
#   3. Reports, per probe: per-era success + avg steps, mean success,
#      WORST-ERA success (the headline number), and max-min era gap.
#   4. Interim per-phase evals stay at n=12 (v2-comparable); probes use n=20/era
#      to reduce the noise that made v2's n=12 evals swing (e.g. 0.75 -> 0.0).
#
# Pre-registered interpretation (written before any 6c run):
#   - Coexistence is a property of the FINAL (post-P4) probe, not any single eval.
#   - An arm "coexists" iff final-probe worst-era success >= 0.50 AND every era
#     beats its RANDOM baseline by >= 0.20 (baselines measured in-script).
#     These are v3 working bars, NOT Sai's program targets (>=95% held-out /
#     >=90% retention), which remain the long-horizon standard in the log.
#   - Comparative claim that matters: RL+REPLAY final worst-era > RL final
#     worst-era. If replay does not raise the WORST era, Exp 5's supervised
#     replay result has NOT transferred to trajectory replay in RL, however
#     good either arm's best era or peak eval looks.
#   - A probe era at/below its random baseline = that era's competence is gone,
#     regardless of the other eras.
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
GAMMA, STEP_PEN, BUMP_PEN, MAX_STEPS = 0.95, -0.01, -0.05, 60

# ------------------------------------------------------------------ environment
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
            # key / switch always spawn on the AGENT's side of the door wall, so the
            # episode is always solvable in principle (P2/P3 would otherwise waste
            # episodes on impossible layouts)
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

def save_policy(model, vhead, arm, phase_name):
    """Persist the trained policy so eras can be re-probed later (v2 gap)."""
    path = f"/content/laya_exp6c_{arm}_{phase_name}.pt"
    try:
        torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "vhead": {k: v.detach().cpu() for k, v in vhead.state_dict().items()},
                    "arm": arm, "phase": phase_name}, path)
        print(f"SAVED {path}", flush=True)
    except Exception as e:
        print(f"SAVE FAILED {path}: {e}", flush=True)
    return path

def probe_all_eras(model, vhead, tok, eras, seed, n_per_era=20):
    """No-training, greedy, interleaved probe across eras. Returns per-era
    success/steps plus mean, worst-era (headline) and era gap."""
    rngs = {era: random.Random(seed + 1000 + i) for i, era in enumerate(eras)}
    counts = {era: [0, 0.0, 0] for era in eras}  # era -> [successes, steps_sum, n]
    for _round in range(n_per_era):
        for era in eras:
            w = World(era, rngs[era])
            s, st, _ = run_episode(model, vhead, tok, w, greedy=True)
            counts[era][0] += int(s); counts[era][1] += st; counts[era][2] += 1
    per_era = {era: {"success_rate": round(c[0] / c[2], 3),
                     "avg_steps": round(c[1] / c[2], 1), "n": c[2]}
               for era, c in counts.items()}
    succ = [v["success_rate"] for v in per_era.values()]
    return {"per_era": per_era, "mean_success": round(sum(succ) / len(succ), 3),
            "worst_era_success": min(succ), "era_gap": round(max(succ) - min(succ), 3),
            "n_per_era": n_per_era}

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
        loss_pg = -(logp * adv).mean()
        loss_v = ((value - rets) ** 2).mean()
        probs = logp_all.exp()
        entropy = -(probs * logp_all).sum(-1).mean()
        loss = (loss_pg + 0.5 * loss_v - 0.01 * entropy) * (len(acts) / T)
        loss.backward()
        total += float(loss.detach())
    nn.utils.clip_grad_norm_(list(model.parameters()) + list(vhead.parameters()), 1.0)
    opt.step()
    return total

PHASES = [("P1_empty", "empty", 80), ("P2_keydoor", "keydoor", 100),
          ("P3_switch_rulechange", "switch", 80), ("P4_return_empty", "empty", 40)]

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
    torch.cuda.empty_cache()
    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
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
                for old in rng.sample(list(memory), 2):
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
                with open("/content/laya_exp6c_partial.json", "w") as f:
                    json.dump({"arm": arm, "evals": history}, f, indent=2)
        # --- 6c additions at each phase end: save policy + coexistence probe ---
        save_policy(model, vhead, arm, pname)
        probe_eras = ["empty", "keydoor"] if pname == "P2_keydoor" else (
            ["empty", "keydoor", "switch"] if pname in ("P3_switch_rulechange", "P4_return_empty") else [])
        if probe_eras:
            probe = probe_all_eras(model, vhead, tok, probe_eras, SEED + 9000 + ep_global)
            rec = {"arm": arm, "phase": pname, "probe": probe}
            history.append(rec)
            print("PROBE " + json.dumps(rec), flush=True)
            with open("/content/laya_exp6c_partial.json", "w") as f:
                json.dump({"arm": arm, "evals": history}, f, indent=2)
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
    # RANDOM baseline: uniform action sampling, no model at all
    rr = random.Random(SEED + 7); rand_res = {}
    for kind in ("empty", "keydoor", "switch"):
        succ, steps = 0, []
        for _ in range(30):
            w = World(kind, rr); done = False
            while not done:
                _r, done = w.step(rr.choice(ACTIONS))
            succ += int(w.agent == w.goal); steps.append(w.steps)
        rand_res[kind] = {"success_rate": succ / 30, "avg_steps": sum(steps) / 30}
        print(f"RANDOM {kind}: {rand_res[kind]}", flush=True)
    results["RANDOM"] = rand_res
    for arm in ["FROZEN", "RL", "RL+REPLAY"]:
        print(f"=== ARM {arm} ===", flush=True)
        t0 = time.time()
        results[arm] = run_arm(arm)
        print(f"arm {arm} done in {time.time()-t0:.0f}s", flush=True)
        with open("/content/laya_exp6c_partial.json", "w") as f:
            json.dump(results, f, indent=2)
    print("\n===== SUMMARY =====", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    print("RESULTS_JSON:" + json.dumps(results), flush=True)
    with open("/content/laya_exp6c_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
