# EXP 17 STAGE 1 (execution cell; canonical copy) — world-model prediction study.
# Registered 2026-10-11 in RESEARCH_LOG.md (Directive sections 5/9). Question: does
# action-conditioned transition prediction extract real, usable knowledge from ALL
# interactions? Arms on ONE fixed dataset: A persistence (rules), B action-blind,
# C action-conditioned, D history-conditioned (1-step), C-shuffled control,
# C-frozen diagnostic. Amendment to registration (collection detail, arm-neutral
# because the dataset is shared): collection exploration = uniform random action
# on 30% of steps instead of the lifetime novelty rule.
# Sanity checks: tokenization skips == 0; per-arm finite eval logits; consistency
# of event counts printed before metrics. Partial results flushed after each arm.
import json, math, random, time
import torch, torch.nn as nn, torch.nn.functional as F

t_start = time.time()
def log(*a): print(*a, flush=True)

CELL_DIR = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}

def cell_passable(world, cell, has_key):
    if cell in world.walls: return False
    if world.door and cell == world.door:
        if world.kind == "keydoor": return has_key
        if world.kind == "switch": return world.switch_on
    return True

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
        dx, dy = CELL_DIR[a]
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

def pooled_h(model, encs):
    batch = collate_items([[e for e in encs]], TOK.pad_token_id)
    ids = batch["input_ids"].to(DEVICE); att = batch["attention_mask"].to(DEVICE)
    qt = batch["qtype"].to(DEVICE)
    with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=(DEVICE.type == "cuda")):
        h = model.encoder(input_ids=ids, attention_mask=att).last_hidden_state
        h = h + model.type_emb(qt)[:, None, :]
        if model.head is not None:
            pad = ~att.bool()
            for layer in model.head.layers:
                h = layer(h, src_key_padding_mask=pad)
    return h[:, 0].float()

class WMPred(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d, 256), nn.GELU())
        self.disp = nn.Linear(256, 9)
        self.rew = nn.Linear(256, 3)
        self.key = nn.Linear(256, 2)
        self.sw = nn.Linear(256, 2)
        self.door = nn.Linear(256, 2)
        self.goal = nn.Linear(256, 2)
        self.done = nn.Linear(256, 2)
    def forward(self, h):
        z = self.trunk(h)
        return {"disp": self.disp(z), "rew": self.rew(z), "key": self.key(z),
                "sw": self.sw(z), "door": self.door(z), "goal": self.goal(z),
                "done": self.done(z)}

def encode_texts(tok, texts):
    encs = []
    for s0 in range(0, len(texts), 512):
        chunk = texts[s0:s0 + 512]
        rows = [{"state": t, "questions": question_for(ACTIONS), "expected": {"act": "north"}} for t in chunk]
        items, skipped = items_from_rows(tok, rows, MAX_LEN, HEAD_MAX_LEN)
        assert skipped == 0 and len(items) == len(chunk), (skipped, len(items), len(chunk))
        encs.extend(encode_item(tok, it, MAX_LEN, HEAD_MAX_LEN) for it in items)
    return encs

def targets_of(rec):
    return {"disp": (rec["dx"] + 1) * 3 + (rec["dy"] + 1), "rew": rec["rew"],
            "key": rec["key_after"], "sw": rec["sw_after"], "door": rec["door_after"],
            "goal": rec["goal"], "done": rec["done"]}

def ece(probs, labels, bins=10):
    probs = torch.tensor(probs, dtype=torch.float64); labels = torch.tensor(labels, dtype=torch.float64)
    out, n = 0.0, max(1, len(labels))
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (probs >= lo) & (probs < hi if b < bins - 1 else probs <= hi)
        if m.any():
            out += (m.sum().item() / n) * abs(probs[m].mean().item() - labels[m].mean().item())
    return round(out, 4)

def metrics_from_preds(recs, pred_cls, pred_prob):
    # pred_cls: dict target->list of argmax classes; pred_prob: {"goal": [...], "blocked": [...]}
    ys = [targets_of(r) for r in recs]
    m = {"n": len(recs)}
    for t in ["disp", "rew", "key", "sw", "door", "goal", "done"]:
        m[t + "_acc"] = round(sum(1 for i, y in enumerate(ys) if pred_cls[t][i] == y[t]) / len(ys), 4)
    def ev(mask, true_fn, pred_fn):
        idx = [i for i, r in enumerate(recs) if mask(r)]
        if not idx: return {"n_base": 0}
        tp = sum(1 for i in idx if true_fn(recs[i]) and pred_fn(i))
        fn_ = sum(1 for i in idx if true_fn(recs[i]) and not pred_fn(i))
        fp = sum(1 for i in idx if not true_fn(recs[i]) and pred_fn(i))
        return {"n_base": len(idx), "n_true": tp + fn_,
                "recall": round(tp / (tp + fn_), 4) if tp + fn_ else None,
                "precision": round(tp / (tp + fp), 4) if tp + fp else None}
    m["blocked"] = ev(lambda r: True, lambda r: (r["dx"], r["dy"]) == (0, 0), lambda i: pred_cls["disp"][i] == 4)
    m["key_gain"] = ev(lambda r: r["key_before"] == 0, lambda r: r["key_after"] == 1, lambda i: pred_cls["key"][i] == 1)
    m["switch_gain"] = ev(lambda r: r["kind"] == "switch" and r["sw_before"] == 0, lambda r: r["sw_after"] == 1, lambda i: pred_cls["sw"][i] == 1)
    m["door_gain"] = ev(lambda r: r["kind"] != "empty" and r["door_before"] == 0, lambda r: r["door_after"] == 1, lambda i: pred_cls["door"][i] == 1)
    m["goal_event"] = ev(lambda r: True, lambda r: r["goal"] == 1, lambda i: pred_cls["goal"][i] == 1)
    if pred_prob is not None:
        m["goal_ece"] = ece(pred_prob["goal"], [r["goal"] for r in recs])
        m["blocked_ece"] = ece(pred_prob["blocked"], [1 if (r["dx"], r["dy"]) == (0, 0) else 0 for r in recs])
    return m

# ---------- Stage 1a: collect ONE fixed transition dataset ----------
def collect():
    model, tok, _c = load_checkpoint("convaiinnovations/laya")
    TOK_HOLDER[0] = tok
    model = model.to(DEVICE); model.eval()
    d = model.encoder.config.hidden_size
    vhead = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1)).to(DEVICE)
    kinds = ["empty", "keydoor", "switch"]
    trans = []
    for i in range(900):
        kind = kinds[i % 3]
        rng = random.Random(910000 + i)
        w = World(kind, rng)
        held = (i % 4 == 3)
        prev_act, prev_desc = None, None
        done = False
        while not done:
            text = ego_text(w)
            enc = encode_state(tok, text)
            with torch.no_grad():
                logits, _v = forward_full(model, vhead, [enc])
            probs = torch.softmax(logits[0].float(), -1)
            ai = rng.randrange(4) if rng.random() < 0.30 else int(torch.multinomial(probs, 1))
            a = ACTIONS[ai]
            kb, sb, db = int(w.has_key), int(w.switch_on), int(w.door_open)
            ax0, ay0 = w.agent
            r, done = w.step(a)
            goal = int(w.agent == w.goal)
            dx, dy = w.agent[0] - ax0, w.agent[1] - ay0
            rew = 2 if r > 0.5 else (1 if r < -0.02 else 0)
            trans.append({"kind": kind, "held": held, "text": text, "action": a,
                          "prev_act": prev_act, "prev_desc": prev_desc,
                          "dx": dx, "dy": dy, "rew": rew,
                          "key_before": kb, "sw_before": sb, "door_before": db,
                          "key_after": int(w.has_key), "sw_after": int(w.switch_on),
                          "door_after": int(w.door_open), "goal": goal, "done": int(done)})
            if goal: prev_desc = "reached the goal"
            elif kb == 0 and w.has_key: prev_desc = "picked up the key"
            elif sb == 0 and w.switch_on: prev_desc = "activated the switch"
            elif (dx, dy) == (0, 0): prev_desc = "was blocked"
            else: prev_desc = "moved"
            prev_act = a
        if (i + 1) % 150 == 0:
            log("COLLECT", i + 1, "episodes; transitions", len(trans), "elapsed_s", round(time.time() - t_start))
    del model, vhead
    torch.cuda.empty_cache()
    return tok, trans

tok, trans = collect()
train_recs = [r for r in trans if not r["held"]]
test_recs = [r for r in trans if r["held"]]
log("DATASET transitions", len(trans), "train", len(train_recs), "heldout", len(test_recs),
    "key_gains", sum(1 for r in trans if r["key_before"] == 0 and r["key_after"] == 1),
    "switch_gains", sum(1 for r in trans if r["sw_before"] == 0 and r["sw_after"] == 1),
    "goals", sum(r["goal"] for r in trans))

def arm_texts(recs, variant, shuffled_map=None):
    out = []
    for j, r in enumerate(recs):
        t = r["text"]
        if variant == "B":
            out.append(t)
        elif variant in ("C", "Cshuf"):
            a = r["action"] if variant == "C" else shuffled_map[j]
            out.append(t + f"\nAction taken: {a}.")
        elif variant == "D":
            pa = r["prev_act"] if r["prev_act"] else "none"
            pd = r["prev_desc"] if r["prev_desc"] else "none"
            out.append(t + f"\nPrevious action: {pa}. Previous result: {pd}." + f"\nAction taken: {r['action']}.")
    return out

def eval_preds(model, pred, encs, recs):
    model.eval(); pred.eval()
    cls = {t: [] for t in ["disp", "rew", "key", "sw", "door", "goal", "done"]}
    prob = {"goal": [], "blocked": []}
    with torch.no_grad():
        for s0 in range(0, len(encs), 64):
            out = pred(pooled_h(model, encs[s0:s0 + 64]))
            assert torch.isfinite(out["disp"]).all(), "non-finite eval logits"
            for t in cls: cls[t].extend(out[t].argmax(-1).tolist())
            prob["goal"].extend(torch.softmax(out["goal"].float(), -1)[:, 1].tolist())
            prob["blocked"].extend(torch.softmax(out["disp"].float(), -1)[:, 4].tolist())
    return metrics_from_preds(recs, cls, prob)

def train_arm(name, variant, encoder_trainable=True):
    shuffled_map = None
    if variant == "Cshuf":
        acts = [r["action"] for r in train_recs]
        shuffled_map = acts[:]
        random.Random(777).shuffle(shuffled_map)
    encs_tr = encode_texts(tok, arm_texts(train_recs, variant, shuffled_map))
    encs_te = encode_texts(tok, arm_texts(test_recs, "C" if variant == "Cshuf" else variant))
    model, tok2, _c = load_checkpoint("convaiinnovations/laya")
    TOK_HOLDER[0] = tok2
    model = model.to(DEVICE)
    d = model.encoder.config.hidden_size
    pred = WMPred(d).to(DEVICE)
    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    rest_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    for p in enc_params: p.requires_grad = encoder_trainable
    groups = [{"params": rest_params + list(pred.parameters()), "lr": 1e-3}]
    if encoder_trainable: groups.append({"params": enc_params, "lr": 1e-5})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    ys = [targets_of(r) for r in train_recs]
    g = torch.Generator().manual_seed(4242)
    for ep in range(3):
        model.train(); pred.train()
        order = torch.randperm(len(encs_tr), generator=g).tolist()
        tot, nb = 0.0, 0
        for s0 in range(0, len(order), 64):
            idx = order[s0:s0 + 64]
            out = pred(pooled_h(model, [encs_tr[i] for i in idx]))
            loss = 0.0
            for t in ["disp", "rew", "key", "sw", "door", "goal", "done"]:
                y = torch.tensor([ys[i][t] for i in idx], device=DEVICE)
                loss = loss + F.cross_entropy(out[t].float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()) + list(pred.parameters()), 1.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
        log("ARM", name, "epoch", ep, "loss", round(tot / max(1, nb), 4), "elapsed_s", round(time.time() - t_start))
    m = eval_preds(model, pred, encs_te, test_recs)
    del model, pred, opt, encs_tr, encs_te
    torch.cuda.empty_cache()
    return m

RESULTS = {}
# Arm A: persistence (no learning)
pcls = {"disp": [4] * len(test_recs), "rew": [0] * len(test_recs),
        "key": [r["key_before"] for r in test_recs], "sw": [r["sw_before"] for r in test_recs],
        "door": [r["door_before"] for r in test_recs], "goal": [0] * len(test_recs),
        "done": [0] * len(test_recs)}
pprob = {"goal": [0.0] * len(test_recs), "blocked": [1.0] * len(test_recs)}
RESULTS["A_persistence"] = metrics_from_preds(test_recs, pcls, pprob)
log("E17S1 A_persistence", json.dumps(RESULTS["A_persistence"]))
with open("/content/laya_exp17s1_results.json", "w") as f: json.dump(RESULTS, f, indent=1)

for name, variant, et in [("B_blind", "B", True), ("C_action", "C", True),
                          ("D_history", "D", True), ("C_shuffled", "Cshuf", True),
                          ("C_frozen", "C", False)]:
    try:
        RESULTS[name] = train_arm(name, variant, et)
    except Exception as e:
        RESULTS[name] = {"error": repr(e)}
        log("ARM", name, "FAILED", repr(e))
    log("E17S1", name, json.dumps(RESULTS[name]))
    with open("/content/laya_exp17s1_results.json", "w") as f: json.dump(RESULTS, f, indent=1)

log("E17S1 SUMMARY disp_acc A/B/C/D/Cshuf/Cfrozen",
    RESULTS["A_persistence"]["disp_acc"], RESULTS["B_blind"].get("disp_acc"),
    RESULTS["C_action"].get("disp_acc"), RESULTS["D_history"].get("disp_acc"),
    RESULTS["C_shuffled"].get("disp_acc"), RESULTS["C_frozen"].get("disp_acc"))
log("RESULTS_JSON:", json.dumps(RESULTS))
log("DONE", round(time.time() - t_start))
