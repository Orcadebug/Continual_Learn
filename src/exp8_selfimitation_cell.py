# ===== EXPERIMENT 8: self-imitation (expert iteration) =====
# Runs in the Colab notebook AFTER the bootstrap cell loads exp7b machinery
# (World, ego_text, encode_state, forward_full, ACTIONS, DEVICE, TOK_HOLDER, SEED).
import json, random, time
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
memory = deque()          # entries: (enc, action_idx, phase_idx)

def mem_push(pairs, ph):
    for e, a in pairs:
        memory.append((e, a, ph))
    while len(memory) > MEM_CAP:
        memory.popleft()

def policy_probs(text):
    enc = encode_state(tok, text)
    with torch.no_grad():
        logits, _v = forward_full(model, vhead, [enc])
    return torch.softmax(logits[0], -1), enc

def train_on_memory(rng, rounds=2, bs=32):
    if len(memory) < 64:
        return
    mem = list(memory)
    for _ in range(rounds):
        batch = rng.sample(mem, bs)
        encs = [b[0] for b in batch]
        acts = torch.tensor([b[1] for b in batch], device=DEVICE)
        logits, _v = forward_full(model, vhead, encs)
        loss = F.cross_entropy(logits, acts)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

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

PHASES8 = [("P1_empty", "empty", 80), ("P2_keydoor", "keydoor", 100),
           ("P3_switch_rulechange", "switch", 80), ("P4_return_empty", "empty", 40)]
rng = random.Random(SEED)
history = []
for ph, (pname, kind, n_eps) in enumerate(PHASES8):
    recent = deque(maxlen=25)
    for ep in range(n_eps):
        w = World(kind, rng)
        pairs, done = [], False
        while not done:
            p, enc = policy_probs(ego_text(w))
            a = int(torch.multinomial(p, 1))
            _r, done = w.step(ACTIONS[a])
            pairs.append((enc, a))
        ok = w.agent == w.goal
        recent.append(int(ok))
        if ok:
            mem_push(pairs, ph)
        train_on_memory(rng)
        if (ep + 1) % 25 == 0 or ep == n_eps - 1:
            gs, gst = greedy_eval(kind)
            ent, mar = probe(kind, 9090 + ph)
            stale = round(sum(1 for m in memory if m[2] < ph) / max(1, len(memory)), 2)
            rec = {"phase": pname, "ep": ep + 1,
                   "train_success_last25": round(sum(recent) / len(recent), 2),
                   "greedy_eval": gs, "greedy_steps": gst, "entropy": ent, "margin": mar,
                   "memory": len(memory), "memory_stale_frac": stale}
            history.append(rec)
            print("E8 " + json.dumps(rec), flush=True)
torch.save({"model": model.state_dict(), "vhead": vhead.state_dict()}, "/content/exp8_policy.pt")
print("policy saved", flush=True)
print("RESULTS_JSON:" + json.dumps(history), flush=True)
with open("/content/laya_exp8_results.json", "w") as f:
    json.dump(history, f, indent=2)
