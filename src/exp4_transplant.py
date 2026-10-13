# Experiment 4: the transplant test -- is forgetting storage failure or readout failure?
# Train the FULL Exp-2 sequence, snapshot weights after task 1 (S1) and after task 6 (S6).
# Then swap head-side params (type_emb + decision head + scorer + act_head) between the
# two states and evaluate all 4 encoder/head combinations on task 1's test set.
# No retraining anywhere: if E6H1 recovers task 1, the final encoder still carries the
# task and only the reader drifted. Corroboration: label-centroid geometry of post-head
# marker vectors at base / S1 / S6.
import copy, json, random, re, time
import torch
from laya.train import (load_checkpoint, train_model, TrainConfig, items_from_rows,
                        encode_item, collate_items, _forward)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN, HEAD_MAX_LEN = 256, 128
SEED = 777
LABELS = ["alpha", "beta", "gamma", "delta"]
WORDS = [
    ["zelvo", "pranix", "tulmar", "osket", "vendra", "quilom", "hastrel", "dunvik"],
    ["merzal", "cobrin", "faskel", "ruvane", "pelto", "snaric", "owdren", "tibash"],
    ["golvik", "nerath", "puskel", "avrim", "tolzan", "weskit", "harbul", "mindra"],
    ["ostrel", "vikane", "darbus", "selmir", "quintal", "rabex", "fulmar", "tovik"],
    ["zendal", "prosk", "talmir", "uvane", "kelbro", "sarnic", "hodrel", "fimash"],
    ["bralto", "mesvik", "ondrel", "kaprin", "sulmar", "tevane", "gorik", "wexal"],
]
PLACES = ["the north depot", "sector nine", "the old harbor", "unit 4", "the east gate", "bay twelve"]
NAMES = ["Alex", "Maria", "Chen", "Priya", "Tom", "Fatima", "Leo", "Nina"]
HEAD_PREFIXES = ("head.", "type_emb.", "scorer.", "act_head.")

def mapping_for(words, shift=0):
    return {w: LABELS[(i // 2 + shift) % 4] for i, w in enumerate(words)}

def gen_rows(words, mapping, n, rng, qid="route"):
    questions = {qid: {"type": "choice",
                       "instructions": "Assign the report to the correct category.",
                       "criteria": {lab: f"the {lab} category" for lab in LABELS}}}
    rows = []
    for _ in range(n):
        w = rng.choice(words); lab = mapping[w]
        who, where = rng.choice(NAMES), rng.choice(PLACES)
        num = rng.randint(2, 98)
        templates = [
            f"Field report {num}: code {w} logged at {where}. Filed by {who}.",
            f"Update from {who}: {w} confirmed near {where}. Reference {num}.",
            f"Log entry {num}. Observer {who} recorded {w} at {where} today.",
            f"Dispatch note: {w} situation developing at {where}, per {who}. Case {num}.",
        ]
        rows.append({"state": rng.choice(templates), "questions": questions, "expected": {qid: lab}})
    return rows

def build(tok, rows):
    items, skipped = items_from_rows(tok, rows, MAX_LEN, HEAD_MAX_LEN)
    assert items, f"no items, skipped={skipped}"
    return items

@torch.no_grad()
def accuracy(model, items, tok):
    model.eval(); right, total = 0, 0
    for start in range(0, len(items), 32):
        chunk = items[start:start + 32]
        batch = collate_items([[encode_item(tok, it, MAX_LEN, HEAD_MAX_LEN) for it in chunk]], tok.pad_token_id)
        logits = _forward(model, batch, DEVICE, amp=False, detach_encoder=False).cpu()
        for row, it in zip(logits, chunk):
            k = it["k"]
            right += int(int(row[:k].argmax()) == int(torch.tensor(it["target"]).argmax())); total += 1
    return right / total

@torch.no_grad()
def marker_reps(model, items, tok):
    """Post-head marker vectors, mean-pooled per item -> (N, D); plus gold label idx."""
    model.eval(); reps, golds = [], []
    for start in range(0, len(items), 32):
        chunk = items[start:start + 32]
        batch = collate_items([[encode_item(tok, it, MAX_LEN, HEAD_MAX_LEN) for it in chunk]], tok.pad_token_id)
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
        m = torch.gather(h, 1, idx)                       # (B, K, D) option markers
        w = mmask.float().unsqueeze(-1)
        reps.append(((m * w).sum(1) / w.sum(1)).cpu())
        golds.extend(int(torch.tensor(it["target"]).argmax()) for it in chunk)
    return torch.cat(reps), golds

def centroid_acc(model, train_items, test_items, tok):
    Xtr, ytr = marker_reps(model, train_items, tok)
    Xte, yte = marker_reps(model, test_items, tok)
    Xtr = torch.nn.functional.normalize(Xtr, dim=-1); Xte = torch.nn.functional.normalize(Xte, dim=-1)
    cents = torch.stack([Xtr[[i for i, y in enumerate(ytr) if y == c]].mean(0) for c in range(4)])
    cents = torch.nn.functional.normalize(cents, dim=-1)
    pred = (Xte @ cents.T).argmax(-1).tolist()
    return sum(int(p == y) for p, y in zip(pred, yte)) / len(yte)

def cfg(epochs):
    return TrainConfig(loss="soft-ce", epochs=epochs, micro_batch=16, grad_accum=1,
                       freeze_encoder=False, shuffle_options=("choice",), seed=SEED, log_every=100000)

def is_head_key(k):
    return k.startswith(HEAD_PREFIXES)

def main():
    print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "", flush=True)
    _m, tok, _c = load_checkpoint("convaiinnovations/laya"); del _m
    rng = random.Random(SEED)
    train_sets, test_sets = [], []
    for words in WORDS:
        mp = mapping_for(words)
        train_sets.append(build(tok, gen_rows(words, mp, 320, rng)))
        test_sets.append(build(tok, gen_rows(words, mp, 128, rng)))
    print("sets built", flush=True)

    model, _t, _c2 = load_checkpoint("convaiinnovations/laya")
    model = model.to(DEVICE)
    base_model = copy.deepcopy(model)
    print("base task1 acc:", round(accuracy(model, test_sets[0], tok), 4), flush=True)

    train_model(model, tok, train_sets[0], cfg(4), DEVICE, MAX_LEN, HEAD_MAX_LEN)
    S1 = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    acc_E1H1 = accuracy(model, test_sets[0], tok)
    print(f"after task 1: E1H1 acc {acc_E1H1:.4f}", flush=True)
    for t in range(1, 6):
        train_model(model, tok, train_sets[t], cfg(4), DEVICE, MAX_LEN, HEAD_MAX_LEN)
        print(f"after task {t+1}: task1 acc {accuracy(model, test_sets[0], tok):.4f}", flush=True)
    S6 = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    acc_E6H6 = accuracy(model, test_sets[0], tok)

    def hybrid(enc_state, head_state):
        sd = {}
        for k in enc_state:
            sd[k] = head_state[k] if is_head_key(k) else enc_state[k]
        m, _t, _c = load_checkpoint("convaiinnovations/laya")
        m = m.to(DEVICE); m.load_state_dict({k: v.to(DEVICE) for k, v in sd.items()}, strict=True)
        return m

    m_E6H1 = hybrid(S6, S1); acc_E6H1 = accuracy(m_E6H1, test_sets[0], tok); del m_E6H1
    m_E1H6 = hybrid(S1, S6); acc_E1H6 = accuracy(m_E1H6, test_sets[0], tok); del m_E1H6
    torch.cuda.empty_cache()

    cent = {}
    cent["base"] = centroid_acc(base_model, train_sets[0], test_sets[0], tok); del base_model
    m1, _t, _c = load_checkpoint("convaiinnovations/laya"); m1 = m1.to(DEVICE)
    m1.load_state_dict({k: v.to(DEVICE) for k, v in S1.items()}, strict=True)
    cent["S1"] = centroid_acc(m1, train_sets[0], test_sets[0], tok); del m1
    cent["S6"] = centroid_acc(model, train_sets[0], test_sets[0], tok)

    out = {"E1H1": round(acc_E1H1, 4), "E6H6": round(acc_E6H6, 4),
           "E6H1_final_encoder_task1_head": round(acc_E6H1, 4),
           "E1H6_task1_encoder_final_head": round(acc_E1H6, 4),
           "centroid_geometry": {k: round(v, 4) for k, v in cent.items()}}
    print("\n===== SUMMARY =====", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    print("RESULTS_JSON:" + json.dumps(out), flush=True)
    with open("/content/laya_exp4_results.json", "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
