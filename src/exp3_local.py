# Experiment 3: LOCAL reorganization -- Sai's hypothesis, depth form.
# Identical protocol/data to Exp 2 (seed 777), plus a LOCAL arm: bottom encoder layers
# frozen, top KEEP_TOP encoder layers + all head-side params trainable. Question: does
# bounding WHERE the representation may reorganize preserve FULL's learning speed
# while cutting its forgetting and its rule-change collateral damage?
import copy, json, random, re, time
import torch
from laya.train import (load_checkpoint, train_model, TrainConfig, items_from_rows,
                        encode_item, collate_items, _forward)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN, HEAD_MAX_LEN = 256, 128
SEED = 777
KEEP_TOP = 10
LABELS = ["alpha", "beta", "gamma", "delta"]
WORDS = [
    ["zelvo", "pranix", "tulmar", "osket", "vendra", "quilom", "hastrel", "dunvik"],
    ["merzal", "cobrin", "faskel", "ruvane", "pelto", "snaric", "owdren", "tibash"],
    ["golvik", "nerath", "puskel", "avrim", "tolzan", "weskit", "harbul", "mindra"],
    ["ostrel", "vikane", "darbus", "selmir", "quintal", "rabex", "fulmar", "tovik"],
    ["zendal", "prosk", "talmir", "uvane", "kelbro", "sarnic", "hodrel", "fimash"],
    ["bralto", "mesvik", "ondrel", "kaprin", "sulmar", "tevane", "gorik", "wexal"],
]
PROBE_WORDS = ["zovrel", "pranik", "talvos", "umarel", "kesbro", "sindal", "hortek", "fimral"]
PLACES = ["the north depot", "sector nine", "the old harbor", "unit 4", "the east gate", "bay twelve"]
NAMES = ["Alex", "Maria", "Chen", "Priya", "Tom", "Fatima", "Leo", "Nina"]

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

def cfg(epochs):
    return TrainConfig(loss="soft-ce", epochs=epochs, micro_batch=16, grad_accum=1,
                       freeze_encoder=False, shuffle_options=("choice",), seed=SEED, log_every=100000)

def freeze_bottom_layers(model, keep_top):
    idx = []
    for name, _p in model.encoder.named_parameters():
        m = re.search(r"layers\.(\d+)\.", name)
        if m: idx.append(int(m.group(1)))
    if not idx:
        raise RuntimeError("no encoder layer indices found in parameter names")
    top = max(idx); cutoff = top - keep_top + 1
    frozen, kept = 0, 0
    for name, p in model.encoder.named_parameters():
        m = re.search(r"layers\.(\d+)\.", name)
        if m is None or int(m.group(1)) < cutoff:
            p.requires_grad_(False); frozen += p.numel()
        else:
            kept += p.numel()
    print(f"LOCAL: encoder layers 0..{cutoff-1} frozen ({frozen/1e6:.1f}M params), "
          f"top {keep_top} layers trainable ({kept/1e6:.1f}M params)", flush=True)

def run_arm(name, tok, train_sets, test_sets, probe_train, probe_test, t1_new_train, t1_new_test):
    model, _tok, _c = load_checkpoint("convaiinnovations/laya")
    model = model.to(DEVICE)
    if name == "LOCAL":
        freeze_bottom_layers(model, KEEP_TOP)
    stages = []
    def snap(sname, m):
        return {"stage": sname, "task_acc": [round(accuracy(m, t, tok), 4) for t in test_sets],
                "probe_base_acc": round(accuracy(m, probe_test, tok), 4)}
    def probe(m):
        pm = copy.deepcopy(m)
        train_model(pm, tok, probe_train, cfg(1), DEVICE, MAX_LEN, HEAD_MAX_LEN)
        acc = round(accuracy(pm, probe_test, tok), 4)
        del pm; torch.cuda.empty_cache(); return acc
    s = snap("base", model); s["probe_trained_acc"] = probe(model); stages.append(s); print("  ", s, flush=True)
    for t in range(len(train_sets)):
        t0 = time.time()
        train_model(model, tok, train_sets[t], cfg(4), DEVICE, MAX_LEN, HEAD_MAX_LEN)
        s = snap(f"after_task_{t+1}", model); s["probe_trained_acc"] = probe(model)
        s["seconds"] = round(time.time() - t0, 1); stages.append(s); print("  ", s, flush=True)
    t0 = time.time()
    train_model(model, tok, t1_new_train, cfg(4), DEVICE, MAX_LEN, HEAD_MAX_LEN)
    s = snap("after_rule_change", model)
    s["task1_new_rule_acc"] = round(accuracy(model, t1_new_test, tok), 4)
    s["task1_stale_rule_acc"] = round(accuracy(model, test_sets[0], tok), 4)
    s["probe_trained_acc"] = probe(model); s["seconds"] = round(time.time() - t0, 1)
    stages.append(s); print("  ", s, flush=True)
    return stages

def main():
    print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "", flush=True)
    _m, tok, _c = load_checkpoint("convaiinnovations/laya"); del _m
    rng = random.Random(SEED)
    train_sets, test_sets = [], []
    for words in WORDS:
        mp = mapping_for(words)
        train_sets.append(build(tok, gen_rows(words, mp, 320, rng)))
        test_sets.append(build(tok, gen_rows(words, mp, 128, rng)))
    probe_train = build(tok, gen_rows(PROBE_WORDS, mapping_for(PROBE_WORDS), 256, rng))
    probe_test = build(tok, gen_rows(PROBE_WORDS, mapping_for(PROBE_WORDS), 128, rng))
    new_map = mapping_for(WORDS[0], shift=1)
    t1_new_train = build(tok, gen_rows(WORDS[0], new_map, 320, rng))
    t1_new_test = build(tok, gen_rows(WORDS[0], new_map, 128, rng))
    print("sets built", flush=True)

    results = {}
    for name in ["FULL", "LOCAL"]:
        print(f"=== ARM {name} ===", flush=True)
        results[name] = run_arm(name, tok, train_sets, test_sets, probe_train, probe_test,
                                t1_new_train, t1_new_test)
    print("\n===== SUMMARY =====", flush=True)
    for name, stages in results.items():
        final = stages[-2]["task_acc"]
        diag = [stages[i + 1]["task_acc"][i] for i in range(6)]
        forget = [round(diag[i] - final[i], 4) for i in range(6)]
        rc = stages[-1]
        print(f"{name}: acc right after learning each task {diag}", flush=True)
        print(f"{name}: final acc before rule change {final}", flush=True)
        print(f"{name}: forgetting per task {forget} (mean {sum(forget)/6:.4f})", flush=True)
        print(f"{name}: plasticity curve {[s['probe_trained_acc'] for s in stages]}", flush=True)
        print(f"{name}: rule change -> new-rule acc {rc['task1_new_rule_acc']}, stale-rule acc {rc['task1_stale_rule_acc']}, tasks2-6 after change {rc['task_acc'][1:]}", flush=True)
    print("RESULTS_JSON:" + json.dumps(results), flush=True)
    with open("/content/laya_exp3_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
