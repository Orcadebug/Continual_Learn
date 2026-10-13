# Experiment 1: Where does forgetting live in Laya -- encoder or decision head?
# Sequential fine-tuning of the shipped English checkpoint on 6 synthetic typed-decision
# tasks, in two arms that differ ONLY in which parameters may move:
#   FULL   : everything trainable (freeze_encoder=False)
#   FROZEN : encoder frozen, only type_emb + decision head + scorer + act_head train
# After every stage we measure (a) accuracy on every task's held-out test set
# (retention matrix) and (b) PLASTICITY: a deepcopy of the current model gets a fixed
# small training budget on a never-seen probe task; its probe accuracy is the stage's
# learning-capacity reading. Same data, same order, same seed, same budget per arm.
import copy, json, random, time
import torch
from laya.train import (load_checkpoint, train_model, TrainConfig, items_from_rows,
                        encode_item, collate_items, _forward)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN, HEAD_MAX_LEN = 256, 128
SEED = 1234

# ---------------------------------------------------------------- synthetic tasks
# Each domain: instructions + 4 labels, each label has cue phrases that decide it.
DOMAINS = [
    ("it_helpdesk", "Which IT team should handle this ticket?", {
        "password_access": ["reset my password", "locked out of my account", "two-factor code never arrives", "cannot sign in since the update"],
        "hardware": ["laptop screen flickers", "battery swells and gets hot", "keyboard keys stopped working", "dock does not detect the monitor"],
        "software": ["the vpn client crashes on launch", "spreadsheet app freezes when saving", "update fails with error 0x2", "browser extension keeps disappearing"],
        "network": ["office wifi drops every hour", "ethernet port shows no link", "video calls stutter on the office network", "cannot reach the shared drive remotely"],
    }),
    ("bank_support", "Which banking team should handle this message?", {
        "fraud": ["a charge I never made appeared", "someone used my card in another country", "suspicious login alerted me", "money left my account without my approval"],
        "billing": ["charged twice for the same fee", "my statement total looks wrong", "a refund is missing after ten days", "interest was applied despite autopay"],
        "cards": ["my debit card was declined at a store", "waiting for a replacement card", "card expires next month and no new one came", "atm swallowed my card yesterday"],
        "loans": ["questions about my mortgage rate", "want to apply for a personal loan", "loan payment date needs changing", "payoff amount for my auto loan"],
    }),
    ("clinic", "Which clinic department should see this patient?", {
        "dental": ["tooth aches when chewing", "a filling fell out last night", "gums bleed when brushing", "wisdom tooth is coming in crooked"],
        "vision": ["vision has become blurry up close", "need a new glasses prescription", "eyes sting after screen work", "floaters appeared in one eye"],
        "skin": ["a rash spread across my arm", "a mole changed color recently", "dry patches that itch at night", "a cut that will not heal"],
        "joints": ["knee hurts climbing stairs", "shoulder clicks when lifting", "lower back stiffness every morning", "ankle still swollen after a sprain"],
    }),
    ("hotel", "Which hotel team should handle this guest request?", {
        "housekeeping": ["need extra towels in room 412", "room was not cleaned today", "requesting more pillows and blankets", "spilled coffee on the carpet"],
        "maintenance": ["the shower only runs cold", "air conditioning rattles loudly", "a light fixture keeps buzzing", "the toilet leaks at the base"],
        "dining": ["book a table for two tonight", "breakfast hours on weekends", "allergic to nuts, what is safe on the menu", "room service never delivered my order"],
        "front_desk": ["can I check out two hours late", "need a second key card", "question about the deposit hold", "extend my stay by one night"],
    }),
    ("airline", "Which airline team should handle this passenger?", {
        "baggage": ["my suitcase did not arrive", "a wheel broke off my checked bag", "bag was delivered to the wrong hotel", "need to add a second checked bag"],
        "booking": ["change my flight to friday", "spelled my name wrong on the ticket", "add a seat selection to my trip", "use my credit for a new booking"],
        "refunds": ["refund for a cancelled flight", "charged for a seat I did not pick", "duplicate payment on my card", "voucher from the delay never arrived"],
        "assistance": ["need a wheelchair at the gate", "traveling with a service animal", "my child flies alone next week", "requesting a bassinet for the baby"],
    }),
    ("university", "Which university office should handle this student?", {
        "admissions": ["when is the application deadline", "my transcripts were sent last week", "how do I appeal a rejection", "questions about transfer credits"],
        "housing": ["roommate conflict in the dorm", "applying for a single room", "my housing deposit refund", "the dorm heater is broken"],
        "records": ["need an official transcript copy", "a grade is missing from my record", "correct the name on my diploma", "how to request enrollment verification"],
        "tuition": ["why was my aid reduced", "set up a payment plan for tuition", "a scholarship was not applied", "questions about a tuition refund"],
    }),
]
PROBE = ("store_orders", "Which store team should handle this order message?", {
    "delivery": ["my package is a week late", "tracking shows delivered but nothing arrived", "the driver left it at the wrong house", "can I change the delivery address"],
    "damaged": ["the item arrived cracked", "the box was crushed and the screen broke", "received the wrong model entirely", "parts are missing from the package"],
    "giftcard": ["gift card balance shows zero", "my gift card code will not redeem", "bought a gift card that never emailed", "can gift cards be combined"],
    "subscription": ["pause my monthly subscription", "charged again after cancelling", "switch my subscription to quarterly", "skip next month's box"],
})
NAMES = ["Alex", "Maria", "Chen", "Priya", "Tom", "Fatima", "Leo", "Nina", "Omar", "Sofia"]
FILLERS = ["Thanks for any help.", "This started yesterday.", "Please advise.", "It is getting urgent.",
           "I already checked the FAQ.", "Appreciate a quick reply.", "This is the second time I ask.", ""]

def gen_rows(domain, n, rng):
    name, instr, labels = domain
    qid = "route"
    questions = {qid: {"type": "choice", "instructions": instr,
                       "criteria": {lab: "requests about: " + "; ".join(cues[:2]) for lab, cues in labels.items()}}}
    rows = []
    labs = list(labels.keys())
    for _ in range(n):
        lab = rng.choice(labs)
        cue = rng.choice(labels[lab])
        person = rng.choice(NAMES)
        templates = [
            f"Hi, this is {person}. I need help: {cue}. {rng.choice(FILLERS)}",
            f"Hello support team, {cue}. {rng.choice(FILLERS)}",
            f"{person} here. Problem: {cue}. Can someone look into it? {rng.choice(FILLERS)}",
            f"Good morning, I am writing because {cue}. {rng.choice(FILLERS)}",
        ]
        rows.append({"state": rng.choice(templates), "questions": questions, "expected": {qid: lab}})
    return rows

def build_items(tok, domains_rows):
    out = []
    for rows in domains_rows:
        items, skipped = items_from_rows(tok, rows, MAX_LEN, HEAD_MAX_LEN)
        assert items, f"no items built, skipped={skipped}"
        out.append(items)
    return out

@torch.no_grad()
def accuracy(model, items):
    model.eval()
    right, total = 0, 0
    for start in range(0, len(items), 32):
        chunk = items[start:start + 32]
        batch = collate_items([[encode_item(tok_global, it, MAX_LEN, HEAD_MAX_LEN) for it in chunk]],
                              tok_global.pad_token_id)
        logits = _forward(model, batch, DEVICE, amp=False, detach_encoder=False).cpu()
        for row, it in zip(logits, chunk):
            k = it["k"]
            pred = int(row[:k].argmax())
            gold = int(torch.tensor(it["target"]).argmax())
            right += int(pred == gold); total += 1
    return right / total

def train_cfg(freeze, epochs):
    return TrainConfig(loss="soft-ce", epochs=epochs, micro_batch=16, grad_accum=1,
                       freeze_encoder=freeze, shuffle_options=("choice",),
                       seed=SEED, log_every=100000)

def run_arm(freeze, task_train, task_test, probe_train, probe_test):
    global tok_global
    model, tok, cfg = load_checkpoint("convaiinnovations/laya")
    tok_global = tok
    model = model.to(DEVICE)
    stages = []
    def snapshot(stage_name, m):
        accs = [accuracy(m, t) for t in task_test]
        return {"stage": stage_name, "task_acc": [round(a, 4) for a in accs],
                "probe_base_acc": round(accuracy(m, probe_test), 4)}
    stages.append(snapshot("base", model))
    # plasticity at base
    pm = copy.deepcopy(model)
    train_model(pm, tok, probe_train, train_cfg(freeze, epochs=1), DEVICE, MAX_LEN, HEAD_MAX_LEN)
    stages[-1]["probe_trained_acc"] = round(accuracy(pm, probe_test), 4)
    del pm; torch.cuda.empty_cache()
    for t in range(len(task_train)):
        t0 = time.time()
        train_model(model, tok, task_train[t], train_cfg(freeze, epochs=3), DEVICE, MAX_LEN, HEAD_MAX_LEN)
        snap = snapshot(f"after_task_{t+1}", model)
        pm = copy.deepcopy(model)
        train_model(pm, tok, probe_train, train_cfg(freeze, epochs=1), DEVICE, MAX_LEN, HEAD_MAX_LEN)
        snap["probe_trained_acc"] = round(accuracy(pm, probe_test), 4)
        del pm; torch.cuda.empty_cache()
        snap["seconds"] = round(time.time() - t0, 1)
        stages.append(snap)
        print(f"  stage done: {snap}", flush=True)
    return stages

def main():
    global tok_global
    print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "", flush=True)
    # tokenize once with a throwaway load to build items (items are tokenizer-bound)
    _m, tok, _cfg = load_checkpoint("convaiinnovations/laya")
    tok_global = tok
    del _m
    rng = random.Random(SEED)
    task_train, task_test = [], []
    for d in DOMAINS:
        task_train.append(build_items(tok, [gen_rows(d, 256, rng)])[0])
        task_test.append(build_items(tok, [gen_rows(d, 128, rng)])[0])
    probe_train = build_items(tok, [gen_rows(PROBE, 256, rng)])[0]
    probe_test = build_items(tok, [gen_rows(PROBE, 128, rng)])[0]
    print("items per task: train", len(task_train[0]), "test", len(task_test[0]), flush=True)

    results = {}
    for name, freeze in [("FULL", False), ("FROZEN", True)]:
        print(f"=== ARM {name} (freeze_encoder={freeze}) ===", flush=True)
        results[name] = run_arm(freeze, task_train, task_test, probe_train, probe_test)

    print("\n===== SUMMARY =====", flush=True)
    for name, stages in results.items():
        final = stages[-1]["task_acc"]
        diag = [stages[i + 1]["task_acc"][i] for i in range(len(DOMAINS))]
        forgetting = [round(diag[i] - final[i], 4) for i in range(len(DOMAINS))]
        probe_curve = [s["probe_trained_acc"] for s in stages]
        print(f"{name}: final task acc {final}", flush=True)
        print(f"{name}: accuracy right after learning each task {diag}", flush=True)
        print(f"{name}: forgetting per task (diag - final) {forgetting}", flush=True)
        print(f"{name}: plasticity curve (probe acc after fixed budget, by stage) {probe_curve}", flush=True)
    print("RESULTS_JSON:" + json.dumps(results), flush=True)
    with open("/content/laya_exp1_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
