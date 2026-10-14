"""Executable Exp18-Reconstructed lifetime.

Run from the repository root, for example:
    python -m exp18_reconstructed.runner --arm valve --seed 1 --output /kaggle/working/run

The only controller distinction across primary arms is `apply_update`: all
arms obtain their proposed update from the same weighted replay objective.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

from .valve import project_gradient
from .replay import eligible_reference_items

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import exp7_sharpening_b as legacy  # noqa: E402
from laya.train import load_checkpoint  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTIONS = legacy.ACTIONS


def corrected_cell_passable(world: Any, cell: tuple[int, int], has_key: bool) -> bool:
    if cell in world.walls:
        return False
    if world.door and cell == world.door:
        return has_key if world.kind == "keydoor" else world.switch_on
    return True


# The original Exp7 observation helper resolves this global at run time.
legacy.cell_passable = corrected_cell_passable


def corrected_ego_text(world: Any) -> str:
    """Exp17's verified sensor-complete observation; no hidden rule identifier."""
    ax, ay = world.agent; gx, gy = world.goal
    def relative(tx: int, ty: int, label: str) -> str:
        dx, dy = tx - ax, ty - ay
        bits = ([f"{-dx} west"] if dx < 0 else [f"{dx} east"] if dx > 0 else [])
        bits += ([f"{-dy} north"] if dy < 0 else [f"{dy} south"] if dy > 0 else [])
        return f"{label}: {', '.join(bits) if bits else 'here'}."
    blocked = []
    for action, (dx, dy) in {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}.items():
        cell = (ax + dx, ay + dy)
        if not (0 <= cell[0] < world.size and 0 <= cell[1] < world.size) or not corrected_cell_passable(world, cell, world.has_key):
            blocked.append(action)
    lines = ["You are in a grid room. Reach the goal G.", relative(gx, gy, "Goal"),
             "Blocked directions: " + (", ".join(blocked) if blocked else "none") + ".",
             "You carry: " + ("the key." if world.has_key else "nothing.")]
    if world.key is not None and not world.has_key:
        lines.append(relative(world.key[0], world.key[1], "Key"))
    if world.switch is not None and not world.switch_on:
        lines.append(relative(world.switch[0], world.switch[1], "Switch"))
    if world.door:
        lines += ["The door is " + ("open." if world.door_open else "locked."), relative(world.door[0], world.door[1], "Door")]
    return "\n".join(lines)


@dataclass
class MemoryItem:
    experience_id: int
    enc: Any
    action: int
    phase_index: int
    admitted_update: int


@dataclass
class RunState:
    phase_index: int = 0
    episode_index: int = 0
    update_index: int = 0
    next_experience_id: int = 1


def atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def parameter_digest(params: list[torch.nn.Parameter]) -> str:
    digest = hashlib.sha256()
    for param in params:
        digest.update(param.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class Lifetime:
    def __init__(self, config: dict[str, Any], arm: str, seed: int, output: Path, pilot: bool, resume: Path | None = None):
        if arm not in config["arms"]:
            raise ValueError(f"unknown arm {arm}")
        self.config, self.arm, self.seed, self.output, self.pilot = config, arm, seed, output, pilot
        self.output.mkdir(parents=True, exist_ok=True)
        self.episodes_path, self.updates_path = output / "episodes.jsonl", output / "updates.jsonl"
        self.checkpoint_path = output / "checkpoint.pt"
        self.rng = random.Random(5000 + seed * 101)
        torch.manual_seed(6000 + seed * 101)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(6000 + seed * 101)
        self.model, self.tokenizer, _ = load_checkpoint(config["model_checkpoint"])
        legacy.TOK_HOLDER[0] = self.tokenizer
        self.model = self.model.to(DEVICE)
        if hasattr(self.model.encoder, "gradient_checkpointing_enable"):
            self.model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        multiplier = config["arms"][arm]["learning_rate_multiplier"]
        opt_config = config["optimizer"]
        groups = [
            {"params": [p for n, p in self.model.named_parameters() if n.startswith("encoder.")],
             "lr": opt_config["encoder_lr"] * multiplier},
            {"params": [p for n, p in self.model.named_parameters() if not n.startswith("encoder.")],
             "lr": opt_config["non_encoder_lr"] * multiplier},
        ]
        self.optimizer = torch.optim.AdamW(groups, weight_decay=opt_config["weight_decay"])
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.memory: deque[MemoryItem] = deque()
        self.protected: list[MemoryItem] = []
        self.success_lengths: dict[int, list[int]] = {i: [] for i in range(4)}
        self.state = RunState()
        self.phase_evaluations: list[dict[str, Any]] = []
        if resume:
            self.restore(resume)

    def restore(self, path: Path) -> None:
        """Restore all mutable training state at an episode checkpoint boundary."""
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["config"] != self.config or saved["arm"] != self.arm or saved["seed"] != self.seed:
            raise ValueError("checkpoint does not belong to this frozen run")
        self.model.load_state_dict(saved["model"])
        self._value_head().load_state_dict(saved["value_head"])
        self.optimizer.load_state_dict(saved["optimizer"])
        for optimizer_state in self.optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(DEVICE)
        self.memory = deque(saved["memory"])
        self.protected = saved["protected"]
        self.success_lengths = saved["success_lengths"]
        self.state = RunState(**saved["run_state"])
        self.phase_evaluations = saved.get("phase_evaluations", [])
        self.rng.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if torch.cuda.is_available() and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])

    def encode(self, text: str) -> Any:
        return legacy.encode_state(self.tokenizer, text)

    def policy(self, text: str) -> tuple[torch.Tensor, Any]:
        enc = self.encode(text)
        with torch.no_grad():
            logits, _ = legacy.forward_full(self.model, self._value_head(), [enc])
        probs = torch.softmax(logits[0], -1)
        probs = torch.nan_to_num(probs, nan=0.25, posinf=0.0, neginf=0.0).clamp(min=0)
        return probs / probs.sum().clamp_min(1e-12), enc

    def _value_head(self):
        # Self-imitation does not train a critic; forward_full still requires one.
        if not hasattr(self, "value_head"):
            d = self.model.encoder.config.hidden_size
            self.value_head = torch.nn.Sequential(torch.nn.LayerNorm(d), torch.nn.Linear(d, 1)).to(DEVICE)
            self.value_head.eval()
            for parameter in self.value_head.parameters():
                parameter.requires_grad_(False)
        return self.value_head

    def admit(self, pairs: list[tuple[Any, int]], phase: int) -> bool:
        length = len(pairs)
        prior = self.success_lengths[phase]
        self.success_lengths[phase].append(length)
        if len(prior) <= 4:
            return True
        ordered = sorted(prior[:-1])
        return length <= max(16, 2.0 * ordered[len(ordered) // 4])

    def push(self, pairs: list[tuple[Any, int]], phase: int, times: int = 1) -> list[MemoryItem]:
        created: list[MemoryItem] = []
        for enc, action in pairs:
            item = MemoryItem(self.state.next_experience_id, enc, action, phase, self.state.update_index)
            self.state.next_experience_id += 1
            created.append(item)
        for _ in range(times):
            self.memory.extend(created)
        while len(self.memory) > self.config["replay"]["memory_capacity"]:
            self.memory.popleft()
        return created

    def weighted_batch(self, phase: int) -> list[MemoryItem]:
        size = self.config["replay"]["batch_size"]
        batch: list[MemoryItem] = []
        protected_n = self.config["replay"]["protected_samples_per_update"]
        if len(self.protected) >= protected_n:
            batch.extend(self.rng.choices(self.protected, k=protected_n))
        memory = list(self.memory)
        if memory:
            weights = [self.config["replay"]["current_phase_weight"] if x.phase_index == phase else 1.0 for x in memory]
            batch.extend(self.rng.choices(memory, weights=weights, k=size - len(batch)))
        return batch

    def reference_batch(self, excluded_ids: set[int]) -> list[MemoryItem]:
        min_age = self.config["replay"]["reference_min_age_updates"]
        candidates = eligible_reference_items(self.memory, excluded_ids, self.state.update_index, min_age)
        size = self.config["replay"]["reference_batch_size"]
        return self.rng.choices(candidates, k=size) if candidates else []

    def loss(self, items: list[MemoryItem]) -> torch.Tensor:
        logits, _ = legacy.forward_full(self.model, self._value_head(), [x.enc for x in items])
        targets = torch.tensor([x.action for x in items], device=DEVICE)
        return F.cross_entropy(logits, targets)

    def apply_update(self, batch: list[MemoryItem], phase: int) -> dict[str, Any] | None:
        if len(batch) < 8:
            return None
        update_ids = [x.experience_id for x in batch]
        references = self.reference_batch(set(update_ids))
        reference_ids = [x.experience_id for x in references]
        if set(update_ids).intersection(reference_ids):
            raise AssertionError("reference/update exact-ID overlap")
        self.optimizer.zero_grad(set_to_none=True)
        new_loss = self.loss(batch)
        if not torch.isfinite(new_loss):
            raise FloatingPointError("non-finite current loss")
        # Reference gradient is separately computed, and never uses phase data.
        ref_loss_before = self.loss(references) if references else None
        ref_grads_cpu: list[torch.Tensor | None] = []
        if references:
            raw_ref = torch.autograd.grad(ref_loss_before, self.params, allow_unused=True)
            ref_grads_cpu = [None if g is None else g.detach().float().cpu() for g in raw_ref]
        new_loss.backward()
        grad_norm_before = float(torch.sqrt(sum((p.grad.detach().float() ** 2).sum() for p in self.params if p.grad is not None)))
        cosine, projected, change_norm, ref_norm = 0.0, False, 0.0, 0.0
        if self.arm == "valve" and references:
            dot = sum(float((p.grad.detach().float() * rg.to(DEVICE)).sum()) for p, rg in zip(self.params, ref_grads_cpu) if p.grad is not None and rg is not None)
            new_sq = sum(float((p.grad.detach().float() ** 2).sum()) for p in self.params if p.grad is not None)
            ref_sq = sum(float((rg.float() ** 2).sum()) for rg in ref_grads_cpu if rg is not None)
            ref_norm = ref_sq ** 0.5
            cosine = dot / max((new_sq * ref_sq) ** 0.5, 1e-12)
            if dot < 0 and ref_sq > 1e-12:
                scale = dot / (ref_sq + 1e-12)
                for p, rg in zip(self.params, ref_grads_cpu):
                    if p.grad is not None and rg is not None:
                        before = p.grad.detach().float().clone()
                        p.grad.sub_(rg.to(DEVICE, dtype=p.grad.dtype), alpha=scale)
                        change_norm += float(((p.grad.detach().float() - before) ** 2).sum())
                projected = True
                change_norm = change_norm ** 0.5
        torch.nn.utils.clip_grad_norm_(self.params, self.config["optimizer"]["gradient_clip_norm"])
        grad_norm_after = float(torch.sqrt(sum((p.grad.detach().float() ** 2).sum() for p in self.params if p.grad is not None)))
        before = [p.detach().cpu().clone() for p in self.params]
        self.optimizer.step()
        delta_sq, ref_dot_delta = 0.0, 0.0
        per_layer: dict[str, float] = {}
        for name, param, prior, rg in zip([n for n, p in self.model.named_parameters() if p.requires_grad], self.params, before, ref_grads_cpu or [None] * len(self.params)):
            delta = param.detach().cpu() - prior
            magnitude = float(torch.linalg.vector_norm(delta))
            per_layer[name] = magnitude
            delta_sq += magnitude * magnitude
            if rg is not None:
                ref_dot_delta += float((rg * delta).sum())
        ref_loss_after = float(self.loss(references).detach()) if references else None
        self.state.update_index += 1
        return {"event": "optimizer_update", "arm": self.arm, "phase_index": phase,
                "update_index": self.state.update_index, "update_ids": update_ids,
                "reference_ids": reference_ids, "exact_id_overlap": 0,
                "new_loss": float(new_loss.detach()), "reference_loss_before": None if ref_loss_before is None else float(ref_loss_before.detach()),
                "reference_loss_after": ref_loss_after, "gradient_norm_before": grad_norm_before,
                "gradient_norm_after": grad_norm_after, "gradient_cosine": cosine,
                "reference_gradient_norm": ref_norm, "projection_active": projected,
                "projection_change_norm": change_norm, "parameter_delta_norm": delta_sq ** 0.5,
                "reference_dot_parameter_delta": ref_dot_delta, "per_layer_delta_norm": per_layer,
                "optimizer": "AdamW", "learning_rate_multiplier": self.config["arms"][self.arm]["learning_rate_multiplier"]}

    def eval_world(self, kind: str, n: int, seed_base: int) -> dict[str, Any]:
        before = parameter_digest(self.params)
        success, steps, milestone = 0, [], 0
        for i in range(n):
            world = legacy.World(kind, random.Random(seed_base + i))
            done = False
            while not done:
                probs, _ = self.policy(corrected_ego_text(world))
                _, done = world.step(ACTIONS[int(torch.argmax(probs))])
                milestone += int((kind == "keydoor" and world.has_key) or (kind == "switch" and world.switch_on))
            success += int(world.agent == world.goal)
            steps.append(world.steps)
        after = parameter_digest(self.params)
        if before != after:
            raise AssertionError("evaluation mutated learned weights")
        return {"successes": success, "episodes": n, "success_rate": success / n,
                "mean_steps": sum(steps) / n, "milestone_events": milestone}

    def save(self) -> None:
        state = {"model": self.model.state_dict(), "value_head": self._value_head().state_dict(),
                 "optimizer": self.optimizer.state_dict(), "memory": list(self.memory), "protected": self.protected,
                 "success_lengths": self.success_lengths, "run_state": asdict(self.state), "python_rng": self.rng.getstate(),
                 "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                 "phase_evaluations": self.phase_evaluations, "config": self.config, "arm": self.arm, "seed": self.seed}
        atomic_torch_save(state, self.checkpoint_path)

    def run(self) -> dict[str, Any]:
        phases = self.config["phases"]
        if self.pilot:
            phases = [[name, kind, min(episodes, 10)] for name, kind, episodes in phases]
        for phase_index, (name, kind, episodes) in enumerate(phases):
            if phase_index < self.state.phase_index:
                continue
            recent, admitted, rejected = deque(maxlen=25), 0, 0
            phase_items: list[MemoryItem] = [item for item in self.memory if item.phase_index == phase_index]
            start_episode = self.state.episode_index if phase_index == self.state.phase_index else 0
            for episode in range(start_episode, episodes):
                world, pairs, done = legacy.World(kind, self.rng), [], False
                positions = [world.agent]
                while not done:
                    probs, enc = self.policy(corrected_ego_text(world))
                    action = int(torch.multinomial(probs, 1))
                    _, done = world.step(ACTIONS[action])
                    pairs.append((enc, action)); positions.append(world.agent)
                succeeded = world.agent == world.goal
                door_used = world.door is not None and any(pos == world.door for pos in positions)
                if succeeded and self.admit(pairs, phase_index):
                    stored = self.push(pairs, phase_index, 3 if door_used else 1)
                    phase_items.extend(stored); admitted += 1
                elif succeeded:
                    rejected += 1
                recent.append(int(succeeded))
                update_count = 0
                for _ in range(self.config["replay"]["rounds_per_episode"]):
                    update = self.apply_update(self.weighted_batch(phase_index), phase_index)
                    if update:
                        append_jsonl(self.updates_path, update); update_count += 1
                append_jsonl(self.episodes_path, {"event": "episode", "phase": name, "phase_index": phase_index,
                                                  "episode": episode + 1, "success": bool(succeeded), "steps": world.steps,
                                                  "door_used": bool(door_used), "admitted": bool(succeeded and update_count >= 0 and admitted),
                                                  "recent_success": sum(recent) / len(recent), "admitted_count": admitted,
                                                  "rejected_count": rejected, "memory_entries": len(self.memory), "optimizer_updates": update_count})
                self.state.phase_index, self.state.episode_index = phase_index, episode + 1
                if (episode + 1) % 25 == 0 or episode + 1 == episodes:
                    self.save()
            self.protected.extend(phase_items[-300:])
            observed = ("empty",) if phase_index == 0 else ("empty", "keydoor") if phase_index == 1 else ("empty", "keydoor", "switch")
            phase_eval = {world: self.eval_world(world, self.config["evaluation"]["episodes_per_world"], self.config["evaluation"]["seed_base"])
                          for world in observed}
            self.phase_evaluations.append({"phase": name, "evaluation": phase_eval})
            self.state.phase_index, self.state.episode_index = phase_index + 1, 0
            self.save()
        evaluation = {kind: self.eval_world(kind, self.config["evaluation"]["episodes_per_world"], self.config["evaluation"]["seed_base"])
                      for kind in ("empty", "keydoor", "switch")}
        self.save()
        result = {"study": "Exp18-Reconstructed", "arm": self.arm, "seed": self.seed,
                  "pilot": self.pilot, "phase_evaluations": self.phase_evaluations,
                  "evaluation": evaluation, "checkpoint": str(self.checkpoint_path)}
        (self.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "exp18_reconstructed/configs/frozen_protocol.json"))
    parser.add_argument("--arm", choices=["normal", "fixed_lr", "valve"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    print(json.dumps(Lifetime(config, args.arm, args.seed, Path(args.output), args.pilot, args.resume).run(), indent=2), flush=True)


if __name__ == "__main__":
    main()
