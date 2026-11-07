"""Host-side candidate selection and budget accounting, outside torch.compile."""

from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch

from envs.physics_protocol import PhysicsContext


@dataclass
class Candidate:
    score: float
    version: int
    origin: tuple
    feature: np.ndarray
    actions: np.ndarray
    context: dict
    context_vector: np.ndarray
    depth: int


class CandidatePool:
    def __init__(self, config, seed=0):
        self.config = config
        if (
            min(
                config.capacity,
                config.top_k,
                config.ttl_updates,
                config.refresh_every_updates,
                config.max_insert,
                config.max_per_origin,
            )
            < 1
        ):
            raise ValueError("Candidate pool limits must be positive")
        self.rng = np.random.default_rng(seed)
        self.items = []
        self.stats = Counter()

    def expire(self, update):
        keep = [c for c in self.items if update - c.version < self.config.ttl_updates]
        self.stats["expired"] += len(self.items) - len(keep)
        self.items = keep

    def insert(self, candidates, update, deter_mean, deter_scale, flat_stoch):
        self.expire(update)
        inserted = 0
        for c in sorted(candidates, key=lambda c: c.score, reverse=True):
            if inserted >= self.config.max_insert:
                break
            if not np.isfinite(c.score) or c.score <= 0:
                continue
            if not np.isfinite(c.feature).all() or not np.isfinite(c.actions).all():
                continue
            duplicates = []
            x = (c.feature[flat_stoch:] - deter_mean) / deter_scale
            if self.items:
                # One vectorized comparison, rather than O(capacity) Python/BLAS calls.
                y = (np.stack([old.feature[flat_stoch:] for old in self.items]) - deter_mean) / deter_scale
                xnorm, ynorm = np.linalg.norm(x), np.linalg.norm(y, axis=1)
                cosine = (y @ x) / np.maximum(ynorm * xnorm, 1e-12)
                cosine = np.where((xnorm < 1e-12) & (ynorm < 1e-12), 1.0, cosine)
                actions = np.stack([old.actions[old.depth] for old in self.items])
                rms = np.sqrt(np.mean((actions - c.actions[c.depth]) ** 2, axis=1))
                near = (cosine >= self.config.deter_cosine_threshold) & (rms <= self.config.action_rms_threshold)
                duplicates = [self.items[i] for i in np.flatnonzero(near)]
                nearest = int(np.argmax(cosine))
                self.stats["nearest_cosine_sum"] += float(cosine[nearest])
                self.stats["nearest_action_rms_sum"] += float(rms[nearest])
                self.stats["comparisons"] += 1
            self.stats["considered"] += 1
            if duplicates:
                self.stats["merged"] += 1
                if max(v.score for v in duplicates) >= c.score:
                    continue
                ids = {id(v) for v in duplicates}
                self.items = [v for v in self.items if id(v) not in ids]
            same = [v for v in self.items if v.origin == c.origin]
            if len(same) >= self.config.max_per_origin:
                worst = min(same, key=lambda v: v.score)
                if worst.score >= c.score:
                    continue
                self.items = [v for v in self.items if v is not worst]
            self.items.append(c)
            inserted += 1
        self.items.sort(key=lambda c: c.score, reverse=True)
        del self.items[self.config.capacity :]
        self.stats["inserted"] += inserted

    def choose(self, update, shortlist=None):
        self.expire(update)
        if shortlist is None:
            shortlist = self.items[: self.config.top_k]
        if not shortlist:
            return None
        candidate = shortlist.pop(int(self.rng.integers(len(shortlist))))
        self.items = [c for c in self.items if c is not candidate]
        return candidate


@dataclass
class Reservation:
    reserved: int
    used: int = 0


class ReconstructionBudget:
    def __init__(self, config):
        self.config = config
        self.enabled = False
        self.credit = 0.0
        self.active = {}
        self.consecutive = Counter()
        self.stats = Counter()
        if not 0 < config.ratio < 1 or config.branch_steps < 1 or config.max_active < 1:
            raise ValueError("Invalid reconstruction budget")

    def account(self, ordinary=0, reconstructed=0):
        self.stats["ordinary_steps"] += ordinary
        self.stats["reconstruction_steps"] += reconstructed
        if self.enabled:
            self.credit = min(
                self.config.credit_cap_steps, self.credit + ordinary * self.config.ratio / (1 - self.config.ratio)
            )

    def allowance(self, slot, remaining):
        if not self.enabled:
            return 0, "warmup_or_guard"
        if len(self.active) >= self.config.max_active:
            return 0, "concurrency"
        if self.consecutive[slot] >= self.config.max_consecutive:
            return 0, "consecutive"
        count = min(self.config.branch_steps, remaining, int(self.credit + 1e-9))
        return (count, None) if count > 0 else (0, "credit")

    def reserve(self, slot, length):
        if slot in self.active or length > self.credit + 1e-9 or length < 1:
            raise RuntimeError("Invalid branch reservation")
        self.active[slot] = Reservation(length)
        self.credit -= length
        self.consecutive[slot] += 1

    def executed(self, slot):
        self.active[slot].used += 1
        if self.active[slot].used > self.active[slot].reserved:
            raise RuntimeError("Branch exceeded reserved budget")

    def finish(self, slot):
        reservation = self.active.pop(slot)
        self.credit = min(self.config.credit_cap_steps, self.credit + reservation.reserved - reservation.used)

    def ordinary_reset(self, slot, reason):
        self.consecutive[slot] = 0
        self.stats["fallback/" + str(reason)] += 1


class ReconstructionController:
    """Coordinate existing environment slots; no extra collection environments."""

    def __init__(self, config, envs, replay, agent, action_repeat):
        from reconstruction_guard import GuardCalibration

        agent.configure_reconstruction(config)
        replay.enable_transactions()
        self.config, self.envs, self.replay, self.agent = config, envs, replay, agent
        self.repeat = int(action_repeat)
        self.pool, self.budget = CandidatePool(config.candidates, config.seed), ReconstructionBudget(config)
        self.guard = GuardCalibration(
            envs.setup_reconstruction(),
            config.calibration_env_steps,
            config.validation_env_steps,
            config.max_false_reject,
        )
        self.contexts, self.slot_origins, self.next_origin = {}, {}, 0
        self.active_candidates, self.stats = {}, Counter()
        self.sent_thresholds = False

    def _request(self, candidate, length):
        timeline = PhysicsContext.from_vector(candidate.context_vector)
        advanced = timeline.advance(candidate.depth, candidate.actions, candidate.context)
        core = self.agent.decode_physics(candidate.feature).cpu().numpy()
        return dict(context=candidate.context, context_vector=advanced.to_vector(), core=core, length=length)

    def _schedule(self, done, env_step):
        requests = [None] * self.envs.env_num
        self.pool.expire(self.agent.reconstruction_updates)
        shortlist = self.pool.items[: self.config.candidates.top_k]
        self.budget.enabled = bool(self.guard.ready and env_step >= self.config.warmup_env_steps)
        for slot, available in enumerate(done.tolist()):
            if not available:
                continue
            length, reason = self.budget.allowance(slot, self.config.branch_steps)
            if length:
                candidate = self.pool.choose(self.agent.reconstruction_updates, shortlist)
                if candidate is None:
                    reason = "no_candidate"
                else:
                    timeline = PhysicsContext.from_vector(candidate.context_vector)
                    length = min(length, timeline.remaining_steps(candidate.context, candidate.depth))
                    if length > 0:
                        requests[slot] = self._request(candidate, length)
                        self.stats["restore_attempts"] += 1
                        self.budget.reserve(slot, length)
                        self.active_candidates[slot] = candidate
                    else:
                        reason = "source_time_limit"
            if requests[slot] is None:
                self.budget.ordinary_reset(slot, reason)
        return requests

    def step(self, actions, done, env_step):
        requests = self._schedule(done, env_step)
        ordinary, reconstructed = 0, 0
        for slot, available in enumerate(done.tolist()):
            if not available:
                if slot in self.budget.active:
                    self.budget.executed(slot)
                    reconstructed += 1
                else:
                    ordinary += 1
        thresholds = None
        if self.guard.thresholds is not None and not self.sent_thresholds:
            thresholds = dict(signature=self.guard.profile["signature"], values=self.guard.thresholds)
            self.sent_thresholds = True
        phase = self.guard.phase
        trans, next_done, reports, contexts = self.envs.reconstruction_step(actions, done, requests, phase, thresholds)
        self.budget.account(ordinary, reconstructed)
        self._process_results(trans, done, reports, contexts, phase)
        self.reports = reports
        return trans, next_done

    def _process_results(self, trans, done, reports, contexts, phase):
        roots, recon = [], []
        for slot, report in enumerate(reports):
            if report.get("guard_failure"):
                self.guard.failure = report["guard_failure"]
            if not done[slot] and not report["branch"]:
                self.guard.observe(report, self.repeat, phase)
            if report["failure"]:
                self.stats["reject/" + report["failure"]] += 1
                if slot in self.budget.active:
                    started = self.replay.has_pending(slot)
                    self.replay.discard(slot)
                    self.budget.finish(slot)
                    self.active_candidates.pop(slot, None)
                    self.stats["discarded_branches" if started else "rejected_restores"] += 1
                    self.budget.ordinary_reset(slot, "rejected")
            root = report["root"] and not report["failure"]
            is_recon = report["branch"] and not report["failure"]
            roots.append(root)
            recon.append(is_recon)
            self.stats["max_penetration"] = max(self.stats["max_penetration"], report["max_depth"])
            if root:
                self.replay.reserve(slot)
                self.stats["started_branches"] += 1
            if contexts[slot] is not None:
                self.contexts[self.next_origin] = (self.replay.generation, contexts[slot])
                self.slot_origins[slot] = self.next_origin
                self.next_origin += 1
        trans["is_branch"] = torch.tensor(roots, dtype=torch.bool)[:, None]
        trans["is_recon"] = torch.tensor(recon, dtype=torch.bool)[:, None]
        trans["physics_valid"] = ~trans["is_branch"]
        trans["origin"] = torch.tensor(
            [
                [
                    self.active_candidates[i].origin[0] if recon[i] else self.slot_origins.get(i, -1),
                    PhysicsContext.from_vector(trans["physics", "context"][i].numpy()).policy_steps,
                ]
                for i in range(self.envs.env_num)
            ],
            dtype=torch.int64,
        )

    @property
    def guard_status(self):
        return self.guard.phase, self.guard.ready, self.guard.failure

    def guard_profile(self):
        return self.guard.export()

    def act(self, trans, state):
        root = trans["is_branch"].reshape(-1)
        candidates = [self.active_candidates[i] for i in root.nonzero().flatten().tolist()]
        features = np.stack([c.feature for c in candidates]) if candidates else None
        actions = np.stack([c.actions[c.depth] for c in candidates]) if candidates else None
        action, result = self.agent.act_with_seeds(trans, state, features, actions)
        trans["branch_stoch"] = torch.where(root[:, None, None], result["stoch"], 0).detach()
        trans["branch_deter"] = torch.where(root[:, None], result["deter"], 0).detach()
        return action, result

    def record(self, trans, done):
        self.replay.add_transition(trans.detach())
        seen_origins = set(self.slot_origins.values()) | {c.origin[0] for c in self.active_candidates.values()}
        for key in seen_origins:
            if key in self.contexts:
                self.contexts[key] = (self.replay.generation, self.contexts[key][1])
        for slot in list(self.budget.active):
            if done[slot]:
                if self.replay.commit(slot):
                    self.stats["committed_branches"] += 1
                else:
                    self.stats["discarded_branches"] += 1
                    self.stats["reject/ring_overwrite"] += 1
                self.budget.finish(slot)
                self.active_candidates.pop(slot, None)
        # Episode contexts are shared, not duplicated with every image/latent row.
        horizon = self.replay.retention_frames(self.envs.env_num) + self.config.branch_steps
        live = set(self.slot_origins.values()) | {c.origin[0] for c in self.pool.items}
        for key, (created, _) in list(self.contexts.items()):
            if key not in live and self.replay.generation - created > horizon:
                del self.contexts[key]

    def ingest(self):
        packet = self.agent.take_reconstruction_candidates()
        if packet is None:
            return
        packet = {k: v.detach().cpu().numpy() for k, v in packet.items()}
        candidates = []
        for i, score in enumerate(packet["score"]):
            origin = tuple(map(int, packet["origin"][i]))
            if not np.isfinite(score) or origin[0] not in self.contexts:
                continue
            candidates.append(
                Candidate(
                    float(score),
                    self.agent.reconstruction_updates,
                    origin,
                    packet["feature"][i].copy(),
                    packet["actions"][i].copy(),
                    self.contexts[origin[0]][1],
                    packet["context"][i].copy(),
                    int(packet["depth"][i]),
                )
            )
        self.pool.insert(candidates, self.agent.reconstruction_updates, **self.agent.candidate_distance_statistics())

    def metrics(self):
        result = {**self.stats, **self.budget.stats}
        result.update({"candidate/" + k: v for k, v in self.pool.stats.items()})
        result["candidate/expiry_rate"] = self.pool.stats["expired"] / max(self.pool.stats["inserted"], 1)
        result["candidate/merge_rate"] = self.pool.stats["merged"] / max(self.pool.stats["considered"], 1)
        total = self.budget.stats["ordinary_steps"] + self.budget.stats["reconstruction_steps"]
        result["interaction_fraction"] = self.budget.stats["reconstruction_steps"] / max(total, 1)
        result["credit"] = self.budget.credit
        result["active"] = len(self.budget.active)
        result["candidate/count"] = len(self.pool.items)
        result["guard_ready"] = float(self.guard.ready)
        result["guard_calibration_env_steps"] = self.guard.calibration_count
        result["guard_validation_env_steps"] = self.guard.validation_count
        result["candidate/nearest_cosine"] = self.pool.stats["nearest_cosine_sum"] / max(
            self.pool.stats["comparisons"], 1
        )
        result["candidate/nearest_action_rms"] = self.pool.stats["nearest_action_rms_sum"] / max(
            self.pool.stats["comparisons"], 1
        )
        result["guard_false_reject_rate"] = self.guard.rejected / max(
            self.guard.profile.get("validation_policy_steps", 0), 1
        )
        result["discard_fraction"] = self.stats["discarded_branches"] / max(self.stats["started_branches"], 1)
        return {"reconstruction/" + k: v for k, v in result.items()}

    def close(self):
        for slot in list(self.budget.active):
            self.replay.discard(slot)
            self.budget.finish(slot)
            self.stats["discarded_branches"] += 1
