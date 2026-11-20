"""CPU regressions for batched comparisons and score-ordered candidate admission."""

import unittest

import numpy as np
from omegaconf import OmegaConf

from reconstruction import Candidate, CandidatePool


class ScalarCandidatePool(CandidatePool):
    """Reference greedy implementation, without the removed origin quota."""

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
                y = (np.stack([old.feature[flat_stoch:] for old in self.items]) - deter_mean) / deter_scale
                xnorm, ynorm = np.linalg.norm(x), np.linalg.norm(y, axis=1)
                cosine = (y @ x) / np.maximum(ynorm * xnorm, 1e-12)
                cosine = np.where((xnorm < 1e-12) & (ynorm < 1e-12), 1.0, cosine)
                actions = np.stack([old.actions[old.depth] for old in self.items])
                rms = np.sqrt(np.mean((actions - c.actions[c.depth]) ** 2, axis=1))
                near = (cosine >= self.config.deter_cosine_threshold) & (rms <= self.config.action_rms_threshold)
                duplicates = [self.items[j] for j in np.flatnonzero(near)]
                nearest = int(np.argmax(cosine))
                self.stats["nearest_cosine_sum"] += float(cosine[nearest])
                self.stats["nearest_action_rms_sum"] += float(rms[nearest])
                self.stats["comparisons"] += 1
            self.stats["considered"] += 1
            if duplicates:
                self.stats["merged"] += 1
                if max(old.score for old in duplicates) >= c.score:
                    continue
                ids = {id(old) for old in duplicates}
                self.items = [old for old in self.items if id(old) not in ids]
            self.items.append(c)
            inserted += 1
        self.items.sort(key=lambda c: c.score, reverse=True)
        del self.items[self.config.capacity :]
        self.stats["inserted"] += inserted


class CandidatePoolTest(unittest.TestCase):
    @staticmethod
    def config(**overrides):
        values = dict(
            capacity=1024,
            top_k=32,
            ttl_updates=128,
            refresh_every_updates=8,
            max_insert=64,
            deter_cosine_threshold=0.98,
            action_rms_threshold=0.1,
        )
        values.update(overrides)
        return OmegaConf.create(values)

    @staticmethod
    def candidate(score, feature, action=0.0, *, origin=(0, 0), version=0, depth=0):
        actions = np.asarray(action)
        if actions.ndim == 0:
            actions = np.full((2, 1), actions, dtype=actions.dtype)
        return Candidate(score, version, origin, np.asarray(feature, dtype=np.float64), actions, {}, np.zeros(3), depth)

    @staticmethod
    def insert(pool, candidates, update=0, mean=None, scale=None, flat_stoch=0):
        mean = np.zeros(2) if mean is None else mean
        scale = np.ones_like(mean) if scale is None else scale
        pool.insert(candidates, update, mean, scale, flat_stoch)

    def assert_same_pool(self, expected, actual):
        self.assertEqual([id(c) for c in expected.items], [id(c) for c in actual.items])
        self.assertEqual(set(expected.stats), set(actual.stats))
        for key, value in expected.stats.items():
            if key.endswith("_sum"):
                self.assertAlmostEqual(value, actual.stats[key], delta=1e-10, msg=key)
            else:
                self.assertEqual(value, actual.stats[key], key)

    def test_distinct_candidates_share_origin_without_quota(self):
        pool = CandidatePool(self.config())
        candidates = [self.candidate(j + 1, [1, 0], j * 0.5, origin=(7, 12)) for j in range(5)]
        self.insert(pool, candidates)
        self.assertEqual([id(c) for c in pool.items], [id(c) for c in reversed(candidates)])

    def test_legacy_origin_quota_is_ignored(self):
        pool = CandidatePool(self.config(max_per_origin=1))
        self.insert(pool, [self.candidate(j + 1, [1, 0], j * 0.5) for j in range(3)])
        self.assertEqual(len(pool.items), 3)

    def test_higher_score_replaces_all_duplicates_and_keeps_dissimilar_same_origin(self):
        pool = CandidatePool(self.config())
        side = np.sqrt(1 - 0.99**2)
        a = self.candidate(2, [0.99, side], 0.04, origin=(1, 0))
        b = self.candidate(4, [0.99, -side], -0.06, origin=(2, 0))
        c = self.candidate(5, [0.6, 0.8], 0.03)
        d = self.candidate(9, [0.75, -np.sqrt(1 - 0.75**2)], 0.3)
        e = self.candidate(8, [0.5, -np.sqrt(0.75)], 0.4, origin=(3, 0))
        new = self.candidate(6, [1, 0])
        self.insert(pool, [a, b, c, d, e])
        self.insert(pool, [new])
        self.assertEqual([id(v) for v in pool.items], [id(v) for v in (d, e, new, c)])

    def test_one_higher_or_equal_duplicate_preserves_all_old_candidates(self):
        for blocking_score in (6, 8):
            with self.subTest(blocking_score=blocking_score):
                pool = CandidatePool(self.config())
                side = np.sqrt(1 - 0.99**2)
                a = self.candidate(2, [0.99, side], 0.04)
                b = self.candidate(blocking_score, [0.99, -side], -0.06)
                new = self.candidate(6, [1, 0])
                self.insert(pool, [a, b])
                self.insert(pool, [new])
                self.assertEqual([id(v) for v in pool.items], [id(b), id(a)])

    def test_acceptance_remains_sequential_inside_a_block(self):
        pool = CandidatePool(self.config())
        candidates = [self.candidate(score, [1, 0]) for score in (3, 5, 4)]
        self.insert(pool, candidates)
        self.assertEqual([id(c) for c in pool.items], [id(candidates[1])])
        self.assertEqual(pool.stats["considered"], 3)
        self.assertEqual(pool.stats["merged"], 2)

    def test_rejections_do_not_exhaust_insert_budget_across_blocks(self):
        pool = CandidatePool(self.config(max_insert=2))
        pool._COMPARE_BLOCK_SIZE = 2
        old = self.candidate(10, [1, 0])
        self.insert(pool, [old])
        incoming = [
            self.candidate(score, [1, 0], action) for score, action in ((9, 0), (8, 0), (7, 0.5), (6, 1), (5, 1.5))
        ]
        self.insert(pool, incoming)
        self.assertEqual([c.score for c in pool.items], [10, 7, 6])
        self.assertEqual(pool.stats["considered"], 5)  # One old insert, then four comparisons.
        self.assertEqual(pool.stats["inserted"], 3)

    def test_normalization_cache_is_rebuilt_on_every_call(self):
        pool = CandidatePool(self.config())
        old = self.candidate(1, [1, 0])
        new = self.candidate(2, [0, 1])
        self.insert(pool, [old])
        # Mean zero gives cosine 0; the updated mean makes these nearly parallel.
        self.insert(pool, [new], update=1, mean=np.full(2, 10.0))
        self.assertEqual([id(c) for c in pool.items], [id(new)])

    def test_invalid_unused_coordinates_and_actions_are_rejected_and_empty_call_expires(self):
        pool = CandidatePool(self.config())
        good = self.candidate(1, [0, 1, 0], depth=1)
        bad_stoch = self.candidate(10, [np.nan, 1, 0])
        bad_actions = self.candidate(9, [0, 1, 0], [[np.nan], [0]], depth=1)
        bad_scores = [self.candidate(score, [0, 1, 0]) for score in (np.nan, np.inf, -np.inf, 0, -1)]
        self.insert(pool, [bad_stoch, bad_actions, *bad_scores, good], flat_stoch=1)
        self.assertEqual([id(c) for c in pool.items], [id(good)])
        self.assertEqual(pool.stats["considered"], 1)
        self.insert(pool, [], update=128, flat_stoch=1)
        self.assertFalse(pool.items)
        self.assertEqual(pool.stats["expired"], 1)

    def test_capacity_is_trimmed_after_successful_insertion_budget(self):
        pool = CandidatePool(self.config(capacity=2, max_insert=3))
        self.insert(pool, [self.candidate(score, [1, 0], score) for score in (1, 2, 3, 4)])
        self.assertEqual([c.score for c in pool.items], [4, 3])
        self.assertEqual(pool.stats["inserted"], 3)
        self.assertEqual(pool.stats["considered"], 3)
        self.insert(pool, [self.candidate(5, [1, 0], 5)])
        self.assertEqual([c.score for c in pool.items], [5, 4])

    def test_cosine_and_action_thresholds_are_inclusive(self):
        cases = (
            (0.98, 0.1, True),
            (0.98 - 1e-8, 0.1, False),
            (0.98 + 1e-8, 0.1, True),
            (1, 0.1 - 1e-8, True),
            (1, 0.1 + 1e-8, False),
        )
        for cosine, action, duplicate in cases:
            with self.subTest(cosine=cosine, action=action):
                pool = CandidatePool(self.config())
                self.insert(pool, [self.candidate(1, [1, 0])])
                self.insert(pool, [self.candidate(2, [cosine, np.sqrt(1 - cosine**2)], action)])
                self.assertEqual(len(pool.items), 1 if duplicate else 2)

    def test_zero_and_tiny_vector_special_case(self):
        for magnitude, expected_count in ((1e-14, 1), (1e-8, 2)):
            with self.subTest(magnitude=magnitude):
                pool = CandidatePool(self.config())
                self.insert(pool, [self.candidate(1, [0, 0])])
                self.insert(pool, [self.candidate(2, [magnitude, 0])])
                self.assertEqual(len(pool.items), expected_count)

    def test_future_action_dtype_does_not_change_current_threshold(self):
        pools = [cls(self.config()) for cls in (ScalarCandidatePool, CandidatePool)]
        old = self.candidate(1, [1, 0], np.float32(0.1))
        new = self.candidate(2, [1, 0], np.float32(0))
        later = self.candidate(1.5, [0, 1], np.float64(0))
        for pool in pools:
            self.insert(pool, [old])
            self.insert(pool, [new, later])
        self.assert_same_pool(*pools)
        self.assertEqual([id(c) for c in pools[1].items], [id(new), id(later)])

    def test_randomized_parity_with_scalar_reference_and_changing_statistics(self):
        for seed in range(12):
            rng = np.random.default_rng(seed)
            cfg = self.config(capacity=16, max_insert=8, ttl_updates=5)
            pools = [cls(cfg) for cls in (ScalarCandidatePool, CandidatePool)]
            pools[1]._COMPARE_BLOCK_SIZE = 4
            features = rng.normal(size=(10, 10)).astype(np.float32)
            actions = rng.uniform(-1, 1, size=(10, 4, 3))
            for update in (0, 1, 6):
                incoming = []
                for j in range(32):
                    base = int(rng.integers(10))
                    feature = features[base] + rng.normal(0, 0.001, 10).astype(np.float32)
                    action = (actions[base] + rng.normal(0, 0.001, (4, 3))).astype(np.float32 if j % 2 else np.float64)
                    incoming.append(
                        Candidate(float(rng.integers(-1, 12)), update, (base, 0), feature, action, {}, np.zeros(3), 1)
                    )
                mean, scale = rng.normal(0, 0.1, 8), rng.uniform(0.5, 2, 8)
                for pool in pools:
                    self.insert(pool, incoming, update, mean, scale, 2)
                with self.subTest(seed=seed, update=update):
                    self.assert_same_pool(*pools)


if __name__ == "__main__":
    unittest.main()
