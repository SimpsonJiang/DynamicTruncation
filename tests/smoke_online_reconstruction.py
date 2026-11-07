"""Bounded GPU integration checks. Run manually; never launches a benchmark.

Uses a known true core ONLY as a test fixture for the successful restore path.
The production controller always calls the learned decoder.
"""

import argparse
import json
import os
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from buffer import Buffer
from dreamer import Dreamer
from envs import make_envs
from reconstruction import Candidate, ReconstructionController


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output", default=str(Path(os.environ.get("TMPDIR", "/mnt/phx2/tmp")) / "online_reconstruction_fixture")
    )
    parser.add_argument("--compile-check", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
        config = compose(
            config_name="configs",
            overrides=[
                "env=dmc_proprio",
                "env.task=dmc_finger_spin",
                "env.env_num=1",
                "env.eval_episode_num=0",
                "env.time_limit=40",
                f"device={args.device}",
                "model.compile=false",
                "model.rep_loss=dreamer",
                "model.deter=32",
                "model.hidden=16",
                "model.units=16",
                "model.discrete=4",
                "model.rssm.stoch=4",
                "model.rssm.blocks=4",
                "model.imag_horizon=3",
                "model.dynamic_trunc.min_horizon=1",
                "model.dynamic_trunc.warmup_steps=0",
                "model.dynamic_trunc.threshold_scale=1e-6",
                "model.critic.outscale=0.01",
                "model.physics_decoder.enabled=true",
                "model.physics_decoder.hidden=16",
                "batch_size=1",
                "batch_length=3",
                "buffer.max_size=100",
                "reconstruction.enabled=true",
                "reconstruction.calibration_env_steps=8",
                "reconstruction.validation_env_steps=8",
                "reconstruction.warmup_env_steps=16",
                "reconstruction.branch_steps=3",
                "reconstruction.candidates.refresh_every_updates=1",
            ],
        )
    envs, _, obs_space, act_space = make_envs(config.env, physics=True)
    try:
        with open_dict(config.model.physics_decoder):
            config.model.physics_decoder.schema = envs.physics_schema()
        from physics_decoder import OnlinePhysicsDecoder

        if torch.device(args.device).type == "cuda":
            before = torch.cuda.get_rng_state(args.device)
            OnlinePhysicsDecoder(8, config.model.physics_decoder)
            assert torch.equal(before, torch.cuda.get_rng_state(args.device)), "Decoder initialization changed CUDA RNG"
        agent = Dreamer(config.model, obs_space, act_space).to(args.device)
        replay = Buffer(config.buffer)
        controller = ReconstructionController(config.reconstruction, envs, replay, agent, 2)
        state = agent.get_initial_state(1)
        actions = torch.zeros_like(state["prev_action"])
        done = torch.ones(1, dtype=torch.bool, device=args.device)
        checkpoint = None
        # Ordinary zero-action fixture: both calibration/validation use the same physical distribution.
        for t in range(10):
            trans, done = controller.step(actions, done, 2 * t)
            trans = trans.to(args.device)
            _, state = controller.act(trans, state)
            state["prev_action"] = actions
            trans.update(
                dict(
                    action=actions,
                    stoch=state["stoch"],
                    deter=state["deter"],
                    episode=torch.zeros(1, dtype=torch.int32, device=args.device),
                )
            )
            controller.record(trans, done)
            if t == 5:
                feature = agent.rssm.get_feat(state["stoch"], state["deter"])[0].detach().cpu().numpy()
                # depth=0 tests the seed handoff itself, independently of imagination accuracy.
                checkpoint = (
                    feature,
                    trans["physics", "core"][0].cpu().numpy().copy(),
                    trans["physics", "context"][0].cpu().numpy().copy(),
                    tuple(trans["origin"][0].cpu().tolist()),
                )
        assert controller.guard.ready, controller.guard.export()
        feature, core, vector, origin = checkpoint
        c = Candidate(
            1.0,
            agent.reconstruction_updates,
            origin,
            feature,
            np.zeros((1, actions.shape[-1])),
            controller.contexts[origin[0]][1],
            vector,
            0,
        )
        controller.pool.items = [c]
        # Credits come from counted ordinary fixture interactions, not an unaccounted extra environment.
        controller.budget.enabled = True
        # Finish the original episode normally; its remaining actions earn credits.
        while not bool(done[0]):
            trans, done = controller.step(actions, done, 100)
            trans = trans.to(args.device)
            _, state = controller.act(trans, state)
            state["prev_action"] = actions
            trans.update(
                dict(
                    action=actions,
                    stoch=state["stoch"],
                    deter=state["deter"],
                    episode=torch.zeros(1, dtype=torch.int32, device=args.device),
                )
            )
            controller.record(trans, done)
        with mock.patch.object(agent.physics_head, "forward", return_value=torch.as_tensor(core, device=args.device)):
            trans, done = controller.step(actions, done, 100)
        assert bool(trans["is_branch"][0]), controller.metrics()
        trans = trans.to(args.device)
        actions, state = controller.act(trans, state)
        trans.update(
            dict(
                action=actions,
                stoch=state["stoch"],
                deter=state["deter"],
                episode=torch.zeros(1, dtype=torch.int32, device=args.device),
            )
        )
        controller.record(trans, done)
        assert replay.has_pending()
        while not bool(done[0]):
            trans, done = controller.step(actions, done, 100)
            trans = trans.to(args.device)
            actions, state = controller.act(trans, state)
            trans.update(
                dict(
                    action=actions,
                    stoch=state["stoch"],
                    deter=state["deter"],
                    episode=torch.zeros(1, dtype=torch.int32, device=args.device),
                )
            )
            controller.record(trans, done)
        assert controller.stats["committed_branches"] == 1, controller.metrics()
        assert not replay.has_pending()
        stored = replay._buffer.storage[:]
        assert "branch_stoch" not in stored.keys() and "branch_deter" not in stored.keys()
        roots = stored["is_branch"].squeeze(-1)
        root_seeds = {key: stored[key][roots].clone() for key in ("stoch", "deter")}
        for _ in range(100):
            batch = replay.sample()
            if batch[0]["is_branch"].any():
                break
        else:
            raise AssertionError("Committed branch was never sampled")
        if args.compile_check:
            agent.physics_grad_to_wm = True
            agent._cal_grad = torch.compile(agent._cal_grad, mode="reduce-overhead")
        updates = 3 if args.compile_check else 1
        for _ in range(updates):
            fresh = (batch[0].clone(), batch[1], tuple(x.clone() for x in batch[2]))
            with mock.patch.object(replay, "sample", return_value=fresh):
                metrics = agent.update(replay, env_step=100)
            assert torch.isfinite(metrics["loss/physics"])
            assert agent.reconstruction_candidates is not None
            controller.ingest()
        assert int(agent.physics_head.norm_updates) == updates
        for key, seed in root_seeds.items():
            torch.testing.assert_close(replay._buffer.storage[:][key][roots], seed, rtol=0, atol=0)
        controller.close()
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "reconstruction_guard.json").write_text(json.dumps(controller.guard_profile(), indent=2))
        (output / "reconstruction_summary.json").write_text(json.dumps(controller.metrics(), indent=2))
        print(json.dumps({"fixture_passed": True, **controller.metrics()}, indent=2))
    finally:
        for env in envs.envs:
            env.close()


if __name__ == "__main__":
    main()
