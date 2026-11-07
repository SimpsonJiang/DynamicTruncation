import atexit
import pathlib
import sys
import warnings

import hydra
import torch

import tools
from buffer import Buffer
from dreamer import Dreamer
from envs import make_envs
from trainer import OnlineTrainer

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))
# torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    # Mirror stdout/stderr to a file under logdir while keeping console output.
    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    print("Logdir", logdir)

    logger = tools.Logger(logdir)
    # save config
    logger.log_hydra_config(config)

    replay_buffer = Buffer(config.buffer)

    print("Create envs.")
    physics_enabled = bool(config.model.get("physics_decoder", {}).get("enabled", False))
    if config.get("reconstruction", {}).get("enabled", False) and not physics_enabled:
        raise ValueError("reconstruction.enabled requires model.physics_decoder.enabled")
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env, physics=physics_enabled)
    if physics_enabled:
        from omegaconf import open_dict

        with open_dict(config.model.physics_decoder):
            config.model.physics_decoder.schema = train_envs.physics_schema()
        logger.log_hydra_config(config)

    print("Simulate agent.")
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
    ).to(config.device)

    policy_trainer = OnlineTrainer(config.trainer, replay_buffer, logger, logdir, train_envs, eval_envs)
    policy_trainer.begin(agent)

    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
    }
    if physics_enabled:
        items_to_save["physics_schema"] = agent.physics_head.codec.schema
        items_to_save["physics_dtype"] = "float64"
    torch.save(items_to_save, logdir / "latest.pt")


if __name__ == "__main__":
    main()
