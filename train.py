import atexit
import copy
import pathlib
import sys
import warnings

import hydra
import torch

import tools
from buffer import Buffer, MixedReplay
from branch_collection import BranchCollector, validate_branch_config
from dreamer import Dreamer
from envs import make_envs
from trainer import OnlineTrainer

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))
# torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(config):
    capture_state, capture_snapshot = validate_branch_config(config)
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
    branch_collector = None
    if config.branch_collection.mode != "off":
        branch_buffer_config = copy.deepcopy(config.buffer)
        branch_buffer_config.max_size = config.branch_collection.replay_capacity
        replay_buffer = MixedReplay(
            replay_buffer, Buffer(branch_buffer_config), config.branch_collection.replay_fraction
        )
        branch_collector = BranchCollector(config.branch_collection, config.env, config.device, config.seed)

    print("Create envs.")
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env, capture_state, capture_snapshot)
    state_spec = train_envs.envs[0].get_state_spec()() if config.model.state_decoder.enabled else None

    print("Simulate agent.")
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
        state_spec=state_spec,
        collect_branch_candidates=config.branch_collection.mode == "imagined_state",
    ).to(config.device)

    policy_trainer = OnlineTrainer(
        config.trainer, replay_buffer, logger, logdir, train_envs, eval_envs,
        branch_collector=branch_collector,
    )
    try:
        policy_trainer.begin(agent)
    finally:
        if branch_collector is not None:
            branch_collector.close()

    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
    }
    torch.save(items_to_save, logdir / "latest.pt")


if __name__ == "__main__":
    main()
