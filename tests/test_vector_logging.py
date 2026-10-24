import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensordict import TensorDict

from tools import Logger
from trainer import OnlineTrainer


class VectorLoggingTest(unittest.TestCase):
    def test_json_keeps_shape_and_tensorboard_records_steps_without_stale_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            logdir = Path(directory)
            logger = Logger(logdir)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    logger.scalar("train/loss", 3.0)
                    logger.vector("train/imag_wasser", np.array([[1.0], [2.0]], dtype=np.float32))
                    logger.write(5)
                    logger.vector("train/imag_wasser", np.array([[7.0]], dtype=np.float32))
                    logger.write(6)
                    logger.scalar("train/loss", 4.0)
                    logger.write(7)
            finally:
                logger._writer.close()
            rows = [json.loads(line) for line in (logdir / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["train/imag_wasser"], [[1.0], [2.0]])
            self.assertEqual(rows[1]["train/imag_wasser"], [[7.0]])
            self.assertNotIn("train/imag_wasser", rows[2])
            self.assertEqual(rows[0]["train/loss"], 3.0)
            events = EventAccumulator(str(logdir)).Reload()
            self.assertNotIn("train/imag_wasser", events.Tags()["scalars"])
            first = events.Scalars("train/imag_wasser/step_00")
            second = events.Scalars("train/imag_wasser/step_01")
            self.assertEqual([(e.step, e.value) for e in first], [(5, 1.0), (6, 7.0)])
            self.assertEqual([(e.step, e.value) for e in second], [(5, 2.0)])

    def test_trainer_routes_wasserstein_to_vector_logger(self):
        config = OmegaConf.create(
            dict(
                steps=4, pretrain=1, eval_every=0, eval_episode_num=0, video_pred_log=False,
                params_hist_log=False, batch_length=1, batch_size=1, train_ratio=1,
                update_log_every=1, action_repeat=1,
            )
        )
        envs = mock.Mock(env_num=1)
        trans = TensorDict({"reward": torch.zeros(1, 1)}, batch_size=(1,))
        envs.step.side_effect = lambda action, done: (trans.clone(), torch.zeros(1, dtype=torch.bool))
        state = TensorDict(
            {"prev_action": torch.zeros(1, 1), "stoch": torch.zeros(1, 2, 2), "deter": torch.zeros(1, 4)},
            batch_size=(1,),
        )
        agent = mock.Mock(device=torch.device("cpu"))
        agent.get_initial_state.return_value = state
        agent.act.return_value = (torch.zeros(1, 1), state)
        agent.update.return_value = {"imag_wasser": torch.tensor([[1.0], [2.0]]), "loss": torch.tensor(3.0)}
        replay = mock.Mock()
        replay.count.return_value = 0
        logger = mock.Mock()
        trainer = OnlineTrainer(config, replay, logger, None, envs, None)
        trainer.begin(agent)
        self.assertEqual(logger.vector.call_count, 2)
        for call in logger.vector.call_args_list:
            self.assertEqual(call.args[0], "train/imag_wasser")
            np.testing.assert_array_equal(call.args[1], np.array([[1.0], [2.0]]))
        self.assertTrue(any(call.args[0] == "train/loss" for call in logger.scalar.call_args_list))
        self.assertFalse(any(call.args[0] == "train/imag_wasser" for call in logger.scalar.call_args_list))


if __name__ == "__main__":
    unittest.main()
