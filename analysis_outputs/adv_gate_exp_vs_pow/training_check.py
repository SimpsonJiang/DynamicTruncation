"""Compare one CPU gradient computation on identical toy batches, without optimizer updates."""

import json
import sys
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_value_ensemble import ValueEnsembleTest
from benchmark import score


def main():
    ValueEnsembleTest.setUpClass()
    case = ValueEnsembleTest()
    results = []
    for mode in ("positive_only", "signed"):
        torch.manual_seed(17)
        kwargs = dict(trunc_enabled=True, min_horizon=1, warmup_steps=0, outscale=0.1,
                      imag_horizon=15, distance_space="symlog", advantage={"mode": mode})
        exp_agent = case.make_agent(**kwargs)
        pow_agent = case.make_agent(**kwargs)
        pow_agent.load_state_dict(exp_agent.state_dict(), strict=True)
        data, initial = case.make_batch(exp_agent)
        rng = torch.get_rng_state()

        def exp_score(u, a, *, mode, max_factor):
            return score(u, a, mode, max_factor, False)

        with (
            mock.patch("dreamer.advantage_truncation_score", side_effect=exp_score),
            mock.patch.object(exp_agent, "_value_loss", wraps=exp_agent._value_loss) as exp_loss,
        ):
            _, exp_metrics = exp_agent._cal_grad(data, initial)
        expected_rng = torch.get_rng_state()
        torch.set_rng_state(rng)

        def power_score(u, a, *, mode, max_factor):
            return score(u, a, mode, max_factor, True)

        with (
            mock.patch("dreamer.advantage_truncation_score", side_effect=power_score),
            mock.patch.object(pow_agent, "_value_loss", wraps=pow_agent._value_loss) as pow_loss,
        ):
            _, pow_metrics = pow_agent._cal_grad(data, initial)
        grad_diff = max(
            (param.grad - pow_agent._named_params[name].grad).abs().max().item()
            for name, param in exp_agent._named_params.items() if param.grad is not None
        )
        row = dict(mode=mode, shape=[8, 15, 1],
                   score_max_abs_diff=(exp_metrics["imag_trunc_score"] - pow_metrics["imag_trunc_score"]).abs().max().item(),
                   loss_max_abs_diff=max((exp_metrics[k] - pow_metrics[k]).abs().item()
                                         for k in exp_metrics if k.startswith("loss/")),
                   gradient_max_abs_diff=grad_diff,
                   mask_equal=torch.equal(exp_loss.call_args_list[0].args[4], pow_loss.call_args_list[0].args[4]),
                   return_equal=torch.equal(exp_loss.call_args_list[0].args[1], pow_loss.call_args_list[0].args[1]),
                   rng_equal=torch.equal(torch.get_rng_state(), expected_rng))
        results.append(row)
        print(json.dumps(row), flush=True)
    Path(__file__).with_name("training_results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
