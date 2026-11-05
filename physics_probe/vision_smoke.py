"""Small real-GPU integration check; temporary datasets/images are removed on exit."""

import argparse
import gc
import shutil
import tempfile
import time
from pathlib import Path

import torch
from PIL import Image

from .common import provenance, select_entry, sha256, write_json
from .decoder import ImageStateDecoder
from .pipeline import collect, fit
from .posterior_restore import evaluate_model
from .prediction import prediction_check
from .render_first_step import render_campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(2 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    entry = select_entry(args.manifest, "finger_spin_s0")
    with tempfile.TemporaryDirectory(prefix="vision_probe_smoke_", dir="/tmp") as temporary:
        root = Path(temporary)
        stage3, stage4 = root / "stage3", root / "stage4"
        stage4.mkdir()
        base = stage3 / entry["id"]
        meta = collect(entry, base / "data", 51000, 3, 8, "cuda:0")
        if meta["policy_modality"] != "vision" or meta["policy_input_keys"] != ["image"]:
            raise ValueError("Smoke collection did not use a vision policy")
        gc.collect()
        torch.cuda.empty_cache()
        inputs, fits, tests = {}, {}, {}
        for source in ("features", "images"):
            trained = fit(base / "data", base / source, source, 17, 2, 4, 8, 1e-3, "cuda:0")
            fits[source] = {k: trained[k] for k in ("decoder_dtype", "label_dtype", "epochs", "history", "split")}
            tests[source] = prediction_check(
                base / "data", base / source / "decoder.pt", base / f"test_{source}.json", "cuda:0", 4
            )["test_loss"]
            for file in (base / "data/dataset.json", base / source / "decoder.pt"):
                inputs[str(file)] = sha256(file)
        # Exercise the actual formal CNN width/batch and double-precision Adam under the 2 GiB cap.
        production = ImageStateDecoder((64, 64, 3), meta["schema"], hidden=256).to("cuda:0")
        pixels = torch.randint(0, 256, (256, 64, 64, 3), dtype=torch.uint8, device="cuda:0")
        targets = torch.randn(256, meta["schema"]["nq"] + meta["schema"]["nv"], dtype=torch.float64, device="cuda:0")
        production.normalize_from_training(pixels, targets)
        optimizer = torch.optim.Adam(production.parameters(), lr=1e-3)
        torch.cuda.reset_peak_memory_stats()
        timings = []
        for index in range(3):
            torch.cuda.synchronize()
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss = production.loss(pixels, targets)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite production-batch CNN loss")
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            if index:
                timings.append(time.monotonic() - started)
        production_check = {
            "batch_size": 256,
            "hidden": 256,
            "dtype": "float64",
            "timed_step_seconds": timings,
            "cuda_peak_bytes": torch.cuda.max_memory_allocated(),
        }
        del production, optimizer, loss, pixels, targets
        gc.collect()
        torch.cuda.empty_cache()
        protocol = {
            "checkpoint_group": "VE_wasser_H15",
            "entries": [entry["id"]],
            "modality": "vision",
            "sources": ["features", "images"],
        }
        write_json(stage3 / "protocol.json", protocol)
        report = evaluate_model(
            stage3,
            entry["id"],
            stage4 / entry["id"],
            [0, 3],
            2,
            ["features", "images"],
            checkpoint_group="VE_wasser_H15",
        )
        if not report["reference_passed"] or any(s["failed"] for s in report["summary"].values()):
            raise ValueError("Smoke restoration failed")
        write_json(
            stage4 / "protocol.json", {**protocol, "points": [0, 3], "source_root": str(stage3), "input_sha256": inputs}
        )
        snapshot = stage4 / "source_snapshot/physics_probe"
        snapshot.mkdir(parents=True)
        for name in ("environment.py", "decoder.py", "prediction.py", "posterior_restore.py"):
            shutil.copy2(Path(__file__).parent / name, snapshot / name)
        images = root / "images"
        render_campaign(stage4, images, 3, frame_all=True, checkpoint_group="VE_wasser_H15")
        saved_images = list(images.glob("*.png"))
        if len(saved_images) != 1:
            raise ValueError("Smoke renderer should save only one task summary image")
        with Image.open(saved_images[0]) as image:
            image.verify()
        write_json(
            args.output,
            {
                "passed": True,
                "entry": entry,
                "episodes": 3,
                "steps_per_episode": 8,
                "fit_epochs": 2,
                "fits": fits,
                "test_losses": tests,
                "restoration": report["summary"],
                "temporary_images_removed": True,
                "production_batch_check": production_check,
                "provenance": provenance(),
                "note": "Pipeline integration only; not evidence of scientific reconstruction quality.",
            },
        )
    print(
        f"PASS vision collection -> two float64 decoders -> held-out test -> restore -> first-step image: {args.output}"
    )


if __name__ == "__main__":
    main()
