"""Small CQ train/restore/imagination pipeline check; no persistent test images."""

import argparse
import json
import tempfile
from pathlib import Path

import torch
from PIL import Image

from .common import provenance, select_entry, write_json
from .pipeline import collect, fit
from .posterior_restore import evaluate_model
from .prediction import prediction_check
from .reconstruction_preflight import CQ_GROUP
from .stage5 import evaluate
from .stage5_report import summarize_campaign


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", required=True)
    args = parser.parse_args()
    root = Path(args.preflight)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(2 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    entry = select_entry(root / "cq_manifest.json", "finger_spin_s1")
    with tempfile.TemporaryDirectory(prefix="cq_stage45_smoke_", dir="/tmp") as temporary:
        stage3 = Path(temporary) / "stage3"
        base = stage3 / entry["id"]
        collect(entry, base / "data", 52000, 3, 24, "cuda:0")
        for source in ("features", "observations"):
            fit(base / "data", base / source, source, 17, 2, 8, 8, 1e-3, "cuda:0")
            prediction_check(base / "data", base / source / "decoder.pt", base / f"test_{source}.json", "cuda:0", 8)
        write_json(
            stage3 / "protocol.json", {"entries": [entry["id"]], "checkpoint_group": CQ_GROUP, "modality": "proprio"}
        )
        fourth = evaluate_model(
            stage3,
            entry["id"],
            Path(temporary) / "stage4",
            [0, 3],
            2,
            ["features", "observations"],
            checkpoint_group=CQ_GROUP,
        )
        if not fourth["reference_passed"]:
            raise ValueError("CQ posterior restoration reference failed")
        stage5 = Path(temporary) / "stage5"
        fifth = evaluate(
            stage3, entry["id"], stage5 / "cq" / entry["id"], CQ_GROUP, points=[3], test_limit=1, example_start=3
        )
        write_json(stage5 / "protocol.json", {"jobs": [{"key": f"cq/{entry['id']}"}]})
        rendered = summarize_campaign(stage5)
        if rendered["images"] != 1:
            raise ValueError("Expected one smoke task sheet")
        image_path = next((stage5 / "first_step_images").glob("*.png"))
        with Image.open(image_path) as image:
            image_size = list(image.size)
            image.verify()
        if fifth["reference_failures"] or any(c["failure"] for c in fifth["cases"] if c["source"] == "core64_control"):
            raise ValueError("Imagined-action true-core control failed")
        report = {
            "passed": True,
            "cq_entry": entry,
            "episodes": 3,
            "steps": 24,
            "epochs": 2,
            "stage4_reference_passed": True,
            "stage5_summary": fifth["summary"],
            "ema_unchanged": fifth["ema_unchanged"],
            "image_shape": image_size,
            "temporary_images_removed": True,
            "provenance": provenance(),
            "note": "Pipeline check only; not evidence of reconstruction quality.",
        }
        write_json(root / "smoke.json", report)
    print(json.dumps({"passed": True, "output": str(root / "smoke.json")}), flush=True)


if __name__ == "__main__":
    main()
