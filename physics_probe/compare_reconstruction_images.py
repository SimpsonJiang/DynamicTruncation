"""Pair physical-state reconstruction and native decoder pixels from existing galleries."""

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage5", required=True)
    args = parser.parse_args()
    root = Path(args.stage5).resolve()
    physical = root / "physics_reconstruction_images"
    native = root / "native_decoder_images"
    output = root / "comparison_images"
    output.mkdir(exist_ok=False)
    import torch

    torch.set_num_threads(1)
    states = torch.load(physical / "physical_states.pt", weights_only=True, map_location="cpu")
    records = torch.load(native / "native_observations.pt", weights_only=True, map_location="cpu")
    pm = json.loads((physical / "manifest.json").read_text())
    nm = json.loads((native / "manifest.json").read_text())
    audits = {m["key"]: m for m in pm["models"]}
    native_audits = {m["key"]: m for m in nm["models"]}
    groups = defaultdict(list)
    for key, record in records.items():
        torch.testing.assert_close(states[key]["features"], record["features"], rtol=0, atol=0)
        assert states[key]["depths"] == record["depths"] == [8, 10, 12, 14]
        a, b = audits[key], native_audits[key]
        for field in ("entry", "episode_id", "start"):
            assert a[field] == b[field], (key, field)
        for left, right in zip(a["details"], b["details"]):
            assert left["imag_depth"] == right["imag_depth"]
            assert left["absolute_step"] == right["absolute_step"]
            assert left["actions_executed"] == right["actions_executed"] == 0
        groups[(record["job"]["group"], record["entry"]["task"])].append(record)

    def font(size):
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)

    images, sources = [], {}
    for (group, task), rows in sorted(groups.items()):
        rows.sort(key=lambda r: r["entry"]["model_seed"])
        p = physical / f"{group}__{task}__physics_imag8_10_12_14.png"
        n = native / f"{group}__{task}__native_decoder_imag8_10_12_14.png"
        for path in (p, n):
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        left, right = Image.open(p).convert("RGB"), Image.open(n).convert("RGB")
        assert left.size == right.size == (1280, 1302)
        # Preserve every 320x320 source pixel; only arrange and label the paired cells.
        pair_width, header, row_height = 656, 120, 424
        canvas = Image.new("RGB", (4 * pair_width, header + 3 * row_height), "#eef2f6")
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 10), f"{group} | {task}", font=font(28), fill="#17212b")
        draw.text((16, 49), "LEFT: feature -> physical state -> simulator    |    RIGHT: native decoder", font=font(25), fill="#17212b")
        vision = rows[0]["policy_modality"] == "vision"
        note = (
            "Native vision: Dreamer image decoder RGB (64x64 enlarged)."
            if vision else "Native proprio: Dreamer proprio decoder -> observation-to-state decoder -> simulator."
        )
        draw.text((16, 86), note + "  Same feature; no extra action.", font=font(22), fill="#465565")
        pixel_checks = []
        for row, record in enumerate(rows):
            y = header + row * row_height
            for col, depth in enumerate(record["depths"]):
                x = col * pair_width
                draw.text((x + 8, y + 4), f"Seed {record['entry']['model_seed']} | IMAG +{depth} | t={record['start']+depth}", font=font(23), fill="#17212b")
                draw.text((x + 8, y + 37), "Physical-state reconstruction", font=font(18), fill="#125c8a")
                draw.text((x + 336, y + 37), "Native decoder", font=font(18), fill="#9a4311")
                source_y = 90 + row * 404 + 61
                box = (col * 320, source_y, (col + 1) * 320, source_y + 320)
                for offset, source in ((0, left), (328, right)):
                    cell = source.crop(box)
                    destination = (x + offset, y + 66)
                    canvas.paste(cell, destination)
                    pixel_checks.append((destination, cell.tobytes()))
                draw.text((x + 8, y + 393), f"Same feature | episode {record['episode_id']} | origin {record['start']}", font=font(18), fill="#465565")
        name = f"{group}__{task}__comparison_imag8_10_12_14.png"
        canvas.save(output / name)
        with Image.open(output / name) as saved:
            for (x, y), pixels in pixel_checks:
                assert saved.crop((x, y, x + 320, y + 320)).tobytes() == pixels
        images.append(name)
    assert len(images) == 10 and len(records) == 30
    (output / "manifest.json").write_text(json.dumps(dict(
        kind="paired_physics_vs_native_decoder", images=images, models=30, paired_states=120,
        source_sha256=sources, features_verified_identical=True,
        pixels_preserved_exactly=True, actions_executed=0,
    ), indent=2) + "\n")
    (output / "README.md").write_text(
        "# 同一 feature 的两条重建链路：并排对比\n\n"
        "**每一对：左侧是 feature→物理状态→仿真；右侧是原生 decoder。**\n\n"
        "每张图三行对应三个seed，四组列依次为想象第8、10、12、14步。"
        "左右使用逐值相同的feature、同一个测试episode和起点，均没有额外执行动作。"
        "共10张任务汇总图、120对重建，直接并排排列已有图片的原始像素，不改变视角或状态。\n\n"
        "右侧vision直接显示Dreamer原生image decoder输出；右侧proprio使用"
        "原生proprio decoder→已训练的observation→物理状态decoder→仿真截图，因此包含第二个decoder的误差。\n\n"
        + "\n\n".join(f"![{name}]({name})" for name in images) + "\n"
    )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Paired reconstruction comparison</title>'
        '<style>body{font-family:sans-serif;background:#eef2f6}img{width:100%;margin:16px 0}</style>'
        '<h1>左：feature → 物理状态 → 仿真；右：原生 decoder</h1>'
        '<p>每组左右使用相同feature；想象第8、10、12、14步。点击图片查看原尺寸。</p>'
        + ''.join(f'<a href="{name}"><img src="{name}" alt="{name}"></a>' for name in images)
    )
    shutil.copy2(__file__, output / Path(__file__).name)
    print(f"Created {len(images)} comparison sheets, 120 matched pairs, exact source pixels.")


if __name__ == "__main__":
    main()
