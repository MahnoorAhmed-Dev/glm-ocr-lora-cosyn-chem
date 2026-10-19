"""Build disjoint train / eval sets from the CoSyn-400K `chemical` subset.

Writes ShareGPT-style json + PNGs into LLaMA-Factory/data/:
    cosyn_chem_train.json, cosyn_chem_train/*.png
    cosyn_chem_eval.json,  cosyn_chem_eval/*.png

Usage:
    python scripts/prepare_data.py --n-train 200 --n-eval 30 --seed 42
"""
import argparse
import json
from pathlib import Path

import datasets

PROMPT = "<image>Code Generation:"


def write_split(name, rows, out_dir):
    img_dir = out_dir / f"cosyn_chem_{name}"
    img_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i, row in enumerate(rows, 1):
        rel = f"cosyn_chem_{name}/{i}.png"
        row["image"].convert("RGB").save(out_dir / rel)
        records.append({
            "messages": [
                {"role": "user", "content": PROMPT},
                {"role": "assistant", "content": f"```rdkit\n{row['code'].strip()}\n```"},
            ],
            "images": [rel],
        })
    with open(out_dir / f"cosyn_chem_{name}.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"{name}: {len(records)} samples -> {out_dir / f'cosyn_chem_{name}.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("LLaMA-Factory/data"))
    args = ap.parse_args()

    ds = datasets.load_dataset("allenai/CoSyn-400K", "chemical", split="train", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=2000)

    need = args.n_train + args.n_eval
    seen, picked = set(), []
    for row in ds:
        smiles = row["code"].strip()
        if not smiles or smiles in seen:  # unique molecules, so eval can never repeat a training target
            continue
        seen.add(smiles)
        picked.append(row)
        if len(picked) == need:
            break
    if len(picked) < need:
        raise SystemExit(f"only found {len(picked)} unique samples, needed {need}")

    write_split("eval", picked[: args.n_eval], args.out)
    write_split("train", picked[args.n_eval:], args.out)


if __name__ == "__main__":
    main()
