"""Build results/summary.md from results/base.json and results/lora.json."""
import json
from pathlib import Path

R = Path("results")
base = json.load(open(R / "base.json", encoding="utf-8"))
lora = json.load(open(R / "lora.json", encoding="utf-8"))
b, l = base["summary"], lora["summary"]


def pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def row(label, key, fmt=pct):
    bv, lv = b[key], l[key]
    delta = "" if bv is None or lv is None else f"{(lv - bv) * 100:+.1f} pts"
    return f"| {label} | {fmt(bv)} | {fmt(lv)} | {delta} |"


lines = [
    "# Results: base GLM-OCR vs LoRA fine-tuned",
    "",
    f"Held-out evaluation on {b['samples']} CoSyn chemical molecules that were never used for training.",
    "",
    "| Metric | Base | LoRA | Change |",
    "|---|---|---|---|",
    row("Exact match (canonical SMILES)", "exact_match"),
    row("Valid SMILES", "valid_smiles"),
    row("Mean character similarity", "mean_similarity"),
    "",
    "## Per-sample outputs",
    "",
    "| # | Target | Base | LoRA |",
    "|---|---|---|---|",
]
for i, (x, y) in enumerate(zip(base["samples"], lora["samples"]), 1):
    mark = lambda s: ("✅ " if s["exact_match"] else "❌ ") + f"`{s['prediction'] or '(empty)'}`"
    lines.append(f"| {i} | `{x['target']}` | {mark(x)} | {mark(y)} |")
(R / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines[:10]))
