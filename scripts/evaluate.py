"""Evaluate GLM-OCR (optionally with a LoRA adapter) on the molecule-to-SMILES task.

Usage:
    python scripts/evaluate.py --data LLaMA-Factory/data/cosyn_chem_eval.json \
        --data-root LLaMA-Factory/data --model GLM-OCR --out results/base.json
    python scripts/evaluate.py ... --adapter LLaMA-Factory/saves/glm-ocr/lora/sft --out results/lora.json
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
except ImportError:  # metrics fall back to plain string comparison
    Chem = None

FENCE = re.compile(r"```(?:rdkit)?\s*\n?(.*?)\n?```", re.S)


def extract_smiles(text):
    """Pull the SMILES out of an ```rdkit fence; fall back to the raw text."""
    m = FENCE.search(text)
    return (m.group(1) if m else text).strip()


def canonical(smiles):
    """Canonical SMILES, or None if RDKit is missing or the string is not a valid molecule."""
    if Chem is None:
        return None
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol) if mol is not None else None


def edit_similarity(a, b):
    """1 - Levenshtein(a, b) / max(len); 1.0 means identical."""
    if not a and not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1 - prev[-1] / max(len(a), len(b))


def load_model(model_path, adapter):
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=dtype, trust_remote_code=True)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return processor, model.to(device).eval(), device


@torch.inference_mode()
def generate(processor, model, device, image, prompt, max_new_tokens):
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(device)
    inputs.pop("token_type_ids", None)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="ShareGPT json (messages + images)")
    ap.add_argument("--data-root", required=True, help="folder the image paths are relative to")
    ap.add_argument("--model", default="GLM-OCR")
    ap.add_argument("--adapter", default=None, help="LoRA adapter folder; omit for the base model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    records = json.load(open(args.data, encoding="utf-8"))[: args.limit]
    processor, model, device = load_model(args.model, args.adapter)

    rows = []
    for n, rec in enumerate(records, 1):
        prompt = rec["messages"][0]["content"].replace("<image>", "").strip()
        target = extract_smiles(rec["messages"][1]["content"])
        image = Image.open(Path(args.data_root) / rec["images"][0]).convert("RGB")
        t0 = time.time()
        raw = generate(processor, model, device, image, prompt, args.max_new_tokens)
        pred = extract_smiles(raw)
        ct, cp = canonical(target), canonical(pred)
        rows.append({
            "image": rec["images"][0], "target": target, "prediction": pred, "raw_output": raw,
            "valid": cp is not None,
            "exact_match": (cp == ct) if ct and cp else pred == target,
            "similarity": round(edit_similarity(pred, target), 4),
            "seconds": round(time.time() - t0, 2),
        })
        print(f"[{n}/{len(records)}] match={rows[-1]['exact_match']} sim={rows[-1]['similarity']}  {pred[:60]}")

    k = len(rows)
    summary = {
        "model": args.model, "adapter": args.adapter, "samples": k,
        "exact_match": round(sum(r["exact_match"] for r in rows) / k, 4),
        "valid_smiles": round(sum(r["valid"] for r in rows) / k, 4) if Chem else None,
        "mean_similarity": round(sum(r["similarity"] for r in rows) / k, 4),
        "rdkit": Chem is not None,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "samples": rows}, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
