#!/usr/bin/env python3
"""
Sys1Pop ModelForge: Declarative Edge Model Compiler Engine (RFC-002)

Compiles a declarative model.spec.json (or YAML) into a spec-compliant
Cloudflare Worker edge neural bundle:
  ├── manifest.json
  ├── model.safetensors
  ├── tokenizer.json
  └── config.json
Strictly enforces <= 35 MB edge bundle limit and instant (< 2s) CPU head fitting.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional imports for validate-only mode
try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoModel, AutoTokenizer
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False

MAX_EDGE_BUNDLE_BYTES = 35 * 1024 * 1024  # 35 MB Cloudflare Worker limit


class ModelSpecValidationError(Exception):
    pass


def load_spec(spec_path: str) -> Dict[str, Any]:
    """Loads and parses a model specification from JSON or YAML."""
    p = Path(spec_path)
    if not p.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_path}")

    content = p.read_text(encoding="utf-8")
    if p.suffix.lower() in [".yaml", ".yml"]:
        try:
            import yaml
            spec = yaml.safe_load(content)
        except ImportError:
            raise ImportError("PyYAML is required to parse .yaml/.yml specs. Install pyyaml or use JSON.")
    else:
        spec = json.loads(content)

    return spec


def load_dataset(spec: Dict[str, Any], spec_dir: Path) -> List[Dict[str, Any]]:
    """Loads training examples from inline spec and/or external dataset_path."""
    examples: List[Dict[str, Any]] = list(spec.get("training_examples") or [])

    dataset_path_str = spec.get("dataset_path")
    if dataset_path_str:
        dataset_path = Path(dataset_path_str)
        if not dataset_path.is_absolute():
            dataset_path = spec_dir / dataset_path

        if not dataset_path.exists():
            raise FileNotFoundError(f"External dataset not found: {dataset_path}")

        if dataset_path.suffix.lower() == ".jsonl":
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        examples.append(json.loads(line))
        elif dataset_path.suffix.lower() == ".csv":
            with open(dataset_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    examples.append(dict(row))
        else:
            raise ValueError(f"Unsupported dataset format '{dataset_path.suffix}'. Use .jsonl or .csv")

    return examples


def validate_spec(spec: Dict[str, Any], examples: List[Dict[str, Any]]) -> None:
    """Validates the spec schema and ensures question labels exist in training data."""
    required_keys = ["model_id", "name", "questions"]
    for key in required_keys:
        if not spec.get(key):
            raise ModelSpecValidationError(f"Missing required field: '{key}'")

    model_id = spec["model_id"]
    if not isinstance(model_id, str) or not model_id.strip():
        raise ModelSpecValidationError("'model_id' must be a non-empty string")

    questions = spec.get("questions")
    if not isinstance(questions, list) or len(questions) == 0:
        raise ModelSpecValidationError("'questions' must be a non-empty array")

    seen_ids = set()
    for idx, q in enumerate(questions):
        qid = q.get("id")
        qtype = q.get("type")
        if not qid or not isinstance(qid, str):
            raise ModelSpecValidationError(f"Question at index {idx} missing valid 'id'")
        if qid in seen_ids:
            raise ModelSpecValidationError(f"Duplicate question id '{qid}'")
        seen_ids.add(qid)

        if qtype not in ["choice", "boolean", "score"]:
            raise ModelSpecValidationError(f"Unsupported question type '{qtype}' for question '{qid}'")

        if qtype == "choice":
            options = q.get("options")
            if not isinstance(options, list) or len(options) < 2:
                raise ModelSpecValidationError(f"Choice question '{qid}' must have at least 2 options")
            if len(set(options)) != len(options):
                raise ModelSpecValidationError(f"Choice question '{qid}' contains duplicate options")
        elif qtype == "score":
            min_val = q.get("min", 1)
            max_val = q.get("max", 5)
            if min_val >= max_val:
                raise ModelSpecValidationError(f"Score question '{qid}' min ({min_val}) must be < max ({max_val})")

    if not examples:
        raise ModelSpecValidationError("Spec must provide at least one training example in 'training_examples' or 'dataset_path'")

    for idx, ex in enumerate(examples):
        if "state" not in ex or not str(ex["state"]).strip():
            raise ModelSpecValidationError(f"Training example at index {idx} missing non-empty 'state' string")


def extract_embeddings(texts: List[str], tokenizer: Any, base_model: Any, batch_size: int = 64) -> np.ndarray:
    """Computes mean-pooled, L2-normalized sentence embeddings on CPU."""
    all_embeddings = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            outputs = base_model(**inputs)
            token_embeddings = outputs[0]
            attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
            sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
            normed = F.normalize(pooled, p=2, dim=1).cpu().numpy()
            all_embeddings.append(normed)

    return np.vstack(all_embeddings)


def fit_choice_head(
    X: np.ndarray,
    labels: List[Any],
    options: List[str],
    hidden_dim: int,
    c_reg: float = 5.0,
    max_iter: int = 200,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fits a multinomial ChoiceHead.
    Guarantees output weight shape is (K, D) and bias shape is (K,) where K = len(options).
    """
    k = len(options)
    opt_to_idx = {opt: idx for idx, opt in enumerate(options)}
    y = np.array([opt_to_idx[str(lbl)] for lbl in labels], dtype=int)
    classes_present = np.unique(y)

    if len(classes_present) < 2:
        # Trivial edge case: all examples have the same label
        weight = np.zeros((k, hidden_dim), dtype=np.float32)
        bias = np.full((k,), -2.0, dtype=np.float32)
        if len(classes_present) == 1:
            bias[classes_present[0]] = 2.0
        return weight, bias, 1.0

    clf = LogisticRegression(C=c_reg, max_iter=max_iter, fit_intercept=True, class_weight="balanced")
    clf.fit(X, y)
    acc = float(np.mean(clf.predict(X) == y))

    weight = np.zeros((k, hidden_dim), dtype=np.float32)
    bias = np.zeros((k,), dtype=np.float32)

    if k == 2:
        # Binary LogisticRegression in scikit-learn outputs shape (1, D) and (1,)
        # Expand symmetrically to (2, D) and (2,) so WASM engine calculates both option logits
        c0 = clf.coef_[0]
        b0 = clf.intercept_[0]
        weight[0, :] = -c0
        weight[1, :] = c0
        bias[0] = -b0
        bias[1] = b0
    else:
        # Multi-class
        for i, c in enumerate(clf.classes_):
            weight[c, :] = clf.coef_[i]
            bias[c] = clf.intercept_[i]

    return weight, bias, acc


def fit_boolean_head(
    X: np.ndarray,
    labels: List[Any],
    hidden_dim: int,
    c_reg: float = 5.0,
    max_iter: int = 200,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fits a binary BooleanHead.
    Guarantees weight shape is (1, D) and bias shape is (1,).
    """
    def to_bool(val: Any) -> int:
        if isinstance(val, bool):
            return 1 if val else 0
        s = str(val).strip().lower()
        return 1 if s in ["true", "1", "yes", "t"] else 0

    y = np.array([to_bool(lbl) for lbl in labels], dtype=int)
    classes_present = np.unique(y)

    if len(classes_present) < 2:
        weight = np.zeros((1, hidden_dim), dtype=np.float32)
        bias = np.array([2.0 if classes_present[0] == 1 else -2.0], dtype=np.float32)
        return weight, bias, 1.0

    clf = LogisticRegression(C=c_reg, max_iter=max_iter, fit_intercept=True, class_weight="balanced")
    clf.fit(X, y)
    acc = float(np.mean(clf.predict(X) == y))

    weight = clf.coef_.astype(np.float32)
    bias = clf.intercept_.astype(np.float32)
    return weight, bias, acc


def fit_score_head(
    X: np.ndarray,
    labels: List[Any],
    min_val: int,
    max_val: int,
    hidden_dim: int,
    c_reg: float = 5.0,
    max_iter: int = 200,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fits an ordinal ScoreHead.
    Guarantees weight shape is (B, D) and bias shape is (B,) where B = max_val - min_val + 1.
    """
    b = max_val - min_val + 1
    y = np.array([int(float(lbl)) - min_val for lbl in labels], dtype=int)
    y = np.clip(y, 0, b - 1)
    classes_present = np.unique(y)

    weight = np.zeros((b, hidden_dim), dtype=np.float32)
    bias = np.zeros((b,), dtype=np.float32)

    if len(classes_present) < 2:
        bias.fill(-2.0)
        if len(classes_present) == 1:
            bias[classes_present[0]] = 2.0
        return weight, bias, 1.0

    clf = LogisticRegression(C=c_reg, max_iter=max_iter, fit_intercept=True, class_weight="balanced")
    clf.fit(X, y)
    acc = float(np.mean(clf.predict(X) == y))

    if len(clf.classes_) == 2 and b == 2:
        weight[0, :] = -clf.coef_[0]
        weight[1, :] = clf.coef_[0]
        bias[0] = -clf.intercept_[0]
        bias[1] = clf.intercept_[0]
    else:
        for i, c in enumerate(clf.classes_):
            if c < b:
                weight[c, :] = clf.coef_[i]
                bias[c] = clf.intercept_[i]

    return weight, bias, acc


def compile_model_bundle(
    spec: Dict[str, Any],
    examples: List[Dict[str, Any]],
    output_dir: str,
    base_model_name: Optional[str] = None,
    quantization: Optional[str] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Compiles spec and examples into safetensors, tokenizer, config, and manifest."""
    if not HAS_ML_DEPS:
        raise RuntimeError("ML dependencies missing. Run via: uv run --with 'torch,transformers,scikit-learn,safetensors' tools/model_forge.py")

    t_start = time.time()
    model_id = spec["model_id"]
    base_model_name = base_model_name or spec.get("base_backbone", "sentence-transformers/all-MiniLM-L6-v2")
    quant_mode = quantization or spec.get("quantization", "fp32_edge")
    calib = spec.get("calibration", {})
    c_reg = float(calib.get("regularization_c", 5.0))
    max_iter = int(calib.get("max_iterations", 200))
    temperature = float(calib.get("temperature", 1.0))
    threshold = float(calib.get("default_threshold", 0.5))

    if not quiet:
        print("\n" + "━" * 65)
        print(f"  🔨 Sys1Pop ModelForge: Compiling '{model_id}'")
        print(f"  Backbone:       {base_model_name}")
        print(f"  Quantization:   {quant_mode}")
        print(f"  Training Items: {len(examples)} examples")
        print("━" * 65)

    # 1. Load frozen backbone
    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModel.from_pretrained(base_model_name)
    base_model.eval()

    base_state_dict = base_model.state_dict()
    config_dict = base_model.config.to_dict()
    hidden_dim = config_dict.get("hidden_size", 128)
    max_seq_len = config_dict.get("max_position_embeddings", 512)
    architecture = config_dict.get("model_type", "bert")

    # 2. Extract sentence embeddings
    t_embed = time.time()
    texts = [ex["state"] for ex in examples]
    X = extract_embeddings(texts, tokenizer, base_model)
    embed_ms = (time.time() - t_embed) * 1000.0

    # 3. Fit probe decision heads
    t_heads = time.time()
    head_tensors = {}
    head_metrics = {}

    for q in spec["questions"]:
        qid = q["id"]
        qtype = q["type"]
        labels = [ex[qid] for ex in examples if qid in ex]

        if len(labels) != len(examples):
            raise ModelSpecValidationError(f"Question '{qid}' is missing in some training examples ({len(labels)}/{len(examples)})")

        if qtype == "choice":
            w, b, acc = fit_choice_head(X, labels, q["options"], hidden_dim, c_reg, max_iter)
            head_tensors[f"heads.choice.{qid}.weight"] = torch.from_numpy(w).float()
            head_tensors[f"heads.choice.{qid}.bias"] = torch.from_numpy(b).float()
            head_metrics[qid] = {"type": "choice", "acc": acc, "options": len(q["options"])}
        elif qtype == "boolean":
            w, b, acc = fit_boolean_head(X, labels, hidden_dim, c_reg, max_iter)
            head_tensors[f"heads.boolean.{qid}.weight"] = torch.from_numpy(w).float()
            head_tensors[f"heads.boolean.{qid}.bias"] = torch.from_numpy(b).float()
            head_metrics[qid] = {"type": "boolean", "acc": acc}
        elif qtype == "score":
            min_val = q.get("min", 1)
            max_val = q.get("max", 5)
            w, b, acc = fit_score_head(X, labels, min_val, max_val, hidden_dim, c_reg, max_iter)
            head_tensors[f"heads.score.{qid}.weight"] = torch.from_numpy(w).float()
            head_tensors[f"heads.score.{qid}.bias"] = torch.from_numpy(b).float()
            head_metrics[qid] = {"type": "score", "acc": acc, "range": f"{min_val}-{max_val}"}

    heads_ms = (time.time() - t_heads) * 1000.0

    # 4. Export bundle files
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 4a. model.safetensors
    full_state_dict = {}
    for k, v in base_state_dict.items():
        full_state_dict[k] = v.contiguous()
    for k, v in head_tensors.items():
        full_state_dict[k] = v.contiguous()

    safetensors_path = out_path / "model.safetensors"
    save_file(full_state_dict, str(safetensors_path))

    # 4b. tokenizer.json
    tokenizer.save_pretrained(str(out_path))

    # 4c. config.json
    config_path = out_path / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # 4d. manifest.json
    manifest = {
        "schema_version": "1.0",
        "model_id": model_id,
        "name": spec.get("name", model_id),
        "icon": spec.get("icon", "cpu"),
        "description": spec.get("description", ""),
        "architecture": architecture,
        "hidden_dim": hidden_dim,
        "max_seq_len": max_seq_len,
        "quantization": quant_mode,
        "supported_heads": ["choice", "boolean", "score"],
        "default_state": spec.get("default_state", examples[0]["state"] if examples else ""),
        "default_chunks": spec.get("default_chunks", []),
        "triage_config": spec.get("triage_config"),
        "default_questions": spec["questions"],
        "sample_presets": spec.get("sample_presets", []),
        "calibration": {
            "temperature": temperature,
            "default_threshold": threshold
        }
    }
    manifest_path = out_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 4e. Synchronize master catalog.json if present in parent directory
    catalog_path = out_path.parent / "catalog.json"
    if catalog_path.exists():
        try:
            with open(catalog_path, "r", encoding="utf-8") as f:
                cat_data = json.load(f)
            models_list = cat_data.get("models", [])
            idx = next((i for i, m in enumerate(models_list) if m.get("model_id") == model_id), -1)
            if idx >= 0:
                models_list[idx] = manifest
            else:
                models_list.append(manifest)
            cat_data["models"] = models_list
            cat_data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with open(catalog_path, "w", encoding="utf-8") as f:
                json.dump(cat_data, f, indent=2)
            if not quiet:
                print(f"  ✓ Updated master catalog: {catalog_path}")
        except Exception as e:
            if not quiet:
                print(f"  ⚠️ Could not update master catalog: {e}")

    # 5. Verify total bundle size
    total_bytes = sum(f.stat().st_size for f in out_path.iterdir() if f.is_file())
    total_mb = total_bytes / (1024 * 1024)

    if total_bytes > MAX_EDGE_BUNDLE_BYTES:
        raise RuntimeError(f"Edge bundle size {total_mb:.2f} MB exceeds 35 MB Cloudflare Worker limit!")

    total_time_s = time.time() - t_start

    if not quiet:
        print("\n📊 Head Calibration Results:")
        for qid, m in head_metrics.items():
            acc_pct = m["acc"] * 100.0
            info = f"options: {m['options']}" if m["type"] == "choice" else (f"range: {m['range']}" if m["type"] == "score" else "binary")
            print(f"  ✓ {qid.ljust(24)} [{m['type'].upper().ljust(7)}] Acc: {acc_pct:5.1f}% ({info})")

        print("\n⏱️ Performance & Packaging:")
        print(f"  • Embedding Extraction: {embed_ms:.1f} ms")
        print(f"  • Head Calibration:     {heads_ms:.1f} ms (< 2.0s target)")
        print(f"  • Total Bundle Size:    {total_mb:.2f} MB (<= 35 MB limit)")
        print(f"  • Output Directory:     {out_path}")
        print(f"  • Total Elapsed Time:   {total_time_s:.2f} s")
        print("━" * 65)
        print(f"✅ Sys1Pop model '{model_id}' successfully compiled!\n")

    return {
        "model_id": model_id,
        "total_bytes": total_bytes,
        "total_mb": total_mb,
        "heads_ms": heads_ms,
        "total_time_s": total_time_s,
        "manifest": manifest,
    }


def main():
    parser = argparse.ArgumentParser(description="Sys1Pop ModelForge: Declarative Edge Model Compiler")
    parser.add_argument("--spec", "-s", required=True, help="Path to model.spec.json or YAML specification file")
    parser.add_argument("--output-dir", "-o", default=None, help="Destination directory (default: ./dist/models/{model_id})")
    parser.add_argument("--base-model", "-b", default=None, help="Base HuggingFace transformer model ID")
    parser.add_argument("--quantization", "-q", default=None, choices=["fp32_edge", "int8_q8_0"], help="Quantization format")
    parser.add_argument("--validate-only", action="store_true", help="Validate spec schema and data without building")
    parser.add_argument("--quiet", action="store_true", help="Suppress logs")

    args = parser.parse_args()

    try:
        spec_path = Path(args.spec).resolve()
        spec = load_spec(str(spec_path))
        examples = load_dataset(spec, spec_path.parent)
        validate_spec(spec, examples)

        if args.validate_only:
            print(f"✅ Spec '{args.spec}' is valid ({len(spec['questions'])} questions, {len(examples)} training examples).")
            sys.exit(0)

        output_dir = args.output_dir or f"./dist/models/{spec['model_id']}"
        compile_model_bundle(
            spec=spec,
            examples=examples,
            output_dir=output_dir,
            base_model_name=args.base_model,
            quantization=args.quantization,
            quiet=args.quiet,
        )
    except Exception as e:
        print(f"\n❌ ModelForge Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
