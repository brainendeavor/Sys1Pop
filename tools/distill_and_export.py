#!/usr/bin/env python3
"""
Sys1Pop Distillation & Export Tool
Packages distilled edge models into Cloudflare Worker Candle WASM compatible bundles
using real transformer backbone embeddings and calibrated decision heads.
Conforms to docs/SPEC.md.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

MAX_BUNDLE_BYTES = 100 * 1024 * 1024  # 100 MB Limit for uncompressed weights

DEFAULT_SAMPLE_DATASET = [
    {
        "text": "Please refund my order #48291, the package never arrived after 3 weeks.",
        "is_spam": False,
        "category": "order_support",
        "urgency": 4
    },
    {
        "text": "Buy high quality backlinks to boost your website Google ranking today!",
        "is_spam": True,
        "category": "seo_sales_pitch",
        "urgency": 1
    },
    {
        "text": "Urgent: I cannot log in and my password reset link is throwing a 500 error.",
        "is_spam": False,
        "category": "account_security",
        "urgency": 5
    },
    {
        "text": "Send 0.5 ETH to this wallet address to claim your automated presale tokens.",
        "is_spam": True,
        "category": "crypto_phishing",
        "urgency": 1
    },
    {
        "text": "Hello, I wanted to inquire about team licensing options and EU data residency.",
        "is_spam": False,
        "category": "general_inquiry",
        "urgency": 2
    },
    {
        "text": "Our production server is unreachable and customer transactions are dropping.",
        "is_spam": False,
        "category": "account_security",
        "urgency": 5
    },
    {
        "text": "Special discount on luxury replica watches with fast worldwide shipping.",
        "is_spam": True,
        "category": "seo_sales_pitch",
        "urgency": 1
    },
    {
        "text": "Can you check the shipping status for tracking ID 98234-US?",
        "is_spam": False,
        "category": "order_support",
        "urgency": 3
    }
]

def encode_texts(tokenizer, base_model, texts):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        outputs = base_model(**inputs)
        token_embeddings = outputs[0]
        attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        return F.normalize(embeddings, p=2, dim=1).numpy()

def export_bundle(model_id, output_dir, dataset=None, manifest_path=None):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📦 Distilling & Exporting Sys1Pop Model Bundle: {model_id}")
    print("=" * 65)

    # 1. Load manifest schema
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {
            "schema_version": "1.0",
            "model_id": model_id,
            "architecture": "minilm_l6_v2",
            "hidden_dim": 384,
            "max_seq_len": 512,
            "quantization": "fp32",
            "supported_heads": ["choice", "boolean", "score"],
            "default_questions": [
                {
                    "type": "boolean",
                    "id": "is_spam"
                },
                {
                    "type": "choice",
                    "id": "category",
                    "options": [
                        "order_support",
                        "account_security",
                        "seo_sales_pitch",
                        "crypto_phishing",
                        "general_inquiry"
                    ]
                },
                {
                    "type": "score",
                    "id": "urgency",
                    "min": 1,
                    "max": 5
                }
            ],
            "calibration": {
                "temperature": 1.0,
                "default_threshold": 0.5
            }
        }

    # 2. Load dataset
    if dataset is None:
        dataset = DEFAULT_SAMPLE_DATASET

    print(f"📥 Loading base model: sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    base_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    base_model.eval()

    texts = [item["text"] for item in dataset]
    embeddings = encode_texts(tokenizer, base_model, texts)
    print(f"  ✓ Encoded {len(texts)} exemplars into {embeddings.shape[1]}-dim space")

    # 3. Calibrate decision heads
    head_tensors = {}
    for q in manifest.get("default_questions", []):
        qid = q["id"]
        qtype = q["type"]

        if qtype == "boolean":
            y = np.array([1 if item.get(qid, False) else 0 for item in dataset])
            if len(np.unique(y)) > 1:
                clf = LogisticRegression(C=5.0, max_iter=200).fit(embeddings, y)
                head_tensors[f"heads.boolean.{qid}.weight"] = torch.from_numpy(clf.coef_).float()
                head_tensors[f"heads.boolean.{qid}.bias"] = torch.from_numpy(clf.intercept_).float()
                print(f"  ✓ Calibrated boolean head: {qid}")

        elif qtype == "choice":
            options = q["options"]
            opt_map = {opt: idx for idx, opt in enumerate(options)}
            y = np.array([opt_map.get(item.get(qid), 0) for item in dataset])
            # Ensure multi-class logistic regression
            clf = LogisticRegression(C=5.0, max_iter=200).fit(embeddings, y)
            coef = clf.coef_
            intercept = clf.intercept_
            if coef.shape[0] < len(options):
                # Expand rows for unobserved classes
                full_coef = np.zeros((len(options), embeddings.shape[1]), dtype=np.float32)
                full_intercept = np.zeros(len(options), dtype=np.float32)
                for i, cls_idx in enumerate(clf.classes_):
                    full_coef[cls_idx] = coef[i] if coef.ndim > 1 else coef[0]
                    full_intercept[cls_idx] = intercept[i] if intercept.ndim > 0 else intercept[0]
                coef = full_coef
                intercept = full_intercept
            head_tensors[f"heads.choice.{qid}.weight"] = torch.from_numpy(coef).float()
            head_tensors[f"heads.choice.{qid}.bias"] = torch.from_numpy(intercept).float()
            print(f"  ✓ Calibrated choice head: {qid} ({len(options)} options)")

        elif qtype == "score":
            min_v = q.get("min", 1)
            max_v = q.get("max", 5)
            num_classes = max_v - min_v + 1
            y = np.array([max(0, min(num_classes - 1, item.get(qid, min_v) - min_v)) for item in dataset])
            clf = LogisticRegression(C=5.0, max_iter=200).fit(embeddings, y)
            coef = clf.coef_
            intercept = clf.intercept_
            if coef.shape[0] < num_classes:
                full_coef = np.zeros((num_classes, embeddings.shape[1]), dtype=np.float32)
                full_intercept = np.zeros(num_classes, dtype=np.float32)
                for i, cls_idx in enumerate(clf.classes_):
                    full_coef[cls_idx] = coef[i] if coef.ndim > 1 else coef[0]
                    full_intercept[cls_idx] = intercept[i] if intercept.ndim > 0 else intercept[0]
                coef = full_coef
                intercept = full_intercept
            head_tensors[f"heads.score.{qid}.weight"] = torch.from_numpy(coef).float()
            head_tensors[f"heads.score.{qid}.bias"] = torch.from_numpy(intercept).float()
            print(f"  ✓ Calibrated score head: {qid} (range {min_v}..{max_v})")

    # 4. Save model.safetensors
    base_state_dict = base_model.state_dict()
    full_state_dict = {}
    for k, v in base_state_dict.items():
        full_state_dict[k] = v.contiguous()
    for k, v in head_tensors.items():
        full_state_dict[k] = v.contiguous()

    safetensors_path = os.path.join(output_dir, "model.safetensors")
    save_file(full_state_dict, safetensors_path)

    # 5. Save tokenizer.json
    tokenizer.save_pretrained(output_dir)

    # 6. Save config.json
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(base_model.config.to_dict(), f, indent=2)

    # 7. Save manifest.json
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Verification
    verify_bundle(output_dir)

def verify_bundle(bundle_dir):
    """Asserts that all bundle files exist and conform to SPEC.md size limits."""
    required_files = ["manifest.json", "model.safetensors", "tokenizer.json", "config.json"]
    total_size = 0

    print(f"\n🔍 Verifying Sys1Pop Model Bundle: {bundle_dir}")
    print("=" * 65)

    for req in required_files:
        path = os.path.join(bundle_dir, req)
        if not os.path.exists(path):
            print(f"❌ Error: Missing required bundle file: {req}")
            sys.exit(1)
        sz = os.path.getsize(path)
        total_size += sz
        print(f"  ✓ {req:<20} ({sz / 1024:.1f} KB)")

    with open(os.path.join(bundle_dir, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)
        assert "model_id" in manifest, "Missing model_id in manifest"
        assert "supported_heads" in manifest, "Missing supported_heads in manifest"

    total_mb = total_size / (1024 * 1024)
    print("=" * 65)
    print(f"Total Bundle Size: {total_mb:.2f} MB (Budget Limit: 100.00 MB)")

    if total_size > MAX_BUNDLE_BYTES:
        print(f"❌ Bundle exceeds limit: {total_mb:.2f} MB")
        sys.exit(1)
    
    print("✅ Verification Passed! Ready for R2 upload.\n")

def main():
    parser = argparse.ArgumentParser(description="Sys1Pop Model Distiller & Bundle Exporter")
    parser.add_argument("--model-id", default="sample-triage-v1", help="Target model identifier")
    parser.add_argument("--output-dir", default="./dist/models/sample-triage-v1", help="Output directory")
    parser.add_argument("--manifest", help="Optional path to custom manifest.json")
    parser.add_argument("--verify", action="store_true", help="Only verify existing bundle directory")
    parser.add_argument("--data", help="Optional path to synthetic training data JSON")

    args = parser.parse_args()

    if args.verify:
        verify_bundle(args.output_dir)
    else:
        dataset = None
        if args.data and os.path.exists(args.data):
            with open(args.data, "r", encoding="utf-8") as f:
                dataset = json.load(f)
        export_bundle(args.model_id, args.output_dir, dataset=dataset, manifest_path=args.manifest)

if __name__ == "__main__":
    main()
