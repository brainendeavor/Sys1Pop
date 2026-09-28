#!/usr/bin/env python3
"""
Sys1Pop Distillation & INT8 Export Tool
Packages distilled edge models into Cloudflare Worker Candle WASM compatible bundles.
Conforms to docs/SPEC.md.
"""

import argparse
import json
import os
import sys
import struct

MAX_BUNDLE_BYTES = 35 * 1024 * 1024  # 35 MB Edge Limit

def create_sample_dataset():
    """Returns a sample domain dataset for quickstart demonstration."""
    return [
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
        }
    ]

def generate_minimal_safetensors(output_path, num_tensors=4, tensor_dim=384):
    """
    Generates a valid, minimal safetensors binary file with INT8 quantized dummy/base weights.
    If safetensors Python library is installed, it uses it; otherwise writes valid binary safetensors.
    """
    try:
        from safetensors.numpy import save_file
        import numpy as np
        tensors = {
            "embeddings.weight": np.zeros((100, tensor_dim), dtype=np.int8),
            "encoder.layer.0.attention.weight": np.zeros((tensor_dim, tensor_dim), dtype=np.int8),
            "heads.choice.weight": np.zeros((5, tensor_dim), dtype=np.float32),
            "heads.boolean.weight": np.zeros((1, tensor_dim), dtype=np.float32),
        }
        save_file(tensors, output_path)
    except ImportError:
        # Fallback binary safetensors writer
        header_dict = {
            "embeddings.weight": {"dtype": "I8", "shape": [100, tensor_dim], "data_offsets": [0, 100 * tensor_dim]},
            "heads.choice.weight": {"dtype": "F32", "shape": [5, tensor_dim], "data_offsets": [100 * tensor_dim, 100 * tensor_dim + 5 * tensor_dim * 4]},
            "__metadata__": {"format": "pt"}
        }
        header_bytes = json.dumps(header_dict).encode("utf-8")
        header_len = len(header_bytes)
        # 8 bytes little-endian header length + header JSON + zeroed data bytes
        total_data_bytes = 100 * tensor_dim + 5 * tensor_dim * 4
        with open(output_path, "wb") as f:
            f.write(struct.pack("<Q", header_len))
            f.write(header_bytes)
            f.write(b"\x00" * total_data_bytes)

def export_bundle(model_id, output_dir, dataset=None):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Generate manifest.json
    manifest = {
        "schema_version": "1.0",
        "model_id": model_id,
        "architecture": "minilm_l6_v2",
        "hidden_dim": 384,
        "max_seq_len": 512,
        "quantization": "int8_q8_0",
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

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 2. Generate config.json (MiniLM backbone config)
    config = {
        "architectures": ["BertForSequenceClassification"],
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_size": 384,
        "initializer_range": 0.02,
        "intermediate_size": 1536,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "type_vocab_size": 2,
        "vocab_size": 30522
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # 3. Generate tokenizer.json skeleton
    tokenizer = {
        "version": "1.0",
        "truncation": {"max_length": 512, "strategy": "longest_first"},
        "padding": {"strategy": "BatchLongest"},
        "model": {"type": "WordPiece", "vocab": {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}}
    }
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    with open(tokenizer_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer, f, indent=2)

    # 4. Generate model.safetensors
    weights_path = os.path.join(output_dir, "model.safetensors")
    generate_minimal_safetensors(weights_path)

    # Verification
    verify_bundle(output_dir)

def verify_bundle(bundle_dir):
    """Asserts that all bundle files exist and conform to SPEC.md size limits."""
    required_files = ["manifest.json", "model.safetensors", "tokenizer.json", "config.json"]
    total_size = 0

    print(f"\nVerifying Sys1Pop Model Bundle: {bundle_dir}")
    print("=" * 60)

    for req in required_files:
        path = os.path.join(bundle_dir, req)
        if not os.path.exists(path):
            print(f"❌ Error: Missing required bundle file: {req}")
            sys.exit(1)
        sz = os.path.getsize(path)
        total_size += sz
        print(f"  ✓ {req:<20} ({sz / 1024:.1f} KB)")

    # Validate manifest.json schema
    with open(os.path.join(bundle_dir, "manifest.json"), "r") as f:
        manifest = json.load(f)
        assert "model_id" in manifest, "Missing model_id in manifest"
        assert "supported_heads" in manifest, "Missing supported_heads in manifest"

    total_mb = total_size / (1024 * 1024)
    print("=" * 60)
    print(f"Total Bundle Size: {total_mb:.2f} MB (Budget Limit: 35.00 MB)")

    if total_size > MAX_BUNDLE_BYTES:
        print(f"❌ Bundle exceeds 35MB edge limit: {total_mb:.2f} MB")
        sys.exit(1)
    
    print("✅ Verification Passed! Ready for R2 upload via 'npx sys1pop model push'.\n")

def main():
    parser = argparse.ArgumentParser(description="Sys1Pop Model Distiller & INT8 Bundle Exporter")
    parser.add_argument("--model-id", default="sample-triage-v1", help="Target model identifier")
    parser.add_argument("--output-dir", default="./dist/models/sample-triage-v1", help="Output directory")
    parser.add_argument("--verify", action="store_true", help="Only verify existing bundle directory")
    parser.add_argument("--data", help="Optional path to synthetic training data JSON")

    args = parser.parse_args()

    if args.verify:
        verify_bundle(args.output_dir)
    else:
        export_bundle(args.model_id, args.output_dir)

if __name__ == "__main__":
    main()
