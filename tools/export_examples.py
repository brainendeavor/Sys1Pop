#!/usr/bin/env python3
"""
Sys1Pop Example Models & Catalog Exporter
Generates INT8 quantized example model bundles and models/catalog.json conforming to docs/SPEC.md.
"""

import argparse
import json
import os
import struct
import sys

MAX_BUNDLE_BYTES = 35 * 1024 * 1024  # 35 MB Edge Limit

EXAMPLE_MODELS = [
    {
        "model_id": "sys1-base",
        "name": "Sys1Pop Base Backbone",
        "icon": "cpu",
        "description": "General semantic representation & System 1 edge decision backbone (33.4M params).",
        "default_state": "Site: example.com | Task: General semantic decision evaluation and intent triage.",
        "default_chunks": [],
        "triage_config": {
            "relevance_threshold": 0.3,
            "sufficiency_threshold": 0.5
        },
        "default_questions": [
            {
                "type": "choice",
                "id": "primary_intent",
                "options": ["information_lookup", "transaction_request", "escalation", "general_feedback"]
            },
            {
                "type": "boolean",
                "id": "requires_action"
            },
            {
                "type": "score",
                "id": "priority_level",
                "min": 1,
                "max": 5
            }
        ],
        "sample_presets": [
            {
                "label": "General Evaluation",
                "state": "Site: example.com | Task: General semantic decision evaluation and intent triage."
            }
        ]
    },
    {
        "model_id": "spam-detector-v1",
        "name": "Web Form Spam & Phishing",
        "icon": "shield",
        "description": "Evaluates user messages, sender emails, and site context to classify spam, intent, and risk.",
        "default_state": "Site: example.com | Name: Alex Taylor | Email: alex.marketing@example.com | Message: Hello team, I noticed your website has great content but low Google ranking. We offer high-quality backlinks and guest posts with DA 80+ to get you to #1 on search engines. Contact my handle: @example_marketer",
        "default_chunks": [],
        "triage_config": None,
        "default_questions": [
            {
                "type": "boolean",
                "id": "is_spam"
            },
            {
                "type": "choice",
                "id": "spam_category",
                "options": [
                    "legitimate_inquiry",
                    "commercial_sales_pitch",
                    "seo_backlink_spam",
                    "crypto_phishing",
                    "automated_bot_gibberish"
                ]
            },
            {
                "type": "score",
                "id": "risk_score",
                "min": 1,
                "max": 5
            }
        ],
        "sample_presets": [
            {
                "label": "🚨 Crypto Phishing",
                "state": "Site: example.com | Name: Reward Bot | Email: airdrop@example.org | Message: Urgent: Send 0.1 ETH to verify wallet and receive 5,000 promo tokens immediately to your balance."
            },
            {
                "label": "📈 SEO Sales Pitch",
                "state": "Site: example.com | Name: Alex Taylor | Email: alex.marketing@example.com | Message: Hello team, I noticed your website has great content but low Google ranking. We offer high-quality backlinks and guest posts with DA 80+ to get you to #1 on search engines. Contact my handle: @example_marketer"
            },
            {
                "label": "✅ Clean Inquiry",
                "state": "Site: example.com | Name: Dr. Jane Doe | Email: jane.doe@example.com | Message: Hello, I would like to schedule a technical briefing on deploying Sys1Pop across our global edge points of presence."
            }
        ]
    },
    {
        "model_id": "rag-triage-v1",
        "name": "Multi-Chunk RAG Triage",
        "icon": "book-open",
        "description": "Batch-evaluates retrieved context passages against a query, prunes irrelevant chunks, and assesses sufficiency.",
        "default_state": "User Query: How do I configure Cloudflare R2 bucket bindings in wrangler.toml?",
        "default_chunks": [
            "Doc 1: Cloudflare R2 allows developers to store unstructured data without egress bandwidth fees. You can create buckets using wrangler r2 bucket create <NAME>.",
            "Doc 2: To bind an R2 bucket to your Worker, add [[r2_buckets]] to wrangler.toml with 'binding' and 'bucket_name' fields. In your Worker code, access it via env.<BINDING>.",
            "Doc 3: Cloudflare KV is an ultra-low latency key-value store optimized for high-volume read workloads with eventual consistency."
        ],
        "triage_config": {
            "relevance_threshold": 0.3,
            "sufficiency_threshold": 0.5
        },
        "default_questions": [
            {
                "type": "boolean",
                "id": "sufficient_context"
            },
            {
                "type": "choice",
                "id": "action_required",
                "options": [
                    "generate_answer",
                    "fetch_more_docs",
                    "clarify_with_user"
                ]
            },
            {
                "type": "score",
                "id": "relevance_rating",
                "min": 1,
                "max": 5
            }
        ],
        "sample_presets": [
            {
                "label": "R2 Wrangler Docs",
                "state": "User Query: How do I configure Cloudflare R2 bucket bindings in wrangler.toml?"
            }
        ]
    },
    {
        "model_id": "intent-router-v1",
        "name": "Agent Intent Routing",
        "icon": "zap",
        "description": "Fast omni-box intent routing, tool dispatch, and confirmation gating under 40ms.",
        "default_state": "User Input: Turn off the living room desk lamp and schedule a focus block for 2 hours.",
        "default_chunks": [],
        "triage_config": None,
        "default_questions": [
            {
                "type": "choice",
                "id": "primary_route",
                "options": [
                    "iot_device_control",
                    "calendar_scheduling",
                    "multi_action_split",
                    "general_conversation"
                ]
            },
            {
                "type": "boolean",
                "id": "requires_confirmation"
            },
            {
                "type": "score",
                "id": "urgency_level",
                "min": 1,
                "max": 3
            }
        ],
        "sample_presets": [
            {
                "label": "💡 IoT Control",
                "state": "User Input: Dim the bedroom lights to 30% and set thermostat to 70 degrees."
            },
            {
                "label": "📅 Calendar Block",
                "state": "User Input: Schedule a 45-minute sync with Sarah tomorrow at 2pm."
            },
            {
                "label": "🔀 Multi-Action Split",
                "state": "User Input: Turn off the living room desk lamp and schedule a focus block for 2 hours."
            }
        ]
    }
]

def generate_minimal_safetensors(output_path, tensor_dim=384):
    """
    Generates a valid binary safetensors file with INT8 base weights.
    Avoids external numpy/safetensors requirements if not installed.
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
        header_dict = {
            "embeddings.weight": {"dtype": "I8", "shape": [100, tensor_dim], "data_offsets": [0, 100 * tensor_dim]},
            "heads.choice.weight": {"dtype": "F32", "shape": [5, tensor_dim], "data_offsets": [100 * tensor_dim, 100 * tensor_dim + 5 * tensor_dim * 4]},
            "__metadata__": {"format": "pt"}
        }
        header_bytes = json.dumps(header_dict).encode("utf-8")
        header_len = len(header_bytes)
        total_data_bytes = 100 * tensor_dim + 5 * tensor_dim * 4
        with open(output_path, "wb") as f:
            f.write(struct.pack("<Q", header_len))
            f.write(header_bytes)
            f.write(b"\x00" * total_data_bytes)

def export_single_bundle(model_meta, base_output_dir):
    model_id = model_meta["model_id"]
    bundle_dir = os.path.join(base_output_dir, model_id)
    os.makedirs(bundle_dir, exist_ok=True)

    # 1. manifest.json
    manifest = {
        "schema_version": "1.0",
        "model_id": model_id,
        "name": model_meta["name"],
        "icon": model_meta["icon"],
        "description": model_meta["description"],
        "architecture": "minilm_l6_v2",
        "hidden_dim": 384,
        "max_seq_len": 512,
        "quantization": "int8_q8_0",
        "supported_heads": ["choice", "boolean", "score"],
        "default_state": model_meta.get("default_state", ""),
        "default_chunks": model_meta.get("default_chunks", []),
        "triage_config": model_meta.get("triage_config"),
        "default_questions": model_meta.get("default_questions", []),
        "sample_presets": model_meta.get("sample_presets", []),
        "calibration": {
            "temperature": 1.0,
            "default_threshold": 0.5
        }
    }
    with open(os.path.join(bundle_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 2. config.json
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
    with open(os.path.join(bundle_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # 3. tokenizer.json
    tokenizer = {
        "version": "1.0",
        "truncation": {"max_length": 512, "strategy": "longest_first"},
        "padding": {"strategy": "BatchLongest"},
        "model": {"type": "WordPiece", "vocab": {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}}
    }
    with open(os.path.join(bundle_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer, f, indent=2)

    # 4. model.safetensors
    generate_minimal_safetensors(os.path.join(bundle_dir, "model.safetensors"))

    print(f"  ✓ Exported model bundle: {model_id} -> {bundle_dir}")
    return manifest

def export_all(base_output_dir):
    print(f"\n📦 Exporting Sys1Pop Example Models & Catalog to: {base_output_dir}")
    print("=" * 65)
    os.makedirs(base_output_dir, exist_ok=True)

    catalog_entries = []
    for model_meta in EXAMPLE_MODELS:
        manifest = export_single_bundle(model_meta, base_output_dir)
        catalog_entries.append(manifest)

    # Write master catalog.json
    catalog_path = os.path.join(base_output_dir, "catalog.json")
    catalog_data = {
        "schema_version": "1.0",
        "updated_at": "2026-09-29T09:00:00Z",
        "models": catalog_entries
    }
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog_data, f, indent=2)
    print(f"  ✓ Generated master catalog: {catalog_path}")
    print("=" * 65)
    print(f"✅ All {len(EXAMPLE_MODELS)} example models successfully generated!\n")

def verify_all(base_output_dir):
    print(f"\n🔍 Verifying Model Catalog in: {base_output_dir}")
    catalog_path = os.path.join(base_output_dir, "catalog.json")
    if not os.path.exists(catalog_path):
        print(f"❌ Error: catalog.json missing in {base_output_dir}")
        sys.exit(1)

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    for item in catalog.get("models", []):
        mid = item["model_id"]
        bundle_dir = os.path.join(base_output_dir, mid)
        for req in ["manifest.json", "model.safetensors", "tokenizer.json", "config.json"]:
            p = os.path.join(bundle_dir, req)
            if not os.path.exists(p):
                print(f"❌ Model '{mid}' missing required file: {req}")
                sys.exit(1)
        print(f"  ✓ Model verified: {mid}")
    print("✅ All catalog models verified!\n")

def main():
    parser = argparse.ArgumentParser(description="Export Sys1Pop example models and catalog")
    parser.add_argument("--output-dir", default="./dist/models", help="Output directory for model bundles")
    parser.add_argument("--verify-all", action="store_true", help="Verify all exported bundles")

    args = parser.parse_args()
    if args.verify_all:
        verify_all(args.output_dir)
    else:
        export_all(args.output_dir)
        verify_all(args.output_dir)

if __name__ == "__main__":
    main()
