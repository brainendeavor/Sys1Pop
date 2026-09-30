#!/usr/bin/env python3
"""
Sys1Pop Production Neural Model Exporter
Downloads real sentence-transformers/all-MiniLM-L6-v2 weights from HuggingFace,
calibrates decision heads for all production model contracts,
and outputs complete, ready-to-run safetensors bundles, manifest.json, and catalog.json
conforming to docs/SPEC.md.
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

MODELS_CONFIG = [
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
            },
            {
                "label": "🚨 Critical Outage",
                "state": "Our production database is experiencing connection exhaustion and 500 errors across all nodes, critical!"
            },
            {
                "label": "💳 Subscription Billing",
                "state": "Please process a refund for our latest monthly invoice #48291 and downgrade to starter."
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

def calibrate_all_heads(tokenizer, base_model):
    def encode_texts(texts):
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            outputs = base_model(**inputs)
            token_embeddings = outputs[0]
            attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
            sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            return F.normalize(embeddings, p=2, dim=1).numpy()

    heads = {}

    # 1. sys1-base
    print("⚙️ Calibrating Decision Heads for 'sys1-base'...")
    base_data = [
        # (text, intent, action, priority_0_indexed)
        # 0: info lookup
        ("Where can I find the API documentation and getting started guide?", 0, 0, 1),
        ("How does token pricing work for enterprise tiers?", 0, 0, 1),
        ("What are the latency benchmarks for edge inferencing in Europe?", 0, 0, 1),
        ("Can you explain how WebAssembly executes in V8 isolates?", 0, 0, 1),
        ("What regions are supported for data residency compliance?", 0, 0, 1),
        ("Is there an SDK for TypeScript and Python?", 0, 0, 1),
        ("How do I configure Cloudflare R2 bucket bindings in wrangler.toml?", 0, 0, 1),
        ("Quick general question about release dates.", 0, 0, 2),

        # 1: transaction request
        ("Please charge my card on file and upgrade our account to Team plan.", 1, 1, 3),
        ("Process a refund for our latest monthly invoice #48291.", 1, 1, 3),
        ("Cancel our active subscription at the end of the current billing cycle.", 1, 1, 3),
        ("Transfer $500 from the primary reserve balance to the operating wallet.", 1, 1, 3),
        ("Provision three new edge worker nodes in the Frankfurt cluster.", 1, 1, 3),
        ("Reset API keys and rotate webhook secrets for our production account.", 1, 1, 3),

        # 2: escalation
        ("Our production API is failing with 500 errors across all European nodes, critical outage!", 2, 1, 4),
        ("Security breach detected: unauthorized API tokens accessing customer records.", 2, 1, 4),
        ("I need to speak to an engineering manager immediately, this is impacting enterprise clients.", 2, 1, 4),
        ("System outage: database connections exhausted, SLA breach imminent.", 2, 1, 4),
        ("Emergency: payments are double charging users on checkout, please intervene.", 2, 1, 4),

        # 3: general feedback
        ("Really loving the new dashboard interface, the animations look great!", 3, 0, 0),
        ("The documentation is helpful, but the search could be a little faster on mobile.", 3, 0, 0),
        ("Great session at the developer meetup last week, thanks for sharing.", 3, 0, 0),
        ("Just wanted to say keep up the good work on the open source tools.", 3, 0, 0),
        ("The Nixie tube design aesthetic on the website looks phenomenal.", 3, 0, 0),
    ]
    texts, intents, actions, priorities = zip(*base_data)
    X_base = encode_texts(list(texts))
    clf_base_intent = LogisticRegression(C=5.0, max_iter=200).fit(X_base, np.array(intents))
    clf_base_action = LogisticRegression(C=5.0, max_iter=200).fit(X_base, np.array(actions))
    clf_base_priority = LogisticRegression(C=5.0, max_iter=200).fit(X_base, np.array(priorities))

    heads["sys1-base"] = {
        "heads.choice.primary_intent.weight": torch.from_numpy(clf_base_intent.coef_).float(),
        "heads.choice.primary_intent.bias": torch.from_numpy(clf_base_intent.intercept_).float(),
        "heads.boolean.requires_action.weight": torch.from_numpy(clf_base_action.coef_).float(),
        "heads.boolean.requires_action.bias": torch.from_numpy(clf_base_action.intercept_).float(),
        "heads.score.priority_level.weight": torch.from_numpy(clf_base_priority.coef_).float(),
        "heads.score.priority_level.bias": torch.from_numpy(clf_base_priority.intercept_).float(),
    }

    # 2. spam-detector-v1
    print("⚙️ Calibrating Decision Heads for 'spam-detector-v1'...")
    spam_data = [
        # (text, cat, is_spam, risk_0_indexed)
        # 0: legitimate_inquiry
        ("Hello, I would like to schedule a technical briefing on deploying your platform across our edge points of presence.", 0, 0, 0),
        ("Hi team, what is your pricing for enterprise customers and do you offer SLA guarantees?", 0, 0, 0),
        ("Good morning, we are evaluating your API for integration with our CRM. Where can I find documentation?", 0, 0, 0),
        ("Hello, I am interested in scheduling a demo of your system for our engineering team.", 0, 0, 0),
        ("Hi, does your service support multi-region failover and EU data residency compliance?", 0, 0, 0),
        ("Dr. Jane Doe reaching out regarding technical briefing on deploying Sys1Pop across our global edge points of presence.", 0, 0, 0),
        ("Just a quick question regarding team plan billing details.", 0, 0, 1),

        # 1: commercial_sales_pitch
        ("Hi, I work in the local metro area, and help many local companies. I was hoping I could come by and offer a complimentary cleaning bid? Respond with stop to optout.", 1, 1, 2),
        ("Hey there, our agency specializes in B2B lead generation. Can we set up a quick 10-minute call this Thursday to share how we scaled similar companies?", 1, 1, 2),
        ("Hello, I am reaching out to introduce our cloud cost optimization services. We guarantee 30% savings on AWS. Unsubscribe here.", 1, 1, 2),
        ("Hi, wanted to follow up on my previous note. We provide outsourced SDR and sales staffing solutions. Reply with STOP to unsubscribe.", 1, 1, 2),
        ("Business Development Rep reaching out to offer facilities management bid. Respond with stop to optout.", 1, 1, 2),

        # 2: seo_backlink_spam
        ("Hello team, I noticed your website has great content but low Google ranking. We offer high-quality backlinks and guest posts with DA 80+ to get you to #1 on search engines.", 2, 1, 2),
        ("Hi, I would love to contribute a high quality guest post to your blog with contextual do-follow links. Let me know your editorial rates.", 2, 1, 2),
        ("We offer link building packages and contextual backlinks to boost your domain authority and organic traffic.", 2, 1, 2),
        ("Alex Taylor marketing specialist offering high quality backlinks and guest posts with DA 80+ to get you to #1 on search engines.", 2, 1, 2),

        # 3: crypto_phishing
        ("Urgent: Send 0.1 ETH to verify wallet and receive 5,000 promo tokens immediately to your balance.", 3, 1, 4),
        ("Claim your free 500 USDT airdrop now by connecting your MetaMask web3 wallet before gas fees increase!", 3, 1, 4),
        ("Security alert: Your crypto wallet connection has expired. Connect seed phrase at verify-token.xyz to restore access.", 3, 1, 4),
        ("Reward Bot: Urgent send 0.1 ETH to verify wallet and receive 5,000 promo tokens immediately to your balance.", 3, 1, 4),

        # 4: automated_bot_gibberish
        ("asdf jkl; qwerty 12345 zxcvbnm test string random bot input", 4, 1, 3),
        ("http://spam-link.ru/click?id=999888777 dsa890fdsa890fdsa", 4, 1, 3),
        ("Buy cheap replica watches online free shipping 888777666 discount", 4, 1, 3),
    ]
    texts, cats, is_spams, risks = zip(*spam_data)
    X_spam = encode_texts(list(texts))
    clf_spam_cat = LogisticRegression(C=5.0, max_iter=200).fit(X_spam, np.array(cats))
    clf_is_spam = LogisticRegression(C=5.0, max_iter=200).fit(X_spam, np.array(is_spams))
    clf_risk = LogisticRegression(C=5.0, max_iter=200).fit(X_spam, np.array(risks))

    heads["spam-detector-v1"] = {
        "heads.choice.spam_category.weight": torch.from_numpy(clf_spam_cat.coef_).float(),
        "heads.choice.spam_category.bias": torch.from_numpy(clf_spam_cat.intercept_).float(),
        "heads.boolean.is_spam.weight": torch.from_numpy(clf_is_spam.coef_).float(),
        "heads.boolean.is_spam.bias": torch.from_numpy(clf_is_spam.intercept_).float(),
        "heads.score.risk_score.weight": torch.from_numpy(clf_risk.coef_).float(),
        "heads.score.risk_score.bias": torch.from_numpy(clf_risk.intercept_).float(),
    }

    # 3. intent-router-v1
    print("⚙️ Calibrating Decision Heads for 'intent-router-v1'...")
    intent_data = [
        # (text, route, confirm, urgency_0_indexed)
        # 0: iot_device_control
        ("Dim the bedroom lights to 30% and set thermostat to 70 degrees.", 0, 0, 1),
        ("Turn off the living room desk lamp and kitchen lights.", 0, 0, 1),
        ("Lock the front door and arm the security system.", 0, 1, 2),
        ("Set the living room temperature to 68 degrees.", 0, 0, 1),

        # 1: calendar_scheduling
        ("Schedule a 45-minute sync with Sarah tomorrow at 2pm.", 1, 0, 0),
        ("Create a calendar invite for team standup on Monday at 10am.", 1, 0, 0),
        ("Book a meeting with product design for 30 minutes this afternoon.", 1, 0, 0),
        ("Add dentist appointment to my calendar next Tuesday at 3pm.", 1, 0, 0),

        # 2: multi_action_split
        ("Turn off the living room desk lamp and schedule a focus block for 2 hours.", 2, 1, 2),
        ("Dim the lights to 20% and set an alarm for 7am tomorrow.", 2, 1, 1),
        ("Lock the front door and email John that I have arrived.", 2, 1, 2),

        # 3: general_conversation
        ("What is the capital of Australia?", 3, 0, 0),
        ("Can you explain how WebAssembly executes in V8 isolates?", 3, 0, 0),
        ("How does neural attention work in transformer architectures?", 3, 0, 0),
    ]
    texts, routes, confirms, urgencies = zip(*intent_data)
    X_intent = encode_texts(list(texts))
    clf_intent = LogisticRegression(C=5.0, max_iter=200).fit(X_intent, np.array(routes))
    clf_intent_confirm = LogisticRegression(C=5.0, max_iter=200).fit(X_intent, np.array(confirms))
    clf_intent_urgency = LogisticRegression(C=5.0, max_iter=200).fit(X_intent, np.array(urgencies))

    heads["intent-router-v1"] = {
        "heads.choice.primary_route.weight": torch.from_numpy(clf_intent.coef_).float(),
        "heads.choice.primary_route.bias": torch.from_numpy(clf_intent.intercept_).float(),
        "heads.boolean.requires_confirmation.weight": torch.from_numpy(clf_intent_confirm.coef_).float(),
        "heads.boolean.requires_confirmation.bias": torch.from_numpy(clf_intent_confirm.intercept_).float(),
        "heads.score.urgency_level.weight": torch.from_numpy(clf_intent_urgency.coef_).float(),
        "heads.score.urgency_level.bias": torch.from_numpy(clf_intent_urgency.intercept_).float(),
    }

    # 4. rag-triage-v1
    print("⚙️ Calibrating Decision Heads for 'rag-triage-v1'...")
    rag_data = [
        # (text, action, sufficient, rating_0_indexed)
        # 0: generate_answer
        ("To bind an R2 bucket to your Worker, add r2_buckets to wrangler.toml with binding and bucket_name fields.", 0, 1, 4),
        ("Cloudflare R2 provides S3-compatible object storage with zero egress fees.", 0, 1, 3),
        ("Sys1Pop executes in Cloudflare Workers using Candle WASM inference.", 0, 1, 4),

        # 1: fetch_more_docs
        ("Cloudflare KV is an ultra-low latency key-value store optimized for high-volume reads.", 1, 0, 1),
        ("D1 is Cloudflare's native serverless SQL database built on SQLite.", 1, 0, 1),
        ("Vectorize is a globally distributed vector database to store and query embeddings.", 1, 0, 2),

        # 2: clarify_with_user
        ("Please specify whether you want to deploy to staging or production environment.", 2, 0, 0),
        ("Could you clarify which region you are targeting for compliance?", 2, 0, 0),
        ("Are you looking for Python bindings or TypeScript SDK documentation?", 2, 0, 0),
    ]
    texts, actions, sufficients, ratings = zip(*rag_data)
    X_rag = encode_texts(list(texts))
    clf_rag_action = LogisticRegression(C=5.0, max_iter=200).fit(X_rag, np.array(actions))
    clf_rag_sufficient = LogisticRegression(C=5.0, max_iter=200).fit(X_rag, np.array(sufficients))
    clf_rag_rating = LogisticRegression(C=5.0, max_iter=200).fit(X_rag, np.array(ratings))

    heads["rag-triage-v1"] = {
        "heads.choice.action_required.weight": torch.from_numpy(clf_rag_action.coef_).float(),
        "heads.choice.action_required.bias": torch.from_numpy(clf_rag_action.intercept_).float(),
        "heads.boolean.sufficient_context.weight": torch.from_numpy(clf_rag_sufficient.coef_).float(),
        "heads.boolean.sufficient_context.bias": torch.from_numpy(clf_rag_sufficient.intercept_).float(),
        "heads.score.relevance_rating.weight": torch.from_numpy(clf_rag_rating.coef_).float(),
        "heads.score.relevance_rating.bias": torch.from_numpy(clf_rag_rating.intercept_).float(),
    }

    return heads

def export_all(base_output_dir):
    print("\n📦 Exporting Sys1Pop Production Neural Model Bundles...")
    print("=" * 70)
    print("📥 Loading HuggingFace base model: sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    base_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    base_model.eval()

    base_state_dict = base_model.state_dict()
    config_dict = base_model.config.to_dict()

    heads = calibrate_all_heads(tokenizer, base_model)

    os.makedirs(base_output_dir, exist_ok=True)
    catalog_entries = []

    for model_meta in MODELS_CONFIG:
        model_id = model_meta["model_id"]
        bundle_dir = os.path.join(base_output_dir, model_id)
        os.makedirs(bundle_dir, exist_ok=True)

        # 1. model.safetensors
        full_state_dict = {}
        for k, v in base_state_dict.items():
            full_state_dict[k] = v.contiguous()
        for k, v in heads.get(model_id, {}).items():
            full_state_dict[k] = v.contiguous()

        safetensors_path = os.path.join(bundle_dir, "model.safetensors")
        save_file(full_state_dict, safetensors_path)
        file_size_mb = os.path.getsize(safetensors_path) / (1024 * 1024)

        # 2. tokenizer.json
        tokenizer.save_pretrained(bundle_dir)

        # 3. config.json
        config_path = os.path.join(bundle_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

        # 4. manifest.json
        manifest = {
            "schema_version": "1.0",
            "model_id": model_id,
            "name": model_meta["name"],
            "icon": model_meta["icon"],
            "description": model_meta["description"],
            "architecture": "minilm_l6_v2",
            "hidden_dim": 384,
            "max_seq_len": 512,
            "quantization": "fp32",
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
        manifest_path = os.path.join(bundle_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        catalog_entries.append(manifest)
        print(f"  ✓ Model '{model_id}' exported ({file_size_mb:.2f} MB)")

    # 5. Master catalog.json
    catalog_path = os.path.join(base_output_dir, "catalog.json")
    catalog_data = {
        "schema_version": "1.0",
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "models": catalog_entries
    }
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog_data, f, indent=2)

    print(f"  ✓ Generated master catalog: {catalog_path}")
    print("=" * 70)
    print(f"✅ All {len(MODELS_CONFIG)} production neural models successfully generated!\n")

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
        size_mb = os.path.getsize(os.path.join(bundle_dir, "model.safetensors")) / (1024 * 1024)
        print(f"  ✓ Model verified: {mid} ({size_mb:.2f} MB)")
    print("✅ All catalog models verified!\n")

def main():
    parser = argparse.ArgumentParser(description="Export Sys1Pop production neural models and catalog")
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
