/**
 * Canonical Cloudflare Worker Service Binding Caller Example
 * 
 * Demonstrates:
 * 1. Binding to Sys1Pop via Cloudflare Service Bindings (<0.2ms isolate-to-isolate IPC).
 * 2. Progressive Enhancement: If Sys1Pop is unconfigured or absent, the caller operates
 *    seamlessly with local heuristics/logic without crashing or failing.
 * 3. Deep Neural Inference via Sys1Pop .decide() when Sys1Pop is bound.
 * 
 * In your caller worker's wrangler.toml (optional):
 * [[services]]
 * binding = "SYS1POP"
 * service = "sys1pop"
 */

import { Sys1Pop } from "@sys1pop/sdk";

export interface Env {
  SYS1POP?: { fetch: typeof fetch }; // Optional Cloudflare Service Binding
}

export interface FormSubmission {
  site: string;
  name: string;
  email: string;
  message: string;
  honeypot_filled?: boolean;
}

export interface TriageVerdict {
  isSpam: boolean;
  confidence: number;
  category: string;
  riskScore: number;
  flags: string[];
}

export async function evaluateFormSubmission(
  submission: FormSubmission,
  env: Env
): Promise<TriageVerdict> {
  const flags: string[] = [];

  // =========================================================================
  // Stage 1: Zero-CPU Instant Heuristics in Caller Worker (Runs Always)
  // =========================================================================
  if (submission.honeypot_filled) {
    flags.push("honeypot_triggered");
    return {
      isSpam: true,
      confidence: 1.0,
      category: "automated_bot_gibberish",
      riskScore: 5.0,
      flags,
    };
  }

  const lowerMsg = submission.message.toLowerCase();
  if (lowerMsg.includes("seo ranking") || lowerMsg.includes("backlink")) {
    flags.push("seo_pitch_detected");
  }
  if (submission.email.endsWith(".ru") || submission.email.includes("tempmail")) {
    flags.push("suspicious_domain");
  }

  // =========================================================================
  // Progressive Enhancement: Work cleanly if Sys1Pop is not configured
  // =========================================================================
  const sys1 = new Sys1Pop(env.SYS1POP);

  if (!sys1.isConfigured()) {
    // Caller operates out of the box with heuristic checks
    const isSpamHeuristic = flags.length > 0;
    return {
      isSpam: isSpamHeuristic,
      confidence: isSpamHeuristic ? 0.8 : 0.5,
      category: isSpamHeuristic ? "commercial_sales_pitch" : "legitimate_inquiry",
      riskScore: isSpamHeuristic ? 3.5 : 1.0,
      flags: [...flags, "sys1pop_unconfigured_heuristic_only"],
    };
  }

  // =========================================================================
  // Stage 2: Deep Neural Inference via Sys1Pop .decide()
  // =========================================================================
  try {
    const res = await sys1.decide({
      state: `Site: ${submission.site} | Name: ${submission.name} | Email: ${submission.email} | Message: ${submission.message}`,
      questions: [
        { type: "boolean", id: "is_spam" },
        {
          type: "choice",
          id: "category",
          options: [
            "legitimate_inquiry",
            "commercial_sales_pitch",
            "seo_backlink_spam",
            "crypto_phishing",
            "automated_bot_gibberish",
          ],
        },
        { type: "score", id: "risk_score", min: 1, max: 5 },
      ],
    });

    let isSpam = res.getBoolean("is_spam") ?? false;
    let confidence = res.getProbability("is_spam") ?? 0.5;
    let category = res.getChoice("category") ?? "legitimate_inquiry";
    let riskScore = res.getScore("risk_score") ?? 1.0;

    // Escalate if heuristic flags triggered on ambiguous neural score
    if (flags.length > 0 && !isSpam && confidence < 0.8) {
      isSpam = true;
      riskScore = Math.max(riskScore, 3.5);
    }

    return {
      isSpam,
      confidence,
      category,
      riskScore,
      flags,
    };
  } catch (err) {
    // Graceful fallback if Sys1Pop service is temporarily unreachable
    console.warn("Sys1Pop decision engine error, falling back to heuristics:", err);
    return {
      isSpam: flags.length > 0,
      confidence: 0.5,
      category: "unverified",
      riskScore: 1.0,
      flags: [...flags, "sys1pop_runtime_fallback"],
    };
  }
}
