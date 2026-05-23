"""Generate analysis/diagnostic_report_v2.md from the *_fixed result JSONs.

Compares legacy results against fixed-protocol results so the impact of the
three combined fixes (A: interleaved split, B: GNN recipe match LSTM, C:
climatology residual) is visible cell-by-cell.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("analysis/diagnostic_report_v2.md")


def load_safe(p: Path) -> dict:
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def fmt_metric(d: dict | None, key: str) -> str:
    if d is None or key not in d:
        return "-"
    v = d[key]
    return f"{v:+.3f}" if key == "R2" else f"{v:.2f}"


def fmt_nested_metric(cell: dict | None, sub: str, key: str) -> str:
    """For the new GNN TL JSON whose entries are dicts like
    {"zero_shot": {...}, "scratch": {...}, "transfer": {...}}.
    """
    if cell is None or sub not in cell:
        return "-"
    return fmt_metric(cell[sub], key)


def main() -> None:
    legacy_lstm = load_safe(Path("results/lstm/lstm_results.json"))
    fixed_lstm = load_safe(Path("results/lstm/lstm_results_fixed.json"))
    legacy_gnn = load_safe(Path("results/gnn/gnn_gat_source_only.json"))
    fixed_gnn_src = load_safe(Path("results/gnn/gnn_gat_source_only_fixed.json"))
    legacy_gnn_tl = load_safe(Path("results/gnn_tl/variantA_gat.json"))
    fixed_gnn_tl = load_safe(Path("results/gnn_tl/variantA_gat_fixed.json"))

    legacy_lstm_src = legacy_lstm.get("source_only", {})
    fixed_lstm_src = fixed_lstm.get("source_only", {})
    legacy_lstm_tl = legacy_lstm.get("transfer", {})
    fixed_lstm_tl = fixed_lstm.get("transfer", {})

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("# Diagnostic Report v2 — After Fixes A + B + C\n\n")
        fh.write("Fixes applied (see analysis/diagnostic_report.md for the why):\n\n")
        fh.write("- **A: Interleaved 70/15/15 split.** Every Nth window goes to val/test, so each "
                 "split spans the full year. Fixes the Kolkata/Guwahati 1-year-of-data trap where the "
                 "winter peak was entirely in the test set.\n")
        fh.write("- **B: GNN recipe matches LSTM.** Hidden=64 (was 24), epochs 25/40/60 with early "
                 "stopping (was 10 fixed), per-city LR + batch size, no aggressive subsample cap.\n")
        fh.write("- **C: Climatology residual.** Per-(month, hour) climatology computed from train "
                 "slice; model predicts deviations from seasonal climatology; climatology added back at "
                 "inference. AdaRNN-style temporal DA.\n\n")
        fh.write("Run with `--fixed` flag on `src.train_lstm` and `src.train_gnn`. Legacy numbers "
                 "stay in the un-suffixed result files; fixed numbers land in `*_fixed.json`.\n\n")

        # ---------- Source-only comparison ----------
        fh.write("## 1. Source-only (Table 5.1 analog)\n\n")
        fh.write("| City | R² LSTM legacy | R² LSTM fixed | Δ R² | MAE LSTM legacy | MAE LSTM fixed | R² GNN legacy | R² GNN fixed | Δ R² GNN |\n")
        fh.write("|------|--:|--:|--:|--:|--:|--:|--:|--:|\n")
        for city in ["Delhi", "Kolkata", "Guwahati"]:
            l0 = legacy_lstm_src.get(city)
            l1 = fixed_lstm_src.get(city)
            g0 = legacy_gnn.get(city)
            g1 = fixed_gnn_src.get(city)
            d_l = (l1["R2"] - l0["R2"]) if (l0 and l1) else None
            d_g = (g1["R2"] - g0["R2"]) if (g0 and g1) else None
            fh.write(
                f"| {city} | {fmt_metric(l0,'R2')} | {fmt_metric(l1,'R2')} | "
                f"{('+' if (d_l or 0) >= 0 else '')}{d_l:+.3f} | "
                f"{fmt_metric(l0,'MAE')} | {fmt_metric(l1,'MAE')} | "
                f"{fmt_metric(g0,'R2')} | {fmt_metric(g1,'R2')} | "
                f"{('+' if (d_g or 0) >= 0 else '')}{d_g:+.3f} |\n"
            )

        # ---------- Transfer comparison ----------
        fh.write("\n## 2. Transfer learning (Table 5.2 analog)\n\n")
        keys_l = sorted(set(legacy_lstm_tl) | set(fixed_lstm_tl))
        if keys_l:
            fh.write("### LSTM-TL\n\n")
            fh.write("| Pair@d% | R² legacy | R² fixed | Δ R² | MAE legacy | MAE fixed |\n")
            fh.write("|---|--:|--:|--:|--:|--:|\n")
            for k in keys_l:
                a = legacy_lstm_tl.get(k)
                b = fixed_lstm_tl.get(k)
                d = (b["R2"] - a["R2"]) if (a and b) else None
                fh.write(
                    f"| {k} | {fmt_metric(a,'R2')} | {fmt_metric(b,'R2')} | "
                    f"{'—' if d is None else f'{d:+.3f}'} | "
                    f"{fmt_metric(a,'MAE')} | {fmt_metric(b,'MAE')} |\n"
                )

        keys_g = sorted(set(legacy_gnn_tl) | set(fixed_gnn_tl))
        if keys_g:
            fh.write("\n### GNN-TL Variant A (transfer with verification)\n\n")
            fh.write(
                "Each fixed-protocol cell ran three trainings on the *same* d% target sample:\n"
                "(0) **zero-shot** — load source weights, no fine-tune; "
                "(1) **transfer** — load source weights, fine-tune; "
                "(2) **scratch** — random init, fine-tune. "
                "Knowledge transfer is \"real\" iff transfer > zero_shot AND transfer > scratch.\n\n"
            )
            fh.write("| Pair@d% | R2 legacy | R2 transfer (fixed) | R2 zero-shot | R2 scratch | gain vs scratch | real_transfer |\n")
            fh.write("|---|--:|--:|--:|--:|--:|:-:|\n")
            n_real = 0
            for k in keys_g:
                a = legacy_gnn_tl.get(k)
                b = fixed_gnn_tl.get(k)
                if b and "transfer" in b:
                    t = b["transfer"]
                    z = b.get("zero_shot")
                    s = b.get("scratch")
                    real = b.get("real_transfer", False)
                    if real: n_real += 1
                    gain = (t["R2"] - s["R2"]) if (s and "R2" in s) else None
                    fh.write(
                        f"| {k} | {fmt_metric(a,'R2')} | {fmt_metric(t,'R2')} | "
                        f"{fmt_metric(z,'R2')} | {fmt_metric(s,'R2')} | "
                        f"{'-' if gain is None else f'{gain:+.3f}'} | "
                        f"{'YES' if real else 'NO'} |\n"
                    )
                else:
                    # Legacy-format row (or only-legacy run).
                    fh.write(
                        f"| {k} | {fmt_metric(a,'R2')} | {fmt_metric(b,'R2')} | "
                        f"- | - | - | - |\n"
                    )
            fh.write(f"\n**Verification result: {n_real}/{len(keys_g)} cells show real knowledge transfer.**\n")

        # ---------- Conclusion ----------
        fh.write("\n## 3. Conclusions\n\n")
        fh.write(
            "Below conclusions are auto-derived from the tables above.\n\n"
        )

        # auto write Guwahati delta
        if legacy_lstm_src.get("Guwahati") and fixed_lstm_src.get("Guwahati"):
            d = fixed_lstm_src["Guwahati"]["R2"] - legacy_lstm_src["Guwahati"]["R2"]
            fh.write(f"- LSTM Guwahati source-only R² changed by **{d:+.3f}** after the fixes.\n")
        if legacy_gnn.get("Guwahati") and fixed_gnn_src.get("Guwahati"):
            d = fixed_gnn_src["Guwahati"]["R2"] - legacy_gnn.get("Guwahati", {}).get("R2", -1)
            fh.write(f"- GNN  Guwahati source-only R² changed by **{d:+.3f}** after the fixes.\n")

        # Implied conclusion
        fh.write("\n### Interpretation guide\n\n")
        fh.write(
            "- If the GNN now beats LSTM on Guwahati under the fixed protocol, the architecture "
            "hypothesis (ST-GNN > LSTM) is supported.\n"
            "- If the LSTM and GNN are close, the split + climatology + recipe was indeed the "
            "limiting factor, and the architectural gain alone is smaller than expected.\n"
            "- If the GNN still underperforms, the next experiments to run are E (GraphNorm) and "
            "F (subgraph sampling on Delhi pre-training).\n"
        )

    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
