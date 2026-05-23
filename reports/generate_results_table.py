"""Generate the all-results-in-one-place tabular markdown report.

Reads every result JSON in results/ and emits reports/RESULTS.md with one
section per experiment plus per-cell side-by-side comparisons. Run from the
project root:

    python reports/generate_results_table.py
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "RESULTS.md"


def load(p: Path) -> dict:
    return json.load(open(p)) if p.exists() else {}


def main() -> None:
    lstm_legacy = load(ROOT / "results/lstm/lstm_results.json")
    lstm_fixed = load(ROOT / "results/lstm/lstm_results_fixed.json")
    gnn_legacy = load(ROOT / "results/gnn/gnn_gat_source_only.json")
    gnn_fixed_src = load(ROOT / "results/gnn/gnn_gat_source_only_fixed.json")
    gnn_fixed_tl = load(ROOT / "results/gnn_tl/variantA_gat_fixed.json")

    lstm_legacy_src = lstm_legacy.get("source_only", {})
    lstm_fixed_src = lstm_fixed.get("source_only", {})
    lstm_legacy_tl = lstm_legacy.get("transfer", {})
    lstm_fixed_tl = lstm_fixed.get("transfer", {})

    # Thesis baseline numbers (from B.Tech thesis Table 5.2, quoted verbatim).
    thesis_src = {
        "Delhi":    {"R2": 0.6570, "MAE": 37.3301},
        "Kolkata":  {"R2": 0.7861, "MAE": 14.6633},
        "Guwahati": {"R2": 0.5723, "MAE": 16.3012},
    }
    thesis_tl = {
        # (source, target) -> { d_pct -> {"R2": ..., "MAE": ...} }
        ("Kolkata", "Guwahati"): {15:(0.598,None), 30:(0.603,None), 45:(0.5823,None), 60:(0.6271,15.1105)},
        ("Kolkata", "Delhi"):    {15:(0.6939,None), 30:(0.6908,None), 45:(0.6881,None), 60:(0.687,35.7143)},
        ("Guwahati", "Kolkata"): {15:(0.8098,None), 30:(0.8174,13.5344), 45:(0.8164,None), 60:(0.8019,None)},
        ("Guwahati", "Delhi"):   {15:(0.6995,35.91), 30:(0.6877,None), 45:(0.6815,None), 60:(0.6837,None)},
        ("Delhi", "Kolkata"):    {15:(0.7437,None), 30:(0.8189,13.3222), 45:(0.8129,None), 60:(0.8051,None)},
        ("Delhi", "Guwahati"):   {15:(0.5797,None), 30:(0.6381,14.5957), 45:(0.621,None), 60:(0.6133,None)},
    }

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("# All Experiment Results — Tabular Report\n\n")
        fh.write("_All results from the PM2.5 GNN-TL project, in one place._\n\n")
        fh.write(
            "This report covers four experiment families:\n\n"
            "1. **Stage I — Thesis LSTM-TL baseline** (verbatim from B.Tech thesis Table 5.1 and Table 5.2). "
            "Numbers reproduced from the PDF; not re-run.\n"
            "2. **Stage II legacy run** — same code that produced thesis-style chronological 70/15/15 split, "
            "as it stood when I diagnosed the underperformance.\n"
            "3. **Stage II fixed-protocol run** — after the three diagnosed fixes (A: interleaved split, B: GNN "
            "recipe matched to LSTM, C: per-(month,hour) climatology residual).\n"
            "4. **Stage II GNN-TL verification** — for every (source, target, d%) cell, three trainings on identical "
            "data: zero-shot (source weights, no fine-tune), scratch (random init + fine-tune), transfer (source "
            "weights + fine-tune). Transfer is considered real iff transfer.R2 > zero_shot.R2 AND transfer.R2 > scratch.R2.\n\n"
            "All metrics are computed on the *target city's held-out test partition* in raw PM2.5 space "
            "(after un-doing the standardization and, where applicable, adding the climatology back).\n"
        )

        # ----------- 1. Thesis baseline -----------
        fh.write("\n## 1. Stage I — Thesis LSTM-TL baseline (reference)\n\n")
        fh.write("### 1.1 Source-only LSTM (Thesis Table 5.1)\n\n")
        fh.write("| Model | City | R² | MAE (µg/m³) |\n|---|---|--:|--:|\n")
        for c, m in thesis_src.items():
            fh.write(f"| M_{c[0]} | {c} | {m['R2']:.4f} | {m['MAE']:.4f} |\n")

        fh.write("\n### 1.2 LSTM-TL fine-tuned (Thesis Table 5.2 — R² scores)\n\n")
        fh.write("| Source | Target | d=15% | d=30% | d=45% | d=60% | best MAE |\n")
        fh.write("|---|---|--:|--:|--:|--:|--:|\n")
        for (s, t), row in thesis_tl.items():
            r15 = row[15][0]
            r30 = row[30][0]
            r45 = row[45][0]
            r60 = row[60][0]
            mae_vals = [v[1] for v in row.values() if v[1] is not None]
            best_mae = min(mae_vals) if mae_vals else None
            fh.write(
                f"| {s} | {t} | {r15:.4f} | {r30:.4f} | {r45:.4f} | {r60:.4f} | "
                f"{'-' if best_mae is None else f'{best_mae:.4f}'} |\n"
            )
        fh.write(
            "\n**Headline thesis cell:** Delhi → Kolkata @ d=30%, R² = **0.8189**, MAE = **13.3222 µg/m³**. "
            "This is the single best target-city result in the entire thesis Table 5.2 and is the number "
            "the new GNN-TL framework must beat.\n"
        )

        # ----------- 2. Stage II legacy run -----------
        fh.write("\n## 2. Stage II legacy run (pre-fix; chronological split, GNN epochs=10, no climatology)\n\n")
        fh.write("### 2.1 Source-only — LSTM and GAT ST-GNN\n\n")
        fh.write("| City | R² LSTM legacy | MAE LSTM legacy | R² GAT-GNN legacy | MAE GAT-GNN legacy |\n")
        fh.write("|---|--:|--:|--:|--:|\n")
        for c in ["Delhi", "Kolkata", "Guwahati"]:
            ll = lstm_legacy_src.get(c, {})
            lg = gnn_legacy.get(c, {})
            fh.write(
                f"| {c} | {ll.get('R2', float('nan')):.4f} | {ll.get('MAE', float('nan')):.4f} | "
                f"{lg.get('R2', float('nan')):.4f} | {lg.get('MAE', float('nan')):.4f} |\n"
            )
        fh.write(
            "\nObservation: under the legacy chronological 70/15/15 split, the GNN's Guwahati R² is **-0.220** "
            "— worse than predicting the test set's mean.\n"
        )

        fh.write("\n### 2.2 LSTM-TL fine-tuned (Stage II legacy)\n\n")
        fh.write("| Pair@d% | R² | MAE | RMSE | MAPE |\n|---|--:|--:|--:|--:|\n")
        for k in sorted(lstm_legacy_tl):
            v = lstm_legacy_tl[k]
            fh.write(f"| {k} | {v['R2']:.4f} | {v['MAE']:.4f} | {v['RMSE']:.4f} | {v['MAPE']:.4f} |\n")

        # ----------- 3. Stage II fixed-protocol run -----------
        fh.write("\n## 3. Stage II fixed-protocol run (interleaved split + matched recipe + climatology residual)\n\n")
        fh.write("### 3.1 Source-only — LSTM and GAT ST-GNN\n\n")
        fh.write("| City | R² LSTM fixed | MAE LSTM fixed | R² GAT-GNN fixed | MAE GAT-GNN fixed |\n")
        fh.write("|---|--:|--:|--:|--:|\n")
        for c in ["Delhi", "Kolkata", "Guwahati"]:
            ll = lstm_fixed_src.get(c, {})
            lg = gnn_fixed_src.get(c, {})
            fh.write(
                f"| {c} | {ll.get('R2', float('nan')):.4f} | {ll.get('MAE', float('nan')):.4f} | "
                f"{lg.get('R2', float('nan')):.4f} | {lg.get('MAE', float('nan')):.4f} |\n"
            )

        fh.write("\n### 3.2 LSTM-TL fine-tuned (fixed protocol, full metrics)\n\n")
        fh.write("| Pair@d% | R² | MAE | RMSE | MAPE |\n|---|--:|--:|--:|--:|\n")
        for k in sorted(lstm_fixed_tl):
            v = lstm_fixed_tl[k]
            fh.write(f"| {k} | {v['R2']:.4f} | {v['MAE']:.4f} | {v['RMSE']:.4f} | {v['MAPE']:.4f} |\n")

        fh.write("\n### 3.3 GNN-TL Variant A fine-tuned (fixed protocol, full metrics)\n\n")
        fh.write("| Pair@d% | R² transfer | MAE transfer | RMSE transfer | MAPE transfer |\n|---|--:|--:|--:|--:|\n")
        for k in sorted(gnn_fixed_tl):
            v = gnn_fixed_tl[k]["transfer"]
            fh.write(f"| {k} | {v['R2']:.4f} | {v['MAE']:.4f} | {v['RMSE']:.4f} | {v['MAPE']:.4f} |\n")

        # ----------- 4. GNN-TL verification -----------
        fh.write("\n## 4. GNN-TL verification — is knowledge actually transferring?\n\n")
        fh.write(
            "Every fixed-protocol GNN TL cell ran three trainings on the same d% target sample:\n\n"
            "- **Zero-shot**: source-pretrained weights loaded, evaluated on target test, **no fine-tune**.\n"
            "- **Scratch**: random initialization, fine-tuned on the same d% target sample using the same "
            "schedule. No knowledge from the source.\n"
            "- **Transfer**: source-pretrained weights loaded, fine-tuned on the same d% target sample.\n\n"
            "Knowledge transfer is \"real\" iff `transfer.R2 > zero_shot.R2` AND `transfer.R2 > scratch.R2`. "
            "If transfer beats scratch the source pre-training is actively helping; if not, the source is "
            "irrelevant or hurting.\n\n"
        )
        fh.write("### 4.1 Per-cell verification table\n\n")
        fh.write("| Pair@d% | zero-shot R² | scratch R² | **transfer R²** | gain vs scratch | gain vs zero-shot | real_transfer |\n")
        fh.write("|---|--:|--:|--:|--:|--:|:-:|\n")
        n_real = 0
        for k in sorted(gnn_fixed_tl):
            cell = gnn_fixed_tl[k]
            z = cell["zero_shot"]["R2"]
            s = cell["scratch"]["R2"]
            t = cell["transfer"]["R2"]
            r = cell.get("real_transfer", False)
            if r: n_real += 1
            fh.write(
                f"| {k} | {z:+.4f} | {s:+.4f} | **{t:+.4f}** | {t-s:+.4f} | {t-z:+.4f} | "
                f"{'YES' if r else 'NO'} |\n"
            )
        fh.write(f"\n**Verification result: {n_real}/{len(gnn_fixed_tl)} cells show real knowledge transfer.**\n")

        # ----------- 5. Cross-method comparison -----------
        fh.write("\n## 5. Cross-method comparison — the headline table\n\n")
        fh.write(
            "Direct apples-to-apples comparison of the *transfer R²* across:\n"
            "- Thesis LSTM-TL (Stage I)\n"
            "- Stage II legacy LSTM-TL\n"
            "- Stage II fixed LSTM-TL\n"
            "- Stage II fixed GNN-TL (Variant A — pre-train + fine-tune)\n\n"
        )
        fh.write("| Pair@d% | Thesis LSTM | Legacy LSTM | Fixed LSTM | Fixed GAT-GNN | best |\n")
        fh.write("|---|--:|--:|--:|--:|:-:|\n")
        for (s, t), row in thesis_tl.items():
            for d in [15, 30, 45, 60]:
                key = f"{s}->{t}@{d}"
                thesis_r2 = row[d][0]
                legacy_r2 = lstm_legacy_tl.get(key, {}).get("R2")
                fixed_lstm_r2 = lstm_fixed_tl.get(key, {}).get("R2")
                fixed_gnn_r2 = gnn_fixed_tl.get(key, {}).get("transfer", {}).get("R2")
                cells = [
                    ("thesis", thesis_r2),
                    ("legacy", legacy_r2),
                    ("LSTMfix", fixed_lstm_r2),
                    ("GNNfix", fixed_gnn_r2),
                ]
                cells_valid = [(n, v) for n, v in cells if v is not None]
                best_name = max(cells_valid, key=lambda x: x[1])[0] if cells_valid else "-"
                fh.write(
                    f"| {key} | {thesis_r2:.4f} | "
                    f"{'-' if legacy_r2 is None else f'{legacy_r2:.4f}'} | "
                    f"{'-' if fixed_lstm_r2 is None else f'{fixed_lstm_r2:.4f}'} | "
                    f"{'-' if fixed_gnn_r2 is None else f'{fixed_gnn_r2:.4f}'} | "
                    f"{best_name} |\n"
                )

        # ----------- 6. Graph structures used -----------
        fh.write("\n## 6. Graph structures (sanity log)\n\n")
        fh.write(
            "All graphs use the k-NN distance kernel with k=3 and Gaussian-decay edge weights "
            "`w_ij = exp(-d_ij² / 2σ²)`, σ=5 km. Edge counts and average degrees:\n\n"
        )
        fh.write("| City | \\|V\\| | \\|E\\| | avg degree |\n|---|--:|--:|--:|\n")
        # Pull from the first GNN TL cell (Delhi->Kolkata@15) for the graph stats.
        first_cell = next(iter(gnn_fixed_tl.values()))
        delhi_g = first_cell["source_graph"]  # Delhi
        kolkata_g = first_cell["target_graph"]  # Kolkata
        fh.write(f"| Delhi    | {delhi_g['V']} | {delhi_g['E']} | {delhi_g['avg_deg']:.2f} |\n")
        fh.write(f"| Kolkata  | {kolkata_g['V']} | {kolkata_g['E']} | {kolkata_g['avg_deg']:.2f} |\n")
        # Guwahati from a Kolkata->Guwahati or Delhi->Guwahati cell.
        gw_cell = gnn_fixed_tl.get("Kolkata->Guwahati@15") or gnn_fixed_tl.get("Delhi->Guwahati@15")
        gw_g = gw_cell["target_graph"]
        fh.write(f"| Guwahati | {gw_g['V']} | {gw_g['E']} | {gw_g['avg_deg']:.2f} |\n")
        fh.write(
            "\nNote: Guwahati has only 4 stations, so a k=3 k-NN graph is effectively a complete directed graph "
            "(minus self-loops). The inductive ST-GNN's parameter count does *not* depend on \\|V\\|, so the same "
            "model trained on Delhi (\\|V\\|=40) runs forward on Guwahati (\\|V\\|=4) without any shape change — "
            "this is the demonstrable resolution of the parameter-shape mismatch failure mode F-i in the "
            "intercity graph reconstruction problem.\n"
        )

        # ----------- 7. Summary deltas -----------
        fh.write("\n## 7. Summary of improvements (Δ R² over baselines)\n\n")
        fh.write("### 7.1 Source-only\n\n")
        fh.write("| City | thesis LSTM | fixed LSTM | Δ vs thesis | fixed GAT-GNN | Δ vs thesis |\n")
        fh.write("|---|--:|--:|--:|--:|--:|\n")
        for c in ["Delhi", "Kolkata", "Guwahati"]:
            t = thesis_src[c]["R2"]
            lf = lstm_fixed_src.get(c, {}).get("R2", float('nan'))
            gf = gnn_fixed_src.get(c, {}).get("R2", float('nan'))
            fh.write(f"| {c} | {t:.4f} | {lf:.4f} | {lf-t:+.4f} | {gf:.4f} | {gf-t:+.4f} |\n")

        fh.write("\n### 7.2 Transfer (headline cells)\n\n")
        for (s, t), label in [
            (("Delhi","Kolkata"), "Delhi→Kolkata @ 30% (thesis headline)"),
            (("Delhi","Guwahati"), "Delhi→Guwahati @ 30%"),
            (("Kolkata","Guwahati"), "Kolkata→Guwahati @ 30%"),
            (("Guwahati","Kolkata"), "Guwahati→Kolkata @ 30%"),
        ]:
            d = 30
            key = f"{s}->{t}@{d}"
            thesis_r2 = thesis_tl[(s,t)][d][0]
            fixed_lstm_r2 = lstm_fixed_tl.get(key, {}).get("R2")
            fixed_gnn_r2 = gnn_fixed_tl.get(key, {}).get("transfer", {}).get("R2")
            fh.write(f"- **{label}**: thesis {thesis_r2:.4f} → fixed-LSTM {fixed_lstm_r2:.4f} → fixed-GNN {fixed_gnn_r2:.4f}\n")

    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
