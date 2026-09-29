from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=21,
            leading=25,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#10233f"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#516274"),
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#0b5c75"),
            spaceBefore=12,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodySmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.8,
            leading=12,
            textColor=colors.HexColor("#253447"),
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableSmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.2,
            leading=9,
            textColor=colors.HexColor("#1e2d3d"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Foot",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=7,
            leading=9,
            textColor=colors.HexColor("#607386"),
        )
    )
    return styles


def _p(text: str, style):
    return Paragraph(text.replace("&", "&amp;"), style)


def _table(rows, widths, styles, header=True):
    wrapped = []
    for row_index, row in enumerate(rows):
        wrapped.append([_p(str(cell), styles["TableSmall"]) for cell in row])
    table = Table(wrapped, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c7d3df")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b5c75")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ]
        )
    for row_index in range(1 if header else 0, len(rows)):
        if row_index % 2:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#f1f6f9")))
    table.setStyle(TableStyle(commands))
    return table


def build(output: Path) -> None:
    styles = _styles()
    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="HAIC 2026 Strategy and Experiment Control Report",
        author="HAIC research operations",
    )
    story = [
        _p("HAIC 2026 Strategy and Experiment Control Report", styles["ReportTitle"]),
        _p("Three-lane comparison, literature mapping, and submission-safe improvement workflow | 2026-09-22", styles["Subtitle"]),
        HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#29a6a6")),
        Spacer(1, 8),
        _p("Executive conclusion", styles["Section"]),
        _p("The current submission baseline is <b>ppo_actor_only</b>. It completed 6 of 8 held-out episodes (0.75 completion rate), with a 19,320 ms median finished lap and 22,420 ms P90 finished lap. It is the only lane currently backed by a packaged submission archive and CPU smoke result.", styles["BodySmall"]),
        _p("The <b>ppo_cem</b> lane is not promoted: in the paired fast-budget held-out comparison it completed 0 of 10 episodes and reached mean progress 0.074530, below PPO-only at 0.077411. The <b>vision_corridor_teacher</b> lane remains a teacher and diagnostic reference, not a submission runtime.", styles["BodySmall"]),
        _p("Three-lane comparison contract", styles["Section"]),
        _table(
            [
                ["Lane", "Role", "Promotion evidence"],
                ["ppo_actor_only", "Submission candidate", "Strict held-out or official improvement"],
                ["ppo_cem", "Model-based candidate", "Same-split improvement after latency and package checks"],
                ["vision_corridor_teacher", "Teacher and diagnostic", "Evidence only; never direct submission runtime"],
            ],
            [35 * mm, 49 * mm, 91 * mm],
            styles,
        ),
        _p("Comparison order: completion rate, median finished lap time, P90 finished lap time, unfinished progress, collisions and damage, then inference cost. Tune selects experiments; held-out and official evidence decide SOTA.", styles["BodySmall"]),
        _p("Evidence and provenance", styles["Section"]),
        _table(
            [
                ["Evidence", "Observation", "Source"],
                ["PPO actor held-out", "6/8 complete; progress 0.788044; median 19320 ms; P90 22420 ms", "artifacts/haic/final-ppo-actor-selection.json"],
                ["PPO plus CEM", "0/10 complete; mean progress 0.074530", "artifacts/haic/task5-eval-fast-fullcap/summary.json"],
                ["Corridor teacher", "3/3 complete; mean progress 0.996622", "artifacts/haic/corridor-controller-candidate-v3/benchmark.json"],
                ["Submission smoke", "Planner disabled; finite action; about 203 MB RSS", "artifacts/haic/final-ppo-actor-selection.json"],
            ],
            [35 * mm, 79 * mm, 61 * mm],
            styles,
        ),
        _p("Literature to experiment mapping", styles["Section"]),
        _p("<b>PPO.</b> The model-free actor baseline is tested with fixed seeds, identical maps, decision cap, CPU budget, and package smoke. The paper supports the algorithm choice, not a claim about a particular HAIC checkpoint.", styles["BodySmall"]),
        _p("<b>DAgGER.</b> Learner-induced observations motivate train-only teacher corrections. Held-out maps must remain separate, and map geometry cannot enter submission inference.", styles["BodySmall"]),
        _p("<b>MBPO and short model rollouts.</b> Model error can harm policy quality. Any new CEM experiment must measure model error, planner latency, and the same held-out episodes before promotion.", styles["BodySmall"]),
        _p("<b>Domain randomization.</b> New local maps, obstacle layouts, and visual perturbations expand train and stress-test coverage. They do not replace official tracks or authorize hidden state at inference.", styles["BodySmall"]),
        _p("Next experiment protocol", styles["Section"]),
        _p("1. Freeze the current SOTA checkpoint and split manifest. 2. Select one hypothesis and one strategy lane. 3. Write to a new timestamped artifact directory. 4. Run train, tune, held-out, official diagnostics, package validation, and CPU smoke. 5. Sync RESULTS.md, compare all lanes, and update SOTA.md only when restriction and strict-improvement gates pass.", styles["BodySmall"]),
        _p("References", styles["Section"]),
        _p("1. Schulman et al., <i>Proximal Policy Optimization Algorithms</i>, 2017. https://arxiv.org/abs/1707.06347<br/>2. Ross, Gordon, and Bagnell, <i>A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning</i>, AISTATS 2011. https://proceedings.mlr.press/v15/ross11a.html<br/>3. Janner et al., <i>When to Trust Your Model: Model-Based Policy Optimization</i>, NeurIPS 2019. https://papers.nips.cc/paper/2019/hash/5faf461eff3099671ad63c6f3f094f7f-Abstract.html<br/>4. Tobin et al., <i>Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World</i>, 2017. https://arxiv.org/abs/1703.06907", styles["BodySmall"]),
        Spacer(1, 8),
        _p("This report records measured evidence and experiment design. It does not claim an improvement or successful external submission before those checks are completed.", styles["Foot"]),
    ]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#607386"))
        canvas.drawString(16 * mm, 8 * mm, "HAIC research operations")
        canvas.drawRightString(A4[0] - 16 * mm, 8 * mm, f"Page {doc.page}")
        canvas.restoreState()

    output.parent.mkdir(parents=True, exist_ok=True)
    document.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("report.pdf"))
    build(parser.parse_args().output)
