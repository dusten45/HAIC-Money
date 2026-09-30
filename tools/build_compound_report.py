"""Build the reconstructed, source-cited current-method report (not a submission receipt)."""
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = "https://github.com/2026-HAIC/Participants/blob/dfb7a2de2178825ca5c5ce20bab01ba67052ba31/"


def build():
    styles = getSampleStyleSheet()
    styles["Normal"].fontSize = 10
    styles["Normal"].leading = 14
    styles["Normal"].spaceAfter = 8
    styles["Heading1"].textColor = colors.HexColor("#123c58")
    story = []

    def p(text, style="Normal"):
        story.append(Paragraph(text, styles[style]))

    p("HAIC: obstacle-curve speed control", "Title")
    p("Reconstructed technical report | 30 September 2026", "Heading2")
    p("The original report.pdf was unavailable. This is a newly authored report from pinned official sources and existing project artifacts, not a recovered original or a claim of competition readiness. The live competition portal could not be inspected; submission-specific schedule and quota remain unknown.")
    p("1. Contract and evidence", "Heading1")
    p("The variables-6 interface provides four 84 x 84 grayscale frames and accepts steering, accelerator and brake. The policy must use CPU inference. Road departures are not universally defined as immediate disqualification: prolonged negative-reward actions and leaving the playfield can terminate a run. Grass friction is lower, so deliberate departures require performance evidence, not an assumption of benefit. [1, 2, 3]")
    p("The active bare-checkpoint path is a camera-based controller. It estimates road centers, obstacle position and speed from the allowed pixels. The baseline neural weights are not the effective road-following policy after road detection. DrQ and explicit exported-policy routes are separate. [4]")
    p("Submission-12 videos report completed times of 23.60, 28.18 and 25.44 seconds. Earlier analysis found rendered motion slowing near obstacles. These videos lack source/package binding and physical speed/action telemetry. They motivate investigation but do not prove that the current candidate is faster. [5]")
    p("2. Mechanism and preregistered change", "Heading1")
    p("The control can apply at least 0.04 supplemental brake when speed exceeds the adaptive compound target by only 0.5 estimated units. This minimum can overcorrect small errors. The candidate ramps the supplement with min(1, excess / 2), where excess = speed - command target - 0.5. The inherited base speed brake remains a floor. At excess of at least 2, behavior is exactly the control. [4, 6]")
    p("Eligibility requires a currently detected far/moderate compound obstacle, target above 30 and at most 38, image-space sweep from 6 inclusive to 12 exclusive, and the existing final steering safety veto. Detector misses, close obstacles, sharp curves and large steering requests retain the existing policy. This does not assume that losing sight of an obstacle means it has been passed.")
    story.append(PageBreak())
    p("3. Validation and limits", "Heading1")
    p("The change is tested first on synthetic observations and stateful action comparisons. Required properties are unchanged targets and lateral commands, exclusive gas/brake, the base-brake floor, maximum relief 0.04, and exact recovery outside the eligible region. Unit tests do not establish closed-loop safety or lap-time gains.")
    p("The registered diagnostic compares one control and one candidate on seven previously consumed or video-recovered cells: (1,42), (1,11), (1,17), (1,21), (1,516237), (2,644062), (3,1007). Each run has at most 1,000 action decisions and frame skip 4. These are development regressions, not independent fresh confirmation. Frozen confirmation/blind reservations are excluded. [6]")
    p("The candidate is rejected on operational failure, lost finishes, greater collision/damage, worse off-track behavior, or lower progress on paired failures. Retention additionally requires lower summed lap time on paired completed cells. All attempted outcomes are recorded; no candidate is retuned after seeing this comparison. No SOTA promotion follows from this diagnostic. [6]")
    p("Outcome updates are recorded in experiments/compound-brake-onset-v1-result.json when available. This report describes the registered method; it does not anticipate the result. No new learning run, official submission, or reserved evaluation is authorized by this report.")
    p("References", "Heading1")
    refs = [
        ("1", "Official participant contract, README.md, pinned variables-6 commit, accessed 2026-09-30", OFFICIAL + "README.md"),
        ("2", "Official observation and termination implementation", OFFICIAL + "env_wrapper.py"),
        ("3", "Official vehicle dynamics and grass friction", OFFICIAL + "core/vendor/car_dynamics.py"),
    ]
    for number, title, url in refs:
        p(f'[{number}] {escape(title)}. <link href="{url}" color="blue">Source</link>.')
    p("[4] Project agent.py: _CompoundClearingBrakeCarryController and _CompoundBrakeOnsetController; source is frozen by the diagnostic receipt before execution.")
    p("[5] experiments/aggressive-compound-pace-v1-result.json and experiments/compound-clearing-brake-carry-v1-result.json. Existing diagnostic evidence, not matched current-policy confirmation.")
    p("[6] experiments/compound-brake-onset-v1.json. Preregistered single-factor protocol; initial protocol commit fc31f21.")

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(0.7 * inch, 0.42 * inch, "HAIC | Reconstructed method report | Diagnostic evidence only")
        canvas.drawRightString(7.55 * inch, 0.42 * inch, str(doc.page))

    SimpleDocTemplate(str(ROOT / "report.pdf"), rightMargin=0.7 * inch,
                      leftMargin=0.7 * inch, topMargin=0.65 * inch,
                      bottomMargin=0.65 * inch).build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build()
