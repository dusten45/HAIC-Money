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
    p("Reconstructed technical report | Updated 1 October 2026", "Heading2")
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
    p("The onset experiment completed 14 runs: both arms finished 4/7, with identical trajectories and zero changed brake calls. It is INCONCLUSIVE and was not activated. Seeds 42, 11 and 21 failed in both arms. The recovered video maps finished in 23.66, 28.12 and 25.26 seconds; the second map had two collisions. Full evidence is in experiments/compound-brake-onset-v1-result.json.")
    p("The separately registered visible-compound-base-brake-v1 tested base-only braking in visible target-30 hazards. It was REJECTED: seed17 changed from a clean 25.32s finish to an off-track failure at 54.82%. At equal progress 53.49%, speed rose from 30.49 to 36.99, but the later steering unwound to zero. Seed42/21 also increased off-road samples. Execution stopped after nine complete runs; the next candidate was interrupted and remaining cells were not run. Neither candidate became the active policy. [7]")
    story.append(PageBreak())
    p("4. Camera geometry and passing commitment", "Heading1")
    p("A source-bound trace reproduced all four baseline trajectories exactly. At seed11 step47 and seed21 step104, row42 disappeared while rows46/50/54 still described a left bend. Substitution of image center for the missing row reversed the road steering request. The observed-target repair uses the actual interpolated or clamped road endpoint without fabricating another detected row or changing visibility confidence. [8]")
    p("This is motivated by choosing a goal on an observed path, not a direct implementation of metric pure pursuit. Coulter describes the relationship between goal displacement and curvature and the limits of lookahead in sharp turns. Nav2 combines path tracking with curvature/proximity regulation. HAIC supplies images rather than a metric path and pose; neither source establishes speed gains here. [9, 10]")
    p("The geometry-only candidate recovered seed11 to a collision-free22.20s finish and seed21 to27.92s, but seed21 gained one collision and damage0.2. The preregistered comparator therefore REJECTED it after two pairs; five remaining cells were not run. At the collision approach, the obstacle stayed left of the vehicle while its relation to the estimated road center changed, reversing the passing side at image row51.45. [8]")
    p("The next registered candidate preserves the observed-road repair and commits the chosen passing side from row44, the existing near-obstacle boundary, instead of allowing reselection until52. Distant reselection and the original four-miss latch remain unchanged. Targets, pedals, model and official environment remain frozen. This has an identity-tracking limitation: adjacent sequential obstacles without a detection gap can inherit the previous passing side. [11]")
    p("This second candidate was also REJECTED after two pairs. Seed11 remained clean22.20s, but seed21 crashed at81.95% progress with five contacts and damage1.0. The other five cells were not run. Neither candidate is active. At steps273/274, tiny left recentering suppressed rightward obstacle avoidance although the far road bent right. The obstacle side remained stable; this was not an identity-switch failure. A separately registered geometry-supported arbitration candidate addresses this mechanism without another distance or speed threshold. [11]")
    p("Offline pixels also reveal white-curb false positives. Whole-component brightness rejection was registered but not run: a real orange obstacle touching a white curb might also be rejected. On seed11/21 the replacement real detection produces identical actions, so curb filtering alone is not claimed to solve their failures.")
    p("The geometry-supported candidate compares the farthest and nearest observed road centers. Opposing recentering is allowed to combine with avoidance when that observed displacement does not support the retained turn. Evidence is cleared every action and reset. Existing preview-transition protection and actual straight/curve classification remain unchanged. Steering-dependent speed gates may still change emitted pedals; unchanged formulas do not imply identical speed actions. Endpoint direction can alias S-curves, and the passing latch still lacks obstacle identity tracking. [12]")
    p("It was REJECTED after six pairs: finishes rose from3/6 to6/6, with seed11/21 clean22.20/27.84s and seed42 at24.98s with three contacts instead of a five-contact crash. However, track2/644062 increased contacts2 to3, damage0.4 to0.6 and time28.12 to28.98s. Three paired completed laps totaled1.64s slower. Track3/1007 was not run. All three candidates remain inactive; the original controller and model are preserved. More robust joint road/obstacle clearance reasoning is needed before claiming an upgrade. [12]")
    story.append(PageBreak())
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
    p("[7] experiments/visible-compound-base-brake-v1.json and its -result.json record. Protocol 7bc76ec, boundary amendment e84a194; source/model hashes and raw outcomes retained.")
    p("[8] experiments/corridor-failure-trace-v1-result.json and experiments/observed-road-target-v1-result.json; matched consumed development evidence, not fresh confirmation.")
    p('[9] R. C. Coulter, Implementation of the Pure Pursuit Path Tracking Algorithm, CMU-RI-TR-92-01, 1992. <link href="https://publications.ri.cmu.edu/storage/publications/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf" color="blue">Author report</link>. Accessed2026-09-30.')
    p('[10] Nav2 Regulated Pure Pursuit Controller, author documentation. <link href="https://github.com/ros-navigation/navigation2/blob/main/nav2_regulated_pure_pursuit_controller/README.md" color="blue">Source</link>. Accessed2026-09-30.')
    p("[11] experiments/observed-road-side-commit-v1.json and its -result.json. Preregistered row44 passing commitment; original control comparator, no tuning during evaluation.")
    p("[12] experiments/observed-curve-arbitration-v1.json and its -result.json. Geometry-supported retention protocol; consumed development cells only, not SOTA or unseen-track certification.")

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
