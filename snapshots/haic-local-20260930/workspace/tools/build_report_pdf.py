"""Build the current HAIC strategy and literature report."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "report.pdf"
FONT = Path(r"C:\Windows\Fonts\malgun.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\malgunbd.ttf")


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("Malgun", str(FONT)))
    pdfmetrics.registerFont(TTFont("Malgun-Bold", str(FONT_BOLD)))


def paragraph_styles():
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleKR", parent=styles["Title"], fontName="Malgun-Bold", fontSize=22,
            leading=28, alignment=TA_CENTER, textColor=colors.HexColor("#16324F"),
            spaceAfter=12,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleKR", parent=styles["Normal"], fontName="Malgun", fontSize=10,
            leading=15, alignment=TA_CENTER, textColor=colors.HexColor("#4B6173"),
            spaceAfter=18,
        ),
        "h1": ParagraphStyle(
            "H1KR", parent=styles["Heading1"], fontName="Malgun-Bold", fontSize=15,
            leading=20, textColor=colors.HexColor("#16324F"), spaceBefore=10, spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "H2KR", parent=styles["Heading2"], fontName="Malgun-Bold", fontSize=11,
            leading=16, textColor=colors.HexColor("#245B7A"), spaceBefore=7, spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "BodyKR", parent=styles["BodyText"], fontName="Malgun", fontSize=9.2,
            leading=14, textColor=colors.HexColor("#1D2730"), spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "SmallKR", parent=styles["BodyText"], fontName="Malgun", fontSize=7.4,
            leading=10, textColor=colors.HexColor("#40505C"), spaceAfter=3,
        ),
        "ref": ParagraphStyle(
            "RefKR", parent=styles["BodyText"], fontName="Malgun", fontSize=8,
            leading=11, leftIndent=8, firstLineIndent=-8, spaceAfter=4,
        ),
    }


def p(text: str, style) -> Paragraph:
    return Paragraph(text, style)


def table(data, widths, header=True, font_size=7.5):
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B7C5CF")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("FONTNAME", (0, 0), (-1, -1), "Malgun"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDEBF2")),
                ("FONTNAME", (0, 0), (-1, 0), "Malgun-Bold"),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#16324F")),
            ]
        )
    table.setStyle(TableStyle(commands))
    return table


def header_footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#C8D4DC"))
    canvas.line(18 * mm, 13 * mm, 192 * mm, 13 * mm)
    canvas.setFont("Malgun", 7)
    canvas.setFillColor(colors.HexColor("#60717D"))
    canvas.drawString(18 * mm, 8 * mm, "HAIC strategy report")
    canvas.drawRightString(192 * mm, 8 * mm, f"{document.page}")
    canvas.restoreState()


def build() -> Path:
    register_fonts()
    styles = paragraph_styles()
    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=19 * mm, title="HAIC 전략 비교 및 장애물 대응 보고서",
        author="Codex",
    )
    story = []
    story.append(Spacer(1, 18 * mm))
    story.append(p("HAIC 전략 비교 및 장애물 대응 보고서", styles["title"]))
    story.append(p("2026-09-22 현재판 · 결과 DB와 제출 제약을 함께 기록", styles["subtitle"]))
    story.append(
        table(
            [
                [p("항목", styles["small"]), p("현재 확인", styles["small"])],
                [p("대회 목표", styles["body"]), p("사용자 제공 선두: Track 1 17초, Track 2 20초", styles["body"])],
                [p("현재 제출 후보", styles["body"]), p("haic-obstacle-risk-ppo-actor.zip · PPO-only · strict load 통과", styles["body"])],
                [p("핵심 실패", styles["body"]), p("공식 트랙에 장애물을 추가하면 PPO와 corridor 모두 DNF", styles["body"])],
                [p("권장 방향", styles["body"]), p("장애물 위험 예측 헤드 + 목표 속도 + 후보 행동 안전 필터", styles["body"])],
            ],
            [38 * mm, 132 * mm],
            font_size=8.2,
        )
    )
    story.append(Spacer(1, 7 * mm))
    story.append(p("요약", styles["h1"]))
    story.append(p(
        "현재 제출 정책은 화면에서 행동을 직접 출력하는 visual PPO입니다. 장애물 보상은 존재하지만 "
        "거리, 횡방향 겹침, 제동 여유, 미래 충돌 확률을 별도 상태로 예측하지 않습니다. 이 때문에 "
        "장애물 앞에서 목표 속도를 바꾸는 동작이 안정적으로 학습됐다고 보기 어렵습니다.", styles["body"]))
    story.append(p(
        "세 전략을 같은 시나리오에서 비교합니다. S1은 실제 제출 가능한 PPO-only, S2는 PPO와 latent "
        "dynamics CEM, S3는 규칙 기반 pixel corridor입니다. S2는 2,000 decision held-out 10회에서 "
        "완주 0/10, 평균 진행 0.074530으로 S1의 0.077411보다 낮았습니다. S3는 일반 공식 트랙에서 "
        "빠르지만 제출 ZIP에 포함되지 않습니다.", styles["body"]))

    story.append(p("현재 결과", styles["h1"]))
    story.append(table(
        [
            [p("전략", styles["small"]), p("조건", styles["small"]), p("결과", styles["small"]), p("판정", styles["small"])],
            [p("S1 PPO-only", styles["body"]), p("held-out 8회", styles["body"]), p("6/8 완료, median 19.32초", styles["body"]), p("현재 제출 후보", styles["body"])],
            [p("S1 PPO-only", styles["body"]), p("Track 1 seed 42 + 추가 장애물", styles["body"]), p("진행 14.84%, 충돌 5회, DNF", styles["body"]), p("장애물 실패", styles["body"])],
            [p("S2 PPO+CEM", styles["body"]), p("held-out 10회", styles["body"]), p("0/10 완료, 평균 진행 7.45%", styles["body"]), p("승격 보류", styles["body"])],
            # The earlier "22.40초 / 20.92초" row reported mean_speed as a lap
            # time. artifacts/haic/vision-racing-official-v1/results.json shows
            # the default corridor teacher finishing track 1 seed 42 in 591
            # decisions (47.28s) and track 2 seed 101 in 613 decisions (49.04s)
            # at mean speeds 20.68 and 20.93.
            [p("S3 corridor teacher", styles["body"]), p("Track 1 seed 42 / Track 2 seed 101", styles["body"]), p("4/4 완주, 47.28초 / 49.04초", styles["body"]), p("오프라인 전용", styles["body"])],
        ],
        [34 * mm, 55 * mm, 55 * mm, 26 * mm],
        font_size=7.8,
    ))
    story.append(p("실험 설계", styles["h1"]))
    story.append(p(
        "각 후보는 동일한 cap에서 패키지 smoke, 공식 Track 1/2, held-out 사용자 맵, 공식 트랙에 "
        "추가 장애물을 넣은 official_plus_custom을 순서대로 실행합니다. 완주율을 최우선으로 두고, "
        "그 다음 충돌·손상, 완주한 랩타임, 진행률, 속도 안정성을 비교합니다. DNF의 부분 주행 속도와 "
        "충돌 순간 속도 급락은 정상 제동 기록으로 사용하지 않습니다.", styles["body"]))
    story.append(p("권장 구현 가설", styles["h1"]))
    story.append(p(
        "정책의 CNN latent에서 장애물 존재·거리·횡방향 겹침·긴급도·목표 속도·충돌 확률을 함께 "
        "예측합니다. PPO가 행동을 제안하면 짧은 후보 행동 집합을 위험 모델로 평가하고, 충돌 확률이 "
        "높은 후보를 제거한 뒤 가장 빠른 안전 후보를 선택합니다. 이 구조는 장애물 판단을 학습하되 "
        "최종 안전 조건만 제한하는 방식이라 기존 PPO와 규칙 기반 교사의 장점을 결합합니다.", styles["body"]))
    story.append(PageBreak())
    story.append(p("논문 근거", styles["h1"]))
    story.append(p(
        "아래 연구는 HAIC에 그대로 복사하는 레시피가 아니라, 실험 가설을 선택하는 근거로 사용합니다. "
        "각 논문의 핵심은 장애물을 단순한 분류 플래그가 아니라 미래 위험·속도·행동 후보와 연결하는 것입니다.", styles["body"]))
    refs = [
        "Kahn, Villaflor, Pong, Abbeel, Levine (2017), <b>Uncertainty-Aware Reinforcement Learning for Collision Avoidance</b>. 충돌 확률과 불확실성을 예측해 낯선 상황에서 속도를 낮추는 모델 기반 접근. https://arxiv.org/abs/1702.01182",
        "Kahn, Abbeel, Levine (2020), <b>BADGR: An Autonomous Self-Supervised Learning-Based Navigation System</b>. 경험에서 미래 충돌과 위치를 예측하고 행동을 계획하는 navigation system. https://arxiv.org/abs/2002.05700",
        "Ebert et al. (2018), <b>Visual Foresight: Model-Based Deep Reinforcement Learning for Vision-Based Robotic Control</b>. action-conditioned video prediction과 visual MPC를 결합. https://arxiv.org/abs/1812.00568",
        "Cheng et al. (2019), <b>End-to-End Safe Reinforcement Learning through Barrier Functions</b>. 학습 정책 위에 안전 제약을 두어 탐색 중 위험 행동을 제한. https://arxiv.org/abs/1903.08792",
        "Evans, Engelbrecht, Jordaan (2023), <b>High-speed Autonomous Racing using Trajectory-aided Deep Reinforcement Learning</b>. 레이싱 line과 속도 프로파일을 학습에 넣어 고속 주행의 감속 구간을 안정화. https://arxiv.org/abs/2306.07003",
    ]
    for ref in refs:
        story.append(p("• " + ref, styles["ref"]))
    story.append(p("실험 연결", styles["h1"]))
    story.append(p(
        "위 근거를 HAIC에 적용할 때 첫 실험은 위험 예측 보조 헤드와 collision-aware action filter입니다. "
        "모델 기반 CEM은 이미 평가했지만 PPO-only보다 개선되지 않았으므로, 다음 CEM 실험은 반드시 "
        "충돌 위험을 예측하는 학습 모델을 추가한 뒤 진행해야 합니다. 규칙 기반 corridor는 교사와 "
        "진단 기준으로 유지하며 제출 정책의 성능으로 혼동하지 않습니다.", styles["body"]))
    story.append(p("재현 경로", styles["h1"]))
    story.append(p(
        "결과 DB: RESULTS.md · 최고 기록: SOTA.md · 대회 기준: COMPETITION_INFO.md · 실격 제한: "
        "RESTRICTIONS.md · 원본 결과: artifacts/haic/** · 지속 운영 절차: RULES.md", styles["body"]))
    story.append(p("주의", styles["h1"]))
    story.append(p(
        "이 보고서는 현재 측정값과 문헌에 근거한 실험 설계 문서입니다. 선두 기록 초과나 제출 성공을 "
        "측정 전에 주장하지 않습니다.", styles["body"]))
    document.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    return OUTPUT


if __name__ == "__main__":
    print(build())
