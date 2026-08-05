"""
미행 탐지 시스템 (Zipkimi) v4 - CCTV 미행 가능성 선별 시스템
Lemon & Mint 테마

화면 구성 (사이드바)
  1) 영상 분석      : 업로드 -> YOLO 추적 -> 규칙 엔진 위험도 -> 결과
  2) 과거 분석 기록 : 이전에 분석한 영상들을 다시 열람
  3) 관제 알림 로그 : 영상별로 그룹화된 위험 알림

실행:
    pip install streamlit ultralytics opencv-python
    streamlit run app_v4.py
"""

import importlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

FPS = 30

# ============================================================
#  theme : Lemon & Mint 팔레트 · 색 결정 · CSS
# ============================================================

#: 화면에서 쓸 수 있는 색은 전부 이 사전에서만 나온다(요구사항 13.1, 13.6).
PALETTE: dict[str, str] = {
    "lemon":      "#F5D547",
    "lemon_soft": "#FFF7D1",
    "lemon_deep": "#C9A400",
    "mint":       "#3EBD93",
    "mint_soft":  "#DFF5EB",
    "mint_deep":  "#1F7A5C",
    "coral":      "#E8705F",
    "coral_soft": "#FDE8E4",
    "coral_deep": "#B23E2E",
    "ink":        "#2C3B36",
    "ink_soft":   "#6B7C75",
    "line":       "#E4EFE9",
    "card":       "#FFFFFF",
    "white":      "#FFFFFF",
}

# 기존 이름은 팔레트에서 파생시킨다. 새 코드는 PALETTE / RiskStyle을 직접 쓴다.
LEMON      = PALETTE["lemon"]
LEMON_SOFT = PALETTE["lemon_soft"]
LEMON_DEEP = PALETTE["lemon_deep"]
MINT       = PALETTE["mint"]
MINT_SOFT  = PALETTE["mint_soft"]
MINT_DEEP  = PALETTE["mint_deep"]
CORAL      = PALETTE["coral"]
CORAL_SOFT = PALETTE["coral_soft"]
CORAL_DEEP = PALETTE["coral_deep"]
INK        = PALETTE["ink"]
INK_SOFT   = PALETTE["ink_soft"]
LINE       = PALETTE["line"]
BG_CARD    = PALETTE["card"]
WHITE      = PALETTE["white"]


@dataclass(frozen=True)
class RiskStyle:
    """등급명과 그 등급에 쓰는 색 3종.

    튜플을 풀어 쓰면 순서를 헷갈리기 쉬우므로 이름 있는 필드로 둔다.
    """

    label: str
    base: str
    soft: str
    deep: str


#: 위험도_등급별 색(요구사항 13.2)
RISK_HIGH = RiskStyle("우선 확인", CORAL, CORAL_SOFT, CORAL_DEEP)
RISK_MID  = RiskStyle("확인 필요", LEMON, LEMON_SOFT, LEMON_DEEP)
RISK_LOW  = RiskStyle("일반 이동", MINT, MINT_SOFT, MINT_DEEP)

#: 항목 점수 비율에 따른 색(요구사항 12.5)과 카드 범례 라벨(요구사항 12.6)
ITEM_HIGH = RiskStyle("높음", CORAL, CORAL_SOFT, CORAL_DEEP)
ITEM_MID  = RiskStyle("중간", LEMON, LEMON_SOFT, LEMON_DEEP)
ITEM_LOW  = RiskStyle("낮음", MINT, MINT_SOFT, MINT_DEEP)

#: 카드 범례에 그리는 순서(요구사항 12.6). 높은 쪽부터 적어 카드 색과 같은 순서로 읽힌다.
#: :func:`score_color`가 고르는 스타일이 곧 이 목록이라, 범례에 없는 색이 카드에 나올 수 없다.
ITEM_LEGEND: tuple[RiskStyle, ...] = (ITEM_HIGH, ITEM_MID, ITEM_LOW)

#: 분석_모드 코드 -> 화면 표시 스타일. ``label``이 곧 화면에 보이는 모드 이름이다.
#: 모드는 등급이 아니지만 색 묶음 구조가 같아 :class:`RiskStyle`을 그대로 쓴다 —
#: 여기서 새 색 상수를 만들면 팔레트 밖 색이 새어 들어올 자리가 생긴다(요구사항 13.6).
MODE_STYLES: dict[str, RiskStyle] = {
    "demo": RiskStyle("데모", LEMON, LEMON_SOFT, LEMON_DEEP),
    "real": RiskStyle("실분석", MINT, MINT_SOFT, MINT_DEEP),
}

#: 사이드바 전환 수단에 쓰는 순서. 첫 값이 기본값이다(요구사항 2.2).
MODE_ORDER: tuple[str, ...] = ("demo", "real")

#: 분석_모드 기본값. 실분석은 무거운 의존(opencv·ultralytics)을 끌어오므로
#: 사용자가 명시적으로 고를 때만 그 경로로 간다.
DEFAULT_MODE = MODE_ORDER[0]

#: 코드 <-> 화면 라벨. 상태에는 코드를, 화면에는 라벨을 쓴다.
MODE_LABELS: dict[str, str] = {code: MODE_STYLES[code].label for code in MODE_ORDER}
MODE_CODES: dict[str, str] = {label: code for code, label in MODE_LABELS.items()}

#: 모드별 한 줄 설명. 사이드바에서 무엇이 달라지는지 알려 준다(요구사항 2.1).
MODE_HINTS: dict[str, str] = {
    "demo": "고정된 예시 데이터로 화면을 확인합니다. 업로드 영상은 그대로 재생됩니다.",
    "real": "업로드 영상을 추적해 실제 결과를 계산합니다. 분석에 시간이 걸립니다.",
}


def mode_style(mode) -> RiskStyle:
    """분석_모드 -> 표시 스타일. 아는 코드가 아니면 기본 모드로 본다."""
    return MODE_STYLES.get(mode, MODE_STYLES[DEFAULT_MODE])


def risk_level(score) -> RiskStyle:
    """위험도 점수 -> 등급 스타일. 70점 이상 코랄, 40점 이상 레몬, 그 미만 민트."""
    if score >= 70:
        return RISK_HIGH
    if score >= 40:
        return RISK_MID
    return RISK_LOW


def score_color(score, maxv) -> RiskStyle:
    """항목 점수 비율 -> 카드 스타일. ``maxv``가 0이거나 없으면 비율을 0으로 본다."""
    ratio = score / maxv if maxv and maxv > 0 else 0.0
    if ratio >= 0.75:
        return ITEM_HIGH
    if ratio >= 0.40:
        return ITEM_MID
    return ITEM_LOW


def hex_to_bgr(color) -> tuple[int, int, int]:
    """``#RRGGBB`` 팔레트 값을 opencv가 쓰는 ``(B, G, R)`` 정수 튜플로 바꾼다.

    결과 영상 박스 색도 화면 색과 같은 :data:`PALETTE`에서 나와야 한다
    (요구사항 13.6). BGR 리터럴을 코드에 박아 두면 팔레트를 고쳐도 영상 색은
    그대로 남아 둘이 조용히 어긋난다. ``#fff`` 같은 3자리 축약도 받아 준다.
    """
    s = str(color).lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    r, g, b = (int(s[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def hex_to_rgb_text(color) -> str:
    """``#RRGGBB`` 팔레트 값을 CSS ``rgba()`` 안에 넣을 ``"R, G, B"`` 문자열로 바꾼다.

    그림자 색도 :data:`PALETTE`에서만 나와야 한다(요구사항 13.6). ``rgba(44,59,54,…)``
    처럼 숫자를 박아 두면 팔레트를 고쳐도 그림자만 옛 색으로 남는다.
    ``hex_to_bgr``를 재사용해 표기 해석 규칙을 한곳에 둔다.
    """
    b, g, r = hex_to_bgr(color)
    return f"{r}, {g}, {b}"


def theme_css() -> str:
    """PALETTE만 참조해 전역 CSS를 만든다."""
    p = PALETTE
    ink_rgb = hex_to_rgb_text(p["ink"])
    return f"""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard-dynamic-subset.css');

/* 한글 본문용 웹폰트. CDN이 막힌 환경에서는 뒤의 시스템 폰트로 그대로 넘어간다. */
html, body, [class*="css"], .block-container, section[data-testid="stSidebar"] {{
  font-family: 'Pretendard', -apple-system, 'Malgun Gothic', sans-serif;
}}

.block-container {{padding-top: 2.2rem; max-width: 1200px;}}
h1, h2, h3 {{color: {p['ink']} !important; letter-spacing: -0.3px;}}
section[data-testid="stSidebar"] {{background: {p['mint_soft']};}}
section[data-testid="stSidebar"] * {{color: {p['ink']};}}
section[data-testid="stSidebar"] label {{font-size:13.5px;}}

.zk-hero {{
  background: linear-gradient(100deg, {p['lemon_soft']} 0%, {p['mint_soft']} 100%);
  border-radius: 18px; padding: 26px 30px; margin-bottom: 20px;
  border: 1px solid {p['line']};
}}
.zk-hero h1 {{margin:0; font-size:32px; font-weight:600; letter-spacing:-0.6px;}}
.zk-hero p {{margin:6px 0 0; color:{p['ink_soft']}; font-size:14px;}}

.zk-card {{
  background:{p['card']}; border:1px solid {p['line']}; border-radius:14px;
  padding:16px 18px; margin-bottom:14px;
  box-shadow: 0 1px 3px rgba({ink_rgb}, .06), 0 1px 2px rgba({ink_rgb}, .04);
}}
.zk-label {{font-size:13px; color:{p['ink_soft']}; margin-bottom:10px; font-weight:600;
           letter-spacing:0.2px;}}

/* 플레이어 앞에 두는 시크 표시자. 값이 바뀌면 마크업이 바뀌어 플레이어가 새로
   그려지지만, 자리는 차지하지 않아야 한다(video_player 참고). */
.zk-seek-mark {{display:none;}}

.zk-metric {{
  background:{p['card']}; border:1px solid {p['line']}; border-radius:14px;
  padding:14px 16px; text-align:center;
  box-shadow: 0 1px 3px rgba({ink_rgb}, .06), 0 1px 2px rgba({ink_rgb}, .04);
}}
.zk-metric .v {{font-size:26px; font-weight:700; color:{p['ink']}; line-height:1.2;}}
.zk-metric .l {{font-size:12px; color:{p['ink_soft']}; margin-top:2px;}}

/* 판정 요약 헤드라인(요구사항 10.8) — 점수를 큰 숫자로 먼저 읽히게 한다.
   색은 호출부가 등급 스타일에서 넘긴다. */
.zk-verdict {{
  background:{p['card']}; border:1px solid {p['line']}; border-radius:16px;
  padding:20px 24px; margin-bottom:12px;
  border-left-width:6px; border-left-style:solid;
  box-shadow: 0 1px 3px rgba({ink_rgb}, .06), 0 1px 2px rgba({ink_rgb}, .04);
}}
.zk-verdict .head {{display:flex; align-items:center; gap:14px; flex-wrap:wrap;}}
.zk-verdict .score {{font-size:52px; font-weight:700; line-height:1;}}
.zk-verdict .score .u {{font-size:16px; font-weight:600; margin-left:2px;}}
.zk-verdict .pair {{font-size:15px; font-weight:600; color:{p['ink']};}}
.zk-verdict .meta {{font-size:12.5px; color:{p['ink_soft']}; margin-top:6px;}}

.zk-row {{
  display:flex; justify-content:space-between; align-items:center;
  padding:11px 13px; border-radius:11px; margin-bottom:8px;
}}
.zk-chip {{
  display:inline-block; padding:3px 10px; border-radius:20px;
  font-size:11px; font-weight:600;
}}
/* 항목별 점수 카드(요구사항 12.4) — 칸 나눔과 테두리 굵기는 여기서만 정한다.
   호출부는 색(border-color/background)만 넘긴다. */
.zk-score-grid {{display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px;}}
.zk-score-legend {{margin-top:9px; font-size:11.5px; color:{p['ink_soft']};}}
.zk-score-box {{
  border-radius:12px; padding:12px 10px; text-align:center; height:100%;
  border-width:1.5px; border-style:solid;
}}
.zk-score-box .n {{font-size:11px; font-weight:600; opacity:.85;}}
.zk-score-box .m {{font-size:15px; font-weight:700; margin:3px 0 1px;}}
.zk-score-box .m .x {{font-size:10px;}}
.zk-score-box .s {{font-size:11px; opacity:.8;}}

/* 판정 근거 표(요구사항 12.1, 12.2) — 열 정렬과 합계 행 모양은 전부 클래스로.
   호출부의 문자열 조립에는 색만 남는다. */
table.zk-table {{width:100%; border-collapse:collapse; font-size:13.5px;}}
table.zk-table th {{
  color:{p['ink_soft']}; font-weight:600; font-size:12px;
  padding:6px 8px; border-bottom:1.5px solid {p['line']};
}}
table.zk-table td {{padding:8px; border-bottom:1px solid {p['line']}; color:{p['ink']};}}
table.zk-table th.item, table.zk-table td.item,
table.zk-table th.measured, table.zk-table td.measured {{text-align:left;}}
table.zk-table th.num, table.zk-table td.num {{text-align:right;}}
table.zk-table td.measured {{color:{p['ink_soft']};}}
table.zk-table td.num {{font-weight:600;}}
table.zk-table tr.total td {{
  border-bottom:none; padding-top:11px; font-weight:700; font-size:14.5px;
}}
table.zk-table tr.total td.num {{font-size:18px;}}

div.stButton > button {{
  border-radius:10px; border:1px solid {p['line']}; background:{p['card']};
  color:{p['ink']}; font-size:13px; font-weight:500;
  padding:6px 14px; transition:all .15s ease;
}}
div.stButton > button:hover {{border-color:{p['mint']}; color:{p['mint_deep']};}}
div.stButton > button[kind="primary"] {{
  background:{p['mint']}; border-color:{p['mint']}; color:{p['white']};
}}
div.stButton > button[kind="primary"]:hover {{background:{p['mint_deep']};}}
</style>
"""


# ============================================================
#  domain : 자료구조와 순수 함수
# ============================================================

def clamp(v, lo, hi):
    """``v``를 ``[lo, hi]`` 안으로 제한한다.

    ``lo``/``hi``는 호출자가 올바른 순서로 넘긴다고 가정한다.
    """
    return lo if v < lo else (hi if v > hi else v)


def fmt_mmss(sec) -> str:
    """초를 ``분:초`` 문자열로 만든다(``754 -> "12:34"``).

    한 시간을 넘어도 ``시:분:초``로 바꾸지 않고 분을 계속 키운다
    (``3661 -> "61:01"``). 그래야 ``parse_mmss``가 분×60+초 규칙
    하나로 원래 초 값을 그대로 되돌릴 수 있다(요구사항 8.3).
    음수는 0으로 취급한다.
    """
    s = int(sec)
    if s < 0:
        s = 0
    return f"{s // 60:02d}:{s % 60:02d}"


def parse_mmss(text) -> int:
    """``분:초`` 문자열을 초로 되돌린다(``"12:34" -> 754``).

    분 값에 60을 곱한 값과 초 값의 합을 사용한다(요구사항 8.3).
    분 자리는 자릿수 제한이 없어 ``"5999:59"``처럼 한 시간을 넘는
    값도 그대로 해석한다. 주변 공백, 각 자리 주변 공백을 허용하고,
    ``시:분:초`` 3토막과 초만 적힌 문자열도 함께 받아 준다.

    해석할 수 없으면 ``ValueError``를 올린다.
    """
    if isinstance(text, (int, float)):
        return int(text)
    parts = [p.strip() for p in str(text).strip().split(":")]
    if not parts or len(parts) > 3 or any(not p.lstrip("-").isdigit() for p in parts):
        raise ValueError(f"시간 문자열로 해석할 수 없습니다: {text!r}")
    nums = [int(p) for p in parts]
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    return nums[0] * 3600 + nums[1] * 60 + nums[2]


def clamp_seek(sec, duration_sec) -> int:
    """재생 시작 위치를 ``[0, duration_sec]`` 안으로 제한한다(요구사항 8.5).

    재생 길이를 모르거나(``None``) 0 이하면 시작 위치를 0으로 되돌린다.
    """
    if duration_sec is None:
        return 0
    limit = int(duration_sec)
    if limit <= 0:
        return 0
    return int(clamp(int(sec), 0, limit))


def fmt_created_at(iso) -> str:
    """ISO 8601 분석 시각을 목록·리포트에 쓸 ``월/일 시:분``으로 만든다.

    저장에는 항상 ISO 8601 전체 시각을 쓰고, 표시 문자열은 이 함수로 그때
    만든다. 해석할 수 없는 값은 원문을 그대로 돌려준다.
    """
    try:
        return datetime.fromisoformat(str(iso)).strftime("%m/%d %H:%M")
    except (TypeError, ValueError):
        return str(iso)


@dataclass(frozen=True)
class ScoreRow:
    """판정 근거 한 줄. 최대 점수를 직접 들고 있어 이름 조회가 필요 없다."""

    name: str
    measured: str
    score: int
    max_score: int


@dataclass(frozen=True)
class TimelineItem:
    """이벤트 타임라인 한 항목. 시각은 초 정수로 보관하고 표시할 때만 변환한다."""

    at_sec: int
    label: str
    tag: str


@dataclass(frozen=True)
class Event:
    """한 쌍(선행자·후행자)에 대한 판정 결과.

    ``suspicion``과 ``score``는 저장하지 않고 ``rows``/``deduction``에서 계산한다.
    저장 필드로 두면 표와 카드가 서로 어긋날 수 있다(요구사항 12.3).
    """

    leader: int
    follower: int
    rows: tuple[ScoreRow, ...]
    deduction: int
    timeline: tuple[TimelineItem, ...]
    curve: tuple[tuple[int, int], ...]   # (초, 0~100 위험도)

    @property
    def suspicion(self) -> int:
        """의심 가점 합(요구사항 12.7)."""
        return sum(r.score for r in self.rows)

    @property
    def score(self) -> int:
        """최종 위험도. 중간값은 음수가 될 수 있고 최종 값만 클램프한다(요구사항 12.8)."""
        return clamp(self.suspicion - self.deduction, 0, 100)


@dataclass(frozen=True)
class Alert:
    """관제 알림 한 건.

    ``at``은 시분초만 담으면 자정을 넘긴 정렬이 뒤집히므로 ISO 8601 전체
    시각으로 보관한다(요구사항 5.4, 6.8). ``score``/``level``은 기록이
    삭제된 뒤에도 목록을 그대로 보여야 하므로 비정규화해 함께 담는다.
    """

    id: str
    at: str                 # ISO 8601 (예: "2026-02-14T14:22:15")
    video: str
    record_id: str
    event_index: int
    pair: str
    score: int
    level: str


@dataclass(frozen=True)
class AnalysisRecord:
    """하나의 영상에 대한 분석 결과 단위.

    ``created_at``은 ISO 8601 전체 시각으로 담는다. 화면에 보여 줄 문자열은
    ``fmt_created_at``으로 그때 만들며, 표시용 문자열을 저장하지 않는다.

    ``video_available``은 파일 존재 여부라 시점에 따라 변하므로 저장하지 않고
    ``load_records()``가 ``Path.exists()``로 채운다(설계 결정). 동치 비교에서도
    빼서 직렬화 왕복이 성립하게 한다(요구사항 4.4).

    ``video_render_failed``는 결과 영상 합성이 실패해 재생 영상 경로가 원본
    영상을 가리키는 경우를 표시한다. 화면이 안내 문구를 띄우는 근거이므로
    저장한다(요구사항 1.5, 1.6).
    """

    id: str
    name: str                       # 원본 파일명
    events: tuple[Event, ...]
    mode: str                       # "demo" | "real"
    video_path: str                 # 재생_영상_경로
    duration_sec: int | None
    created_at: str                 # ISO 8601 (예: "2026-02-14T14:22:15")
    video_render_failed: bool = False
    video_available: bool = field(default=False, compare=False)


@dataclass(frozen=True)
class PairFeatures:
    """한 쌍(선행자·후행자)에서 뽑아낸 측정값. 점수표의 입력이다."""

    rear_ratio: float      # 후방 근접 유지 비율 (0~1)
    duration_sec: float    # 근접 유지 지속 시간 (초)
    heading: float         # 이동 방향 유사도 (0~1)
    path_ratio: float      # 지연 경로 유사도 (0~1)
    stop_follows: int      # 선행자 정지 뒤 후행자도 멈춘 횟수
    turn_follows: int      # 선행자 방향 전환 뒤 후행자도 전환한 횟수


@dataclass(frozen=True)
class PairTiming:
    """측정값이 '언제' 관측됐는지. 프레임 인덱스로만 담는다.

    :class:`PairFeatures`가 점수표의 입력(얼마나)이라면 이쪽은 타임라인의
    입력(언제)이다. 초로 환산하지 않고 프레임 인덱스로 두는 이유는, fps를
    아는 쪽(타임라인을 만드는 쪽)에서 한 번만 환산하면 되고 측정 단계가
    fps 추정 오차를 결과에 굳혀 넣지 않기 때문이다.

    ``approach_frame``은 후행자가 근접 범위에 처음 들어온 프레임이며, 근접
    프레임이 아예 없으면 ``None``이다. ``stop_frames``/``turn_frames``는
    선행자의 정지·방향 전환에 후행자가 반응한 프레임으로, 길이가 각각
    ``PairFeatures.stop_follows``/``turn_follows``와 같다.
    """

    approach_frame: int | None
    stop_frames: tuple[int, ...]
    turn_frames: tuple[int, ...]


@dataclass(frozen=True)
class PairCompanion:
    """동행(같이 다니는 사이)으로 보이는 정도. 감점의 근거다.

    :class:`PairFeatures`가 의심 가점의 입력이라면 이쪽은 감점의 입력이다.
    점수표와 섞지 않고 따로 두는 이유는 부호가 반대이고, 화면에서도 '의심
    가점'과 '동행 감점'을 나눠 보여 주기 때문이다(요구사항 12.7).

    ``deduction``은 아래 두 측정값에서 계산된 감점이며 상한
    :data:`COMPANION_DEDUCTION_MAX`까지다. 최종 이벤트에서는 가점 합을
    넘지 않도록 한 번 더 자른다.
    """

    side_ratio: float      # 뒤가 아니라 옆에 있던 프레임 비율 (0~1)
    sync_stops: int         # 거의 동시에 멈춘 횟수
    deduction: int          # 위 두 값에서 계산한 감점


# ------------------------------------------------------------
#  점수표 (측정값 -> (점수, 표시용 측정값))
# ------------------------------------------------------------

def score_rear_keep(r):
    """후방 근접 유지 비율(0~1) -> 최대 15점."""
    p = r * 100
    if p >= 90: return 15, f"{p:.0f}%"
    if p >= 75: return 12, f"{p:.0f}%"
    if p >= 60: return 9,  f"{p:.0f}%"
    if p >= 40: return 5,  f"{p:.0f}%"
    return 0, f"{p:.0f}%"


def score_duration(s):
    """근접 유지 지속 시간(초) -> 최대 15점."""
    if s >= 20: return 15, f"{s:.0f}초"
    if s >= 15: return 12, f"{s:.0f}초"
    if s >= 10: return 9,  f"{s:.0f}초"
    if s >= 5:  return 5,  f"{s:.0f}초"
    return 0, f"{s:.0f}초"


def score_heading(v):
    """이동 방향 유사도(0~1) -> 최대 10점."""
    if v >= 0.90: return 10, f"{v:.2f}"
    if v >= 0.75: return 8,  f"{v:.2f}"
    if v >= 0.60: return 5,  f"{v:.2f}"
    return 0, f"{v:.2f}"


def score_path(r):
    """지연 경로 유사도(0~1) -> 최대 20점."""
    p = r * 100
    if p >= 75: return 20, f"{p:.0f}%"
    if p >= 60: return 15, f"{p:.0f}%"
    if p >= 45: return 10, f"{p:.0f}%"
    if p >= 30: return 5,  f"{p:.0f}%"
    return 0, f"{p:.0f}%"


def score_stop(c):
    """정지 후 추종 횟수 -> 최대 20점."""
    return (20, f"{c}회") if c >= 2 else ((12, "1회") if c == 1 else (0, "0회"))


def score_turn(c):
    """방향 전환 추종 횟수 -> 최대 20점."""
    return (20, f"{c}회") if c >= 2 else ((12, "1회") if c == 1 else (0, "0회"))


#: 점수표 항목명 -> 최대 점수. 합계 100이며 항목은 6개다(요구사항 12.1).
MAXES = {"후방 근접 유지": 15, "지속 시간": 15, "방향 유사도": 10,
         "지연 경로 유사도": 20, "정지 후 추종": 20, "방향 전환 추종": 20}

#: 항목명 -> 측정값에서 (점수, 표시 문자열)을 뽑는 함수. 순서는 MAXES와 같다.
SCORERS: tuple[tuple[str, object], ...] = (
    ("후방 근접 유지",     lambda f: score_rear_keep(f.rear_ratio)),
    ("지속 시간",         lambda f: score_duration(f.duration_sec)),
    ("방향 유사도",        lambda f: score_heading(f.heading)),
    ("지연 경로 유사도",   lambda f: score_path(f.path_ratio)),
    ("정지 후 추종",       lambda f: score_stop(f.stop_follows)),
    ("방향 전환 추종",     lambda f: score_turn(f.turn_follows)),
)


def build_rows(features) -> tuple[ScoreRow, ...]:
    """측정값 묶음 -> 판정 근거 행 6개(요구사항 12.1, 12.3).

    각 행의 ``max_score``는 ``MAXES``에서 주입하므로 표와 카드가 같은
    최대값을 보고, 항목명 조회 실패 경로가 생기지 않는다.
    행 순서는 ``MAXES``의 선언 순서와 같다.
    """
    return tuple(
        ScoreRow(name=name, measured=measured, score=score, max_score=MAXES[name])
        for name, (score, measured) in (
            (name, scorer(features)) for name, scorer in SCORERS
        )
    )


# ------------------------------------------------------------
#  관제 알림 그룹화
# ------------------------------------------------------------

def alert_at(alert) -> datetime:
    """알림 발생 시각을 ``datetime``으로 해석한다(요구사항 6.8).

    시분초 문자열만 비교하면 날짜가 다른 알림의 순서가 뒤집히므로
    항상 ISO 8601 전체 시각을 파싱해 비교한다. 해석할 수 없는 값은
    가장 오래된 것으로 취급해 목록 끝으로 보낸다.
    """
    try:
        return datetime.fromisoformat(str(alert.at))
    except (TypeError, ValueError):
        return datetime.min


def group_alerts(alerts) -> list[tuple[str, list[Alert]]]:
    """알림을 영상명으로 묶어 ``(영상명, 알림목록)`` 리스트로 돌려준다.

    - 그룹 내부: 발생 시각 내림차순(요구사항 6.2)
    - 그룹 순서: 그룹의 가장 최근 발생 시각 내림차순(요구사항 6.6)
    - 총 개수 보존: 그룹별 항목 수의 합은 입력 길이와 같다(요구사항 6.8)

    딕셔너리 삽입 순서에는 의존하지 않는다. 시각이 같아 순서가 갈리지
    않을 때는 그룹 내부에서 ``(id, pair, event_index)`` 오름차순,
    그룹 사이에서는 영상명 오름차순을 동점 기준으로 써서 같은 입력이
    항상 같은 결과를 내도록 한다(전순서).
    """
    buckets: dict[str, list[Alert]] = {}
    for a in alerts:
        buckets.setdefault(a.video, []).append(a)

    groups: list[tuple[str, list[Alert]]] = []
    for video, items in buckets.items():
        # 동점 기준을 먼저 적용하고, 안정 정렬로 시각 내림차순을 덮어쓴다.
        ordered = sorted(items, key=lambda a: (str(a.id), str(a.pair), int(a.event_index)))
        ordered.sort(key=alert_at, reverse=True)
        groups.append((video, ordered))

    groups.sort(key=lambda g: g[0])
    groups.sort(key=lambda g: alert_at(g[1][0]), reverse=True)
    return groups


def group_top(items) -> tuple[int, RiskStyle]:
    """그룹 머리글용 최고 점수와 그 등급(요구사항 6.3, 6.4).

    전체 최고 점수가 아니라 넘겨받은 그룹의 알림만 보고 계산한다.
    """
    top = max((int(a.score) for a in items), default=0)
    return top, risk_level(top)


# ============================================================
#  store : 직렬화 (항목 단위. version 봉투는 파일 I/O 쪽 책임)
# ============================================================
#
# 규칙 세 가지만 지키면 왕복 동치(요구사항 4.4)가 성립한다.
#
#   1) ``curve`` 좌표는 ``[초, 위험도]`` 순서로 고정한다(요구사항 10.4).
#   2) JSON은 튜플을 리스트로 바꾸므로, 역직렬화에서 모든 묶음을 튜플로
#      되돌린다. ``curve`` 좌표 하나하나도 ``tuple[int, int]``여야
#      frozen dataclass의 동치 비교가 성립한다.
#   3) ``score``/``suspicion``은 ``rows``/``deduction``에서 계산되는
#      속성이므로 직렬화하지 않는다. 저장 값과 계산 값이 어긋날 여지를 없앤다.

def score_row_to_dict(r: ScoreRow) -> dict:
    """판정 근거 한 줄 -> dict."""
    return {"name": r.name, "measured": r.measured,
            "score": int(r.score), "max_score": int(r.max_score)}


def score_row_from_dict(d: dict) -> ScoreRow:
    """dict -> 판정 근거 한 줄."""
    return ScoreRow(name=str(d["name"]), measured=str(d["measured"]),
                    score=int(d["score"]), max_score=int(d["max_score"]))


def timeline_item_to_dict(i: TimelineItem) -> dict:
    """타임라인 항목 -> dict. 시각은 초 정수로 담는다."""
    return {"at_sec": int(i.at_sec), "label": i.label, "tag": i.tag}


def timeline_item_from_dict(d: dict) -> TimelineItem:
    """dict -> 타임라인 항목."""
    return TimelineItem(at_sec=int(d["at_sec"]), label=str(d["label"]),
                        tag=str(d["tag"]))


def event_to_dict(e: Event) -> dict:
    """이벤트 -> dict. ``score``/``suspicion``은 계산 속성이라 담지 않는다."""
    return {
        "leader": int(e.leader),
        "follower": int(e.follower),
        "deduction": int(e.deduction),
        "rows": [score_row_to_dict(r) for r in e.rows],
        "timeline": [timeline_item_to_dict(i) for i in e.timeline],
        # [초, 위험도] 순서(요구사항 10.4)
        "curve": [[int(sec), int(risk)] for sec, risk in e.curve],
    }


def event_from_dict(d: dict) -> Event:
    """dict -> 이벤트. 모든 묶음을 튜플로 되돌린다."""
    return Event(
        leader=int(d["leader"]),
        follower=int(d["follower"]),
        deduction=int(d["deduction"]),
        rows=tuple(score_row_from_dict(r) for r in d["rows"]),
        timeline=tuple(timeline_item_from_dict(i) for i in d["timeline"]),
        # 리스트 -> 튜플. 좌표 하나하나도 튜플이어야 동치가 성립한다.
        curve=tuple((int(sec), int(risk)) for sec, risk in d["curve"]),
    )


def record_to_dict(rec: AnalysisRecord) -> dict:
    """분석 기록 -> dict. ``video_available``은 읽을 때 계산하므로 담지 않는다."""
    return {
        "id": rec.id,
        "name": rec.name,
        "mode": rec.mode,
        "video_path": rec.video_path,
        "duration_sec": None if rec.duration_sec is None else int(rec.duration_sec),
        "created_at": rec.created_at,
        "video_render_failed": bool(rec.video_render_failed),
        "events": [event_to_dict(e) for e in rec.events],
    }


def record_from_dict(d: dict) -> AnalysisRecord:
    """dict -> 분석 기록. ``video_available``은 기본값(False)으로 두고 나중에 채운다."""
    duration = d["duration_sec"]
    return AnalysisRecord(
        id=str(d["id"]),
        name=str(d["name"]),
        events=tuple(event_from_dict(e) for e in d["events"]),
        mode=str(d["mode"]),
        video_path=str(d["video_path"]),
        duration_sec=None if duration is None else int(duration),
        created_at=str(d["created_at"]),
        video_render_failed=bool(d.get("video_render_failed", False)),
    )


def alert_to_dict(a: Alert) -> dict:
    """관제 알림 -> dict."""
    return {
        "id": a.id,
        "at": a.at,
        "video": a.video,
        "record_id": a.record_id,
        "event_index": int(a.event_index),
        "pair": a.pair,
        "score": int(a.score),
        "level": a.level,
    }


def alert_from_dict(d: dict) -> Alert:
    """dict -> 관제 알림."""
    return Alert(
        id=str(d["id"]),
        at=str(d["at"]),
        video=str(d["video"]),
        record_id=str(d["record_id"]),
        event_index=int(d["event_index"]),
        pair=str(d["pair"]),
        score=int(d["score"]),
        level=str(d["level"]),
    )


# ============================================================
#  store : 파일 I/O (version 봉투 · 원자적 쓰기)
# ============================================================
#
# ``st.session_state``는 브라우저 세션 단위라 새로고침이면 사라진다. 분석 기록과
# 관제 알림은 JSON 두 개로 저장하고, 결과 영상은 ``videos/``에 보관한다
# (요구사항 4.1, 5.1).
#
# 경로 상수는 모듈 전역으로 두고 함수 안에서 그때그때 이름으로 읽는다. 기본값
# 인자로 붙잡아 두면 테스트가 디렉터리를 갈아 끼울 수 없다.

DATA_DIR = Path(".zipkimi_data")
VIDEO_DIR = DATA_DIR / "videos"
RECORDS = DATA_DIR / "records.json"
ALERTS = DATA_DIR / "alerts.json"

#: 파일 봉투에 적는 스키마 버전. 이 값과 다르면 내용을 해석하지 않는다.
STORE_VERSION = 1


def ensure_dirs() -> None:
    """데이터 디렉터리와 영상 보관 디렉터리를 만든다. 여러 번 불러도 안전하다.

    임포트 시점이 아니라 저장소 함수를 실제로 부를 때만 만든다. 그래야 순수
    계층만 쓰는 테스트가 작업 디렉터리에 ``.zipkimi_data/``를 남기지 않는다.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)


def _write_json_atomic(path, payload) -> None:
    """같은 디렉터리의 임시 파일에 쓴 뒤 ``os.replace``로 갈아 끼운다.

    쓰는 중에 프로세스가 죽어도 목적지 파일은 이전 내용을 온전히 유지한다.
    ``os.replace``가 원자적이려면 임시 파일이 목적지와 같은 파일시스템에
    있어야 하므로 같은 디렉터리에 만든다. 한글이 그대로 읽히도록
    ``ensure_ascii=False``로 저장한다.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=target.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _load_items(path, key, from_dict) -> tuple[list, int]:
    """``{"version": n, key: [...]}`` 봉투를 읽어 ``(항목 목록, 해석 실패 건수)``.

    요구사항 4.5에 따라 한 항목이 깨졌다고 파일 전체를 버리지 않는다.

    - 파일이 없으면 실패가 아니다. ``([], 0)``을 돌려준다.
    - ``version``이 아는 값이 아니면 항목 전부를 건너뛰고 실패로 센다. 항목 수를
      셀 수 없는 경우에도 최소 1건은 보고해, 파일이 무시됐다는 사실이 화면에
      드러나게 한다.
    - 항목 하나의 역직렬화가 실패하면 그 항목만 건너뛰고 실패 건수를 늘린다.
    """
    p = Path(path)
    if not p.exists():
        return [], 0
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return [], 1
    if not isinstance(raw, dict):
        return [], 1
    items = raw.get(key)
    if not isinstance(items, list):
        return [], 1
    if raw.get("version") != STORE_VERSION:
        return [], max(len(items), 1)

    out, failed = [], 0
    for item in items:
        try:
            out.append(from_dict(item))
        except (AttributeError, KeyError, TypeError, ValueError):
            failed += 1
    return out, failed


def _created_key(rec) -> datetime:
    """정렬용 분석 시각. 해석할 수 없으면 가장 오래된 것으로 취급한다."""
    try:
        return datetime.fromisoformat(str(rec.created_at))
    except (TypeError, ValueError):
        return datetime.min


def load_records() -> tuple[list[AnalysisRecord], int]:
    """저장된 분석 기록 전체를 ``(최신순 목록, 해석 실패 건수)``로 돌려준다.

    ``video_available``은 저장하지 않고 여기서 ``Path.exists()``로 채운다.
    파일이 사라진 기록도 목록에는 그대로 남긴다(요구사항 4.6).
    시각이 같아 순서가 갈리지 않을 때는 ``id`` 오름차순을 동점 기준으로 쓴다.
    """
    recs, failed = _load_items(RECORDS, "records", record_from_dict)
    filled = [
        replace(r, video_available=bool(r.video_path) and Path(r.video_path).exists())
        for r in recs
    ]
    filled.sort(key=lambda r: str(r.id))
    filled.sort(key=_created_key, reverse=True)
    return filled, failed


def _write_records(recs) -> None:
    """기록 목록을 파일에 쓴다."""
    ensure_dirs()
    _write_json_atomic(RECORDS, {
        "version": STORE_VERSION,
        "records": [record_to_dict(r) for r in recs],
    })


def save_record(rec: AnalysisRecord) -> None:
    """분석 기록 하나를 저장한다. 같은 ``id``가 이미 있으면 새 값으로 바꾼다.

    해석할 수 없던 항목은 다시 쓸 방법이 없으므로 이 시점에 파일에서 사라진다.
    """
    kept, _ = _load_items(RECORDS, "records", record_from_dict)
    kept = [r for r in kept if r.id != rec.id]
    kept.append(rec)
    _write_records(kept)


def find_record(record_id) -> AnalysisRecord | None:
    """``id``로 기록 하나를 찾는다. 없으면 ``None``.

    화면은 기록 객체를 들고 있지 않고 ``id``만 들고 있다가 렌더 시점에 이걸로
    다시 읽는다. 그래야 삭제된 기록을 계속 붙들지 않는다.
    """
    wanted = str(record_id)
    for rec in load_records()[0]:
        if rec.id == wanted:
            return rec
    return None


def delete_record(record_id) -> None:
    """기록의 메타 항목과 보관된 결과 영상을 함께 지운다(요구사항 9.5).

    영상 파일이 이미 없어도 예외를 올리지 않는다.
    """
    wanted = str(record_id)
    existing, _ = _load_items(RECORDS, "records", record_from_dict)
    kept = [r for r in existing if r.id != wanted]
    if len(kept) != len(existing):
        _write_records(kept)
    try:
        (VIDEO_DIR / f"{wanted}.mp4").unlink(missing_ok=True)
    except OSError:
        pass


def adopt_video(src_path, record_id) -> str:
    """결과 영상을 ``videos/{record_id}.mp4``로 옮기고 그 경로를 돌려준다(요구사항 4.2).

    합성 결과는 임시 디렉터리에 만들어지므로 그대로 두면 OS가 언제든 지울 수
    있다. 기록과 같은 수명을 갖도록 보관 위치로 옮긴다.

    순서는 세 단계다.

      1) ``shutil.move``. 같은 파일시스템이면 이름만 바뀌어 가장 싸다.
      2) 실패하면(임시 디렉터리가 다른 드라이브인 경우 등) ``shutil.copy2`` 후
         원본 삭제로 대체한다.
      3) 복사까지 실패하면 예외를 그대로 올린다. 호출자는 이를 받아 원본 영상
         경로를 재생 영상 경로로 기록한다(요구사항 1.5).

    복사는 됐지만 원본 삭제가 실패한 경우는 실패가 아니다. 영상은 보관 위치에
    안전히 들어갔고 남은 임시 파일은 OS가 정리하므로 그냥 넘어간다.
    """
    ensure_dirs()
    src = Path(src_path)
    dst = VIDEO_DIR / f"{record_id}.mp4"

    # 이미 보관 위치에 있으면 옮길 것이 없다(재저장 경로에서 원본을 잃지 않도록).
    if src == dst or (src.exists() and dst.exists() and src.samefile(dst)):
        return str(dst)

    # 목적지를 먼저 비워 두면 이동/복사 동작이 플랫폼에 관계없이 같아진다.
    dst.unlink(missing_ok=True)

    try:
        shutil.move(str(src), str(dst))
        return str(dst)
    except (OSError, shutil.Error):
        pass

    shutil.copy2(str(src), str(dst))   # 여기서 실패하면 예외를 그대로 올린다
    try:
        src.unlink(missing_ok=True)
    except OSError:
        pass
    return str(dst)


def load_alerts() -> tuple[list[Alert], int]:
    """저장된 관제 알림 전체를 ``(목록, 해석 실패 건수)``로 돌려준다.

    표시 순서는 ``group_alerts``가 정하므로 여기서는 저장 순서를 그대로 둔다.
    """
    return _load_items(ALERTS, "alerts", alert_from_dict)


def _write_alerts(items) -> None:
    """알림 목록을 파일에 쓴다."""
    ensure_dirs()
    _write_json_atomic(ALERTS, {
        "version": STORE_VERSION,
        "alerts": [alert_to_dict(a) for a in items],
    })


def append_alerts(items) -> None:
    """관제 알림을 기존 목록 뒤에 덧붙여 저장한다(요구사항 5.1)."""
    new = list(items)
    if not new:
        return
    existing, _ = _load_items(ALERTS, "alerts", alert_from_dict)
    _write_alerts(existing + new)


def clear_alerts() -> None:
    """저장된 관제 알림 전체를 지운다(요구사항 5.5)."""
    _write_alerts([])


# ============================================================
#  analysis : 영상 메타 읽기
# ============================================================

# ``cv2``는 이 계층에서만 필요한 무거운 선택 의존이라 모듈 최상단에서 임포트하지
# 않는다. 순수 계층 테스트와 ``import app_v4``가 opencv 없이도 성립해야 한다
# (요구사항 2.7).

@dataclass(frozen=True)
class VideoMeta:
    """영상 컨테이너에서 읽은 기본 정보.

    ``total_frames``와 ``duration_sec``은 "모른다"를 ``None``으로 표현한다.
    컨테이너 메타데이터는 0이나 음수처럼 못 믿을 값을 내놓을 때가 있는데,
    그걸 0으로 받아 두면 진행률 계산(요구사항 3.6)과 시크 상한(요구사항 8.5)이
    조용히 틀어진다. 프레임 수를 못 믿으면 재생 길이도 못 믿으므로 둘을 같이
    ``None``으로 떨어뜨린다 — 나쁜 프레임 수로 길이를 지어내지 않는다.
    """
    width: int
    height: int
    fps: float
    total_frames: int | None
    duration_sec: int | None


#: 읽기에 실패했을 때 돌려줄 값. 예외를 올리지 않는 이유는 호출자(업로드 직후)가
#: 메타를 몰라도 데모/실분석을 계속 진행할 수 있어야 하기 때문이다.
UNKNOWN_VIDEO_META = VideoMeta(width=0, height=0, fps=float(FPS),
                               total_frames=None, duration_sec=None)


def probe(video_path) -> VideoMeta:
    """영상의 해상도 · fps · 프레임 수 · 재생 길이를 읽는다.

    opencv가 없거나 파일을 열 수 없으면 :data:`UNKNOWN_VIDEO_META`를 돌려준다.
    프레임 수가 0 이하이거나 fps가 0 이하면 ``total_frames``와
    ``duration_sec``을 함께 ``None``으로 둔다.
    """
    try:
        import cv2
    except Exception:
        return UNKNOWN_VIDEO_META

    cap = None
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return UNKNOWN_VIDEO_META
        raw_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        raw_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        raw_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    except Exception:
        return UNKNOWN_VIDEO_META
    finally:
        # 성공이든 실패든 캡처는 반드시 놓아 준다.
        if cap is not None:
            cap.release()

    width = _safe_int(raw_w)
    height = _safe_int(raw_h)
    fps = _safe_float(raw_fps)
    total_frames = _safe_int(raw_frames)

    if fps <= 0 or total_frames <= 0:
        # fps나 프레임 수 하나라도 못 믿으면 길이 계산을 포기한다.
        return VideoMeta(width=width, height=height,
                         fps=fps if fps > 0 else float(FPS),
                         total_frames=None, duration_sec=None)

    return VideoMeta(width=width, height=height, fps=fps,
                     total_frames=total_frames,
                     duration_sec=int(total_frames / fps))


def _safe_float(v) -> float:
    """``None``/``nan``/변환 불가 값을 0.0으로 눌러 준다."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) else 0.0


def _safe_int(v) -> int:
    """``None``/``nan``/변환 불가 값을 0으로 눌러 준다."""
    return int(_safe_float(v))


# ============================================================
#  analysis : 진행률 보고
# ============================================================

class ProgressReporter:
    """진행률 계산을 한곳에 모아 막대에 넘길 값을 만든다.

    두 패스(1차 좌표 수집, 2차 영상 합성)가 각자 자기 프레임 수만 알고 있으면
    되도록, 패스별 가중치 누적은 이 객체가 맡는다. ``phase_weights``는 각 패스가
    전체 진행률에서 차지하는 몫이고, ``advance(phase, done_frames)``의 ``phase``는
    그 가중치 목록의 인덱스다(0 = 좌표 수집, 1 = 영상 합성).

    막대에 넘기는 값은 두 가지를 항상 지킨다(요구사항 3.2, 3.4).

    * ``clamp(ratio, 0, 1)`` — 0 이상 1 이하
    * 직전에 넘긴 값보다 작아지지 않음 — 되돌아가는 진행률은 보여 주지 않는다

    처리한 프레임 수가 예상 총 프레임 수를 넘으면(컨테이너 메타데이터가 실제와
    다를 때) 막대에는 제한한 값을 그대로 넘기고, 처리 프레임을 총 프레임으로 나눈
    실제 비율을 백분율 문자열로 캡션에 노출한다
    (요구사항 3.3). 총 프레임 수를 모르면(``total_frames`` 가 ``None``) 막대에
    수치를 아예 넘기지 않고 진행 중 문구만 유지한다(요구사항 3.6).

    ``bar``는 덕 타이핑으로 다룬다. ``progress(value)``와 ``caption(text)``가
    있으면 쓰고 없으면 조용히 넘긴다. ``bar=None``도 허용하므로 Streamlit 런타임
    없이 분석 경로를 그대로 돌릴 수 있다.
    """

    #: 총 프레임 수를 모를 때 캡션에 남기는 문구. 수치를 쓰지 않는다(요구사항 3.6).
    INDETERMINATE_TEXT = "분석 진행 중..."

    def __init__(self, total_frames, bar, phase_weights=(0.55, 0.45)):
        self.bar = bar
        weights = tuple(_safe_float(w) for w in phase_weights)
        self.phase_weights = weights if weights else (1.0,)
        self.total_frames = self._usable_total(total_frames)
        #: 막대에 마지막으로 넘긴 값. 아직 없으면 ``None``.
        self.last_value = None
        #: 캡션으로 마지막에 내보낸 문구. 같은 문구를 반복해 쓰지 않으려고 들고 있다.
        self.last_caption = None

    @staticmethod
    def _usable_total(total_frames):
        """믿고 나눌 수 있는 총 프레임 수만 남기고 나머지는 ``None``으로."""
        if total_frames is None:
            return None
        total = _safe_int(total_frames)
        return total if total > 0 else None

    def advance(self, phase, done_frames) -> None:
        """``phase`` 패스에서 ``done_frames``까지 처리했음을 알린다."""
        if self.total_frames is None:
            # 총 프레임 수를 모르는 경로 — 수치 대신 진행 중 표시만(요구사항 3.6).
            self._say(self.INDETERMINATE_TEXT)
            return
        done = max(0, _safe_int(done_frames))
        self._push(self._ratio(phase, done))
        frame_ratio = done / self.total_frames
        if frame_ratio > 1.0:
            # 실제 비율을 감추면 사용자는 막대가 1에 붙어 멈춘 것으로 읽는다.
            self._say(f"진행률 {frame_ratio * 100:.1f}% · "
                      f"예상 총 프레임({self.total_frames})보다 많은 "
                      f"{done}프레임을 처리했습니다")

    def finish(self) -> None:
        """완료 상태(1.0)로 만든다(요구사항 3.5). 두 번 불러도 안전하다."""
        self._push(1.0)

    def _ratio(self, phase, done):
        """앞 패스들의 가중치를 누적한 뒤 현재 패스 몫을 얹은 진행 비율.

        클램프하지 않은 값을 돌려준다. 제한은 :meth:`_push`가 한다.
        ``phase``는 가중치 목록 범위 안으로 눌러 잘못된 인덱스에도 터지지 않게 한다.
        """
        weights = self.phase_weights
        i = int(clamp(_safe_int(phase), 0, len(weights) - 1))
        return sum(weights[:i]) + done / self.total_frames * weights[i]

    def _push(self, value) -> None:
        """0~1로 제한하고 단조 증가를 보장한 값을 막대에 넘긴다."""
        v = float(clamp(value, 0.0, 1.0))
        if self.last_value is not None and v < self.last_value:
            v = self.last_value
        self.last_value = v
        fn = getattr(self.bar, "progress", None)
        if callable(fn):
            fn(v)

    def _say(self, text) -> None:
        """캡션 문구를 내보낸다. 같은 문구가 이어지면 다시 쓰지 않는다."""
        if text == self.last_caption:
            return
        self.last_caption = text
        fn = getattr(self.bar, "caption", None)
        if callable(fn):
            fn(text)


# ============================================================
#  실제 분석 : YOLO 2-pass (좌표 수집 -> 위험 판정 -> 색깔 박스)
# ============================================================

#: 1차 패스에서 진행률을 갱신하는 프레임 간격. 매 프레임 갱신은 Streamlit
#: 왕복이 너무 잦다. 마지막 갱신은 루프 뒤에서 한 번 더 한다.
TRACK_PROGRESS_STEP = 20


@dataclass(frozen=True)
class AnalysisResult:
    """분석 한 번의 결과.

    ``video_path``는 재생 대상 경로다. 합성이 성공하면 합성 영상, 실패하면
    원본 영상을 가리킨다. 둘을 경로만 보고 구분하려면 호출자가 원본 경로와
    비교해야 하는데, 그러면 "합성을 시도하지 않았다"와 "시도했지만 실패했다"가
    같은 모양이 된다. ``render_failed``를 따로 두어 화면이 안내 문구를 띄울
    근거를 명시적으로 넘긴다(요구사항 1.5).

    ``browser_safe``는 ``video_path``가 브라우저에서 재생되는 형식(H.264)인지다.
    합성은 mp4v로 쓰기 때문에 그대로 두면 ``st.video``에서 화면이 비는데,
    :func:`to_browser_mp4`가 변환에 실패했는지는 경로만 보고 알 수 없다.
    저장되는 :class:`AnalysisRecord`에는 넣지 않는다 — 파일 형식은 화면을 한 번
    그릴 때의 안내 근거일 뿐이고, 기록으로 남길 분석 결과가 아니다.
    """

    events: tuple[Event, ...]
    video_path: str
    render_failed: bool = False
    browser_safe: bool = True


def analyze_video(video_path, progress=None, total_frames=None, duration_sec=None,
                  fps=None) -> AnalysisResult:
    """영상을 분석해 :class:`AnalysisResult`를 돌려준다.

    ``progress``는 진행률 막대이거나 이미 만들어진 :class:`ProgressReporter`다.
    막대를 받으면 ``total_frames``(``probe()``의 값)를 분모로 삼아 리포터를
    직접 만든다. 분모를 모르면 리포터가 수치 없이 진행 중 표시만 유지한다.

    ``duration_sec``도 ``probe()``가 읽은 값을 그대로 받는다. 타임라인 시각과
    곡선 x축을 제한하는 상한이라, 컨테이너가 알려 준 값이 있으면 그걸 쓰고
    없을 때만 처리한 프레임 수로 되짚어 추정한다(요구사항 8.5).

    ``fps``도 ``probe()``가 읽은 값을 그대로 받는다. 모듈 상수 :data:`FPS`를
    그대로 쓰면 25fps·60fps 영상에서 타임라인 초, 곡선 x 값, 그리고
    :func:`measure_pair`의 모든 시간 기준(측정 창, h/s 속도, 정지 지속 시간,
    반응 지연)이 한꺼번에 어긋난다. 못 믿을 값이면 :func:`_usable_fps`가
    :data:`FPS`로 되돌린다.

    2차 패스(영상 합성)가 실패하면 이벤트는 그대로 살리고 재생 경로만 원본으로
    되돌린다. 합성은 근거를 눈으로 확인하는 수단일 뿐이고, 점수와 타임라인은
    1차 패스에서 이미 다 나왔기 때문이다(요구사항 1.5, 1.6).
    """
    from ultralytics import YOLO

    reporter = (progress if hasattr(progress, "advance")
                else ProgressReporter(total_frames, progress))
    use_fps = _usable_fps(fps)

    model = YOLO("yolov8n.pt")

    # --- 1차: 좌표 수집 ---
    tracks = {}
    results = model.track(video_path, classes=[0], tracker="bytetrack.yaml",
                          persist=True, stream=True, verbose=False)
    frame_idx = 0
    boxes_per_frame = {}
    for r in results:
        frame_idx += 1
        b = r.boxes
        rec = []
        if b is not None and b.id is not None:
            for pid, xyxy, xywh in zip(b.id.tolist(), b.xyxy.tolist(), b.xywh.tolist()):
                pid = int(pid)
                cx, cy = xywh[0], xywh[1] + xywh[3] / 2
                tracks.setdefault(pid, []).append((frame_idx, cx, cy, xywh[3]))
                rec.append((pid, list(map(int, xyxy))))
        boxes_per_frame[frame_idx] = rec
        if frame_idx % TRACK_PROGRESS_STEP == 0:
            reporter.advance(0, frame_idx)
    # 간격 갱신만 두면 마지막 1~19프레임과 20프레임 미만 영상이 phase 0을 한 번도
    # 갱신하지 못한다. 2차 패스와 같이 루프 뒤에 한 번 더 알린다(요구사항 3.1).
    reporter.advance(0, frame_idx)

    # --- 위험도 계산 ---
    # probe()가 읽은 재생 길이를 먼저 쓰고, 없으면 처리한 프레임 수로 추정한다.
    span_sec = duration_sec if duration_sec is not None else (
        int(frame_idx / use_fps) if frame_idx else None)
    events = compute_events(tracks, use_fps, span_sec)
    # 임계값은 MIN_EVENT_SCORE 하나만 쓴다. 여기에 40을 다시 적어 두면
    # compute_events의 기준과 조용히 갈라진다(요구사항 1.2).
    danger_ids = {e.follower for e in events if e.score >= MIN_EVENT_SCORE}

    # --- 2차: 색깔 박스 영상 생성 ---
    out = _temp_video_path()
    rendered = (render_boxes(video_path, out, boxes_per_frame, danger_ids, reporter)
                if out is not None else None)
    reporter.finish()
    if rendered is None:
        return AnalysisResult(events=tuple(events), video_path=str(video_path),
                              render_failed=True)
    # 합성 결과는 mp4v다. 브라우저가 못 읽으므로 H.264로 다시 인코딩한다.
    # 변환이 실패해도 render_failed는 False다 — 박스 합성 자체는 성공했고,
    # 남은 mp4v 파일도 내려받으면 볼 수 있다.
    playable, converted = to_browser_mp4(rendered)
    return AnalysisResult(events=tuple(events), video_path=playable,
                          render_failed=False, browser_safe=converted)


# ------------------------------------------------------------
#  결과 영상 합성
# ------------------------------------------------------------
#
# 박스 색은 화면 등급 색과 같은 팔레트에서 가져온다(요구사항 1.2, 13.6).
# MIN_EVENT_SCORE(40점) 이상 이벤트의 후행자는 코랄, 그 외 사람은 민트다.

#: 위험(코랄) · 일반(민트) 박스 색과 라벨 글자색을 BGR로 미리 바꿔 둔다.
BOX_BGR_DANGER = hex_to_bgr(RISK_HIGH.base)
BOX_BGR_NORMAL = hex_to_bgr(RISK_LOW.base)
BOX_BGR_TEXT = hex_to_bgr(PALETTE["white"])

#: 라벨 글자 크기와 굵기. 프레임 해상도와 무관하게 고정한다.
LABEL_SCALE = 0.55
LABEL_THICK = 2

#: 진행률을 갱신하는 프레임 간격. 매 프레임 갱신은 Streamlit 왕복이 너무 잦다.
RENDER_PROGRESS_STEP = 20


def box_label(pid, danger) -> str:
    """박스 라벨 문자열. ASCII만 쓴다(요구사항 1.4).

    ``cv2.putText``의 Hershey 폰트는 한글 글리프가 없어 한글을 넣으면 물음표
    사각형으로 렌더된다. 의미는 화면 범례(요구사항 1.7)가 한글로 설명하고,
    영상 안에서는 ``id7 SUSPECT`` / ``id7``로 고정한다.
    """
    return f"id{int(pid)} SUSPECT" if danger else f"id{int(pid)}"


def render_boxes(video_path, out_path, boxes_per_frame, danger_ids, progress=None):
    """원본 프레임에 박스와 라벨을 합성해 ``out_path``에 쓴다(요구사항 1.1).

    성공하면 쓴 파일 경로를, 실패하면 ``None``을 돌려준다. 예외를 올리지 않는
    이유는 합성 실패가 분석 실패가 아니기 때문이다. 호출자는 ``None``을 받으면
    재생 경로를 원본으로 돌리고 안내 문구를 띄우면 된다(요구사항 1.5).

    실패로 보는 경우는 네 가지다 — opencv 없음, 원본을 열 수 없음,
    ``VideoWriter``를 열 수 없음(코덱 미지원 등), 읽기/쓰기 중 예외.
    실패하면 반쯤 쓰인 파일을 지운다. 크기 0이거나 깨진 파일을 재생 경로로
    넘기면 화면에는 "영상이 있는데 안 나온다"로 보여 원인 파악이 더 어렵다.
    """
    if _render_pass(video_path, out_path, boxes_per_frame, danger_ids, progress):
        return str(out_path)
    _discard(out_path)
    return None


#: H.264 재인코딩에 실패했을 때 화면에 띄우는 안내. 합성 자체는 성공했으므로
#: 오류가 아니라 경고다 — 파일은 있고, 브라우저가 못 읽을 수 있을 뿐이다.
H264_FALLBACK_NOTE = ("결과 영상을 브라우저 재생 형식(H.264)으로 변환하지 못했습니다. "
                      "영상이 재생되지 않으면 파일을 내려받아 확인해 주세요.")


def to_browser_mp4(src):
    """합성 영상을 브라우저가 재생할 수 있는 H.264 mp4로 다시 인코딩한다.

    ``(경로, 변환했는지)``를 돌려준다.

    :func:`_render_pass`는 ``mp4v``(MPEG-4 Part 2)로 쓴다. opencv 기본 빌드가
    확실히 열어 주는 코덱이라 합성에는 안전하지만, 크롬·파이어폭스는 이 코덱을
    디코드하지 못한다. 그래서 ``st.video``에 넘기면 파일이 정상이어도 화면이
    비어 있다. 재생되는 건 H.264(avc1)뿐이라 여기서 한 번 더 인코딩한다.

    ffmpeg 실행 파일은 ``PATH``에서 먼저 찾고, 없으면 ``imageio-ffmpeg``가
    들고 있는 바이너리를 쓴다(지연 임포트 — 이 패키지는 필수가 아니다).
    둘 다 없거나 인코딩이 실패하면 원본 경로를 그대로 돌려준다. 변환 실패는
    분석 실패가 아니다. 최악의 경우 mp4v 파일이 남고, 호출부가
    :data:`H264_FALLBACK_NOTE`로 내려받기를 안내한다.
    """
    ff = shutil.which("ffmpeg")
    if not ff:
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return src, False
    if not ff:
        return src, False

    dst = None
    try:
        dst = tempfile.NamedTemporaryFile(delete=False, suffix="_h264.mp4").name
        subprocess.run(
            [ff, "-y", "-i", str(src), "-c:v", "libx264", "-preset", "veryfast",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # 크기 하한을 두는 이유: ffmpeg가 0을 돌려주고도 헤더만 있는 파일을
        # 남기는 경우가 있다. 그걸 채택하면 재생 불가가 조용히 이어진다.
        if Path(dst).stat().st_size > 1000:
            _discard(src)       # mp4v 임시파일은 더 쓸 데가 없다
            return dst, True
    except Exception:
        pass
    if dst is not None:
        _discard(dst)
    return src, False


def _render_pass(video_path, out_path, boxes_per_frame, danger_ids, progress) -> bool:
    """합성 본체. 성공 여부만 돌려주고 뒷정리는 :func:`render_boxes`가 한다."""
    try:
        import cv2
    except Exception:
        return False

    cap = None
    writer = None
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False
        w = _safe_int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = _safe_int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = _safe_float(cap.get(cv2.CAP_PROP_FPS)) or float(FPS)
        if w <= 0 or h <= 0:
            return False

        writer = cv2.VideoWriter(str(out_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            # 코덱을 못 찾으면 write()는 조용히 아무 것도 하지 않는다.
            # 여기서 걸러 내지 않으면 크기 0인 파일이 재생 경로로 흘러간다.
            return False

        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            for pid, box in boxes_per_frame.get(idx, ()):
                _draw_box(cv2, frame, pid, box, pid in danger_ids)
            writer.write(frame)
            if idx % RENDER_PROGRESS_STEP == 0:
                _advance(progress, idx)
        if idx == 0:
            return False        # 한 프레임도 못 읽었다 — 재생할 것이 없다
        _advance(progress, idx)
    except Exception:
        return False
    finally:
        # 성공이든 실패든 놓아 준다. writer는 release() 전까지 파일을 마무리하지 않는다.
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()

    return _has_bytes(out_path)


def _draw_box(cv2, frame, pid, box, danger) -> None:
    """사람 하나의 박스와 라벨을 그린다.

    라벨 배경은 글자 폭을 재서 맞춘다. 폭을 고정하면 ``id12 SUSPECT``처럼 긴
    라벨이 배경을 넘어가 읽기 어려워진다. 프레임 위쪽에 자리가 없으면 배경을
    프레임 안으로 눌러 잘리지 않게 한다.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    color = BOX_BGR_DANGER if danger else BOX_BGR_NORMAL
    label = box_label(pid, danger)
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                     LABEL_SCALE, LABEL_THICK)
    band = th + base + 6
    top = max(0, y1 - band)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if danger else 2)
    cv2.rectangle(frame, (x1, top), (x1 + tw + 8, top + band), color, -1)
    cv2.putText(frame, label, (x1 + 4, top + th + 3), cv2.FONT_HERSHEY_SIMPLEX,
                LABEL_SCALE, BOX_BGR_TEXT, LABEL_THICK)


def _advance(progress, done_frames) -> None:
    """진행률 리포터가 있으면 2차 패스(phase 1) 진행을 알린다(요구사항 3.1)."""
    fn = getattr(progress, "advance", None)
    if callable(fn):
        fn(1, done_frames)


def _temp_video_path():
    """합성 결과를 쓸 임시 파일 경로. 만들 수 없으면 ``None``."""
    try:
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tf.close()
        return tf.name
    except OSError:
        return None


def _has_bytes(path) -> bool:
    """파일이 있고 크기가 0보다 큰지."""
    try:
        return Path(path).stat().st_size > 0
    except OSError:
        return False


def _discard(path) -> None:
    """실패한 합성 결과를 지운다. 없거나 못 지워도 넘어간다."""
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def _time_limit(duration_sec):
    """시각을 제한할 상한을 돌려준다. 길이를 모르거나 0 이하면 ``None``."""
    if duration_sec is None:
        return None
    limit = int(duration_sec)
    return limit if limit > 0 else None


def bound_timeline(items, duration_sec):
    """타임라인 시각을 ``[0, duration_sec]`` 안으로 제한한다(요구사항 8.5)."""
    limit = _time_limit(duration_sec)
    if limit is None:
        return tuple(items)
    return tuple(
        TimelineItem(at_sec=int(clamp(int(i.at_sec), 0, limit)), label=i.label, tag=i.tag)
        for i in items
    )


def bound_curve(curve, duration_sec):
    """곡선의 x(초)를 ``[0, duration_sec]`` 안으로 제한한다.

    상한에 눌려 같은 x가 겹치면 나중 값이 남으므로 마지막 좌표의
    위험도(= 최종 점수)가 그대로 곡선 끝에 남는다.
    """
    limit = _time_limit(duration_sec)
    if limit is None:
        return tuple((int(x), int(y)) for x, y in curve)
    merged: dict[int, int] = {}
    for x, y in curve:
        merged[int(clamp(int(x), 0, limit))] = int(y)
    return tuple(sorted(merged.items()))


# ------------------------------------------------------------
#  쌍 측정 : 트랙 좌표 -> PairFeatures · PairTiming
# ------------------------------------------------------------
#
# 거리는 전부 사람 박스 높이로 나눠 쓴다. 픽셀 거리는 카메라 화각과 피사체
# 거리에 따라 의미가 완전히 달라지지만, 박스 높이는 "그 화면에서 사람 한 명
# 크기"라 같은 장면 안에서 비교할 수 있는 자연스러운 기준이 된다. 속도도 같은
# 이유로 초당 박스 높이(h/s) 단위로 잰다.
#
# 아래 상수는 모두 튜닝 손잡이다. 값을 코드 안에 흩뿌리지 않고 여기 모아 둔다.

#: 이동 벡터를 뽑는 창(초). 검출 좌표는 프레임마다 흔들리므로 한 프레임 차이가
#: 아니라 이만큼 떨어진 두 지점을 이어 방향과 속도를 잰다.
MEASURE_WINDOW_SEC = 0.5

#: 근접으로 볼 거리 범위(박스 높이 배수). 아래는 박스가 겹칠 만큼 붙은 오검출,
#: 위는 그냥 같은 화면에 있는 무관한 통행으로 본다.
#:
#: 상한을 3.0에서 1.5로 좁혔다. 사람 키를 1.7m로 보면 3.0h는 5m로 웬만한 보도·
#: 복도 폭을 그대로 덮는다. 즉 "같은 화면에 있으면 근접"에 가까워서 무관한 통행이
#: 후방 근접 유지 점수를 그대로 받아 갔다. 1.5h(약 2.5m)는 뒤에서 따라붙어 걷는
#: 사람과 그냥 같은 길을 지나가는 사람이 갈리는 거리다.
NEAR_MIN_H = 0.2
NEAR_MAX_H = 1.5

#: 첫 접근 시각(타임라인의 '근접 진입')에만 쓰는 느슨한 상한.
#: 점수는 위 ``NEAR_MAX_H`` 안에서만 쌓지만, "언제부터 뒤에 있었나"는 조금 멀리서
#: 따라붙기 시작한 시점을 보여 주는 편이 정확하다. 이 값은 점수에 영향을 주지
#: 않으므로 예전 상한을 그대로 남겨 둔다.
APPROACH_MAX_H = 3.0

#: 후방으로 인정할 최소 각도(도). 선행자의 이동 방향과 '선행자→후행자' 벡터
#: 사이 각도가 이 값 이상이면 후행자가 뒤에 있다고 본다. 90도(내적 0)가 정확히
#: 옆이므로, 그보다 10도 더 뒤를 요구해 옆으로 나란히 걷는 프레임이 후방으로
#: 새는 것을 막는다. 거리만 보던 예전 판정은 나란히 걷기·마주 오는 사람까지
#: 후방 근접으로 셌다.
REAR_ANGLE_DEG = 100.0

#: 나란히(동행)로 볼 각도 범위의 아래쪽 경계(도). ``SIDE_MIN_ANGLE_DEG`` 이상
#: ``REAR_ANGLE_DEG`` 미만이면 뒤가 아니라 옆이다. 이보다 작으면 후행자가 앞서
#: 가고 있다는 뜻이라 미행도 동행도 아니다.
SIDE_MIN_ANGLE_DEG = 60.0

#: 선행자가 잠깐 멈춘 동안 직전 이동 방향을 '뒤'의 기준으로 유지할 시간(초).
#: 멈춰 있는 순간에는 '뒤'가 정의되지 않지만, 걷다가 잠깐 선 사람의 뒤는 방금
#: 걸어온 방향의 반대쪽이라고 보는 것이 자연스럽다. 이 시간을 넘겨도 이동
#: 방향을 못 찾은 프레임은 기준이 없으므로 후방 판정에서 아예 제외한다
#: (근접 프레임으로도, 나란히 비율의 분모로도 세지 않는다).
REAR_HEADING_HOLD_SEC = 1.5

#: 이동 중으로 인정할 최소 속도(h/s). 이 미만은 제자리 흔들림으로 본다.
MOVING_SPEED_H = 0.30

#: 정지 판정(h/s)과 정지로 인정할 최소 지속 시간(초).
STOP_SPEED_H = 0.15
STOP_MIN_SEC = 0.8

#: 지연 경로 유사도 — 선행자 위치를 몇 초 전으로 되짚을지, 그 지점에서 이만큼
#: 안에 있으면 "지나간 자리를 그대로 밟았다"로 본다(박스 높이 배수).
PATH_LAG_SEC = 2.0
PATH_NEAR_H = 0.8

#: 추종으로 인정할 반응 지연 상한(초). 선행자 사건 뒤 이 시간 안에 후행자
#: 사건이 일어나야 짝으로 센다.
FOLLOW_LAG_MAX_SEC = 4.0

#: 방향 전환 판정 각도(도)와, 후행자 전환이 선행자 전환과 같다고 볼 각도 차이.
TURN_ANGLE_DEG = 35.0
TURN_MATCH_DEG = 60.0

#: 같은 사건을 두 번 세지 않기 위한 선행자 사건 최소 간격(초).
EVENT_GAP_SEC = 3.0

#: 쌍을 판정 대상으로 삼을 최소 겹침 시간(초)과 최소 근접 유지 비율.
MIN_OVERLAP_SEC = 5.0
MIN_REAR_RATIO = 0.4

# --- 동행 감점 (측정값) ---
#
# 예전에는 모든 쌍에 5점을 똑같이 뺐다. 상수 감점은 동행을 걸러 내지도 못하고
# (5점으로는 판정이 바뀌지 않는다) 근거로 보여 줄 것도 없다. 이제 두 가지를
# 실제로 재서 더한다.
#
#   1) 나란히 비율 — 후행자가 뒤가 아니라 옆에 있던 프레임 비율. 후방 판정과
#      같은 각도 기준(``REAR_ANGLE_DEG``·``SIDE_MIN_ANGLE_DEG``)을 쓴다.
#   2) 거의 동시 정지 — 두 사람이 아주 짧은 시간 안에 함께 멈춘 횟수.
#      동행은 같이 멈추고, 미행은 앞사람이 멈춘 뒤에 멈춘다.

#: 나란히 비율 1.0일 때의 감점. 비율에 곱해 쓴다.
COMPANION_SIDE_WEIGHT = 12

#: '거의 동시'로 볼 정지 시각 차이(초). 추종 판정의 ``FOLLOW_LAG_MAX_SEC``(4초)
#: 보다 훨씬 짧게 둔다 — 이 창이 넓으면 정지 추종을 동시 정지로 잘못 읽는다.
COMPANION_SYNC_STOP_SEC = 0.8

#: 거의 동시 정지 1회당 감점.
COMPANION_SYNC_STOP_WEIGHT = 5

#: 측정 감점 상한. 이 위로는 올라가지 않는다(감점 하나가 판정을 전부 덮지
#: 못하게 한다). 실제로 남기는 값은 가점 합으로 한 번 더 자른다.
COMPANION_DEDUCTION_MAX = 20

#: 결과로 남길 최소 위험도 점수.
MIN_EVENT_SCORE = 40

#: 의도를 보여 주는 점수 항목. 후방 근접·지속 시간·방향 유사도는 같은 길을 같은
#: 방향으로 걷기만 해도 쌓이므로, 이 세 항목 중 하나라도 관측되지 않으면
#: 이벤트로 남기지 않는다(:func:`has_intent_evidence`).
INTENT_ROWS: tuple[str, ...] = ("지연 경로 유사도", "정지 후 추종", "방향 전환 추종")


def has_intent_evidence(rows) -> bool:
    """의도 신호가 하나라도 관측됐는지.

    ``INTENT_ROWS``에 든 항목 중 점수가 0보다 큰 것이 하나라도 있으면 참이다.
    나머지 세 항목(후방 근접 유지·지속 시간·방향 유사도)은 같은 시간에 같은
    길을 걷는 것만으로도 채워지므로, 그것만으로 합계 임계값을 넘긴 쌍은
    미행의 근거가 없다고 본다. :data:`MIN_EVENT_SCORE`와 함께 두 관문이
    모두 걸린다.
    """
    return any(int(r.score) > 0 for r in rows if r.name in INTENT_ROWS)


def _usable_fps(fps) -> float:
    """나눗셈에 쓸 수 있는 fps만 남긴다. 못 믿을 값은 기본 fps로 대체한다."""
    f = _safe_float(fps)
    return f if f > 0 else float(FPS)


def _track_series(track) -> dict[int, tuple[float, float, float]]:
    """트랙 목록을 ``프레임 -> (x, y, 박스높이)`` 사전으로 만든다.

    같은 프레임이 두 번 들어오면 나중 값이 남는다(추적기 출력에서는 프레임마다
    한 점이므로 정상 경로에서는 일어나지 않는다).
    """
    return {int(f): (float(x), float(y), float(h)) for f, x, y, h in track}


def _motion(series, window, window_sec) -> dict[int, tuple[float, float, float]]:
    """프레임별 ``(dx, dy, 정규화 속도)``. 속도 단위는 초당 박스 높이다.

    ``window`` 프레임 전 위치와의 차이를 쓰므로, 트랙 앞부분과 좌표가 끊긴
    구간은 결과에서 빠진다. 박스 높이가 0 이하인 프레임도 나눌 수 없어 뺀다.
    """
    out = {}
    for f, (x, y, h) in series.items():
        prev = series.get(f - window)
        if prev is None:
            continue
        scale = (h + prev[2]) / 2
        if scale <= 0:
            continue
        dx, dy = x - prev[0], y - prev[1]
        out[f] = (dx, dy, math.hypot(dx, dy) / scale / window_sec)
    return out


def _wrap_deg(deg) -> float:
    """각도를 ``(-180, 180]`` 범위로 접는다."""
    return (float(deg) + 180.0) % 360.0 - 180.0


def _angle_delta(a, b) -> float:
    """벡터 ``a``에서 ``b``까지의 회전 각도(도). 왼쪽/오른쪽이 부호로 갈린다."""
    return _wrap_deg(math.degrees(math.atan2(b[1], b[0]) - math.atan2(a[1], a[0])))


def _lead_headings(lead_motion, fps) -> dict[int, tuple[float, float]]:
    """프레임 -> 선행자의 '앞' 방향 벡터. 방향을 못 정하는 프레임은 빠진다.

    이동 중(``MOVING_SPEED_H`` 이상)인 프레임은 그 프레임의 이동 벡터를 쓴다.
    멈춰 있는 프레임은 :data:`REAR_HEADING_HOLD_SEC` 안의 직전 이동 방향을
    이어서 쓴다 — 걷다가 잠깐 선 사람의 '뒤'는 방금 걸어온 방향의 반대쪽이다.
    그 시간을 넘도록 이동 방향이 없으면(처음부터 서 있던 사람 등) 이 사전에
    담기지 않으며, 후방·나란히 판정에서 그 프레임은 아예 빠진다.
    """
    hold = max(1, int(round(REAR_HEADING_HOLD_SEC * fps)))
    out: dict[int, tuple[float, float]] = {}
    last: tuple[int, tuple[float, float]] | None = None
    for f in sorted(lead_motion):
        dx, dy, sp = lead_motion[f]
        if sp >= MOVING_SPEED_H and (dx or dy):
            last = (f, (dx, dy))
            out[f] = (dx, dy)
        elif last is not None and f - last[0] <= hold:
            out[f] = last[1]
    return out


def _rear_angle(heading, lead_pos, follow_pos) -> float:
    """선행자의 진행 방향과 '선행자→후행자' 벡터 사이 각도(도, 0~180).

    0도면 후행자가 정면에, 90도면 옆에, 180도면 정확히 뒤에 있다는 뜻이다.
    두 점이 겹치면 방향이 없으므로 ``180``으로 본다(가장 가까운 뒤).
    """
    vx, vy = follow_pos[0] - lead_pos[0], follow_pos[1] - lead_pos[1]
    norm = math.hypot(heading[0], heading[1]) * math.hypot(vx, vy)
    if norm <= 0:
        return 180.0
    cos = clamp((heading[0] * vx + heading[1] * vy) / norm, -1.0, 1.0)
    return math.degrees(math.acos(cos))


def _positions(lead, follow, common, headings, max_h):
    """거리 조건을 통과한 프레임의 ``(프레임, 후방 각도)`` 목록.

    거리는 ``NEAR_MIN_H``~``max_h``(박스 높이 배수) 사이여야 하고, 그 프레임에
    선행자의 진행 방향이 정의돼 있어야 한다. 방향이 없는 프레임은 뒤인지 옆인지
    가릴 수 없으므로 통째로 빠진다(:func:`_lead_headings` 주석 참고).
    """
    out = []
    for f in common:
        heading = headings.get(f)
        if heading is None:
            continue
        lx, ly, lh = lead[f]
        fx, fy, fh = follow[f]
        scale = (lh + fh) / 2
        if scale <= 0:
            continue
        if not (NEAR_MIN_H <= math.hypot(lx - fx, ly - fy) / scale <= max_h):
            continue
        out.append((int(f), _rear_angle(heading, (lx, ly), (fx, fy))))
    return out


def _near_frames(lead, follow, common, headings) -> tuple[int, ...]:
    """후행자가 '뒤에서' 근접해 있던 프레임(요구사항 12.1의 '후방 근접 유지').

    거리만 보면 나란히 걷는 일행과 마주 오는 사람도 후방 근접으로 잡힌다.
    그래서 거리(``NEAR_MIN_H``~``NEAR_MAX_H``)와 함께 위치도 본다 — 선행자의
    진행 방향 기준으로 :data:`REAR_ANGLE_DEG` 이상 뒤에 있어야 한다.
    """
    return tuple(f for f, angle in _positions(lead, follow, common, headings, NEAR_MAX_H)
                 if angle >= REAR_ANGLE_DEG)


def _approach_frame(lead, follow, common, headings) -> int | None:
    """후행자가 뒤에서 처음 따라붙은 프레임. 없으면 ``None``.

    거리 상한만 :data:`APPROACH_MAX_H`로 넓힌 :func:`_near_frames`다. 점수에는
    쓰지 않고 타임라인의 '근접 진입' 시각으로만 쓴다.
    """
    for f, angle in _positions(lead, follow, common, headings, APPROACH_MAX_H):
        if angle >= REAR_ANGLE_DEG:
            return f
    return None


def _side_ratio(lead, follow, common, headings) -> float:
    """뒤가 아니라 옆에 있던 프레임 비율(0~1).

    분모는 후방·나란히를 가릴 수 있었던 프레임(거리 조건을 통과하고 선행자
    진행 방향이 정의된 프레임)이고, 분자는 그중 각도가
    :data:`SIDE_MIN_ANGLE_DEG` 이상 :data:`REAR_ANGLE_DEG` 미만인 프레임이다.
    후방 판정과 같은 각도 기준을 쓰므로 두 값이 같은 잣대 위에 있다.
    가릴 수 있는 프레임이 없으면 0을 돌려준다(동행이라고 볼 근거가 없다).
    """
    angles = [a for _, a in _positions(lead, follow, common, headings, NEAR_MAX_H)]
    if not angles:
        return 0.0
    side = sum(1 for a in angles if SIDE_MIN_ANGLE_DEG <= a < REAR_ANGLE_DEG)
    return side / len(angles)


def _sync_stops(lead_stops, follow_stops, fps) -> int:
    """두 사람이 거의 동시에 멈춘 횟수.

    정지 시작 시각 차이가 :data:`COMPANION_SYNC_STOP_SEC` 안이면 한 번으로 센다
    (앞뒤 순서는 보지 않는다 — 동시에 멈추는 것 자체가 동행 신호다). 한 번 짝지은
    정지는 다시 쓰지 않는다.
    """
    window = COMPANION_SYNC_STOP_SEC * fps
    used = set()
    count = 0
    for lf in sorted(lead_stops):
        for ff in sorted(follow_stops):
            if ff in used or abs(ff - lf) > window:
                continue
            used.add(ff)
            count += 1
            break
    return count


def _companion(lead, follow, common, headings, lead_stops, follow_stops, fps):
    """동행 신호를 재서 :class:`PairCompanion`으로 담는다.

    감점은 ``나란히 비율 × COMPANION_SIDE_WEIGHT + 동시 정지 × COMPANION_SYNC_STOP_WEIGHT``
    이며 :data:`COMPANION_DEDUCTION_MAX`에서 자른다.
    """
    side = _side_ratio(lead, follow, common, headings)
    sync = _sync_stops(lead_stops, follow_stops, fps)
    raw = side * COMPANION_SIDE_WEIGHT + sync * COMPANION_SYNC_STOP_WEIGHT
    return PairCompanion(side_ratio=side, sync_stops=sync,
                         deduction=int(clamp(round(raw), 0, COMPANION_DEDUCTION_MAX)))


def _heading_similarity(lead_motion, follow_motion) -> float:
    """이동 방향 유사도(0~1).

    두 사람이 함께 움직이는 프레임에서만 방향 코사인을 재고, ``(1+cos)/2``로
    0~1에 옮겨 평균한다. 같은 방향이면 1, 직각이면 0.5, 반대 방향이면 0이다.
    한쪽이라도 멈춰 있는 프레임은 방향이 정의되지 않으므로 제외한다. 함께
    움직인 프레임이 없으면 0을 돌려준다(유사하다고 볼 근거가 없다).
    """
    sims = []
    for f, (lx, ly, lsp) in lead_motion.items():
        fol = follow_motion.get(f)
        if fol is None or lsp < MOVING_SPEED_H or fol[2] < MOVING_SPEED_H:
            continue
        norm = math.hypot(lx, ly) * math.hypot(fol[0], fol[1])
        if norm <= 0:
            continue
        cos = clamp((lx * fol[0] + ly * fol[1]) / norm, -1.0, 1.0)
        sims.append((cos + 1.0) / 2.0)
    return sum(sims) / len(sims) if sims else 0.0


def _path_ratio(lead, follow, lag) -> float:
    """후행자가 ``lag`` 프레임 전 선행자 자리 근처에 있던 프레임 비율(0~1).

    분모는 비교가 가능한 프레임 수(후행자 좌표와 ``lag`` 전 선행자 좌표가 모두
    있는 프레임)다. 비교할 프레임이 없으면 0을 돌려준다.
    """
    hit = seen = 0
    for f, (fx, fy, fh) in follow.items():
        past = lead.get(f - lag)
        if past is None or fh <= 0:
            continue
        seen += 1
        if math.hypot(fx - past[0], fy - past[1]) / fh <= PATH_NEAR_H:
            hit += 1
    return hit / seen if seen else 0.0


def _stop_starts(motion, fps) -> tuple[int, ...]:
    """정지가 시작된 프레임 목록.

    ``STOP_SPEED_H`` 미만이 ``STOP_MIN_SEC`` 이상 이어진 구간 하나를 정지 한
    번으로 보고 그 시작 프레임을 담는다. 좌표가 끊긴 자리(프레임 간격이 측정
    창보다 큰 지점)에서는 구간을 끊는다 — 추적이 빠진 동안을 정지로 오해하면
    안 된다.
    """
    gap_limit = max(1, int(round(MEASURE_WINDOW_SEC * fps)))
    need = STOP_MIN_SEC * fps
    out = []
    start = prev = None
    for f in sorted(motion):
        slow = motion[f][2] < STOP_SPEED_H
        if slow and start is not None and f - prev <= gap_limit:
            prev = f
            continue
        if start is not None and prev - start >= need:
            out.append(int(start))
        start = f if slow else None
        prev = f
    if start is not None and prev - start >= need:
        out.append(int(start))
    return tuple(out)


def _turns(motion, window) -> tuple[tuple[int, float], ...]:
    """방향이 크게 꺾인 ``(프레임, 회전 각도)`` 목록.

    ``window`` 프레임 간격을 둔 두 이동 벡터 사이 각도가 ``TURN_ANGLE_DEG``
    이상이면 전환으로 본다. 양쪽 다 이동 중이어야 하므로, 멈춰 서서 몸만 돌린
    구간은 방향 전환으로 잡히지 않는다.
    """
    out = []
    for f, (dx, dy, sp) in motion.items():
        nxt = motion.get(f + window)
        if nxt is None or sp < MOVING_SPEED_H or nxt[2] < MOVING_SPEED_H:
            continue
        delta = _angle_delta((dx, dy), (nxt[0], nxt[1]))
        if abs(delta) >= TURN_ANGLE_DEG:
            out.append((int(f), delta))
    return tuple(sorted(out, key=lambda t: t[0]))


def _follow_matches(lead_events, follow_events, fps, accept=None) -> tuple[int, ...]:
    """선행자 사건에 뒤이은 후행자 사건을 짝지어 그 프레임을 모은다.

    ``lead_events``/``follow_events``는 ``(프레임, 값)`` 목록이다. 짝이 되는
    조건은 두 가지다 — 후행자 사건이 선행자 사건 이후 ``FOLLOW_LAG_MAX_SEC``
    안에 있어야 하고, ``accept(선행값, 후행값)``이 참이어야 한다.

    한 번 쓴 후행자 사건은 다시 쓰지 않는다(한 번의 반응이 여러 번으로 세는 것을
    막는다). 선행자 사건도 ``EVENT_GAP_SEC`` 안에 몰려 있으면 같은 사건으로 보고
    첫 번째만 남긴다.
    """
    lag_max = FOLLOW_LAG_MAX_SEC * fps
    gap = EVENT_GAP_SEC * fps
    follows = sorted(follow_events, key=lambda e: e[0])
    used = set()
    out = []
    last_lead = None
    for lf, lval in sorted(lead_events, key=lambda e: e[0]):
        if last_lead is not None and lf - last_lead < gap:
            continue
        last_lead = lf
        for ff, fval in follows:
            if ff in used or not (0 <= ff - lf <= lag_max):
                continue
            if accept is not None and not accept(lval, fval):
                continue
            used.add(ff)
            out.append(int(ff))
            break
    return tuple(out)


def _same_turn(lead_delta, follow_delta) -> bool:
    """후행자 전환이 선행자 전환과 같은 방향·비슷한 크기인지."""
    return abs(_wrap_deg(follow_delta - lead_delta)) <= TURN_MATCH_DEG


def measure_pair(lead_track, follow_track, fps):
    """한 쌍의 트랙 좌표에서 ``(PairFeatures, PairTiming, PairCompanion)``을 뽑는다.

    겹치는 시간이 ``MIN_OVERLAP_SEC`` 미만이거나 후방 근접 유지 비율이
    ``MIN_REAR_RATIO`` 미만이면 판정 대상이 아니므로 ``None``을 돌려준다.

    이동 벡터를 먼저 뽑는다 — 후방 판정이 선행자의 진행 방향을 쓰기 때문이다.

    순수 함수다 — 좌표만 보고 계산하며 파일이나 화면에 손대지 않는다.
    """
    fps = _usable_fps(fps)
    lead = _track_series(lead_track)
    follow = _track_series(follow_track)
    common = sorted(set(lead) & set(follow))
    if len(common) < MIN_OVERLAP_SEC * fps:
        return None

    window = max(1, int(round(MEASURE_WINDOW_SEC * fps)))
    lead_motion = _motion(lead, window, window / fps)
    follow_motion = _motion(follow, window, window / fps)
    headings = _lead_headings(lead_motion, fps)

    near = _near_frames(lead, follow, common, headings)
    rear_ratio = len(near) / len(common)
    if rear_ratio < MIN_REAR_RATIO:
        return None

    lead_stops = _stop_starts(lead_motion, fps)
    follow_stops = _stop_starts(follow_motion, fps)
    stop_frames = _follow_matches(
        [(f, None) for f in lead_stops],
        [(f, None) for f in follow_stops],
        fps,
    )
    turn_frames = _follow_matches(
        _turns(lead_motion, window), _turns(follow_motion, window), fps,
        accept=_same_turn,
    )

    features = PairFeatures(
        rear_ratio=rear_ratio,
        duration_sec=len(near) / fps,
        heading=_heading_similarity(lead_motion, follow_motion),
        path_ratio=_path_ratio(lead, follow, max(1, int(round(PATH_LAG_SEC * fps)))),
        stop_follows=len(stop_frames),
        turn_follows=len(turn_frames),
    )
    timing = PairTiming(
        approach_frame=_approach_frame(lead, follow, common, headings),
        stop_frames=stop_frames,
        turn_frames=turn_frames,
    )
    companion = _companion(lead, follow, common, headings,
                           lead_stops, follow_stops, fps)
    return features, timing, companion


# ------------------------------------------------------------
#  타임라인과 위험도 곡선 : PairTiming -> 화면에 쓸 값
# ------------------------------------------------------------
#
# 여기부터는 '언제'를 사람이 읽을 수 있는 값으로 옮기는 단계다. 데모 생성기
# (``timeline_demo`` / ``curve_demo``)가 하던 자리이며, 이제는 추적 좌표에서 나온
# :class:`PairTiming`과 점수 행만 보고 만든다(요구사항 2.3).

#: 곡선 샘플 개수(양 끝 포함). 사건 시점은 이 격자와 별도로 항상 넣는다.
CURVE_SAMPLES = 12

#: 사건 시점에 한 번에 붙는 점수 항목 -> :class:`PairTiming`의 프레임 목록 이름.
#: 나머지 항목은 근접이 이어지는 동안 서서히 쌓이는 값으로 본다.
STEP_ROWS: tuple[tuple[str, str], ...] = (
    ("정지 후 추종", "stop_frames"),
    ("방향 전환 추종", "turn_frames"),
)


def _frame_to_sec(frame, fps) -> int:
    """프레임 인덱스를 초로 환산한다.

    추적 결과의 프레임 번호는 1부터 시작하므로 첫 프레임이 0초가 되도록
    1을 뺀다. 음수는 0으로 눌러 준다.
    """
    return max(0, int((int(frame) - 1) / _usable_fps(fps)))


def _pair_end_sec(lead_track, follow_track, fps) -> int:
    """두 사람이 함께 보인 마지막 시각(초). 겹치는 프레임이 없으면 0.

    곡선의 x축 상한으로 쓴다. 영상 전체 길이가 아니라 쌍이 관측된 구간까지만
    그리는 편이 정확하다 — 관측이 끝난 뒤 구간은 판정 근거가 없다.
    """
    common = ({int(f) for f, *_ in lead_track}
              & {int(f) for f, *_ in follow_track})
    return _frame_to_sec(max(common), fps) if common else 0


def build_timeline(leader, follower, timing, fps) -> tuple[TimelineItem, ...]:
    """관측 시점을 타임라인 항목으로 옮긴다(요구사항 2.3).

    근접 진입은 한 번(처음 들어온 시점), 정지 추종과 방향 전환 추종은 관측된
    횟수만큼 항목이 생긴다. 시각 오름차순으로 정렬하며, 같은 초에 겹치면
    근접 → 정지 → 방향 순서가 유지된다(안정 정렬).
    """
    items: list[TimelineItem] = []
    if timing.approach_frame is not None:
        items.append(TimelineItem(
            _frame_to_sec(timing.approach_frame, fps),
            f"id{follower}이 id{leader} 후방 근접 범위에 진입", "근접"))
    for f in timing.stop_frames:
        items.append(TimelineItem(
            _frame_to_sec(f, fps),
            f"id{leader} 정지 → id{follower}도 멈춤", "정지 추종"))
    for f in timing.turn_frames:
        items.append(TimelineItem(
            _frame_to_sec(f, fps),
            f"id{leader} 방향 전환 → id{follower}도 같은 방향으로 전환", "방향 추종"))
    items.sort(key=lambda i: i.at_sec)
    return tuple(items)


def build_curve(rows, timing, fps, final_score, end_sec) -> tuple[tuple[int, int], ...]:
    """구간 누적 점수를 시간축으로 샘플링해 ``(초, 0~100)`` 좌표를 만든다.

    모양은 누적 가점이 정하고, 크기는 최종 점수에 맞춘다. 누적 가점 비율에
    최종 점수를 곱하므로 마지막 좌표의 위험도가 최종 점수와 정확히 같다
    (곡선 끝과 표의 최종 점수가 어긋나면 화면이 서로 다른 말을 한다).

    쌓이는 방식은 두 가지로 나눈다.

    * 정지 후 추종 · 방향 전환 추종 — 사건이 관측된 시각에 그 항목 점수를
      횟수만큼 나눠 한 번에 붙인다(계단).
    * 나머지 네 항목 — 근접 진입부터 관측 종료까지 고르게 쌓는다(경사).
      비율·지속 시간·방향 유사도는 특정 순간의 사건이 아니라 구간 전체에서
      나온 값이라 특정 시점에 붙일 근거가 없다.

    가점이 하나도 없으면(감점만 있는 쌍) 모양을 정할 근거가 없으므로 0부터
    최종 점수까지 직선으로 잇는다.
    """
    fin = int(clamp(int(final_score), 0, 100))
    span = max(1, int(end_sec))
    start = int(clamp(_frame_to_sec(timing.approach_frame, fps)
                      if timing.approach_frame is not None else 0, 0, span))

    by_name = {r.name: int(r.score) for r in rows}
    steps: list[tuple[int, float]] = []
    step_total = 0.0
    for name, attr in STEP_ROWS:
        score = by_name.get(name, 0)
        frames = tuple(getattr(timing, attr, ()) or ())
        if score <= 0 or not frames:
            continue
        step_total += score
        share = score / len(frames)
        for f in frames:
            steps.append((int(clamp(_frame_to_sec(f, fps), 0, span)), share))
    gradual = max(0.0, sum(by_name.values()) - step_total)
    total = gradual + step_total

    def accrued(t):
        """``t``초까지 쌓인 가점 비율(0~1)."""
        if total <= 0:
            return t / span
        got = sum(w for at, w in steps if at <= t)
        if t >= start:
            got += (gradual if span <= start
                    else gradual * clamp((t - start) / (span - start), 0.0, 1.0))
        return got / total

    times = {0, start, span}
    times.update(at for at, _ in steps)
    times.update(int(round(span * i / (CURVE_SAMPLES - 1)))
                 for i in range(CURVE_SAMPLES))
    points = [(t, int(clamp(round(fin * accrued(t)), 0, 100)))
              for t in sorted(times)]
    # 부동소수 누적 오차로 마지막 값이 1점 어긋나는 것을 막는다.
    points[-1] = (points[-1][0], fin)
    return tuple(points)


def compute_events(tracks, fps, duration_sec):
    """트랙 좌표에서 이벤트를 만든다.

    측정값은 전부 :func:`measure_pair`가 좌표에서 재고, 이 함수는 점수 행 생성,
    타임라인·곡선 조립, 쌍 정리(방향별 중복 제거·정렬)를 맡는다.

    ``duration_sec``은 타임라인 시각과 곡선 x축을 영상 길이 안으로
    제한하는 데 쓴다. 길이를 모르면 ``None``을 넘기고 제한하지 않는다.
    """
    ids = list(tracks)
    out = []
    for leader in ids:
        for follower in ids:
            if leader == follower:
                continue
            measured = measure_pair(tracks[leader], tracks[follower], fps)
            if measured is None:
                continue
            features, timing, companion = measured
            rows = build_rows(features)
            # 관문은 두 개다. (1) 의도 신호가 하나라도 있어야 한다 — 후방 근접·
            # 지속 시간·방향 유사도만으로 임계값을 넘긴 쌍은 그냥 같은 길을 걷는
            # 사람이다. (2) 그 위에 합계 임계값(MIN_EVENT_SCORE)도 넘어야 한다.
            if not has_intent_evidence(rows):
                continue
            suspicion = sum(r.score for r in rows)
            # 감점이 가점 합보다 커지면 화면에 '가점보다 큰 감점'이 남는다.
            # 측정 감점은 가점 합에서 자른다(요구사항 12.7).
            ded = min(companion.deduction, suspicion)
            fin = clamp(suspicion - ded, 0, 100)
            if fin < MIN_EVENT_SCORE:
                continue
            end_sec = _pair_end_sec(tracks[leader], tracks[follower], fps)
            out.append(Event(
                leader=leader, follower=follower, rows=rows, deduction=ded,
                timeline=bound_timeline(
                    build_timeline(leader, follower, timing, fps), duration_sec),
                curve=bound_curve(
                    build_curve(rows, timing, fps, fin, end_sec), duration_sec),
            ))
    best = {}
    for e in out:
        k = tuple(sorted([e.leader, e.follower]))
        if k not in best or e.score > best[k].score:
            best[k] = e
    return sorted(best.values(), key=lambda e: -e.score)


# ============================================================
#  analysis : 의존 패키지 확인과 모드 확정
# ============================================================
#
# 실분석 경로는 ``cv2``·``ultralytics``와 가중치 파일 ``yolov8n.pt``가 있어야
# 성립한다. 하나라도 없으면 오류 내용을 화면에 알리고 데모 경로로 계속 진행한다
# (요구사항 2.7). 여기서 확인하지 않으면 ``analyze_video`` 안의 임포트나
# ``YOLO("yolov8n.pt")``가 스택트레이스로 터져 사용자는 원인도 대안도 못 본다.

#: 확인 로직이 들어 있는 스크립트. ``scripts/``는 패키지가 아니라 ``import``로
#: 끌어올 수 없으므로 파일 경로로 읽어 온다.
CHECK_DEPS_PATH = Path(__file__).resolve().parent / "scripts" / "check_deps.py"

#: 위 스크립트를 올릴 때 쓰는 모듈 이름. ``sys.modules``에 등록해야 스크립트
#: 안의 dataclass가 자기 모듈을 찾을 수 있다.
CHECK_DEPS_MODULE = "zipkimi_check_deps"

#: 실분석에 반드시 필요한 임포트 — (임포트 이름, 배포 이름).
#: ``streamlit``은 이 코드가 도는 시점에 이미 올라와 있으므로 보지 않는다.
ANALYSIS_PACKAGES: tuple[tuple[str, str], ...] = (
    ("cv2", "opencv-python"),
    ("ultralytics", "ultralytics"),
)

#: 실분석에 반드시 필요한 가중치 파일. ultralytics가 첫 실행 때 내려받지만
#: 그건 네트워크가 있을 때 이야기다. 파일이 없고 네트워크도 없으면
#: ``YOLO("yolov8n.pt")``가 그 자리에서 예외로 터지므로, 패키지 확인과 같은
#: 시점에 파일도 함께 본다(요구사항 2.7).
ANALYSIS_WEIGHTS: tuple[str, ...] = ("yolov8n.pt",)

#: 가중치 파일을 찾는 기준 경로. ``scripts/check_deps.py``의 ``ROOT``와 같다.
ANALYSIS_ROOT = Path(__file__).resolve().parent


def _load_check_deps():
    """``scripts/check_deps.py``를 모듈로 읽어 온다. 못 읽으면 ``None``.

    확인 로직을 앱 안에 다시 쓰지 않고 스크립트 쪽 하나만 쓰기 위한 통로다
    (CLI와 앱이 같은 판단을 하도록). 스크립트가 없거나 실행에 실패하면
    :func:`analysis_deps_error`가 자체 임포트 확인으로 되돌아간다.
    """
    cached = sys.modules.get(CHECK_DEPS_MODULE)
    if cached is not None:
        return cached
    try:
        spec = importlib.util.spec_from_file_location(
            CHECK_DEPS_MODULE, CHECK_DEPS_PATH)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        # dataclass 데코레이터가 ``sys.modules[cls.__module__]``를 들여다보므로
        # 실행 전에 등록해 둔다. 등록 없이 exec_module하면 그 자리에서 터진다.
        sys.modules[CHECK_DEPS_MODULE] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(CHECK_DEPS_MODULE, None)
            raise
        return mod
    except Exception:
        return None


def _import_failure(import_name, dist_name) -> str | None:
    """패키지 하나를 임포트해 보고 실패 내용을 돌려준다. 되면 ``None``.

    :func:`_load_check_deps`가 실패했을 때 쓰는 최소 대체 경로다.
    """
    label = import_name if dist_name == import_name else f"{import_name} ({dist_name})"
    try:
        importlib.import_module(import_name)
    except BaseException as exc:   # 임포트는 ImportError 말고도 무엇이든 낼 수 있다
        return f"{label}: 임포트 실패 — {type(exc).__name__}: {exc}"
    return None


def _file_failure(name, root=None) -> str | None:
    """가중치 파일 하나의 존재를 확인해 실패 내용을 돌려준다. 있으면 ``None``.

    :func:`_load_check_deps`가 실패했을 때 쓰는 최소 대체 경로다. 문구는
    스크립트 쪽 ``check_file``과 같은 모양으로 맞춘다.
    """
    path = (root or ANALYSIS_ROOT) / str(name)
    return None if path.is_file() else f"{name}: 파일 없음 — {path}"


def analysis_deps_error(packages=ANALYSIS_PACKAGES, weights=ANALYSIS_WEIGHTS,
                        root=None) -> str | None:
    """실분석 의존 항목을 확인해 문제가 있으면 오류 내용을, 없으면 ``None``.

    확인 대상은 두 가지다 — 임포트해야 하는 패키지(``packages``)와 있어야 하는
    가중치 파일(``weights``). 가중치가 없고 네트워크도 없으면
    ``YOLO("yolov8n.pt")``가 스택트레이스로 터지므로 패키지와 같은 확인에 넣는다.

    돌려주는 문자열은 그대로 화면에 띄울 수 있는 사람이 읽는 내용이다
    (요구사항 2.7). 실패 항목이 여럿이면 줄바꿈으로 이어 붙이며, 패키지 실패와
    파일 실패가 ``이름: 내용`` 같은 형식을 쓴다.
    """
    checker = _load_check_deps()
    check = getattr(checker, "check_package", None)
    check_path = getattr(checker, "check_file", None)
    problems = []
    for import_name, dist_name in packages:
        if callable(check):
            item = check(import_name, dist_name)
            if not item.ok:
                problems.append(f"{item.name}: {item.detail}")
            continue
        failure = _import_failure(import_name, dist_name)
        if failure:
            problems.append(failure)
    for name in weights:
        if callable(check_path):
            item = check_path(name, root)
            if not item.ok:
                problems.append(f"{item.name}: {item.detail}")
            continue
        failure = _file_failure(name, root)
        if failure:
            problems.append(failure)
    return "\n".join(problems) if problems else None


def requested_mode() -> str:
    """화면이 요청한 분석_모드(``"demo"`` 또는 ``"real"``).

    사이드바 전환 수단이 ``st.session_state.mode``에 코드를 넣는다. 상태가 비어
    있거나 아는 코드가 아니면 :data:`DEFAULT_MODE`(``"demo"``)를 쓴다
    (요구사항 2.2). 모듈 상수로 모드를 박아 두지 않기 때문에, 코드를 고치지
    않고 화면에서 모드를 바꿀 수 있다.

    Streamlit 런타임 없이 임포트한 경우에도 성립해야 하므로 세션 상태 접근은
    전부 방어적으로 감싼다.
    """
    state = getattr(st, "session_state", None)
    value = None
    if state is not None:
        try:
            value = state.get("mode")
        except Exception:
            value = None
    return value if value in MODE_LABELS else DEFAULT_MODE


def resolve_mode(mode=None) -> tuple[str, str | None]:
    """요청된 모드를 실제로 쓸 모드로 확정한다 — ``(모드, 오류 내용 또는 None)``.

    ``실분석``을 요청했는데 의존 항목(패키지 또는 가중치 파일)이 없으면
    ``"demo"``와 오류
    내용을 함께 돌려준다. 호출자는 오류 내용을 화면에 띄우고 데모 경로로 계속
    진행한다(요구사항 2.7). 오류 내용을 예외로 올리지 않는 이유가 여기 있다 —
    분석을 멈추는 것이 아니라 경로를 바꾸는 판단이다.
    """
    wanted = mode if mode in ("demo", "real") else requested_mode()
    if wanted != "real":
        return "demo", None
    detail = analysis_deps_error()
    if detail:
        return "demo", detail
    return "real", None


# ============================================================
#  데모 : 화면 확인용 예시 이벤트 (실분석 경로에서는 쓰지 않는다)
# ============================================================
#
# 데모 이벤트의 시각은 손으로 적은 값이라 업로드 영상과 아무 관계가 없다. 그대로
# 쓰면 30초짜리 클립에 36초 타임라인이 붙어 시크가 끝으로 튀고 그래프 x축이 영상
# 밖까지 뻗는다. 그래서 기준 폭(:data:`DEMO_BASE_SPAN_SEC`)으로 적어 두고, 화면에
# 낼 때 재생 길이에 비례해 늘리거나 줄인다(요구사항 2.5).

#: 데모 시각을 손으로 적을 때 기준으로 쓴 시간 폭(초). 스케일의 분모다.
DEMO_BASE_SPAN_SEC = 36

#: 재생 길이를 모를 때 쓰는 보수적인 상한(초).
#: 넉넉히 잡으면 짧은 클립에서 시각이 영상 밖을 가리키므로 짧게 잡는다.
#: 반대로 실제 영상이 이보다 길면 데모 이벤트가 앞부분에 몰릴 뿐 어긋나지 않는다.
DEMO_FALLBACK_DURATION_SEC = 20


def demo_curve(final):
    """데모 이벤트용 곡선. 실분석 경로는 :func:`build_curve`를 쓴다.

    좌표는 손으로 적은 예시 값이고 마지막 값만 최종 점수에 맞춘다.
    x는 :data:`DEMO_BASE_SPAN_SEC` 기준이며 스케일은 :func:`demo_events`가 한다.
    """
    return ((0, 5), (3, 22), (8, 30), (11, 48), (16, 52), (20, 58),
            (24, 70), (28, 74), (31, final), (36, final))


def demo_curve_companion(final):
    """동행으로 판정되는 쌍의 곡선. 오르다가 감점이 쌓이며 최종 점수로 내려온다.

    :func:`demo_curve`와 마찬가지로 마지막 값이 최종 점수다. 중간에 40점 선을
    넘었다가 내려오는 모양이라 '감점 때문에 걸러졌다'가 그래프에서 읽힌다.
    """
    fin = int(clamp(int(final), 0, 100))
    peak = max(fin, 52)
    mid = (peak + fin) // 2
    return ((0, 4), (5, 28), (10, 40), (14, peak), (20, mid), (26, fin), (32, fin))


def _scale_demo_sec(sec, limit) -> int:
    """데모 기준 시각을 ``[0, limit]`` 안으로 비례 변환한다."""
    if DEMO_BASE_SPAN_SEC <= 0:
        return 0
    return int(clamp(round(int(sec) * limit / DEMO_BASE_SPAN_SEC), 0, limit))


def demo_limit(duration_sec) -> int:
    """데모 시각의 상한(초). 재생 길이를 모르거나 0 이하면 보수적인 기본값을 쓴다."""
    return _time_limit(duration_sec) or DEMO_FALLBACK_DURATION_SEC


def _demo_event(leader, follower, rows, deduction, timeline, curve_of, limit) -> Event:
    """데모 이벤트 하나를 만들고 시각을 재생 길이에 맞춘다.

    곡선은 최종 점수가 정해진 뒤에 ``curve_of(score)``로 만든다. 곡선 끝 값을
    손으로 적어 두면 점수 행을 손볼 때 그래프와 카드가 조용히 어긋난다.

    스케일한 값은 :func:`bound_timeline` / :func:`bound_curve`에 그대로 넘긴다.
    상한 클램프뿐 아니라, 짧은 영상에서 여러 좌표가 같은 초로 눌렸을 때 x를
    엄격히 증가하도록 정리하는 일까지 그쪽이 이미 한다.
    """
    base = Event(leader=leader, follower=follower, rows=rows, deduction=deduction,
                 timeline=timeline, curve=())
    scaled_timeline = tuple(
        replace(i, at_sec=_scale_demo_sec(i.at_sec, limit)) for i in timeline)
    scaled_curve = tuple(
        (_scale_demo_sec(x, limit), int(y)) for x, y in curve_of(base.score))
    return replace(
        base,
        timeline=bound_timeline(scaled_timeline, limit),
        curve=bound_curve(scaled_curve, limit),
    )


def demo_events(duration_sec=None) -> list[Event]:
    """화면 확인용 예시 이벤트 두 개(요구사항 2.5, 2.6).

    타임라인 시각과 곡선 x 좌표는 모두 ``[0, duration_sec]`` 안에 들어가고,
    스케일이 단조 증가라 서로의 앞뒤 순서도 그대로 유지된다. 재생 길이를 모르면
    :data:`DEMO_FALLBACK_DURATION_SEC`를 상한으로 쓴다.

    타임라인과 곡선은 출처가 다르다 — 타임라인은 손으로 적은 문구, 곡선은 최종
    점수에서 만든 모양이다. 데모에서는 이 둘이 같은 계산에서 나오지 않아도 된다
    (요구사항 2.6). 실분석 경로는 :func:`compute_events`가 둘 다 좌표에서 만든다.
    """
    limit = demo_limit(duration_sec)

    e1 = _demo_event(
        leader=3, follower=7,
        rows=(ScoreRow("후방 근접 유지", "87%", 15, 15),
              ScoreRow("지속 시간", "18초", 12, 15),
              ScoreRow("방향 유사도", "0.91", 10, 10),
              ScoreRow("지연 경로 유사도", "76%", 20, 20),
              ScoreRow("정지 후 추종", "2회", 20, 20),
              ScoreRow("방향 전환 추종", "1회", 12, 20)),
        deduction=5,
        timeline=(TimelineItem(3, "id7이 id3 후방 3m 이내 진입", "근접"),
                  TimelineItem(11, "id3 정지 → 1.8초 뒤 id7 정지", "정지 추종"),
                  TimelineItem(24, "id3 정지 → 1.5초 뒤 id7 정지", "정지 추종"),
                  TimelineItem(31, "id3 좌회전 → 2.0초 뒤 id7 좌회전", "방향 추종")),
        curve_of=demo_curve,
        limit=limit,
    )

    # e2는 '동행 감점이 어떻게 작동하는지'를 화면에서 보여 주려고 두는 쌍이다.
    # 가점 66점에 감점 35점이라 최종 31점이고, 실분석 경로는 MIN_EVENT_SCORE(40)
    # 미만을 버리므로 이 이벤트에 대응하는 실분석 결과는 존재하지 않는다.
    # 곡선 끝 값을 최종 점수에서 만들어 표·카드와 어긋나지 않게 한다.
    e2 = _demo_event(
        leader=3, follower=12,
        rows=(ScoreRow("후방 근접 유지", "72%", 9, 15),
              ScoreRow("지속 시간", "16초", 12, 15),
              ScoreRow("방향 유사도", "0.88", 8, 10),
              ScoreRow("지연 경로 유사도", "40%", 5, 20),
              ScoreRow("정지 후 추종", "2회", 20, 20),
              ScoreRow("방향 전환 추종", "1회", 12, 20)),
        deduction=35,
        timeline=(TimelineItem(5, "id12 나란히 이동 시작", "동행"),
                  TimelineItem(14, "id3·id12 거의 동시 정지 (0.3초)", "동행 신호")),
        curve_of=demo_curve_companion,
        limit=limit,
    )
    return [e1, e2]


# ============================================================
#  UI 조각
# ============================================================

def chip(label, style, extra="") -> str:
    """등급·모드 표시용 칩 HTML.

    색은 :class:`RiskStyle`에서만 가져온다. 칩을 쓰는 자리마다 배경·글자색을 손으로
    적으면 팔레트 밖 색이 새어 들어오기 쉽다(요구사항 13.6).
    """
    return (f"<span class='zk-chip' style='background:{style.soft};"
            f"color:{style.deep};border:1px solid {style.base};{extra}'>{label}</span>")


def hero(title, sub, badge=None):
    """모든 화면 상단의 히어로 영역(요구사항 13.3).

    ``badge``에 :class:`RiskStyle`을 주면 제목 옆에 그 라벨을 칩으로 붙인다.
    분석_모드 표시(요구사항 2.1)가 이 자리를 쓴다 — 사이드바를 접어 둔 상태에서도
    지금 어떤 모드로 보고 있는지 결과 화면에서 바로 읽히게 하려는 목적이다.
    색은 넘겨받은 스타일에서만 나오므로 히어로 자체는 색을 정하지 않는다.
    """
    mark = "" if badge is None else chip(
        badge.label, badge, extra="margin-left:10px;vertical-align:middle")
    st.markdown(f"<div class='zk-hero'><h1>{title}{mark}</h1><p>{sub}</p></div>",
                unsafe_allow_html=True)


def metric_row(items):
    """요약 지표 카드 한 줄(요구사항 6.7, 10.5).

    각 항목은 ``(라벨, 값, 색)``이다. 색 자리에는 팔레트 HEX 문자열도, 그 값을
    들고 있는 :class:`RiskStyle`도 넣을 수 있다. 호출부가 ``style.base``를 매번
    풀어 쓰지 않아도 되고, 등급 색을 쓰는 지표(최고 위험도 등)는 스타일을 그대로
    넘기면 된다.

    항목이 없으면 아무 것도 그리지 않는다. ``st.columns(0)``은 예외를 낸다.
    """
    if not items:
        return
    cols = st.columns(len(items))
    for c, (label, value, color) in zip(cols, items):
        base = color.base if isinstance(color, RiskStyle) else color
        c.markdown(
            f"<div class='zk-metric' style='border-color:{base}55'>"
            f"<div class='v' style='color:{base}'>{value}</div>"
            f"<div class='l'>{label}</div></div>", unsafe_allow_html=True)


def evidence_table(event) -> str:
    """판정 근거 표 HTML을 만든다(요구사항 12.1, 12.2).

    열은 항목명 · 측정값 · 점수 세 개이며, 정렬(항목 좌측, 측정값 좌측, 점수 우측)과
    합계 행 모양은 ``theme_css()``의 ``zk-table`` 클래스가 정한다. 여기서는 클래스
    이름만 붙이고 인라인 스타일에는 색만 남긴다 — 정렬을 문자열 조립에 섞으면 열마다
    값이 갈려 표가 어긋난 것처럼 보인다.

    마지막 두 행은 동행 감점과 최종 위험도다. 점수 값은 전부 ``event.rows``와
    ``event.deduction``에서 나오므로 :func:`score_cards`가 보여 주는 값과 같은
    원본을 쓴다(요구사항 12.3).

    화면에 직접 쓰지 않고 문자열을 돌려주는 이유는 이 조각이 어느 열(``st.columns``)에
    들어갈지, 카드로 감쌀지를 호출부가 정하기 때문이다.
    """
    style = risk_level(event.score)
    body = "".join(
        f"<tr><td class='item'>{r.name}</td>"
        f"<td class='measured'>{r.measured}</td>"
        f"<td class='num' style='color:{score_color(r.score, r.max_score).deep}'>"
        f"+{r.score}</td></tr>"
        for r in event.rows
    )
    # 감점은 등급이 아니라 '깎인 값'이라 일반 이동 등급 색(민트)을 쓴다.
    body += (f"<tr><td class='item'>동행 감점</td>"
             f"<td class='measured'>나란히·동시행동</td>"
             f"<td class='num' style='color:{RISK_LOW.deep}'>−{event.deduction}</td></tr>")
    body += (f"<tr class='total'><td class='item'>최종 위험도</td>"
             f"<td class='measured'></td>"
             f"<td class='num' style='color:{style.deep}'>{event.score}점</td></tr>")
    return (f"<div class='zk-card'><table class='zk-table'>"
            f"<tr><th class='item'>항목</th><th class='measured'>측정값</th>"
            f"<th class='num'>점수</th></tr>{body}</table></div>")


def score_cards(rows) -> str:
    """항목별 점수 카드와 색 범례 HTML을 만든다(요구사항 12.4, 12.5, 12.6).

    카드 색은 ``ScoreRow.max_score``와 :func:`score_color`가 정한다 — 항목명으로
    최대 점수를 되짚지 않으므로 이름이 바뀌어도 색 결정이 깨지지 않는다. 점수 값은
    :func:`evidence_table`과 같은 ``event.rows``에서 나온다(요구사항 12.3).

    격자·테두리 굵기는 ``zk-score-grid``/``zk-score-box`` 클래스가 정하고, 여기서는
    색만 인라인으로 넣는다. 범례는 카드와 같은 :class:`RiskStyle` 목록
    (:data:`ITEM_LEGEND`)에서 :func:`chip`으로 만들어, 카드에 나올 수 있는 색과
    범례에 적힌 색이 갈라지지 않게 한다.
    """
    cards = ""
    for r in rows:
        cs = score_color(r.score, r.max_score)
        cards += (f"<div class='zk-score-box' style='background:{cs.soft};"
                  f"border-color:{cs.base};color:{cs.deep}'>"
                  f"<div class='n'>{r.name}</div>"
                  f"<div class='m'>{r.score}<span class='x'>/{r.max_score}</span></div>"
                  f"<div class='s'>{r.measured}</div></div>")
    legend = " ".join(chip(s.label, s) for s in ITEM_LEGEND)
    return (f"<div class='zk-score-grid'>{cards}</div>"
            f"<div class='zk-score-legend'>{legend}</div>")


class ProgressSlots:
    """진행률 막대와 캡션을 서로 다른 자리에 그리는 어댑터.

    :class:`ProgressReporter`는 ``progress(value)``와 ``caption(text)``를 같은
    객체에 부른다. Streamlit 요소 핸들에 ``caption()``을 부르면 그 자리의 내용이
    캡션으로 **교체**되므로, 막대 핸들을 그대로 넘기면 캡션이 나가는 순간 막대가
    사라진다. 캡션이 나가는 경로가 바로 총 프레임 수를 모를 때(요구사항 3.6)와
    실제 비율이 1을 넘을 때(요구사항 3.3)라, 정작 진행 상황을 알고 싶은 순간에
    막대가 없어지는 모양이 된다.

    그래서 막대와 캡션에 각자의 자리를 준다. 리포터는 덕 타이핑으로 이 객체를
    받으므로 리포터 쪽 코드는 손대지 않는다.
    """

    def __init__(self, bar, caption_slot):
        self.bar = bar
        self.caption_slot = caption_slot

    def progress(self, value) -> None:
        """막대 자리에 진행률을 그린다."""
        self._call(self.bar, "progress", value)

    def caption(self, text) -> None:
        """막대와 별개인 캡션 자리에 문구를 그린다."""
        self._call(self.caption_slot, "caption", text)

    @staticmethod
    def _call(target, name, arg) -> None:
        """대상에 그 메서드가 있으면 부르고 없으면 조용히 넘어간다."""
        fn = getattr(target, name, None)
        if callable(fn):
            fn(arg)


#: 그래프에 그리는 기준선(요구사항 11.2). 등급 경계와 같은 값이라 색도 그 등급
#: 스타일에서 가져온다 — 여기에 색을 직접 적으면 :func:`risk_level`과 조용히 어긋난다.
RISK_GUIDES: tuple[tuple[int, str, RiskStyle], ...] = (
    (70, "우선확인 70", RISK_HIGH),
    (40, "확인필요 40", RISK_MID),
)

#: 꺾은선을 그리는 데 필요한 최소 좌표점 수. 그 미만은 안내 문구로 대체한다(요구사항 11.5).
CHART_MIN_POINTS = 2

#: 좌표점이 부족할 때 그래프 위에 겹쳐 쓰는 문구(요구사항 11.5).
CHART_EMPTY_NOTE = "데이터 부족 · 좌표점이 2개 미만입니다"


def chart_span(curve, duration_sec) -> int:
    """그래프 x축의 오른쪽 끝(초)을 정한다.

    재생 길이를 우선 쓴다(요구사항 11.4 눈금이 이벤트마다 흔들리지 않게). 재생
    길이를 모르거나(``None``) 0 이하면 곡선의 최대 x로 내려오고, 그것도 없으면
    1을 쓴다. 0을 그대로 쓰면 좌표 변환에서 0으로 나누게 된다.
    """
    for candidate in (duration_sec, max((p[0] for p in curve), default=0)):
        try:
            value = int(candidate)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 1


def risk_chart_svg(curve, final_score, duration_sec) -> str:
    """시간축 x, 위험도 y 꺾은선 그래프를 SVG 문자열로 만든다.

    x축 범위는 곡선의 최대 x가 아니라 ``duration_sec``으로 고정한다(요구사항 10.3).
    선 색은 ``final_score``의 위험도_등급 색이고(요구사항 11.3), 40점·70점 기준선을
    함께 그린다(요구사항 11.2). x축 눈금 라벨은 ``분:초``다(요구사항 11.4).

    모든 좌표는 그래프 영역 안으로 잘라 넣는다(요구사항 11.1). 재생 길이보다 뒤에
    있는 좌표점이나 0~100을 벗어난 위험도가 들어와도 축 밖으로 삐져나가지 않는다.

    좌표점이 :data:`CHART_MIN_POINTS` 미만이면 축과 기준선을 그린 뒤 그 위에 안내
    문구를 겹쳐 그린다(요구사항 11.5).
    """
    W, H = 640, 190
    pad_l, pad_r, pad_t, pad_b = 38, 14, 12, 26
    x_left, x_right = pad_l, W - pad_r
    y_top, y_bottom = pad_t, H - pad_b
    xmax = chart_span(curve, duration_sec)

    def px(sec) -> float:
        """초 -> x 좌표. 그래프 영역 밖은 경계로 붙인다."""
        return clamp(x_left + (float(sec) / xmax) * (x_right - x_left),
                     x_left, x_right)

    def py(score) -> float:
        """위험도 -> y 좌표. 0~100 밖은 경계로 붙인다."""
        return clamp(y_top + (1 - float(score) / 100) * (y_bottom - y_top),
                     y_top, y_bottom)

    style = risk_level(final_score)
    points = [(px(x), py(y)) for x, y in curve]

    # ---- 축 ----
    axes = (f"<line x1='{x_left}' y1='{y_top}' x2='{x_left}' y2='{y_bottom}' "
            f"stroke='{LINE}' stroke-width='1'/>"
            f"<line x1='{x_left}' y1='{y_bottom}' x2='{x_right}' y2='{y_bottom}' "
            f"stroke='{LINE}' stroke-width='1'/>")

    # ---- 기준선(요구사항 11.2) ----
    guides = ""
    for value, label, gs in RISK_GUIDES:
        gy = py(value)
        guides += (f"<line x1='{x_left}' y1='{gy:.1f}' x2='{x_right}' y2='{gy:.1f}' "
                   f"stroke='{gs.base}' stroke-width='1' stroke-dasharray='4 4' "
                   f"opacity='.7'/>"
                   f"<text x='{x_right - 2}' y='{gy - 4:.1f}' font-size='10' "
                   f"fill='{gs.base}' text-anchor='end'>{label}</text>")

    # ---- 눈금(요구사항 11.4) ----
    ticks = ""
    for sec in range(0, xmax + 1, max(1, xmax // 6)):
        ticks += (f"<text x='{px(sec):.1f}' y='{H - 8}' font-size='10' "
                  f"fill='{INK_SOFT}' text-anchor='middle'>{fmt_mmss(sec)}</text>")
    for value in (0, 50, 100):
        ticks += (f"<text x='{x_left - 6}' y='{py(value) + 3:.1f}' font-size='10' "
                  f"fill='{INK_SOFT}' text-anchor='end'>{value}</text>")

    # ---- 데이터 ----
    dots = "".join(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3' fill='{style.base}'/>"
                   for x, y in points)
    if len(points) >= CHART_MIN_POINTS:
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        area = (f"{points[0][0]:.1f},{y_bottom:.1f} {pts} "
                f"{points[-1][0]:.1f},{y_bottom:.1f}")
        body = (f"<polygon points='{area}' fill='{style.soft}' opacity='.75'/>"
                f"<polyline points='{pts}' fill='none' stroke='{style.base}' "
                f"stroke-width='2.5' stroke-linejoin='round' stroke-linecap='round'/>"
                + dots)
    else:
        # 축·기준선은 그대로 두고 문구만 겹친다(요구사항 11.5).
        box_w, box_h = 260, 30
        bx = (x_left + x_right) / 2 - box_w / 2
        by = (y_top + y_bottom) / 2 - box_h / 2
        body = (dots
                + f"<rect x='{bx:.1f}' y='{by:.1f}' width='{box_w}' height='{box_h}' "
                  f"rx='9' fill='{BG_CARD}' stroke='{LINE}'/>"
                  f"<text x='{(x_left + x_right) / 2:.1f}' y='{by + 20:.1f}' "
                  f"font-size='12' fill='{INK_SOFT}' text-anchor='middle'>"
                  f"{CHART_EMPTY_NOTE}</text>")

    return f"""<svg viewBox="0 0 {W} {H}" width="100%" style="display:block">
<rect x="0" y="0" width="{W}" height="{H}" rx="12" fill="{BG_CARD}" stroke="{LINE}"/>
{axes}{guides}{body}{ticks}
</svg>"""


def risk_ribbon_svg(curve, duration_sec) -> str:
    """시간축을 등급 색으로 칠한 리스크 리본을 SVG 문자열로 만든다.

    판정 요약 블록의 헤드라인 아래에 두는 압축 표현이다(요구사항 10.11). 꺾은선은
    값을 읽어야 하지만 리본은 "언제 붉어졌는지"만 보여 주므로 목록을 훑기 전에
    판단할 수 있다.

    가로 좌표 변환은 :func:`risk_chart_svg`와 같은 :func:`chart_span`을 쓴다 — 두
    표현의 시간축이 어긋나면 같은 곡선인데 다른 시각처럼 읽힌다(요구사항 10.12).
    ``duration_sec``을 모르거나 0 이하면 ``chart_span``이 곡선의 마지막 x 좌표로
    내려온다(요구사항 10.13).

    색은 인접한 두 좌표점 사이를 사각형 하나로 칠하고, 그 구간의 색은 두 끝점 중
    **높은** 위험도를 :func:`risk_level`에 넣어 얻는다. 올라가는 구간을 낮은 쪽
    색으로 칠하면 위험이 올라간 시점이 한 칸 늦게 보인다. 색은 등급 스타일에서만
    나오므로 팔레트 밖 색이 섞이지 않는다(요구사항 13.6).

    좌표점이 :data:`CHART_MIN_POINTS` 미만이면 칠할 구간을 만들 수 없다. 그때는
    등급 색을 쓰지 않은 빈 띠에 :data:`CHART_EMPTY_NOTE`를 겹쳐 그린다
    (요구사항 10.14) — 그래프의 같은 상황 처리(요구사항 11.5)와 표현을 맞춘다.
    """
    W, H = 640, 46
    band_t, band_b = 1.0, 27.0
    x_left, x_right = 1.0, W - 1.0
    xmax = chart_span(curve, duration_sec)

    def px(sec) -> float:
        """초 -> x 좌표. 띠 밖은 경계로 붙인다."""
        return clamp(x_left + (float(sec) / xmax) * (x_right - x_left),
                     x_left, x_right)

    points = tuple(curve)
    fills = ""
    if len(points) >= CHART_MIN_POINTS:
        for (x0, r0), (x1, r1) in zip(points, points[1:]):
            a, b = px(x0), px(x1)
            if b <= a:
                # 같은 초에 놓인 두 점(또는 축 밖으로 잘린 점) — 폭이 0이라 건너뛴다.
                continue
            base = risk_level(max(r0, r1)).base
            fills += (f"<rect x='{a:.1f}' y='{band_t:.1f}' width='{b - a:.1f}' "
                      f"height='{band_b - band_t:.1f}' fill='{base}'/>")
        note = ""
    else:
        # 등급 색은 한 칸도 쓰지 않는다. 빈 띠 위에 문구만 얹는다(요구사항 10.14).
        note = (f"<text x='{W / 2:.1f}' y='{(band_t + band_b) / 2 + 4:.1f}' "
                f"font-size='11.5' fill='{INK_SOFT}' text-anchor='middle'>"
                f"{CHART_EMPTY_NOTE}</text>")

    # 시간축 양 끝 라벨. 리본이 재생 길이 전체를 덮는다는 걸 숫자로 확인시킨다.
    labels = (f"<text x='{x_left:.1f}' y='{H - 4}' font-size='10' "
              f"fill='{INK_SOFT}' text-anchor='start'>{fmt_mmss(0)}</text>"
              f"<text x='{x_right:.1f}' y='{H - 4}' font-size='10' "
              f"fill='{INK_SOFT}' text-anchor='end'>{fmt_mmss(xmax)}</text>")

    return f"""<svg viewBox="0 0 {W} {H}" width="100%" style="display:block">
<defs><clipPath id="zk-ribbon-clip"><rect x="{x_left}" y="{band_t}" \
width="{x_right - x_left}" height="{band_b - band_t}" rx="7"/></clipPath></defs>
<rect x="{x_left}" y="{band_t}" width="{x_right - x_left}" \
height="{band_b - band_t}" rx="7" fill="{BG_CARD}"/>
<g clip-path="url(#zk-ribbon-clip)">{fills}</g>
<rect x="{x_left}" y="{band_t}" width="{x_right - x_left}" \
height="{band_b - band_t}" rx="7" fill="none" stroke="{LINE}"/>
{note}{labels}
</svg>"""


# ------------------------------------------------------------
#  재생 위치 이동
# ------------------------------------------------------------

def _passthrough_fragment(func=None, **_kwargs):
    """``st.fragment``가 없는 버전에서 데코레이터 자리만 채운다.

    ``@fragment``와 ``@fragment(run_every=...)`` 두 표기를 모두 받아 주고, 어느
    쪽이든 원래 함수를 그대로 돌려준다. 프래그먼트는 "이 블록만 다시 그린다"는
    최적화이므로 없어도 화면은 성립한다 — 전체 rerun으로 그려질 뿐이다.
    """
    if func is None:
        return lambda f: f
    return func


#: 프래그먼트 데코레이터. 정의 시점에 한 번 고른다.
#:
#: ``st.fragment``는 Streamlit 1.33 이후 이름이고 그 앞에는 실험 이름으로 있었거나
#: 아예 없다. 호출 시점마다 고르면 데코레이터가 매번 새로 감싸 프래그먼트 신원이
#: 흔들리고, 무엇보다 여기서 결정을 끝내 두면 Streamlit 런타임 없이 이 모듈을
#: 임포트하는 경로(테스트)에서도 이름이 그대로 성립한다.
fragment = (getattr(st, "fragment", None)
            or getattr(st, "experimental_fragment", None)
            or _passthrough_fragment)

#: 재생 영상 파일을 찾을 수 없을 때의 안내(요구사항 4.7). 이 경우에도 호출부는
#: 나머지 결과 블록을 그대로 그린다 — 영상만 없고 분석 결과는 남아 있다.
VIDEO_MISSING_NOTE = ("재생할 영상 파일을 찾을 수 없습니다. "
                      "나머지 분석 결과는 그대로 확인할 수 있습니다.")

#: 결과 영상 박스 색 범례(요구사항 1.7). 색은 등급 스타일에서만 가져온다 —
#: 여기서 색을 직접 적으면 실제 박스 색(:func:`hex_to_bgr`)과 조용히 어긋난다.
BOX_LEGEND: tuple[tuple[str, RiskStyle], ...] = (
    ("빨강 · 미행 의심", RISK_HIGH),
    ("초록 · 일반 보행자", RISK_LOW),
)


def box_legend() -> str:
    """영상 아래에 붙이는 박스 색 범례 HTML(요구사항 1.7)."""
    chips = " ".join(chip(label, style) for label, style in BOX_LEGEND)
    return (f"<div style='font-size:12px;color:{INK_SOFT};margin-top:6px'>"
            f"{chips}</div>")


def video_playable(path) -> bool:
    """재생 대상 파일이 실제로 있는지 본다(요구사항 4.7).

    기록은 파일이 사라져도 목록에 남으므로(요구사항 4.6) 그릴 때마다 확인한다.
    """
    if not path:
        return False
    try:
        return Path(path).exists()
    except (OSError, TypeError, ValueError):
        return False


def seek_limit(rec) -> int | None:
    """시크 상한(초)을 정한다.

    기록의 재생 길이를 우선 쓴다. ``probe()``가 총 프레임 수를 못 읽으면 재생 길이가
    ``None``이 되는데, 그때 그대로 :func:`clamp_seek`에 넘기면 상한을 몰라 시작 위치가
    항상 0이 되어 타임라인 이동이 통째로 죽는다. 그래서 이벤트가 가리키는 가장 늦은
    시각(타임라인·곡선)을 상한으로 대신 쓴다. 두 값 모두 분석이 실제로 관측한 범위
    안이라 요구사항 8.5의 "재생 길이 이내"를 벗어나지 않는다.
    """
    if rec is None:
        return None
    if rec.duration_sec:
        return int(rec.duration_sec)
    latest = 0
    for e in rec.events:
        latest = max([latest]
                     + [int(i.at_sec) for i in e.timeline]
                     + [int(x) for x, _ in e.curve])
    return latest or None


def current_seek(duration_sec) -> int:
    """세션에 담긴 시작 위치를 재생 길이 안으로 잘라 돌려준다(요구사항 8.5)."""
    return clamp_seek(st.session_state.get("seek", 0), duration_sec)


def _bump_seek_nonce() -> int:
    """플레이어 재생성 카운터를 1 올린다.

    같은 자리에 같은 내용이 다시 오면 브라우저가 기존 미디어 요소를 그대로 두어
    새 ``start_time``을 무시하는 경우가 있다. 이 값이 플레이어 마크업에 섞여
    들어가 렌더 대상이 매번 달라지게 만든다.
    """
    value = int(st.session_state.get("seek_nonce", 0)) + 1
    st.session_state.seek_nonce = value
    return value


def set_seek(sec, duration_sec) -> int:
    """시작 위치를 ``sec``으로 옮긴다(요구사항 8.2, 8.5). 적용된 초를 돌려준다."""
    value = clamp_seek(sec, duration_sec)
    st.session_state.seek = value
    _bump_seek_nonce()
    return value


def reset_seek() -> None:
    """시작 위치를 0으로 되돌린다(요구사항 8.6).

    이벤트를 바꿀 때는 현재 재생 위치와 무관하게 처음부터 다시 본다.
    """
    st.session_state.seek = 0
    _bump_seek_nonce()


def video_player(path, seek, nonce) -> None:
    """재생 시작 위치가 적용된 플레이어를 그린다(요구사항 1.3, 8.4).

    플레이스홀더를 하나 두고 렌더 직전에 비운다. rerun에서 같은 자리의 미디어
    요소가 재사용되면 브라우저가 새 ``start_time``을 반영하지 않는 일이 있어,
    자리를 비워 DOM 노드가 새로 생기도록 밀어 준다. ``nonce``는 그 위에 얹는
    표시로, 값이 바뀌면 플레이어 앞의 마크업도 함께 바뀌어 같은 시각을 두 번
    눌렀을 때와 다른 시각을 눌렀을 때가 구분된다.

    그래도 브라우저에 따라 시크가 즉시 반영되지 않을 수 있으므로, 아래에 현재
    시작 위치를 ``분:초``로 적어 사용자가 상태를 확인할 수 있게 한다.
    """
    start = int(seek)
    slot = st.empty()
    slot.empty()
    with slot.container():
        st.markdown(f"<div class='zk-seek-mark' data-nonce='{int(nonce)}' "
                    f"data-seek='{start}'></div>", unsafe_allow_html=True)
        st.video(path, start_time=start)
        st.caption(f"재생 시작 위치 · {fmt_mmss(start)}")


@fragment
def player_block(rec, seek) -> None:
    """분석 영상 블록(요구사항 1.3, 1.7, 4.7).

    프래그먼트로 감싸 시크가 이 블록만 다시 그리게 한다. 지원하지 않는 버전에서는
    :func:`_passthrough_fragment`가 데코레이터를 통과시키므로 전체 rerun으로 그려진다.

    재생 영상 파일이 없으면 안내만 남긴다. 여기서 예외를 올리거나 호출부가 일찍
    돌아가면 그래프·지표·근거까지 함께 사라지므로, 이 함수는 항상 정상 종료한다
    (요구사항 4.7).
    """
    st.markdown("<div class='zk-label'>분석 영상 · YOLO 추적 결과</div>",
                unsafe_allow_html=True)
    if not video_playable(getattr(rec, "video_path", None)):
        st.warning(VIDEO_MISSING_NOTE)
        return
    video_player(rec.video_path, seek, st.session_state.get("seek_nonce", 0))
    st.markdown(box_legend(), unsafe_allow_html=True)


# ============================================================
#  화면 1 : 영상 분석
# ============================================================

#: 결과_화면 블록 순서(요구사항 10.1).
#:
#: 순서를 코드 흐름에 묻어 두지 않고 상수로 꺼낸다. 블록을 옮길 때 고칠 자리가
#: 한 곳으로 모이고, 규정된 순서와 같은지도 이 튜플 비교로 확인된다.
#:
#: 판정 요약(``verdict``)이 맨 앞이다. 점수와 등급이 구간 목록 안의 작은 칩으로만
#: 보이면 담당자가 판정을 알기까지 목록을 훑어야 한다.
BLOCKS: tuple[str, ...] = ("verdict", "player", "chart", "metrics", "evidence",
                           "actions")

#: 데모 기록임을 알리는 표시(요구사항 2.4).
#:
#: 모드 칩은 히어로에도 있지만 그건 '지금 고른 모드'다. 기록은 저장 시점의 모드를
#: 들고 있으므로, 실분석으로 바꾼 뒤 과거 데모 기록을 열면 두 값이 갈린다. 화면에
#: 보이는 숫자의 출처는 기록 쪽 모드라 안내도 기록에서 나온다.
DEMO_DATA_NOTE = ("데모 데이터입니다. 화면 확인용 예시 값이며 업로드한 영상을 "
                  "실제로 분석한 결과가 아닙니다.")

#: 결과 영상 합성(또는 보관)이 실패한 기록의 안내(요구사항 1.5).
RENDER_FAILED_NOTE = ("결과 영상을 만들지 못해 원본 영상을 재생합니다. "
                      "박스가 합성된 결과 영상은 사용할 수 없습니다.")

#: 활성 분석_기록이 없을 때의 안내(요구사항 9.7).
NO_ACTIVE_NOTE = ("표시할 분석 결과가 없습니다. 영상을 올려 분석을 시작하거나 "
                  "'과거 분석 기록'에서 기록을 열어 주세요.")

#: 해석하지 못한 기록이 있을 때의 안내(요구사항 4.5).
LOAD_FAILED_NOTE = "불러오지 못한 기록 {n}건을 건너뛰었습니다."

#: 이벤트가 하나도 없는 기록의 안내.
NO_EVENT_NOTE = "확인이 필요한 구간이 감지되지 않았습니다."


@dataclass(frozen=True)
class ResultView:
    """결과 블록들이 함께 보는 값 묶음.

    블록마다 세션 상태를 다시 읽으면 한 번의 렌더 안에서 블록끼리 다른 값을 볼 수
    있다(선택 이벤트를 바꾼 직후가 그렇다). 페이지가 한 번 정해 준 값을 그대로
    돌려 쓰도록 묶어서 넘긴다.

    기록 객체를 여기 담긴 것과 별개로 세션에 남기지는 않는다. 이 값은 한 번의
    렌더 동안만 살고, 다음 렌더는 ``active_id``로 저장소에서 다시 읽는다.
    """

    rec: AnalysisRecord
    sel: int                  # 선택된 이벤트 인덱스(범위 안으로 잘린 값)
    limit: int | None         # 시크 상한(초)
    seek: int                 # 지금 적용된 재생 시작 위치(초)

    @property
    def events(self) -> tuple[Event, ...]:
        return self.rec.events

    @property
    def event(self) -> Event:
        """선택된 이벤트."""
        return self.rec.events[self.sel]

    @property
    def style(self) -> RiskStyle:
        """선택된 이벤트의 위험도_등급 스타일."""
        return risk_level(self.event.score)


def active_record() -> tuple[AnalysisRecord | None, int]:
    """활성 분석_기록과 해석 실패 건수 — ``(기록 또는 None, 실패 건수)``.

    화면은 ``st.session_state.active_id``만 들고 있고 기록 객체는 그릴 때마다
    저장소에서 다시 읽는다. 그래야 다른 화면에서 지운 기록을 계속 붙들지 않고,
    지워진 뒤에는 곧바로 안내 문구 경로로 내려온다(요구사항 9.7).

    실패 건수는 :func:`load_records`가 알려 준다 — 활성 기록만 찾아서는 옆에서
    몇 건이 버려졌는지 알 수 없다(요구사항 4.5).
    """
    failed = load_records()[1]
    aid = st.session_state.get("active_id")
    rec = find_record(aid) if aid else None
    return rec, failed


def result_notices(rec) -> None:
    """결과_화면 상단 안내. 기록의 상태만 보고 결정한다.

    - 데모 기록이면 데모 데이터임을 알린다(요구사항 2.4).
    - 결과 영상 합성이나 보관이 실패했으면 결과 영상을 쓸 수 없다고 알린다
      (요구사항 1.5). 재생 자체는 원본 영상으로 계속된다.
    """
    if getattr(rec, "mode", None) == "demo":
        st.info(DEMO_DATA_NOTE)
    if getattr(rec, "video_render_failed", False):
        st.warning(RENDER_FAILED_NOTE)


def record_line(rec) -> None:
    """어느 파일의 언제 분석인지 한 줄로 적는다."""
    st.markdown(f"<div style='color:{INK_SOFT};font-size:13px;margin:-6px 0 14px'>"
                f"분석 파일 · <b style='color:{INK}'>{rec.name}</b> · "
                f"{fmt_created_at(rec.created_at)}</div>",
                unsafe_allow_html=True)


def build_view(rec) -> ResultView:
    """기록과 세션 상태에서 :class:`ResultView`를 만든다.

    선택 인덱스는 이벤트 수 안으로 자른다. 기록을 바꿔 열면 이전 화면의 인덱스가
    남아 범위를 넘길 수 있다. 시작 위치도 :func:`current_seek`가 재생 길이 안으로
    잘라 넣는다(요구사항 8.5).
    """
    try:
        sel = int(st.session_state.get("sel", 0) or 0)
    except (TypeError, ValueError):
        sel = 0
    sel = int(clamp(sel, 0, max(len(rec.events) - 1, 0)))
    limit = seek_limit(rec)
    return ResultView(rec=rec, sel=sel, limit=limit, seek=current_seek(limit))


# ------------------------------------------------------------
#  결과 블록
# ------------------------------------------------------------

def event_picker(view) -> None:
    """확인 필요 구간 목록. 다른 구간을 고르면 시작 위치를 0으로 되돌린다(요구사항 8.6)."""
    st.markdown("<div class='zk-label'>확인 필요 구간</div>", unsafe_allow_html=True)
    events = view.events
    for i, e in enumerate(events):
        es = risk_level(e.score)
        picked = (i == view.sel)
        st.markdown(
            f"<div class='zk-row' style='background:{es.soft};"
            f"border:{'2px' if picked else '1px'} solid {es.base if picked else LINE}'>"
            f"<span style='color:{es.deep};font-size:13.5px;font-weight:600'>"
            f"id{e.leader} ← id{e.follower}</span>"
            f"<span style='color:{es.deep};font-weight:700'>{es.label} {e.score}</span></div>",
            unsafe_allow_html=True)
        if len(events) > 1 and not picked:
            if st.button("이 구간 보기", key=f"pick{i}"):
                st.session_state.sel = i
                reset_seek()
                st.rerun()


def timeline_list(view) -> None:
    """선택된 이벤트의 타임라인. 시각을 누르면 그 장면으로 이동한다(요구사항 8.1, 8.2)."""
    st.markdown("<div class='zk-label' style='margin-top:14px'>이벤트 타임라인</div>",
                unsafe_allow_html=True)
    st.caption("시간을 누르면 해당 장면으로 이동합니다")
    style = view.style
    for k, item in enumerate(view.event.timeline):
        tc1, tc2 = st.columns([1, 3.4])
        with tc1:
            if st.button(fmt_mmss(item.at_sec), key=f"tl{view.sel}_{k}"):
                # 시작 위치를 옮기면서 플레이어 재생성 카운터도 함께 올린다
                # (요구사항 8.2, 8.4, 8.5).
                set_seek(item.at_sec, view.limit)
                st.rerun()
        with tc2:
            st.markdown(
                f"<div style='padding-top:7px;font-size:13px;color:{INK}'>{item.label}"
                f"<span class='zk-chip' style='background:{style.soft};"
                f"color:{style.deep};"
                f"margin-left:6px'>{item.tag}</span></div>", unsafe_allow_html=True)


def worst_event(record) -> tuple[int, Event] | None:
    """기록에서 위험도가 가장 높은 이벤트 — ``(인덱스, 이벤트)`` 또는 ``None``.

    판정 요약 헤드라인이 선택 이벤트에 흔들리지 않게 하려면(요구사항 10.9, 10.10)
    "기록 전체의 최악 사례"를 한 곳에서만 정해야 한다. 그 한 곳이 이 함수다.

    인덱스를 함께 돌려주는 이유는 화면이 그 이벤트를 가리킬 수 있어야 하기
    때문이다(구간 목록의 몇 번째인지). 점수가 같으면 앞쪽 이벤트를 고른다 —
    ``max``의 기본 동작이라 같은 기록을 두 번 그려도 결과가 흔들리지 않는다.

    Streamlit을 부르지 않는 순수 함수다. 세션 상태를 읽지 않으므로 헤드라인 값이
    선택 상태에 영향받을 경로 자체가 없다.
    """
    events = tuple(getattr(record, "events", ()) or ())
    if not events:
        return None
    index = max(range(len(events)), key=lambda i: events[i].score)
    return index, events[index]


def verdict_block(record, curve=None) -> None:
    """블록 1 — 판정 요약: 기록의 최고 위험도 · 등급 · 쌍 · 구간 건수 + 리스크 리본.

    헤드라인 값은 :func:`worst_event`가 돌려주는 이벤트에서만 나온다. 선택 이벤트를
    바꿔도 이 블록의 점수·등급·쌍 표기는 그대로다(요구사항 10.8, 10.9, 10.10).
    선택 이벤트의 세부 수치는 요약 지표 블록이 담당한다(요구사항 10.5, 10.15).

    ``curve``는 리본에 쓸 위험도_곡선이다. 위험도 변화 그래프와 같은 곡선을 넘겨
    두 표현이 같은 데이터를 보게 한다(요구사항 10.12). 생략하면 최고 위험도
    이벤트의 곡선을 쓴다 — 기록만으로도 이 블록을 그릴 수 있게 해 둔다.
    """
    worst = worst_event(record)
    if worst is None:
        st.info(NO_EVENT_NOTE)
        return
    _, event = worst
    style = risk_level(event.score)
    count = len(record.events)
    st.markdown(
        f"<div class='zk-verdict' "
        f"style='border-color:{style.base}55;border-left-color:{style.base}'>"
        f"<div class='head'>"
        f"<span class='score' style='color:{style.deep}'>{event.score}"
        f"<span class='u'>점</span></span>"
        f"{chip(style.label, style)}"
        f"<span class='pair'>id{event.leader} ← id{event.follower}</span>"
        f"</div>"
        f"<div class='meta'>감지 구간 {count}건 중 가장 높은 위험도입니다</div>"
        f"</div>", unsafe_allow_html=True)
    ribbon = event.curve if curve is None else curve
    st.markdown(risk_ribbon_svg(ribbon, record.duration_sec),
                unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


def block_verdict(view) -> None:
    """판정 요약 블록을 :class:`ResultView`에 맞춰 그린다(요구사항 10.1).

    리본에는 선택 이벤트의 곡선을 넘긴다 — 아래 그래프 블록이 그리는 곡선과 같은
    데이터여야 한다(요구사항 10.12). 헤드라인은 기록에서만 계산되므로 선택을
    바꿔도 변하지 않는다(요구사항 10.10).
    """
    verdict_block(view.rec, curve=view.event.curve)


def block_player(view) -> None:
    """블록 2 — 좌: 분석 영상과 박스 색 범례 / 우: 구간 목록과 타임라인(요구사항 10.2, 1.7)."""
    left, right = st.columns([1.45, 1])
    with left:
        # 파일이 없으면 이 블록 안에서 안내만 하고, 아래 블록은 계속 그린다(요구사항 4.7).
        player_block(view.rec, view.seek)
    with right:
        event_picker(view)
        timeline_list(view)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)


def block_chart(view) -> None:
    """블록 3 — 시간축 위험도 꺾은선(요구사항 10.3).

    x축 범위는 곡선이 아니라 기록의 재생 길이로 고정한다. 그래야 구간을 번갈아
    골라도 시간축 눈금이 흔들리지 않는다.
    """
    st.markdown("<div class='zk-label'>시간에 따른 위험도 변화</div>",
                unsafe_allow_html=True)
    st.markdown(risk_chart_svg(view.event.curve, view.event.score,
                               view.rec.duration_sec), unsafe_allow_html=True)
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)


def block_metrics(view) -> None:
    """블록 4 — 선택된 이벤트의 위험도 점수 · 의심 가점 · 동행 감점(요구사항 10.5).

    최고 위험도와 감지 구간 건수는 판정 요약 블록으로 옮겼다(요구사항 10.8). 같은
    숫자를 두 블록에 두면 선택을 바꿀 때 한쪽만 움직여 어느 쪽이 기록 전체 값인지
    읽는 사람이 구분할 수 없다. 여기 값은 전부 선택 이벤트에서 나오고 선택이
    바뀌면 함께 갱신된다(요구사항 10.15).
    """
    e = view.event
    metric_row([
        ("위험도 점수", f"{e.score}", view.style.base),
        ("의심 가점", f"+{e.suspicion}", CORAL),
        ("동행 감점", f"−{e.deduction}", MINT),
    ])
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)


def block_evidence(view) -> None:
    """블록 5 — 좌: 판정 근거 표 / 우: 항목별 점수 카드(요구사항 12.1~12.6).

    표와 카드가 같은 원본(``event.rows``)을 본다(요구사항 12.3).
    """
    b1, b2 = st.columns([1.1, 1])
    with b1:
        st.markdown("<div class='zk-label'>판정 근거</div>", unsafe_allow_html=True)
        st.markdown(evidence_table(view.event), unsafe_allow_html=True)
    with b2:
        st.markdown("<div class='zk-label'>항목별 점수</div>", unsafe_allow_html=True)
        st.markdown(score_cards(view.event.rows), unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


def block_actions(view) -> None:
    """블록 6 — 관제 알림 전송과 리포트 다운로드(요구사항 10.6)."""
    rec = view.rec
    a1, a2 = st.columns(2)
    with a1:
        if st.button("관제 알림으로 보내기", type="primary", use_container_width=True):
            sent = make_alerts(rec)
            append_alerts(sent)
            st.success(f"{len(sent)}건을 관제 알림으로 보냈습니다.")
    with a2:
        st.download_button("리포트 다운로드", build_report(rec),
                           file_name="미행탐지시스템_리포트.txt", use_container_width=True)


#: 블록 이름 -> 렌더 함수. :data:`BLOCKS`가 순서를, 이 사전이 내용을 정한다.
BLOCK_RENDERERS = {
    "verdict": block_verdict,
    "player": block_player,
    "chart": block_chart,
    "metrics": block_metrics,
    "evidence": block_evidence,
    "actions": block_actions,
}


def render_blocks(view, names=BLOCKS) -> None:
    """블록을 :data:`BLOCKS` 순서대로 그린다(요구사항 10.1).

    이름에 해당하는 렌더 함수가 없으면 조용히 넘어간다 — 블록 이름을 바꾸는 중에
    화면 전체가 죽는 편보다, 그 블록만 비는 편이 고치기 쉽다.
    """
    for name in names:
        render = BLOCK_RENDERERS.get(name)
        if render is not None:
            render(view)


# ------------------------------------------------------------
#  업로드와 분석 실행
# ------------------------------------------------------------

def keep_result_video(result_video, src_path, record_id) -> tuple[str, bool]:
    """결과 영상을 보관 위치로 옮긴다 — ``(재생_영상_경로, 보관 실패 여부)``.

    합성 영상이 없는 경로(데모 또는 합성 실패)는 원본을 그대로 재생 대상으로 쓴다.
    보관까지 실패하면 원본 경로를 쓰고 안내 문구용 플래그를 올린다(요구사항 1.5).
    """
    if result_video == src_path:
        return src_path, False
    try:
        return adopt_video(result_video, record_id), False
    except (OSError, shutil.Error):
        return src_path, True


def analyze_upload(name, vpath, meta) -> AnalysisRecord:
    """업로드된 영상을 분석해 분석_기록을 만든다. 저장은 호출부가 한다."""
    # 실분석을 요청했어도 의존 패키지가 없으면 데모로 내려온다(요구사항 2.7).
    mode, deps_error = resolve_mode()
    if deps_error:
        st.error("분석 의존 항목을 준비할 수 없어 데모 모드로 진행합니다.\n\n" + deps_error)

    with st.spinner("영상을 분석하고 있습니다..."):
        if mode == "demo":
            # 데모는 합성 영상이 없으므로 원본을 그대로 재생한다.
            events = demo_events(meta.duration_sec)
            result_video, render_failed = vpath, False
        else:
            # 막대와 캡션은 각자의 자리를 갖는다. 한 자리에 둘을 쓰면 캡션이
            # 나가는 순간 막대가 사라진다(ProgressSlots 참고).
            bar = st.progress(0)
            note = st.empty()
            # probe()로 읽은 총 프레임 수가 진행률의 분모다(요구사항 3.1).
            # fps도 함께 넘긴다. 모듈 상수로 대신하면 25fps·60fps 영상에서
            # 타임라인 초와 시간 기준 임계값이 전부 어긋난다.
            res = analyze_video(vpath, ProgressSlots(bar, note),
                                meta.total_frames, meta.duration_sec, meta.fps)
            events = res.events
            result_video = res.video_path
            # 합성 실패는 경로만 보고는 알 수 없다. 결과가 알려 준다(요구사항 1.5).
            render_failed = res.render_failed
            # H.264 변환 실패는 분석 실패가 아니라 재생 형식 문제다. 경고만 띄우고
            # 기록에는 남기지 않는다.
            if not res.render_failed and not res.browser_safe:
                st.warning(H264_FALLBACK_NOTE)

    rid = str(uuid.uuid4())[:8]
    video_path, adopt_failed = keep_result_video(result_video, vpath, rid)
    return AnalysisRecord(
        id=rid,
        name=name,
        events=tuple(events),
        mode=mode,
        video_path=video_path,
        duration_sec=meta.duration_sec,
        created_at=datetime.now().isoformat(timespec="seconds"),
        video_render_failed=render_failed or adopt_failed,
        video_available=Path(video_path).exists(),
    )


def upload_block() -> str | None:
    """업로드 영역을 그리고 업로드된 임시 파일 경로를 돌려준다(없으면 ``None``).

    분석 시작을 누르면 분석·저장까지 끝내고 활성 기록을 그 기록으로 바꾼 뒤
    ``st.rerun()``으로 결과_화면을 다시 그린다. 세션에는 기록 객체가 아니라
    식별자만 남긴다 — 다음 렌더가 저장소에서 다시 읽으므로 화면이 낡은 객체를
    붙들 일이 없다.
    """
    up = st.file_uploader("분석할 CCTV 영상", type=["mp4", "avi", "mov"])
    if up is None:
        return None

    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tf.write(up.read())
    tf.close()          # 버퍼를 내려야 probe()가 완전한 파일을 읽는다
    vpath = tf.name
    # 업로드 직후 재생 길이를 읽어 둔다. 시크 상한(요구사항 8.5)과 그래프
    # x축 범위가 이 값에 걸려 있다. 못 읽으면 None이 정상 값이다.
    meta = probe(vpath)

    if st.button("분석 시작", type="primary"):
        rec = analyze_upload(up.name, vpath, meta)
        save_record(rec)
        st.session_state.active_id = rec.id
        st.session_state.sel = 0
        reset_seek()
        st.rerun()
    return vpath


def page_analyze():
    """'영상 분석' 화면. 업로드 영역 아래에 :data:`BLOCKS` 순서로 결과를 그린다.

    판정 방식을 풀어 쓰는 설명 섹션은 두지 않는다(요구사항 10.7).
    """
    # 히어로 칩으로 현재 분석_모드를 함께 보여 준다(요구사항 2.1).
    hero("미행 탐지 시스템",
         "CCTV 영상 속 두 사람의 상대적 이동 관계를 분석해 확인이 필요한 구간을 선별합니다",
         badge=mode_style(requested_mode()))

    pending = upload_block()

    # 활성 기록은 렌더 시점에 저장소에서 읽는다(객체를 들고 있지 않는다).
    rec, failed = active_record()
    if failed:
        st.warning(LOAD_FAILED_NOTE.format(n=failed))

    if rec is None:
        if pending:
            # 아직 분석하지 않은 업로드 — 원본만 미리 보여 준다.
            st.video(pending)
        st.info(NO_ACTIVE_NOTE)
        return

    result_notices(rec)
    record_line(rec)

    if not rec.events:
        st.success(NO_EVENT_NOTE)
        return

    render_blocks(build_view(rec))


#: 리포트 첫 줄. 파일을 열었을 때 무엇인지 바로 알 수 있어야 한다.
REPORT_TITLE = "미행 탐지 시스템 분석 리포트"

#: 리포트 꼬리말(요구사항 14.4).
#:
#: 리포트는 보고에 첨부되어 앱 화면 밖에서 읽히므로, 화면에서 본 안내가 따라오지
#: 않는다. 결과의 성격(확인 우선순위 제시)과 판단 주체(담당자)를 리포트 자체에
#: 적어 둔다.
REPORT_DISCLAIMER: tuple[str, ...] = (
    "※ 이 리포트는 확인 우선순위를 제시하는 자료입니다. 미행 확정 판정이 아닙니다.",
    "   최종 판단은 담당자가 수행합니다.",
)

#: 타임라인 항목이 없는 이벤트에 적는 한 줄. 머리글만 남겨 두면 항목이 빠진 것인지
#: 없는 것인지 구분되지 않는다.
REPORT_NO_TIMELINE = "(타임라인 항목 없음)"


def report_head(rec) -> list[str]:
    """리포트 머리말 — 원본 파일명 · 분석 시각 · 분석 모드 · 감지 구간 건수.

    파일명과 분석 시각은 요구사항 14.3이 요구하는 항목이고, 모드는 데모 기록이
    보고에 섞였을 때 값의 출처를 알 수 있게 함께 적는다(요구사항 2.4).
    """
    lines = [REPORT_TITLE, "=" * 44,
             f"원본 파일: {rec.name}",
             f"분석 시각: {fmt_created_at(rec.created_at)}",
             f"분석 모드: {MODE_LABELS.get(rec.mode, rec.mode)}",
             f"감지 구간: {len(rec.events)}건"]
    if rec.mode == "demo":
        lines.append(f"※ {DEMO_DATA_NOTE}")
    return lines


def report_event(no, e) -> list[str]:
    """이벤트 한 건의 본문(요구사항 14.2).

    등급·점수는 :class:`Event` 계산 속성에서, 항목별 점수의 최대값은
    ``ScoreRow.max_score``에서 나온다 — 이름으로 최대 점수를 되찾지 않는다.
    타임라인 시각은 초로 보관하고 여기서 :func:`fmt_mmss`로만 표시한다.
    """
    lv = risk_level(e.score)
    lines = ["",
             f"[{no}] {lv.label} · id{e.leader} ← id{e.follower} · 위험도 {e.score}점",
             f"  의심 가점 +{e.suspicion} / 동행 감점 −{e.deduction}",
             "  [항목별 점수]"]
    lines += [f"    - {r.name}: {r.measured} → {r.score}/{r.max_score}점"
              for r in e.rows]
    lines.append("  [타임라인]")
    lines += ([f"    {fmt_mmss(t.at_sec)}  {t.label}  ({t.tag})" for t in e.timeline]
              or [f"    {REPORT_NO_TIMELINE}"])
    return lines


def build_report(rec) -> str:
    """활성 분석_기록 전체를 담은 텍스트 리포트(요구사항 14.1~14.4).

    이벤트가 하나도 없는 기록도 리포트를 만든다 — 머리말과 꼬리말만 있는 파일이
    '감지된 구간이 없었다'는 기록으로 쓰인다.
    """
    lines = report_head(rec)
    for no, e in enumerate(rec.events, start=1):
        lines += report_event(no, e)
    if not rec.events:
        lines += ["", NO_EVENT_NOTE]
    return "\n".join([*lines, "", *REPORT_DISCLAIMER])


# ============================================================
#  화면 2 : 과거 분석 기록
# ============================================================

#: 저장된 기록이 없을 때의 안내(요구사항 9.4).
NO_HISTORY_NOTE = "아직 분석 기록이 없습니다. '영상 분석'에서 영상을 분석해 보세요."

#: 삭제 확인 문구(요구사항 9.5). 어느 기록을 지우는지 이름으로 확인시킨다 —
#: 목록이 길어지면 어느 줄의 삭제를 눌렀는지 헷갈리기 쉽다.
DELETE_CONFIRM_NOTE = ("'{name}' 기록을 삭제합니다. 저장된 결과 영상도 함께 지워지며 "
                       "되돌릴 수 없습니다.")


def record_top(rec) -> tuple[int, RiskStyle]:
    """기록의 최고 위험도 점수와 그 등급 스타일(요구사항 9.2).

    이벤트가 없는 기록도 목록에 남으므로 그때는 0점으로 본다.
    """
    score = max((e.score for e in rec.events), default=0)
    return score, risk_level(score)


def history_card(rec) -> str:
    """기록 한 줄의 카드 HTML — 파일명 · 분석 시각 · 이벤트 건수 · 최고 점수 · 등급.

    색은 전부 :func:`risk_level`이 준 :class:`RiskStyle`과 팔레트 파생 상수에서만
    나온다(요구사항 13.6).
    """
    score, rs = record_top(rec)
    return (f"<div class='zk-card' style='border-left:5px solid {rs.base};margin-bottom:8px'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"<div><b style='font-size:15px;color:{INK}'>{rec.name}</b>"
            f"<div style='font-size:12px;color:{INK_SOFT};margin-top:3px'>"
            f"{fmt_created_at(rec.created_at)} · 감지 {len(rec.events)}건</div></div>"
            f"<div style='text-align:right'>"
            f"<div style='font-size:22px;font-weight:700;color:{rs.deep}'>{score}</div>"
            f"{chip(rs.label, rs)}"
            f"</div></div></div>")


def open_record(record_id, sel=0) -> None:
    """기록을 활성으로 만들고 '영상 분석' 화면으로 넘긴다(요구사항 9.3, 7.2, 7.3).

    기록 객체가 아니라 식별자만 넘긴다. 결과_화면이 렌더 시점에 저장소에서 다시
    읽으므로 낡은 객체가 남지 않는다. 시작 위치는 0으로 되돌린다 — 다른 기록을
    열었는데 이전 영상의 재생 위치가 남으면 엉뚱한 장면에서 시작한다.

    ``sel``은 열자마자 고를 이벤트 인덱스다. 기록 목록에서 열 때는 첫 이벤트(0)이고,
    관제 알림에서 열 때는 그 알림이 가리키는 이벤트다(요구사항 7.3). 범위를 넘는
    값은 :func:`build_view`가 이벤트 수 안으로 잘라 준다.
    """
    st.session_state.active_id = record_id
    st.session_state.sel = max(int(sel), 0)
    reset_seek()
    st.session_state.page = PAGES[0]


def delete_pending_record(record_id) -> None:
    """확인을 받은 기록을 지운다(요구사항 9.5).

    :func:`delete_record`가 메타 항목과 보관한 결과 영상 파일을 함께 지운다. 지운
    기록이 활성이었다면 활성 표시도 비운다 — 그대로 두면 '영상 분석' 화면이 없는
    기록을 계속 찾다가 매 렌더마다 저장소를 헛되이 읽는다(요구사항 9.7의 안내
    경로로 내려가게 한다).
    """
    delete_record(record_id)
    if st.session_state.get("active_id") == record_id:
        st.session_state.active_id = None
    st.session_state.pending_delete = None


def delete_controls(rec) -> None:
    """삭제 조작 — 한 번 누르면 확인 단계로 들어가고, 확인해야 실제로 지운다.

    확인 대상은 ``st.session_state.pending_delete``에 담은 기록 식별자 하나다.
    줄마다 별도 플래그를 두면 두 줄이 동시에 확인 상태로 열려 어느 기록을 지우는지
    화면에서 갈린다.
    """
    pending = st.session_state.get("pending_delete")
    if pending != rec.id:
        if st.button("삭제", key=f"hd{rec.id}", use_container_width=True):
            st.session_state.pending_delete = rec.id
            st.rerun()
        return

    if st.button("삭제 확인", key=f"hdy{rec.id}", type="primary",
                 use_container_width=True):
        delete_pending_record(rec.id)
        st.rerun()
    if st.button("취소", key=f"hdn{rec.id}", use_container_width=True):
        st.session_state.pending_delete = None
        st.rerun()


def page_history():
    """'과거 분석 기록' 화면. 저장소가 정렬해 준 최신순 목록을 그린다(요구사항 9.1)."""
    hero("과거 분석 기록", "이전에 분석한 영상을 다시 열어볼 수 있습니다")

    # load_records()가 created_at 기준 최신순으로 정렬해 돌려준다.
    hist, failed = load_records()
    if failed:
        # 해석하지 못한 기록이 몇 건인지 알려 준다(요구사항 4.5).
        st.warning(LOAD_FAILED_NOTE.format(n=failed))
    if not hist:
        st.info(NO_HISTORY_NOTE)
        return

    for rec in hist:
        c1, c2 = st.columns([4, 1])
        with c1:
            st.markdown(history_card(rec), unsafe_allow_html=True)
        with c2:
            st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
            if st.button("열기", key=f"h{rec.id}", use_container_width=True):
                open_record(rec.id)
                st.rerun()
            delete_controls(rec)
        if st.session_state.get("pending_delete") == rec.id:
            st.warning(DELETE_CONFIRM_NOTE.format(name=rec.name))


# ============================================================
#  화면 3 : 관제 알림 로그 (영상별 그룹)
# ============================================================

#: 저장된 알림이 없을 때의 안내.
NO_ALERT_NOTE = ("아직 알림이 없습니다. '영상 분석'에서 '관제 알림으로 보내기'를 "
                 "누르면 쌓입니다.")

#: 해석하지 못한 알림이 있을 때의 안내(요구사항 5.2).
ALERT_LOAD_FAILED_NOTE = "불러오지 못한 알림 {n}건을 건너뛰었습니다."

#: 알림이 가리키는 기록을 저장소에서 찾을 수 없을 때의 안내(요구사항 7.4, 9.6).
ALERT_ORPHAN_NOTE = "연결된 분석 기록이 없습니다."

#: 로그 비우기 확인 문구(요구사항 5.6). 몇 건이 사라지는지 숫자로 확인시킨다.
CLEAR_CONFIRM_NOTE = "저장된 관제 알림 {n}건을 모두 삭제합니다. 되돌릴 수 없습니다."

#: 예시 알림이 가리킬 데모 기록의 영상명. 영상별 그룹화를 화면에서 보려면 둘 이상이어야 한다.
DEMO_ALERT_VIDEOS: tuple[str, ...] = ("cctv_역출구_0731.mp4", "cctv_공원_0731.mp4")

#: 예시 알림용 데모 기록의 재생 길이(초). 타임라인·곡선 시각이 이 안으로 스케일된다.
DEMO_ALERT_DURATION_SEC = 45


def alert_time_text(alert) -> str:
    """알림 목록에 보여 줄 시분초 문자열. 날짜는 그룹 안에서 중복되므로 뗀다."""
    at = alert_at(alert)
    return at.strftime("%H:%M:%S") if at != datetime.min else str(alert.at)


def make_alerts(rec, at=None) -> list[Alert]:
    """분석_기록의 이벤트를 관제_알림 목록으로 만든다(요구사항 5.1, 7.2, 7.3).

    ``record_id``와 ``event_index``를 함께 담는 이유는 이동 조작이 '어느 기록의 몇
    번째 이벤트'까지 되짚어야 하기 때문이다. ``event_index``는 ``rec.events``의
    인덱스이므로 결과_화면의 선택 인덱스와 같은 좌표계를 쓴다.

    ``score``/``level``은 계산해서 담는다 — 기록이 지워진 뒤에도 목록은 그대로
    보여야 하고(요구사항 9.6), 그때는 되계산할 원본이 없다.

    ``at``은 ISO 8601 문자열이며 기본값은 지금 시각이다(요구사항 5.4). 한 번의
    전송에서 만든 알림은 같은 시각을 갖는다.
    """
    stamp = at or datetime.now().isoformat(timespec="seconds")
    return [
        Alert(id=str(uuid.uuid4())[:8], at=stamp, video=rec.name,
              record_id=rec.id, event_index=i,
              pair=f"id{e.leader} ← id{e.follower}",
              score=e.score, level=risk_level(e.score).label)
        for i, e in enumerate(rec.events)
    ]


def seed_demo_alerts() -> list[Alert]:
    """예시 알림을 채우고, 그 알림이 가리키는 데모 기록도 함께 저장한다(요구사항 7.6).

    알림만 채우면 ``record_id``가 가리킬 기록이 없어 이동 조작이 전부 비활성으로
    떨어진다(요구사항 7.4). 그러면 정작 확인하려던 화면 — 알림에서 분석 화면으로
    넘어가는 흐름 — 을 예시 데이터로 볼 수 없다. 그래서 기록을 먼저 만들어
    저장하고 그 식별자로 알림을 만든다.

    영상 파일은 없다. 재생만 안내로 대체되고 나머지 결과 블록은 그대로 그려진다
    (요구사항 4.7).
    """
    base = datetime.now()
    made: list[Alert] = []
    for i, name in enumerate(DEMO_ALERT_VIDEOS):
        stamp = (base - timedelta(minutes=5 * i)).isoformat(timespec="seconds")
        rec = AnalysisRecord(
            id=str(uuid.uuid4())[:8],
            name=name,
            events=tuple(demo_events(DEMO_ALERT_DURATION_SEC)),
            mode="demo",
            video_path="",
            duration_sec=DEMO_ALERT_DURATION_SEC,
            created_at=stamp,
        )
        save_record(rec)
        made.extend(make_alerts(rec, at=stamp))
    append_alerts(made)
    return made


def alert_summary(alerts) -> None:
    """요약 지표 — 총 알림 건수와 ``우선 확인`` 건수(요구사항 6.7).

    그룹 머리글과는 별개로 그린다. 이 지표가 무엇을 보여 주든 머리글은 그대로
    나온다(요구사항 6.5).
    """
    high = sum(1 for a in alerts if a.level == RISK_HIGH.label)
    latest = max(alerts, key=alert_at, default=None)
    items = [("총 알림", f"{len(alerts)}건", INK),
             (RISK_HIGH.label, f"{high}건", RISK_HIGH)]
    if latest is not None:
        items.append(("최근 알림", alert_time_text(latest), RISK_LOW))
    metric_row(items)
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)


def group_header(vname, items) -> str:
    """그룹 머리글 HTML — 영상명 · 그룹 내 건수 · 그룹 내 최고 점수의 등급.

    등급은 :func:`group_top`이 그 그룹의 알림만 보고 정한다(요구사항 6.3, 6.4).
    """
    top_score, gs = group_top(items)
    return (f"<div style='display:flex;align-items:center;gap:10px;margin:6px 0 8px'>"
            f"<b style='font-size:15px;color:{INK}'>{vname}</b>"
            f"{chip(f'{len(items)}건', gs)}"
            f"{chip(f'{gs.label} {top_score}', gs)}</div>")


def alert_line(alert) -> str:
    """알림 한 줄의 HTML — 시각 · 쌍 표기 · 점수 · 등급."""
    als = risk_level(alert.score)
    return (f"<div class='zk-row' style='background:{als.soft};"
            f"border-left:4px solid {als.base}'>"
            f"<code style='font-size:13px;color:{als.deep}'>"
            f"{alert_time_text(alert)}</code>"
            f"<span style='flex:1;margin-left:14px;color:{INK};font-size:13.5px'>"
            f"{alert.pair}</span>"
            f"<b style='color:{als.deep};font-size:17px'>{alert.score}</b>"
            f"{chip(alert.level, als, extra='margin-left:8px')}</div>")


def alert_nav(alert, rec) -> None:
    """이동 조작(요구사항 7.1~7.5).

    모든 알림에 조작을 그린다(요구사항 7.1). 기록이 있으면 활성 상태이고 추가
    안내를 붙이지 않는다(요구사항 7.5). 없으면 누를 수 없는 상태로 두고 왜 못
    누르는지 한 줄 적는다(요구사항 7.4) — 조작을 아예 지우면 '기록이 사라졌다'와
    '이 알림에는 원래 이동이 없다'가 화면에서 구분되지 않는다.
    """
    if rec is None:
        st.button("영상 보기", key=f"al{alert.id}", disabled=True,
                  use_container_width=True)
        st.caption(ALERT_ORPHAN_NOTE)
        return
    if st.button("영상 보기", key=f"al{alert.id}", use_container_width=True):
        # 식별자와 이벤트 인덱스만 넘긴다(결과_화면이 저장소에서 다시 읽는다).
        open_record(alert.record_id, sel=alert.event_index)
        st.rerun()


def alert_group(vname, items, records) -> None:
    """한 영상의 그룹 머리글과 알림 줄들을 그린다(요구사항 6.1, 6.2).

    줄 순서는 :func:`group_alerts`가 정한 그대로 쓴다 — 여기서 다시 정렬하면
    정렬 기준이 두 곳으로 갈린다.
    """
    st.markdown(group_header(vname, items), unsafe_allow_html=True)
    for a in items:
        c1, c2 = st.columns([4, 1])
        with c1:
            st.markdown(alert_line(a), unsafe_allow_html=True)
        with c2:
            alert_nav(a, records.get(a.record_id))


def clear_controls(alerts) -> None:
    """로그 비우기 — 확인 단계를 거친 뒤에 지운다(요구사항 5.5, 5.6).

    확인 상태는 ``st.session_state.pending_clear`` 하나로 들고 있다.
    """
    if not st.session_state.get("pending_clear"):
        if st.button("로그 비우기"):
            st.session_state.pending_clear = True
            st.rerun()
        return

    st.warning(CLEAR_CONFIRM_NOTE.format(n=len(alerts)))
    c1, c2 = st.columns(2)
    with c1:
        if st.button("비우기 확인", type="primary", use_container_width=True):
            clear_alerts()
            st.session_state.pending_clear = False
            st.rerun()
    with c2:
        if st.button("취소", key="clear_no", use_container_width=True):
            st.session_state.pending_clear = False
            st.rerun()


def page_alerts():
    """'관제 알림 로그' 화면. 저장소에서 읽어 영상별로 묶어 그린다.

    화면을 옮겼다 돌아와도 같은 목록이 나온다 — 세션이 아니라 저장소가 원본이다
    (요구사항 5.3).
    """
    hero("관제 알림 로그", "실제 운영 시에는 CCTV가 자동 분석되어 위험 구간만 알림으로 쌓입니다")

    alerts, failed = load_alerts()
    if failed:
        st.warning(ALERT_LOAD_FAILED_NOTE.format(n=failed))
    if not alerts:
        st.info(NO_ALERT_NOTE)
        if st.button("예시 알림 채우기"):
            seed_demo_alerts()
            st.rerun()
        return

    alert_summary(alerts)

    # 알림이 가리키는 기록을 한 번만 읽어 둔다. 줄마다 저장소를 다시 읽으면
    # 목록이 길어질수록 같은 파일을 몇십 번 읽는다.
    records = {r.id: r for r in load_records()[0]}

    # 영상별 그룹화 — 정렬 기준은 group_alerts가 명시적으로 정한다(요구사항 6.6).
    for vname, items in group_alerts(alerts):
        alert_group(vname, items, records)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    clear_controls(alerts)


# ============================================================
#  메인
# ============================================================

PAGES = ["영상 분석", "과거 분석 기록", "관제 알림 로그"]

#: 세션 상태 기본값. 상태 키를 여기 한 곳에 모아 둔다.
#:
#: 키를 쓰는 자리마다 ``if "x" not in st.session_state`` 같은 초기화를 붙이면,
#: 같은 키의 기본값이 여러 곳에 생겨 어느 쪽이 먼저 도느냐에 따라 값이 갈린다.
#: :func:`init_state`가 렌더 맨 앞에서 한 번 심고, 이후 코드는 값이 있다고
#: 보고 읽는다. 범위·형식 방어(예: 이벤트 수를 넘는 ``sel``)는 초기화와 별개
#: 문제라 읽는 자리에 그대로 남는다.
#:
#: - ``page``       : 지금 보고 있는 화면 이름. 사이드바 라디오와 코드가 함께 쓴다.
#: - ``active_id``  : 결과_화면이 그릴 분석_기록 식별자(객체가 아니라 id만 든다).
#: - ``sel``        : 선택된 이벤트 인덱스.
#: - ``seek``       : 재생 시작 위치(초).
#: - ``seek_nonce`` : 플레이어 재생성 카운터.
#: - ``mode``       : 요청된 분석_모드 코드.
#: - ``pending_clear``  : 알림 로그 비우기 확인 대기 여부.
#: - ``pending_delete`` : 삭제 확인 대기 중인 기록 식별자.
SESSION_DEFAULTS: dict = {
    "page": PAGES[0],
    "active_id": None,
    "sel": 0,
    "seek": 0,
    "seek_nonce": 0,
    "mode": DEFAULT_MODE,
    "pending_clear": False,
    "pending_delete": None,
}


def init_state(state=None) -> None:
    """세션 상태 기본값을 심는다. 이미 값이 있는 키는 건드리지 않는다.

    ``main()`` 맨 앞에서 한 번 부른다. rerun마다 다시 불리지만 기존 값을 덮지
    않으므로 화면 이동·선택·확인 대기 같은 상태가 살아남는다.

    Streamlit 런타임 없이 임포트·호출되는 경우(테스트, 스크립트)에도 성립해야
    하므로 세션 상태 접근은 방어적으로 감싼다. 상태 저장소가 없으면 할 일도 없다.
    """
    if state is None:
        state = getattr(st, "session_state", None)
    if state is None:
        return
    for key, value in SESSION_DEFAULTS.items():
        try:
            if key not in state:
                state[key] = value
        except Exception:      # 런타임이 없거나 상태 저장소가 쓰기를 막는 경우
            return


@dataclass(frozen=True)
class StoreStatus:
    """앱 시작 시 저장소에서 읽어 온 요약 — 건수와 해석 실패 건수.

    사이드바 건수 표시(요구사항 13.5)와 해석 실패 안내(요구사항 4.5, 5.2)가 같은
    한 번의 읽기를 나눠 쓴다. 표시할 때마다 파일을 다시 읽으면 한 렌더 안에서
    건수와 안내가 서로 다른 스냅샷을 가리킬 수 있다.
    """

    records: int
    alerts: int
    records_failed: int
    alerts_failed: int


def store_status() -> StoreStatus:
    """저장소를 읽어 :class:`StoreStatus`를 만든다(요구사항 4.3, 5.2, 5.3).

    앱이 뜰 때마다 세션이 아니라 저장소를 원본으로 읽는다. 새로고침해도 지난
    분석_기록과 관제_알림이 그대로 화면에 붙는 이유가 여기다.
    """
    recs, records_failed = load_records()
    alerts, alerts_failed = load_alerts()
    return StoreStatus(records=len(recs), alerts=len(alerts),
                       records_failed=records_failed, alerts_failed=alerts_failed)


def sidebar_nav() -> str:
    """화면 이동 라디오(요구사항 13.4). 선택된 화면 이름을 돌려준다.

    인덱스를 ``st.session_state.page``에서 계산하므로, 기록 열기·알림 이동처럼
    코드가 화면을 바꾼 경우에도 라디오가 같은 상태를 가리킨다. 라디오에 별도
    ``key``를 주지 않는 것도 같은 이유다 — 위젯이 자기 상태를 따로 들면 화면
    이동에 상태가 둘이 되어 어긋난다.
    """
    page = st.session_state.get("page", PAGES[0])
    if page not in PAGES:
        # 범위 방어. init_state가 기본값을 심어도, 코드가 넣은 값이 목록에 없는
        # 이름일 수 있다(화면 이름을 고친 뒤 남은 세션 등).
        page = PAGES[0]
        st.session_state.page = page
    choice = st.sidebar.radio("화면", PAGES, index=PAGES.index(page),
                              label_visibility="collapsed")
    if choice != page:
        st.session_state.page = choice
        st.rerun()
    return choice


def sidebar_mode() -> str:
    """분석_모드 전환 수단(요구사항 2.1, 2.2). 확정된 모드 코드를 돌려준다.

    라디오에는 사람이 읽는 라벨(``데모``/``실분석``)을 두고 상태에는 코드를 담는다.
    기본값은 ``데모``이며, 상태에 값이 없거나 아는 코드가 아니면
    :func:`requested_mode`가 기본값으로 되돌린다.
    """
    current = requested_mode()
    if st.session_state.get("mode") != current:
        # 아는 코드가 아니면 기본 모드로 되돌린다(형식 방어).
        st.session_state.mode = current

    labels = [MODE_LABELS[c] for c in MODE_ORDER]
    picked = st.sidebar.radio("분석 모드", labels,
                              index=MODE_ORDER.index(current), horizontal=True)
    code = MODE_CODES.get(picked, DEFAULT_MODE)
    if code != current:
        st.session_state.mode = code
        st.rerun()

    st.sidebar.caption(MODE_HINTS[code])
    return code


def sidebar_counts(status) -> None:
    """저장소에서 읽은 분석_기록·관제_알림 건수를 사이드바에 표시한다(요구사항 13.5).

    세션에 들고 있는 값이 아니라 그 렌더에서 저장소를 읽은 결과다. 다른 화면에서
    기록을 지우거나 알림을 비운 결과가 다음 렌더에 곧바로 반영된다.
    """
    st.sidebar.markdown(
        f"<div style='font-size:12.5px;color:{INK_SOFT};line-height:1.9'>"
        f"분석 기록 <b style='color:{INK}'>{status.records}</b>건<br>"
        f"관제 알림 <b style='color:{INK}'>{status.alerts}</b>건</div>",
        unsafe_allow_html=True)


def sidebar_notices(status) -> None:
    """해석하지 못한 항목이 있으면 사이드바에서 먼저 알린다(요구사항 4.5, 5.2).

    각 화면도 자기 목록의 실패 건수를 알리지만, 그건 그 화면에 들어가야 보인다.
    파일이 깨진 사실은 어느 화면에 있든 눈에 띄어야 하므로 사이드바에도 적는다.
    """
    if status.records_failed:
        st.sidebar.warning(LOAD_FAILED_NOTE.format(n=status.records_failed))
    if status.alerts_failed:
        st.sidebar.warning(ALERT_LOAD_FAILED_NOTE.format(n=status.alerts_failed))


def render_sidebar(status=None):
    """사이드바를 그리고 선택된 화면 이름을 돌려준다.

    ``status``는 :func:`store_status`가 읽어 둔 저장소 요약이다. 넘기지 않으면
    여기서 읽는다.
    """
    if status is None:
        status = store_status()

    st.sidebar.markdown(
        f"<div style='font-size:18px;font-weight:700;color:{INK};padding:6px 0 2px'>미행 탐지 시스템</div>"
        f"<div style='font-size:12px;color:{INK_SOFT};margin-bottom:14px'>"
        f"미행 가능성 선별 시스템</div>", unsafe_allow_html=True)

    page = sidebar_nav()

    st.sidebar.divider()
    sidebar_mode()

    st.sidebar.divider()
    sidebar_counts(status)
    sidebar_notices(status)

    return page


#: 화면 이름 -> 그 화면을 그리는 함수(요구사항 13.4).
#:
#: ``if page == ...`` 사슬 대신 표로 둔다. 화면을 더할 때 고칠 자리가
#: :data:`PAGES`와 이 표 두 곳으로 모이고, 둘의 키가 어긋나면 바로 드러난다.
PAGE_VIEWS: dict = {
    PAGES[0]: page_analyze,
    PAGES[1]: page_history,
    PAGES[2]: page_alerts,
}


def main():
    """Streamlit 진입점. 임포트 시에는 실행되지 않는다.

    순서를 지킨다 — ``set_page_config``는 다른 Streamlit 호출보다 앞이어야 하고,
    테마 CSS는 화면 요소가 붙기 전에 주입돼야 한다. 그다음 세션 상태 기본값을
    한 번에 심고(:func:`init_state`), 저장소를 읽어(:func:`store_status`)
    사이드바에 반영한 뒤, 고른 화면을 그린다.
    """
    st.set_page_config(page_title="미행 탐지 시스템", layout="wide")
    st.markdown(theme_css(), unsafe_allow_html=True)

    init_state()

    # 저장 데이터를 한 번 읽어 사이드바 건수·실패 안내에 함께 쓴다
    # (요구사항 4.3, 5.2, 5.3).
    page = render_sidebar(store_status())

    PAGE_VIEWS.get(page, page_analyze)()


if __name__ == "__main__":
    main()
