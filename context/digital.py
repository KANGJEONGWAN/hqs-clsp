from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv
"""
context/digital.py

PC(app_switch) / Mobile(app_start+app_close) JSONL 이벤트 파일을 읽어서
GPT-4o-mini로 digital_summary 문장을 생성하는 모듈.

사용법:
    from context.digital import summarize_events, match_events_to_window, process_jsonl

    # PPG window에 해당하는 이벤트 찾아서 요약
    summary = match_events_to_window(
        events=events,
        window_start="2026-04-24T02:33:52+00:00",
        window_end="2026-04-24T02:34:02+00:00",
    )

    # 파일 전체 배치 처리
    process_jsonl("data/events_20260424.jsonl", "data/events_20260424_summarized.jsonl")
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from openai import OpenAI

# .env 파일 로드
for _env in [
    Path(__file__).parent.parent / ".env",
]:
    if _env.exists():
        load_dotenv(_env)
        break
    

# ── 앱 이름 정리 ──────────────────────────────────────────
_APP_LABEL: dict[str, str] = {
    # PC
    "chrome.exe":   "Chrome",
    "Code.exe":     "VSCode",
    "msedge.exe":   "Edge",
    "explorer.exe": "File Explorer",
    # Mobile
    "com.google.android.youtube":  "YouTube",
    "com.nhn.android.search":      "Naver",
    "com.android.settings":        "Settings",
    "com.hqs.tracker":             "HQS Tracker",
}

_IGNORE_APPS = {"File Explorer", "HQS Tracker", "Settings"}

def _clean_app(app: str) -> str:
    return _APP_LABEL.get(app, app)


# ── GPT 프롬프트 ──────────────────────────────────────────
_SYSTEM_PROMPT = (
    "You summarize a user's recent digital activity for an XR personalization system. "
    "Given a list of apps used and their durations, write ONE natural English sentence (15-25 words). "
    "Rules: "
    "1. Do NOT mention emotions, stress, cognitive load, or mental state. "
    "2. Focus only on what apps were used and for how long. "
    "3. Be concise and factual. "
    "4. Output the sentence only, no extra text."
)

_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# ── timestamp 파싱 ────────────────────────────────────────
def _parse_dt(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ── 유효한 이벤트 추출 ────────────────────────────────────
def _extract_valid_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    PC / Mobile 이벤트에서 유효한 사용 구간만 추출.

    PC (app_switch):
      - 같은 app + 같은 timestamp로 dur=0.0 / dur>0 쌍이 찍힘
      - 같은 app + 같은 timestamp 쌍 확인 후 dur>0인 것만 사용
      - 사용 구간 = timestamp - duration ~ timestamp

    Mobile (app_start / app_close):
      - app_start: dur=0.0, app_close: dur>0
      - 같은 app의 직전 app_start와 app_close 쌍으로 묶음
      - 사용 구간 = app_start.timestamp ~ app_close.timestamp
    """
    valid: list[dict[str, Any]] = []

    # ── PC: app_switch ────────────────────────────────────
    pc_events = [e for e in events if e.get("event_type") == "app_switch"]

    # (app, timestamp) 기준으로 그룹핑
    pc_groups: dict[tuple[str, str], list[dict]] = {}
    for e in pc_events:
        key = (e.get("app", ""), e.get("timestamp", ""))
        pc_groups.setdefault(key, []).append(e)

    for (app, ts), group in pc_groups.items():
        has_start = any(float(e.get("duration_seconds", 0)) == 0.0 for e in group)
        end_events = [e for e in group if float(e.get("duration_seconds", 0)) > 0]

        if has_start and end_events:
            e = end_events[0]
            dur = float(e["duration_seconds"])
            end_dt = _parse_dt(ts)
            start_dt = end_dt - timedelta(seconds=dur)
            valid.append({
                "app":        app,
                "start_time": start_dt,
                "end_time":   end_dt,
                "duration":   dur,
                "title":      e.get("title", ""),
                "source":     "pc",
            })

    # ── Mobile: app_start / app_close ────────────────────
    pending_starts: dict[str, datetime] = {}  # app → start_time

    for e in sorted(events, key=lambda x: x.get("timestamp", "")):
        etype = e.get("event_type", "")
        app   = e.get("app", "")

        if etype == "app_start":
            pending_starts[app] = _parse_dt(e["timestamp"])

        elif etype == "app_close":
            dur = float(e.get("duration_seconds", 0))
            if dur > 0 and app in pending_starts:
                start_dt = pending_starts.pop(app)
                end_dt   = _parse_dt(e["timestamp"])
                valid.append({
                    "app":        app,
                    "start_time": start_dt,
                    "end_time":   end_dt,
                    "duration":   dur,
                    "title":      e.get("title", ""),
                    "source":     "mobile",
                })

    return valid


# ── PPG window 기준 이벤트 매칭 ───────────────────────────
def match_events_to_window(
    events: list[dict[str, Any]],
    window_start: str,
    window_end: str,
) -> str:
    """
    PPG window (window_start ~ window_end) 구간과 겹치는 이벤트를 찾아서
    llm_summary를 반환. 겹치는 이벤트 없으면 빈 문자열 반환.
    """
    w_start = _parse_dt(window_start)
    w_end   = _parse_dt(window_end)

    valid = _extract_valid_events(events)

    # window와 겹치는 이벤트 필터링
    matched = [
        e for e in valid
        if e["start_time"] <= w_end and e["end_time"] >= w_start
    ]

    if not matched:
        return ""

    return summarize_events(matched)


# ── 이벤트 리스트 → llm_summary ──────────────────────────
def summarize_events(valid_events: list[dict[str, Any]]) -> str:
    """
    유효한 이벤트 리스트 ({app, start_time, end_time, duration, title})를 받아서
    llm_summary 문장을 반환.
    """
    if not valid_events:
        return ""

    # 앱별 총 사용 시간 집계
    app_durations: dict[str, float] = {}
    for e in valid_events:
        app = _clean_app(e["app"])
        if app not in _IGNORE_APPS:
            app_durations[app] = app_durations.get(app, 0) + e["duration"]

    if not app_durations:
        return ""

    lines = [
        f"- {app}: {dur:.0f}s"
        for app, dur in sorted(app_durations.items(), key=lambda x: -x[1])
    ]

    # 가장 긴 이벤트의 title 추가
    pc_events = [e for e in valid_events if e.get("source") == "pc"]
    if pc_events:
        longest = max(pc_events, key=lambda e: e["duration"])
        title = longest.get("title", "")
        if title:
            lines.append(f"- Most viewed: '{title[:60]}'")

    user_prompt = (
        "Apps used:\n"
        + "\n".join(lines)
        + "\n\nWrite one sentence summarizing this digital activity."
    )

    client = _get_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=60,
        temperature=0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


# ── 배치 처리 ─────────────────────────────────────────────
def process_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    window_seconds: float = 300.0,
) -> None:
    """
    JSONL 파일 전체를 time window로 묶어서 세션화하고
    각 세션에 llm_summary를 채운 뒤 output_path에 저장.
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, encoding="utf-8") as f:
        raw_events = [json.loads(l) for l in f if l.strip()]

    if not raw_events:
        print("No events found.")
        return

    # 유효한 이벤트 추출
    valid_events = _extract_valid_events(raw_events)

    if not valid_events:
        print("No valid events after pairing.")
        return

    # time window로 세션 분리
    valid_events.sort(key=lambda e: e["start_time"])
    sessions: list[list[dict]] = []
    current: list[dict] = [valid_events[0]]

    for e in valid_events[1:]:
        prev_end = current[-1]["end_time"]
        curr_start = e["start_time"]
        if (curr_start - prev_end).total_seconds() > window_seconds:
            sessions.append(current)
            current = [e]
        else:
            current.append(e)
    sessions.append(current)

    with open(output_path, "w", encoding="utf-8") as fout:
        for i, sess_events in enumerate(sessions):
            summary = summarize_events(sess_events)
            record = {
                "session_id":  f"session_{i:04d}",
                "device_type": raw_events[0].get("device_type", "unknown"),
                "start_time":  sess_events[0]["start_time"].isoformat(),
                "end_time":    sess_events[-1]["end_time"].isoformat(),
                "llm_summary": summary,
            }
            print(f"session_{i:04d} → {summary}")
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved: {output_path}")


# ── CLI ───────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl",  type=str, required=True, help="이벤트 JSONL 경로")
    parser.add_argument("--out",    type=str, required=True, help="출력 JSONL 경로")
    parser.add_argument("--window", type=float, default=300.0, help="세션 분리 기준(초), 기본 300")
    args = parser.parse_args()

    process_jsonl(args.jsonl, args.out, window_seconds=args.window)