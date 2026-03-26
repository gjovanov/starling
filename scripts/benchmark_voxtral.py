#!/usr/bin/env python3
"""
Benchmark Voxtral transcription quality against broadcast_1.wav reference.

Extends benchmark_broadcast1.py with:
- Voxtral-server defaults (port 8090, speedy mode)
- WebSocket "ready" handshake (triggers audio pipeline)
- Segment-level discrepancy analysis (FINAL vs PARTIAL coverage)
- Detailed word-level diff showing exactly where words are swallowed

Usage:
  python3 scripts/benchmark_voxtral.py                          # 5-min benchmark
  python3 scripts/benchmark_voxtral.py --duration 600           # full 10 min
  python3 scripts/benchmark_voxtral.py --compare-baseline       # compare with previous

Requirements:
  pip install websockets
"""

import argparse
import asyncio
import json
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
MAGENTA = "\033[0;35m"
BOLD = "\033[1m"
DIM = "\033[2m"
NC = "\033[0m"

# ---------------------------------------------------------------------------
# Key phrases from broadcast_1.txt first 10 minutes
# ---------------------------------------------------------------------------
KEY_PHRASES = [
    "Bischofshofen", "ORF", "Manuel Rubay", "Simon Schwarz", "Hallwang",
    "Alighieri", "Domquartier", "Salzachblume", "Nussdorf", "Wirtschaftskammer",
    "EU Kommission", "Kartellverfahren", "Red Bull", "Walsberg",
    "Videoüberwachung", "Grenzkontrollen", "Salzburg", "Österreich", "Mara",
    "Kulturzentrum", "November", "Filialen", "Arzneimittel", "Eigenmarke",
    "Rezeptfreie", "Wille", "Marke", "Pointe", "Restaurant", "Societat",
]

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "benchmark_results"
REFERENCE_FILE = PROJECT_DIR / "media" / "broadcast_1.txt"

# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = text.lower()
    out = []
    for ch in text:
        if ch.isalnum() or ch.isspace():
            out.append(ch)
        else:
            out.append(" ")
    return " ".join("".join(out).split())


def normalize_words(text: str) -> list:
    return normalize_text(text).split()


# ---------------------------------------------------------------------------
# Levenshtein / WER / CER
# ---------------------------------------------------------------------------

def levenshtein(a: list, b: list) -> int:
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[n]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = normalize_words(reference)
    hyp_words = normalize_words(hypothesis)
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return levenshtein(ref_words, hyp_words) / len(ref_words)


def char_error_rate(reference: str, hypothesis: str) -> float:
    ref_chars = list(normalize_text(reference))
    hyp_chars = list(normalize_text(hypothesis))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return levenshtein(ref_chars, hyp_chars) / len(ref_chars)


def key_phrase_recall(hypothesis: str, phrases: list) -> tuple:
    hyp_lower = hypothesis.lower()
    found, missed = [], []
    for phrase in phrases:
        (found if phrase.lower() in hyp_lower else missed).append(phrase)
    recall = len(found) / len(phrases) if phrases else 1.0
    return recall, found, missed


# ---------------------------------------------------------------------------
# Reference transcript parser
# ---------------------------------------------------------------------------

def parse_reference(path: Path, max_seconds: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    segments = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        if not lines[i].strip().startswith("SPK_"):
            i += 1
            continue
        i += 1
        if i >= len(lines):
            break
        ts_line = lines[i].strip()
        i += 1
        m = re.match(r"^(\d+):(\d{2})$", ts_line)
        if not m:
            continue
        seconds = int(m.group(1)) * 60 + int(m.group(2))
        if i >= len(lines):
            break
        text = lines[i].strip()
        i += 1
        if i < len(lines) and not lines[i].strip():
            i += 1
        if seconds <= max_seconds:
            segments.append(text)
    return " ".join(segments)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def api_get(server: str, path: str):
    url = f"{server}{path}"
    with urllib.request.urlopen(urllib.request.Request(url), timeout=15) as resp:
        return json.loads(resp.read().decode())


def api_post(server: str, path: str, body: dict = None):
    url = f"{server}{path}"
    if body:
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    else:
        req = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def get_git_sha() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, cwd=str(PROJECT_DIR), timeout=5)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Segment-level discrepancy analysis
# ---------------------------------------------------------------------------

def analyze_segments(finals: list, partials: list, reference_text: str):
    """Analyze FINAL and PARTIAL segments for coverage gaps."""
    print()
    print(f"{CYAN}{'=' * 72}{NC}")
    print(f"{CYAN}{BOLD}  SEGMENT ANALYSIS{NC}")
    print(f"{CYAN}{'=' * 72}{NC}")
    print()

    # --- FINAL segment analysis ---
    finals_text = " ".join(f.get("text", "") for f in finals)
    finals_words = normalize_words(finals_text)
    ref_words = normalize_words(reference_text)

    print(f"{YELLOW}  FINAL Segments ({len(finals)} total){NC}")
    total_final_words = 0
    for i, f in enumerate(finals):
        text = f.get("text", "")
        words = len(text.split())
        total_final_words += words
        ts = f.get("start", 0)
        if i < 5 or i >= len(finals) - 3:
            print(f"    {DIM}[{i+1:>3d}] t={ts:>6.1f}s ({words:>3d}w){NC} {text[:90]}")
        elif i == 5:
            print(f"    {DIM}  ... ({len(finals) - 8} more segments) ...{NC}")

    print(f"\n    Total FINAL words: {total_final_words}")
    print(f"    Reference words:   {len(ref_words)}")
    coverage = total_final_words / len(ref_words) * 100 if ref_words else 0
    cov_color = GREEN if coverage > 80 else (YELLOW if coverage > 50 else RED)
    print(f"    Word coverage:     {cov_color}{coverage:.1f}%{NC}")

    # --- Gap analysis: find reference words missing from FINALs ---
    print(f"\n{YELLOW}  Gap Analysis (words in reference but missing from FINALs){NC}")
    finals_set = set(normalize_words(finals_text))
    ref_unique = set(ref_words)
    missing_words = ref_unique - finals_set
    present_words = ref_unique & finals_set

    miss_pct = len(missing_words) / len(ref_unique) * 100 if ref_unique else 0
    print(f"    Unique ref words:     {len(ref_unique)}")
    print(f"    Found in FINALs:      {GREEN}{len(present_words)}{NC}")
    print(f"    Missing from FINALs:  {RED}{len(missing_words)}{NC} ({miss_pct:.1f}%)")

    if missing_words:
        # Show top missing words (sorted by frequency in reference)
        from collections import Counter
        ref_counter = Counter(ref_words)
        missing_by_freq = sorted(missing_words, key=lambda w: ref_counter.get(w, 0), reverse=True)
        show = missing_by_freq[:30]
        print(f"    Top missing: {', '.join(show)}")

    # --- PARTIAL analysis ---
    print(f"\n{YELLOW}  PARTIAL Segments ({len(partials)} total){NC}")
    if partials:
        # Check if PARTIALs contain words missing from FINALs
        all_partial_text = " ".join(p.get("text", "") for p in partials)
        partial_set = set(normalize_words(all_partial_text))
        rescued = missing_words & partial_set
        if rescued:
            print(f"    Words in PARTIALs but NOT in FINALs: {YELLOW}{len(rescued)}{NC}")
            print(f"    Rescued words: {', '.join(sorted(rescued)[:20])}")
            print(f"    {YELLOW}→ These words were transcribed but lost in FINAL emission{NC}")
        else:
            print(f"    No additional words found in PARTIALs")

        # Check full_transcript coverage
        last_ft = ""
        for p in reversed(partials):
            ft = p.get("full_transcript", "")
            if ft:
                last_ft = ft
                break
        if not last_ft:
            for f in reversed(finals):
                ft = f.get("full_transcript", "")
                if ft:
                    last_ft = ft
                    break

        if last_ft:
            ft_words = set(normalize_words(last_ft))
            ft_rescued = missing_words & ft_words
            ft_coverage = len(ft_words & ref_unique) / len(ref_unique) * 100 if ref_unique else 0
            print(f"\n    full_transcript coverage: {ft_coverage:.1f}%")
            if ft_rescued - rescued:
                extra = ft_rescued - rescued
                print(f"    Additional words in full_transcript: {len(extra)}")

    # --- Timing gaps ---
    print(f"\n{YELLOW}  Timing Analysis{NC}")
    if len(finals) > 1:
        gaps = []
        for i in range(1, len(finals)):
            prev_end = finals[i-1].get("end", 0)
            curr_start = finals[i].get("start", 0)
            gap = curr_start - prev_end
            if gap > 2.0:  # gaps > 2 seconds
                gaps.append((i, prev_end, curr_start, gap))

        if gaps:
            print(f"    Gaps > 2s between FINALs: {RED}{len(gaps)}{NC}")
            for idx, prev_end, curr_start, gap in gaps[:10]:
                print(f"      Gap {gap:.1f}s between FINAL {idx} (end={prev_end:.1f}s) and FINAL {idx+1} (start={curr_start:.1f}s)")
        else:
            print(f"    {GREEN}No large gaps between FINALs{NC}")

    # --- Consecutive FINAL dedup check ---
    print(f"\n{YELLOW}  FINAL Dedup Check{NC}")
    dedup_issues = 0
    for i in range(1, len(finals)):
        prev = normalize_text(finals[i-1].get("text", ""))
        curr = normalize_text(finals[i].get("text", ""))
        if prev and curr:
            # Check containment
            prev_words = set(prev.split())
            curr_words = set(curr.split())
            overlap = prev_words & curr_words
            if len(overlap) >= 3:
                min_size = min(len(prev_words), len(curr_words))
                containment = len(overlap) / min_size if min_size > 0 else 0
                if containment >= 0.75:
                    dedup_issues += 1
                    if dedup_issues <= 5:
                        print(f"    {RED}High overlap ({containment:.0%}) between FINAL {i} and {i+1}:{NC}")
                        print(f"      Prev: {prev[:80]}")
                        print(f"      Curr: {curr[:80]}")

    if dedup_issues > 5:
        print(f"    ... and {dedup_issues - 5} more")
    if dedup_issues == 0:
        print(f"    {GREEN}No dedup issues between consecutive FINALs{NC}")

    print()
    return {
        "total_final_words": total_final_words,
        "word_coverage_pct": round(coverage, 2),
        "unique_ref_words": len(ref_unique),
        "missing_word_count": len(missing_words),
        "missing_word_pct": round(miss_pct, 2),
        "partial_rescued_count": len(rescued) if partials and missing_words else 0,
        "timing_gaps_gt_2s": len(gaps) if len(finals) > 1 else 0,
        "dedup_issues": dedup_issues,
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

async def run_benchmark(args):
    import websockets

    print(f"{CYAN}{'=' * 72}{NC}")
    print(f"{CYAN}{BOLD}  VOXTRAL BROADCAST_1 BENCHMARK{NC}")
    print(f"{CYAN}{'=' * 72}{NC}")
    print()

    # --- Parse reference ---
    print(f"{YELLOW}[1/7] Parsing reference transcript...{NC}")
    if not REFERENCE_FILE.exists():
        print(f"{RED}Error: Reference file not found: {REFERENCE_FILE}{NC}")
        sys.exit(1)
    reference_text = parse_reference(REFERENCE_FILE, args.duration)
    ref_word_count = len(normalize_words(reference_text))
    print(f"  Reference: {ref_word_count} words (first {args.duration}s)")

    # --- Check server ---
    print(f"\n{YELLOW}[2/7] Checking server at {args.server}...{NC}")
    try:
        url = f"{args.server}/health"
        with urllib.request.urlopen(url, timeout=10) as resp:
            resp.read()
    except Exception as e:
        print(f"{RED}Error: Server not reachable at {args.server}: {e}{NC}")
        sys.exit(1)

    modes_resp = api_get(args.server, "/api/modes")
    mode_ids = [m["id"] for m in modes_resp.get("data", [])]
    if args.mode not in mode_ids:
        print(f"{RED}Error: Mode '{args.mode}' not available. Available: {mode_ids}{NC}")
        sys.exit(1)
    print(f"  {GREEN}Server OK. Mode '{args.mode}' available.{NC}")

    # --- Detect model ---
    print(f"\n{YELLOW}[3/7] Detecting model...{NC}")
    model = args.model
    if not model:
        models_resp = api_get(args.server, "/api/models")
        models = models_resp.get("data", [])
        if not models:
            print(f"{RED}Error: No models available.{NC}")
            sys.exit(1)
        model = models[0]["id"]
    print(f"  {GREEN}Using model: {model}{NC}")

    # --- Create session ---
    print(f"\n{YELLOW}[4/7] Creating session...{NC}")
    session_body = {
        "model_id": model,
        "mode": args.mode,
        "language": args.language,
        "media_id": "broadcast_1",
    }
    session_resp = api_post(args.server, "/api/sessions", session_body)
    if not session_resp.get("success"):
        print(f"{RED}Error creating session: {session_resp.get('error')}{NC}")
        sys.exit(1)
    session_id = session_resp["data"]["id"]
    print(f"  Session: {session_id}")

    # --- Start session ---
    print(f"\n{YELLOW}[5/7] Starting session...{NC}")
    start_resp = api_post(args.server, f"/api/sessions/{session_id}/start")
    if not start_resp.get("success"):
        print(f"{RED}Error starting session: {start_resp.get('error')}{NC}")
        sys.exit(1)
    print(f"  {GREEN}Session started.{NC}")

    # --- Connect WebSocket and collect ---
    print(f"\n{YELLOW}[6/7] Collecting transcription via WebSocket...{NC}")
    ws_url = f"{args.server.replace('http', 'ws')}/ws/{session_id}"

    finals = []
    partials = []
    inference_times = []
    last_full_transcript = ""
    wall_start = time.monotonic()
    max_wall_secs = args.duration + 30  # stop collecting after duration + 30s buffer

    try:
        async with websockets.connect(ws_url, close_timeout=10) as ws:
            # Wait for welcome
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            welcome = json.loads(raw)
            print(f"  {DIM}Welcome: {welcome.get('type', 'unknown')}{NC}")

            # Send "ready" to trigger WebRTC + audio pipeline
            await ws.send(json.dumps({"type": "ready"}))
            print(f"  {DIM}Sent 'ready' signal{NC}")

            # Collect messages until duration exceeded or session ends
            while True:
                elapsed = time.monotonic() - wall_start
                remaining = max(1.0, max_wall_secs - elapsed)

                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    print(f"  {YELLOW}Duration limit reached ({args.duration}s + buffer){NC}")
                    break

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "subtitle":
                    is_final = msg.get("is_final", False)
                    text = msg.get("text", "")
                    ft = msg.get("full_transcript", "")
                    if ft:
                        last_full_transcript = ft
                    inf_time = msg.get("inference_time_ms")
                    if inf_time is not None:
                        inference_times.append(inf_time)

                    if is_final:
                        finals.append(msg)
                        display = text[:100] + ("..." if len(text) > 100 else "")
                        print(f"  {GREEN}[FINAL {len(finals):>3d}]{NC} {display}")
                    else:
                        partials.append(msg)
                        if len(partials) % 20 == 0:
                            display = (msg.get("growing_text", "") or text)[:80]
                            print(f"  {DIM}[PARTIAL {len(partials):>4d}]{NC} {display}")

                elif msg_type == "end":
                    print(f"  {CYAN}[END]{NC} Session complete. duration={msg.get('total_duration', '?')}s")
                    break

                elif msg_type in ("offer", "ice-candidate", "welcome", "ping", "start"):
                    pass  # signaling messages, ignore

                elif msg_type == "error":
                    print(f"  {RED}[ERROR]{NC} {msg.get('message', msg)}")

    except Exception as e:
        print(f"{RED}WebSocket error: {e}{NC}")
        if not finals and not partials:
            sys.exit(1)

    wall_secs = time.monotonic() - wall_start

    # Stop the session to free resources
    try:
        api_post(args.server, f"/api/sessions/{session_id}", None)
    except Exception:
        pass
    try:
        urllib.request.urlopen(
            urllib.request.Request(f"{args.server}/api/sessions/{session_id}", method="DELETE"),
            timeout=5,
        )
    except Exception:
        pass

    # --- Compute metrics ---
    print(f"\n{YELLOW}[7/7] Computing metrics...{NC}")

    finals_text = " ".join(f.get("text", "") for f in finals)
    hypothesis = last_full_transcript if last_full_transcript else finals_text

    wer = word_error_rate(reference_text, hypothesis)
    cer = char_error_rate(reference_text, hypothesis)
    recall, found_phrases, missed_phrases = key_phrase_recall(hypothesis, KEY_PHRASES)

    avg_inf_time = sum(inference_times) / len(inference_times) if inference_times else 0.0
    partial_final_ratio = len(partials) / len(finals) if finals else 0.0
    hyp_word_count = len(normalize_words(hypothesis))

    finals_wer = word_error_rate(reference_text, finals_text)
    finals_cer = char_error_rate(reference_text, finals_text)
    finals_recall, _, _ = key_phrase_recall(finals_text, KEY_PHRASES)

    # --- Segment analysis ---
    seg_analysis = analyze_segments(finals, partials, reference_text)

    # --- Summary ---
    print(f"{CYAN}{'=' * 72}{NC}")
    print(f"{CYAN}{BOLD}  BENCHMARK RESULTS{NC}")
    print(f"{CYAN}{'=' * 72}{NC}")
    print()
    print(f"  {DIM}Model:{NC}          {model}")
    print(f"  {DIM}Mode:{NC}           {args.mode}")
    print(f"  {DIM}Language:{NC}        {args.language}")
    print(f"  {DIM}Duration:{NC}        {args.duration}s")
    print()

    print(f"{YELLOW}  Message Counts{NC}")
    print(f"  {'FINAL count':<28} {len(finals):>8d}")
    print(f"  {'PARTIAL count':<28} {len(partials):>8d}")
    print(f"  {'Partial/Final ratio':<28} {partial_final_ratio:>8.1f}x")
    print()

    print(f"{YELLOW}  Timing{NC}")
    print(f"  {'Wall clock time':<28} {wall_secs:>8.1f}s")
    print(f"  {'Avg inference time':<28} {avg_inf_time:>8.1f}ms")
    print()

    wer_pct = wer * 100
    cer_pct = cer * 100
    recall_pct = recall * 100
    wer_color = GREEN if wer_pct < 40 else (YELLOW if wer_pct < 60 else RED)
    cer_color = GREEN if cer_pct < 30 else (YELLOW if cer_pct < 50 else RED)
    recall_color = GREEN if recall_pct > 70 else (YELLOW if recall_pct > 50 else RED)

    print(f"{YELLOW}  Quality (full_transcript){NC}")
    print(f"  {'WER':<28} {wer_color}{wer_pct:>8.1f}%{NC}")
    print(f"  {'CER':<28} {cer_color}{cer_pct:>8.1f}%{NC}")
    print(f"  {'Key Phrase Recall':<28} {recall_color}{recall_pct:>8.1f}%{NC}")
    print(f"  {'Reference words':<28} {ref_word_count:>8d}")
    print(f"  {'Hypothesis words':<28} {hyp_word_count:>8d}")
    print()

    print(f"{YELLOW}  Quality (FINALs only){NC}")
    print(f"  {'WER':<28} {finals_wer * 100:>8.1f}%")
    print(f"  {'CER':<28} {finals_cer * 100:>8.1f}%")
    print(f"  {'Key Phrase Recall':<28} {finals_recall * 100:>8.1f}%")
    print(f"  {'FINAL words':<28} {len(normalize_words(finals_text)):>8d}")
    print()

    print(f"{YELLOW}  Segment Coverage{NC}")
    print(f"  {'Word coverage':<28} {seg_analysis['word_coverage_pct']:>8.1f}%")
    print(f"  {'Missing words':<28} {seg_analysis['missing_word_count']:>8d} ({seg_analysis['missing_word_pct']:.1f}%)")
    print(f"  {'Rescued by PARTIALs':<28} {seg_analysis['partial_rescued_count']:>8d}")
    print(f"  {'Timing gaps >2s':<28} {seg_analysis['timing_gaps_gt_2s']:>8d}")
    print(f"  {'Dedup overlap issues':<28} {seg_analysis['dedup_issues']:>8d}")
    print()

    if missed_phrases:
        print(f"{YELLOW}  Key Phrases Missed ({len(missed_phrases)}/{len(KEY_PHRASES)}):{NC}")
        for phrase in missed_phrases:
            print(f"    {RED}- {phrase}{NC}")
        print()

    print(f"{CYAN}{'=' * 72}{NC}")

    # --- Save ---
    git_sha = get_git_sha()
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    metrics = {
        "model": model,
        "mode": args.mode,
        "language": args.language,
        "duration": args.duration,
        "enable_formatting": False,
        "tone": "casual",
        "git_sha": git_sha,
        "timestamp": timestamp,
        "final_count": len(finals),
        "partial_count": len(partials),
        "partial_final_ratio": round(partial_final_ratio, 2),
        "avg_inference_time_ms": round(avg_inf_time, 1),
        "wall_clock_secs": round(wall_secs, 1),
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "key_phrase_recall": round(recall, 4),
        "key_phrases_found": found_phrases,
        "key_phrases_missed": missed_phrases,
        "reference_word_count": ref_word_count,
        "hypothesis_word_count": hyp_word_count,
        "finals_only_wer": round(finals_wer, 4),
        "finals_only_cer": round(finals_cer, 4),
        "finals_only_key_phrase_recall": round(finals_recall, 4),
        "hypothesis_source": "full_transcript" if last_full_transcript else "finals_concat",
        "segment_analysis": seg_analysis,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_safe = model.replace("/", "-").replace(" ", "_")
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"broadcast1_{model_safe}_{args.mode}_noformat_{ts_file}.json"
    result_path = RESULTS_DIR / filename
    result_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  {GREEN}Results saved: {result_path}{NC}")

    # Save raw segments for debugging
    raw_path = result_path.with_suffix(".segments.json")
    raw_path.write_text(json.dumps({
        "finals": finals,
        "partials": partials[-50:],  # last 50 partials only
        "full_transcript": last_full_transcript,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {GREEN}Raw segments saved: {raw_path}{NC}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark Voxtral transcription quality")
    parser.add_argument("--server", default="http://localhost:8090",
                        help="Voxtral server URL (default: http://localhost:8090)")
    parser.add_argument("--model", default="", help="Model ID (default: auto-detect)")
    parser.add_argument("--mode", default="speedy", help="Latency mode (default: speedy)")
    parser.add_argument("--language", default="de", help="Language (default: de)")
    parser.add_argument("--duration", type=int, default=300,
                        help="Duration in seconds (default: 300 = 5 min)")
    args = parser.parse_args()

    if args.duration < 10:
        print(f"{RED}Error: Duration must be at least 10 seconds.{NC}")
        sys.exit(1)

    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
