from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from faster_whisper import WhisperModel

TZ = timezone(timedelta(hours=8))
DATA_DIR = Path(os.getenv("NORDEN_DATA_DIR", "data/norden"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small.en")
IOS_UA = "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1"
SEEDS = [x.strip() for x in os.getenv("NORDEN_SEED_VIDEO_IDS", "").split(",") if x.strip()]
KNOWN = {
    "7679312806105156916": ("沃什放鹰金银大跌，风险管理的价值再次验证", "2026-08-29"),
    "7678516691764071732": ("贝森特把球踢给沃什，周五沃什会亮出底牌吗", "2026-08-27"),
    "7678180835652029730": ("沃什或成金银牛市最大考验，如何应对？", "2026-08-26"),
    "7677804082693672242": ("三重因素影响科技股，英伟达能否对冲逆风？", "2026-08-25"),
}
_model = None

@dataclass
class Meta:
    video_id: str
    title: str
    published_at: str | None
    share_url: str
    status: str = "discovered"
    media_url: str | None = None
    updated_at: str = ""


def now() -> str:
    return datetime.now(TZ).isoformat(timespec="seconds")


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), "utf-8")


def run(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=False, timeout=timeout)


def fetch_share_html(video_id: str) -> bytes:
    url = f"https://www.iesdouyin.com/share/video/{video_id}/"
    p = run([
        "curl", "-sS", "-L", "--compressed", "--retry", "2", "--max-time", "45",
        "-A", IOS_UA,
        "-H", "Accept-Language: zh-CN,zh;q=0.9,en;q=0.8",
        url,
    ], 75)
    if p.returncode != 0:
        raise RuntimeError(f"share curl failed rc={p.returncode}: {p.stderr.decode('utf-8','ignore')[:300]}")
    return p.stdout


def normalize_url(s: str) -> str:
    return (
        s.replace("\\u002F", "/")
         .replace("\\u002f", "/")
         .replace("\\/", "/")
         .replace("&amp;", "&")
    )


def extract_play_urls(raw_bytes: bytes) -> list[str]:
    raw = raw_bytes.decode("utf-8", "replace")
    blocks = []
    for pat in [
        r'"play_addr"\s*:\s*\{([^}]+)\}',
        r'\\"play_addr\\"\s*:\s*\{([^}]+)\}',
        r'"playAddr"\s*:\s*\{([^}]+)\}',
    ]:
        blocks.extend(re.findall(pat, raw, re.S))
    urls: list[str] = []
    for block in blocks:
        for u in re.findall(r'https?[^"\s<]+', block):
            u = normalize_url(u)
            if u.startswith("http") and u not in urls:
                urls.append(u)
    if not urls:
        # Fallback for layout changes: find media-looking URLs anywhere in SSR document.
        for u in re.findall(r'https?[^"\s<]+', raw):
            u = normalize_url(u)
            if ("/play/" in u or "/playwm/" in u or "douyinvod" in u or "bytecdn" in u) and u not in urls:
                urls.append(u)
    out: list[str] = []
    for u in urls:
        no_wm = u.replace("playwm", "play")
        if "ratio=" in no_wm:
            no_wm = re.sub(r"ratio=[A-Za-z0-9]+", "ratio=720p", no_wm)
        else:
            no_wm += ("&" if "?" in no_wm else "?") + "ratio=720p"
        for candidate in (no_wm, u):
            if candidate not in out:
                out.append(candidate)
    return out


def download_video(urls: list[str], out: Path, referer: str) -> str | None:
    for i, url in enumerate(urls[:12], 1):
        tmp = out.with_suffix(".part")
        p = run([
            "curl", "-f", "-sS", "-L", "--retry", "2", "--max-time", "120",
            "-A", IOS_UA,
            "-e", referer,
            "-o", str(tmp),
            url,
        ], 150)
        size = tmp.stat().st_size if tmp.exists() else 0
        print(f"[download] try={i} rc={p.returncode} bytes={size}")
        if p.returncode == 0 and size > 300_000:
            tmp.replace(out)
            return url
        tmp.unlink(missing_ok=True)
    return None


def extract_audio(video: Path, wav: Path) -> None:
    p = subprocess.run([
        "ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(wav)
    ], capture_output=True, timeout=120)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + p.stderr.decode("utf-8", "ignore")[-500:])


def transcribe(wav: Path):
    global _model
    if _model is None:
        print("[whisper] loading", WHISPER_MODEL)
        _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    prompt = (
        "Gary Norden market commentary. Scott Bessent. Kevin Warsh. Federal Reserve. "
        "Treasury yields. Bond yields. Jackson Hole. Gold. Silver. US dollar. Yen. "
        "Carry trade. Nvidia. Nasdaq. Liquidity. Leverage. Put options. Fiscal discipline. Debasement trade."
    )
    segments, info = _model.transcribe(
        str(wav), language="en", vad_filter=True, beam_size=5, initial_prompt=prompt
    )
    rows = []
    parts = []
    for s in segments:
        text = s.text.strip()
        if text:
            rows.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": text})
            parts.append(text)
    return " ".join(parts).strip(), rows


def process(video_id: str) -> dict:
    title, date = KNOWN.get(video_id, (f"Gary Norden {video_id}", None))
    share = f"https://www.iesdouyin.com/share/video/{video_id}/"
    meta = Meta(video_id, title, date, share, updated_at=now())
    d = DATA_DIR / video_id
    d.mkdir(parents=True, exist_ok=True)
    try:
        raw = fetch_share_html(video_id)
        (d / "share_debug.html").write_bytes(raw[:2_000_000])
        urls = extract_play_urls(raw)
        save_json(d / "media_candidates.json", urls)
        print(f"[share] {video_id} html={len(raw)} media_candidates={len(urls)}")
        if not urls:
            meta.status = "media_unavailable"
            save_json(d / "meta.json", asdict(meta))
            return asdict(meta)

        video = d / "video.mp4"
        media = download_video(urls, video, share)
        if not media:
            meta.status = "download_failed"
            save_json(d / "meta.json", asdict(meta))
            return asdict(meta)
        meta.media_url = media
        meta.status = "downloaded"
        save_json(d / "meta.json", asdict(meta))

        wav = d / "audio.wav"
        extract_audio(video, wav)
        text, rows = transcribe(wav)
        (d / "transcript_raw.txt").write_text(text + "\n", "utf-8")
        save_json(d / "segments.json", rows)
        meta.status = "transcribed"
        meta.updated_at = now()
        save_json(d / "meta.json", asdict(meta))
        wav.unlink(missing_ok=True)
        video.unlink(missing_ok=True)
    except Exception as e:
        meta.status = "error"
        meta.updated_at = now()
        save_json(d / "meta.json", asdict(meta))
        (d / "error.txt").write_text(f"{type(e).__name__}: {e}\n", "utf-8")
        print("[error]", video_id, type(e).__name__, e)
    return asdict(meta)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ids = SEEDS or list(KNOWN.keys())
    results = [process(v) for v in ids]
    save_json(DATA_DIR / "index.json", {"updated_at": now(), "mode": "seed-ssr-smoke", "videos": results})


if __name__ == "__main__":
    main()
