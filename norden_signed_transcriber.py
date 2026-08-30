from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from faster_whisper import WhisperModel

TZ = timezone(timedelta(hours=8))
INPUT = Path(os.getenv("NORDEN_RESOLVED_JSON", "/tmp/norden_urls.json"))
OUT = Path(os.getenv("NORDEN_ARTIFACT_DIR", "/tmp/norden_transcripts"))
MODEL_NAME = os.getenv("WHISPER_MODEL", "small.en")

PROMPT = (
    "Gary Norden market commentary. Scott Bessent. Kevin Warsh. Federal Reserve. Treasury yields. "
    "Bond yields. Jackson Hole. Gold. Silver. US dollar. Yen. Carry trade. Nvidia. Nasdaq. "
    "Liquidity. Leverage. Put options. Fiscal discipline. Debasement trade. Risk management."
)


def now() -> str:
    return datetime.now(TZ).isoformat(timespec="seconds")


def extract_audio(video: Path, wav: Path) -> None:
    p = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
        capture_output=True,
        timeout=240,
    )
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "ignore")[-1500:])


def download(url: str, out: Path) -> int:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
        "Referer": "https://www.douyin.com/",
        "Accept": "*/*",
    }
    total = 0
    with httpx.Client(headers=headers, follow_redirects=True, timeout=120) as client:
        with client.stream("GET", url) as r:
            r.raise_for_status()
            with out.open("wb") as f:
                for chunk in r.iter_bytes(1024 * 512):
                    f.write(chunk)
                    total += len(chunk)
                    if total > 250 * 1024 * 1024:
                        raise RuntimeError("video larger than 250 MB")
    return total


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    items = json.loads(INPUT.read_text("utf-8"))
    model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
    index = []

    for item in items:
        vid = str(item["video_id"])
        d = OUT / vid
        d.mkdir(parents=True, exist_ok=True)
        meta = {
            "video_id": vid,
            "title": item.get("title", ""),
            "published_at": item.get("published_at"),
            "source_url": item.get("source_url"),
            "status": "resolved",
            "updated_at": now(),
        }
        try:
            video = d / "video.mp4"
            wav = d / "audio.wav"
            size = download(item["video_url"], video)
            print(f"[download] {vid}: {size/1024/1024:.2f} MB")
            extract_audio(video, wav)
            segments, info = model.transcribe(
                str(wav),
                language="en",
                vad_filter=True,
                beam_size=5,
                initial_prompt=PROMPT,
                word_timestamps=False,
            )
            rows = []
            text_parts = []
            for s in segments:
                text = s.text.strip()
                if not text:
                    continue
                text_parts.append(text)
                rows.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": text})
            transcript = " ".join(text_parts).strip()
            (d / "transcript_raw.txt").write_text(transcript + "\n", "utf-8")
            (d / "segments.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
            meta.update({
                "status": "transcribed",
                "language": getattr(info, "language", "en"),
                "duration_seconds": round(getattr(info, "duration", 0.0) or 0.0, 2),
                "segment_count": len(rows),
                "transcript_chars": len(transcript),
                "updated_at": now(),
            })
            video.unlink(missing_ok=True)
            wav.unlink(missing_ok=True)
        except Exception as e:
            meta["status"] = "error"
            meta["error"] = f"{type(e).__name__}: {e}"
            print(f"[error] {vid}: {meta['error']}")
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
        index.append(meta)

    (OUT / "index.json").write_text(json.dumps({"updated_at": now(), "items": index}, ensure_ascii=False, indent=2), "utf-8")


if __name__ == "__main__":
    main()
