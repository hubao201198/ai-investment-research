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


def has_audio(video: Path) -> bool:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(video)],
        capture_output=True,
        timeout=60,
    )
    return p.returncode == 0 and bool(p.stdout.strip())


def select_muxed_video(urls: list[str], out: Path) -> tuple[str, int]:
    seen = set()
    errors = []
    for i, url in enumerate(urls[:16], 1):
        if not isinstance(url, str) or not url.startswith("http") or url in seen:
            continue
        seen.add(url)
        out.unlink(missing_ok=True)
        try:
            size = download(url, out)
            audio = has_audio(out)
            print(f"[candidate] {i}: {size/1024/1024:.2f} MB audio={audio}")
            if audio:
                return url, size
            errors.append(f"candidate {i}: no audio")
        except Exception as e:
            errors.append(f"candidate {i}: {type(e).__name__}: {e}")
            print(f"[candidate-error] {i}: {type(e).__name__}: {e}")
    out.unlink(missing_ok=True)
    raise RuntimeError("no muxed video with audio found; " + "; ".join(errors[-8:]))


def extract_audio(video: Path, wav: Path) -> None:
    p = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
        capture_output=True,
        timeout=240,
    )
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "ignore")[-1500:])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    items = json.loads(INPUT.read_text("utf-8"))
    model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
    index = []

    for item in items:
        vid = str(item["video_id"])
        d = OUT / vid
        d.mkdir(parents=True, exist_ok=True)
        video = d / "video.mp4"
        wav = d / "audio.wav"
        meta = {
            "video_id": vid,
            "title": item.get("title", ""),
            "published_at": item.get("published_at"),
            "source_url": item.get("source_url"),
            "status": "resolved",
            "updated_at": now(),
        }
        try:
            urls = item.get("video_urls") or ([item["video_url"]] if item.get("video_url") else [])
            selected_url, size = select_muxed_video(urls, video)
            meta["selected_stream_kind"] = next(
                (x.get("kind") for x in item.get("stream_candidates", []) if x.get("url") == selected_url),
                None,
            )
            print(f"[download] {vid}: selected muxed stream, {size/1024/1024:.2f} MB")
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
        except Exception as e:
            meta["status"] = "error"
            meta["error"] = f"{type(e).__name__}: {e}"
            print(f"[error] {vid}: {meta['error']}")
        finally:
            video.unlink(missing_ok=True)
            wav.unlink(missing_ok=True)
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
        index.append(meta)

    (OUT / "index.json").write_text(json.dumps({"updated_at": now(), "items": index}, ensure_ascii=False, indent=2), "utf-8")


if __name__ == "__main__":
    main()
