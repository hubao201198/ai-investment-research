from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from faster_whisper import WhisperModel
from playwright.async_api import async_playwright

TZ = timezone(timedelta(hours=8))
DATA_DIR = Path(os.getenv("NORDEN_DATA_DIR", "data/norden"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small.en")
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
    source_url: str
    status: str = "discovered"
    video_url: str | None = None
    updated_at: str = ""
    diagnostics: dict | None = None


def now():
    return datetime.now(TZ).isoformat(timespec="seconds")


def dump(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), "utf-8")


def first_http(urls):
    if not isinstance(urls, list):
        return None
    return next((u for u in urls if isinstance(u, str) and u.startswith("http")), None)


def extract_video_url(payload: dict | None):
    if not isinstance(payload, dict):
        return None
    aweme = payload.get("aweme_detail")
    if not isinstance(aweme, dict):
        # some responses wrap payload
        for v in payload.values():
            if isinstance(v, dict):
                u = extract_video_url(v)
                if u:
                    return u
        return None
    video = aweme.get("video") or {}
    bitrates = video.get("bit_rate") or []
    ranked = []
    for br in bitrates:
        if not isinstance(br, dict):
            continue
        addr = br.get("play_addr") or {}
        u = first_http(addr.get("url_list"))
        if u:
            ranked.append((int(br.get("bit_rate") or 0), u))
    if ranked:
        ranked.sort(reverse=True)
        return ranked[0][1]
    for k in ("play_addr_h264", "play_addr", "download_addr", "play_addr_265"):
        addr = video.get(k) or {}
        u = first_http(addr.get("url_list")) if isinstance(addr, dict) else None
        if u:
            return u
    return None


def extract_title(payload: dict | None):
    if not isinstance(payload, dict):
        return ""
    aweme = payload.get("aweme_detail")
    if isinstance(aweme, dict):
        return str(aweme.get("desc") or "").strip()
    for v in payload.values():
        if isinstance(v, dict):
            t = extract_title(v)
            if t:
                return t
    return ""


async def resolve_video(video_id: str):
    source_url = f"https://www.douyin.com/discover?modal_id={video_id}"
    diagnostics = {"api": [], "media": [], "landed": None, "body_text": ""}
    payload = None
    direct_media = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-features=IsolateOrigins,site-per-process,TranslateUI",
            ],
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            locale="zh-CN",
            timezone_id="Asia/Shanghai",
        )
        await context.set_extra_http_headers({
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "sec-ch-ua": '"Chromium";v="149", "Not_A Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
        })
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            window.chrome = {runtime: {}, loadTimes: function(){}, csi: function(){}};
            Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
            Object.defineProperty(navigator, 'languages', {get: () => ['zh-CN','zh','en']});
        """)
        page = await context.new_page()

        async def handle_response(response):
            nonlocal payload
            try:
                url = response.url
                status = response.status
                if "/aweme/v1/web/aweme/detail/" in url:
                    diagnostics["api"].append({"status": status, "url": url[:600]})
                    if status == 200 and payload is None:
                        try:
                            data = await response.json()
                            if isinstance(data, dict) and data.get("aweme_detail"):
                                payload = data
                        except Exception as e:
                            diagnostics["api"][-1]["json_error"] = str(e)[:200]
                if status in (200, 206) and (
                    "douyinvod" in url or "aweme.snssdk" in url or ".mp4" in url or ".m3u8" in url
                ):
                    direct_media.append(url)
                    if len(diagnostics["media"]) < 15:
                        diagnostics["media"].append({"status": status, "url": url[:600]})
            except Exception:
                pass

        page.on("response", handle_response)
        try:
            await page.goto(source_url, wait_until="domcontentloaded", timeout=60000)
            diagnostics["landed"] = page.url
            await page.wait_for_timeout(10000)
            try:
                diagnostics["body_text"] = (await page.locator("body").inner_text(timeout=3000))[:2000]
            except Exception:
                pass
            # trigger lazy playback/network activity
            for selector in ["video", '[class*="play"]', "button"]:
                try:
                    await page.click(selector, timeout=1200, force=True)
                    await page.wait_for_timeout(3000)
                    if payload or direct_media:
                        break
                except Exception:
                    continue
            await page.wait_for_timeout(5000)
            if payload is None:
                # Current browser session may be allowed to call detail API without manually generating signatures.
                # Do this in-page so browser cookies/fingerprint are reused.
                try:
                    res = await page.evaluate(f"""
                    async () => {{
                      try {{
                        const r = await fetch('/aweme/v1/web/aweme/detail/?aweme_id={video_id}&aid=6383', {{credentials:'include'}});
                        return {{status:r.status, text:await r.text()}};
                      }} catch(e) {{ return {{status:0,text:String(e)}}; }}
                    }}
                    """)
                    diagnostics["direct_api"] = {"status": res.get("status"), "length": len(res.get("text", ""))}
                    if res.get("status") == 200 and res.get("text"):
                        try:
                            data = json.loads(res["text"])
                            if data.get("aweme_detail"):
                                payload = data
                        except Exception:
                            pass
                except Exception as e:
                    diagnostics["direct_api_error"] = str(e)[:300]
        finally:
            await context.close()
            await browser.close()

    video_url = extract_video_url(payload)
    if not video_url and direct_media:
        video_url = direct_media[0]
    title = extract_title(payload)
    return video_url, title, diagnostics


async def download(url: str, referer: str, out: Path):
    headers = {
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_6 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1",
        "Referer": referer,
        "Accept": "*/*",
    }
    async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=120) as c:
        async with c.stream("GET", url) as r:
            r.raise_for_status()
            total = 0
            with out.open("wb") as f:
                async for chunk in r.aiter_bytes(262144):
                    f.write(chunk)
                    total += len(chunk)
                    if total > 250 * 1024 * 1024:
                        raise RuntimeError("video exceeds 250 MB")
    return total


def extract_audio(video: Path, wav: Path):
    p = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
        capture_output=True,
        timeout=180,
    )
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "ignore")[-1000:])


def transcribe(wav: Path):
    global _model
    if _model is None:
        print("[whisper] loading", WHISPER_MODEL)
        _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    prompt = (
        "Gary Norden. Scott Bessent. Kevin Warsh. Federal Reserve. Treasury yields. Bond yields. "
        "Jackson Hole. Gold. Silver. US dollar. Yen. Carry trade. Nvidia. Nasdaq. Liquidity. "
        "Leverage. Put options. Fiscal discipline. Debasement trade."
    )
    segs, _ = _model.transcribe(str(wav), language="en", vad_filter=True, beam_size=5, initial_prompt=prompt)
    rows, parts = [], []
    for s in segs:
        text = s.text.strip()
        if text:
            rows.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": text})
            parts.append(text)
    return " ".join(parts).strip(), rows


async def process(video_id: str):
    title, date = KNOWN.get(video_id, (f"Gary Norden {video_id}", None))
    source = f"https://www.douyin.com/discover?modal_id={video_id}"
    d = DATA_DIR / video_id
    d.mkdir(parents=True, exist_ok=True)
    meta = Meta(video_id, title, date, source, updated_at=now())
    try:
        video_url, live_title, diag = await resolve_video(video_id)
        meta.diagnostics = diag
        dump(d / "browser_diagnostics.json", diag)
        if live_title:
            meta.title = live_title
        if not video_url:
            meta.status = "media_unavailable"
            dump(d / "meta.json", asdict(meta))
            return asdict(meta)
        meta.video_url = video_url
        meta.status = "resolved"
        dump(d / "meta.json", asdict(meta))
        video = d / "video.mp4"
        size = await download(video_url, source, video)
        print(f"[download] {video_id} {size/1024/1024:.2f} MB")
        wav = d / "audio.wav"
        extract_audio(video, wav)
        text, rows = transcribe(wav)
        (d / "transcript_raw.txt").write_text(text + "\n", "utf-8")
        dump(d / "segments.json", rows)
        meta.status = "transcribed"
        meta.updated_at = now()
        dump(d / "meta.json", asdict(meta))
        wav.unlink(missing_ok=True)
        video.unlink(missing_ok=True)
    except Exception as e:
        meta.status = "error"
        meta.updated_at = now()
        dump(d / "meta.json", asdict(meta))
        (d / "browser_error.txt").write_text(f"{type(e).__name__}: {e}\n", "utf-8")
        print("[error]", video_id, type(e).__name__, e)
    return asdict(meta)


async def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ids = SEEDS or list(KNOWN.keys())
    results = []
    for vid in ids:
        results.append(await process(vid))
    dump(DATA_DIR / "index.json", {"updated_at": now(), "mode": "discover-browser", "videos": results})


if __name__ == "__main__":
    asyncio.run(main())
