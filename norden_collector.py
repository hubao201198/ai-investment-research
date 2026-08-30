from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote

import httpx
from dateutil import parser as date_parser
from faster_whisper import WhisperModel
from playwright.async_api import BrowserContext, Page, async_playwright

PROFILE_URL = os.getenv("DOUYIN_PROFILE_URL", "https://v.douyin.com/JaidNObikLo/")
AUTHOR_KEYWORDS = [x.strip() for x in os.getenv("NORDEN_AUTHOR_KEYWORDS", "Gary Norden,交易员Gary Norden").split(",") if x.strip()]
LOOKBACK_DAYS = int(os.getenv("NORDEN_LOOKBACK_DAYS", "7"))
MAX_VIDEOS = int(os.getenv("NORDEN_MAX_VIDEOS", "12"))
DATA_DIR = Path(os.getenv("NORDEN_DATA_DIR", "data/norden"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small.en")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-5-mini")
TZ = timezone(timedelta(hours=8))

VIDEO_ID_RE = re.compile(r"(?:/video/|aweme_id[=\"': ]+)(\d{15,22})")
DATE_PATTERNS = [
    re.compile(r"(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})日?(?:\s+(\d{1,2}):(\d{2}))?"),
    re.compile(r"(\d{1,2})[-/.月](\d{1,2})日?(?:\s+(\d{1,2}):(\d{2}))?"),
]
MEDIA_HINTS = ("bytecdn", "douyinvod", "pstatp", "snssdk", ".mp4", "/video/tos/")


@dataclass
class VideoMeta:
    video_id: str
    page_url: str
    title: str = ""
    author: str = ""
    published_at: str | None = None
    discovered_at: str = ""
    source_profile: str = PROFILE_URL
    media_url: str | None = None
    status: str = "discovered"


def now_iso() -> str:
    return datetime.now(TZ).isoformat(timespec="seconds")


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def normalize_media_url(url: str) -> str:
    url = unquote(url.replace("\\u002F", "/").replace("\\/", "/"))
    return url.replace("&amp;", "&")


def candidate_video_ids(text: str) -> list[str]:
    ids = VIDEO_ID_RE.findall(text or "")
    out: list[str] = []
    seen = set()
    for vid in ids:
        if vid not in seen:
            seen.add(vid)
            out.append(vid)
    return out


def parse_date_from_text(text: str) -> datetime | None:
    text = text or ""
    now = datetime.now(TZ)
    for i, p in enumerate(DATE_PATTERNS):
        m = p.search(text)
        if not m:
            continue
        vals = [int(x) if x else None for x in m.groups()]
        if i == 0:
            y, mo, d, hh, mm = vals
        else:
            mo, d, hh, mm = vals
            y = now.year
        try:
            dt = datetime(y, mo, d, hh or 0, mm or 0, tzinfo=TZ)
            if i == 1 and dt > now + timedelta(days=2):
                dt = dt.replace(year=y - 1)
            return dt
        except ValueError:
            pass
    return None


def is_recent(dt: datetime | None) -> bool:
    if dt is None:
        return True
    return dt >= datetime.now(TZ) - timedelta(days=LOOKBACK_DAYS + 1)


async def apply_storage_state(context: BrowserContext) -> None:
    raw = os.getenv("DOUYIN_COOKIES_JSON", "").strip()
    b64 = os.getenv("DOUYIN_STORAGE_STATE_B64", "").strip()
    if b64:
        try:
            state = json.loads(base64.b64decode(b64).decode("utf-8"))
            cookies = state.get("cookies", [])
            if cookies:
                await context.add_cookies(cookies)
        except Exception as e:
            print(f"[warn] failed to load DOUYIN_STORAGE_STATE_B64: {e}")
    elif raw:
        try:
            cookies = json.loads(raw)
            if isinstance(cookies, dict):
                cookies = cookies.get("cookies", [])
            if cookies:
                await context.add_cookies(cookies)
        except Exception as e:
            print(f"[warn] failed to load DOUYIN_COOKIES_JSON: {e}")


async def discover_recent_video_ids(context: BrowserContext) -> list[str]:
    page = await context.new_page()
    try:
        print(f"[discover] open {PROFILE_URL}")
        await page.goto(PROFILE_URL, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(5000)
        ids: list[str] = []
        seen = set()
        for _ in range(8):
            html = await page.content()
            hrefs = await page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
            for source in [html, "\n".join(hrefs), page.url]:
                for vid in candidate_video_ids(source):
                    if vid not in seen:
                        seen.add(vid)
                        ids.append(vid)
            if len(ids) >= MAX_VIDEOS * 2:
                break
            await page.mouse.wheel(0, 1800)
            await page.wait_for_timeout(1800)
        print(f"[discover] found {len(ids)} candidate ids")
        return ids[: MAX_VIDEOS * 3]
    finally:
        await page.close()


async def inspect_video(context: BrowserContext, video_id: str) -> tuple[VideoMeta, list[str]]:
    urls_seen: list[str] = []
    page = await context.new_page()
    page.on("request", lambda req: urls_seen.append(req.url))
    page_url = f"https://www.douyin.com/video/{video_id}"
    meta = VideoMeta(video_id=video_id, page_url=page_url, discovered_at=now_iso())
    try:
        try:
            await page.goto(page_url, wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_timeout(7000)
        except Exception as e:
            print(f"[warn] douyin page failed {video_id}: {e}")

        text = ""
        try:
            text = await page.locator("body").inner_text(timeout=5000)
        except Exception:
            pass
        try:
            meta.title = (await page.title()).strip()
        except Exception:
            pass

        if text:
            lines = [x.strip() for x in text.splitlines() if x.strip()]
            for line in lines[:100]:
                if any(k.lower() in line.lower() for k in AUTHOR_KEYWORDS):
                    meta.author = line[:160]
                    break
            dt = parse_date_from_text(text)
            if dt:
                meta.published_at = dt.isoformat(timespec="minutes")
            if not meta.title or meta.title in {"抖音", "Douyin"}:
                meta.title = next((x for x in lines if 6 <= len(x) <= 120 and not any(k in x for k in AUTHOR_KEYWORDS)), meta.title)

        try:
            current_src = await page.eval_on_selector("video", "v => v.currentSrc || v.src || ''")
            if current_src:
                urls_seen.append(current_src)
        except Exception:
            pass
        try:
            resources = await page.evaluate("performance.getEntriesByType('resource').map(e => e.name)")
            urls_seen.extend(resources or [])
        except Exception:
            pass

        # Mobile share page is a useful fallback and often exposes direct bytecdn media URLs.
        share = await context.new_page()
        share.on("request", lambda req: urls_seen.append(req.url))
        try:
            await share.goto(f"https://www.iesdouyin.com/share/video/{video_id}/", wait_until="domcontentloaded", timeout=45000)
            await share.wait_for_timeout(3500)
            try:
                src = await share.eval_on_selector("video", "v => v.currentSrc || v.src || ''")
                if src:
                    urls_seen.append(src)
            except Exception:
                pass
            try:
                share_html = await share.content()
                for match in re.findall(r"https?:[^\"'<> ]+", share_html):
                    if any(h in match.lower() for h in MEDIA_HINTS):
                        urls_seen.append(match)
            except Exception:
                pass
        except Exception as e:
            print(f"[warn] share page failed {video_id}: {e}")
        finally:
            await share.close()

        media = []
        seen = set()
        for u in urls_seen:
            u = normalize_media_url(u)
            low = u.lower()
            if u.startswith("http") and any(h in low for h in MEDIA_HINTS) and u not in seen:
                seen.add(u)
                media.append(u)
        # Prefer master/original/play endpoints over thumbnails and tiny resources.
        media.sort(key=lambda u: ("playwm" in u, "image" in u or "jpeg" in u, -len(u)))
        return meta, media
    finally:
        await page.close()


async def download_media(context: BrowserContext, meta: VideoMeta, candidates: Iterable[str], out_path: Path) -> str | None:
    cookies = await context.cookies()
    cookie_map = {c["name"]: c["value"] for c in cookies}
    headers = {
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_6 like Mac OS X) AppleWebKit/605.1.15 Version/17.0 Mobile/15E148 Safari/604.1",
        "Referer": meta.page_url,
        "Accept": "*/*",
    }
    async with httpx.AsyncClient(headers=headers, cookies=cookie_map, follow_redirects=True, timeout=60) as client:
        for url in list(candidates)[:30]:
            tmp = out_path.with_suffix(".part")
            try:
                print(f"[download] try {url[:140]}")
                total = 0
                async with client.stream("GET", url) as r:
                    if r.status_code >= 400:
                        continue
                    ctype = r.headers.get("content-type", "")
                    with tmp.open("wb") as f:
                        async for chunk in r.aiter_bytes(1024 * 256):
                            f.write(chunk)
                            total += len(chunk)
                            if total > 250 * 1024 * 1024:
                                raise RuntimeError("media too large")
                if total >= 100_000 and ("video" in ctype or "octet-stream" in ctype or total > 500_000):
                    tmp.replace(out_path)
                    print(f"[download] success {meta.video_id}: {total/1024/1024:.1f} MB")
                    return url
            except Exception as e:
                print(f"[warn] candidate failed: {e}")
            finally:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
    return None


def extract_audio(video_path: Path, wav_path: Path) -> None:
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def transcribe_audio(wav_path: Path) -> tuple[str, list[dict]]:
    print(f"[transcribe] loading whisper {WHISPER_MODEL}")
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    prompt = (
        "Gary Norden market commentary. Financial terms include Scott Bessent, Kevin Warsh, Federal Reserve, "
        "Treasury yields, bond yields, Jackson Hole, gold, silver, dollar, yen, carry trade, Nvidia, Nasdaq, "
        "liquidity, leverage, puts, options, fiscal discipline and debasement trade."
    )
    segments, info = model.transcribe(str(wav_path), language="en", vad_filter=True, initial_prompt=prompt, beam_size=5)
    segs = []
    parts = []
    for s in segments:
        text = s.text.strip()
        if not text:
            continue
        parts.append(text)
        segs.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": text})
    return " ".join(parts).strip(), segs


def load_ciro_latest() -> dict | None:
    path = Path(os.getenv("CIRO_LATEST_PATH", "data/ciro/latest.json"))
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:
        return None


def analyze_with_openai(meta: VideoMeta, transcript: str, ciro: dict | None) -> dict | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    ciro_text = json.dumps(ciro, ensure_ascii=False) if ciro else "暂无已锁定的 CIRO 独立判断，跳过对照，只分析 Norden。"
    prompt = f"""
你是 CIRO 的 Gary Norden 方法论校准器。下面是 Gary Norden 本人视频的英文语音转写。

要求：
1. 先校正明显的语音识别人名和金融术语错误，但不得改变他的观点；给出 cleaned_transcript_en。
2. 用中文提取：市场、时间尺度、方向、核心驱动、关键变量、因果链、风险事件、仓位/对冲动作、触发条件、反证、哪些是事实判断/哪些是交易建议。
3. 特别识别 Norden 方法论：市场当前交易什么、仓位是否拥挤、杠杆、被迫买卖、流动性、风险管理、寻找反证。
4. 如果提供了 CIRO 独立判断，比较二者：一致点、不一致点、Norden 可能看到了 CIRO 漏掉的变量、CIRO 是否应修改决策链。不要因为 Norden 是专家就默认他正确。
5. 只输出合法 JSON，不要 markdown。

视频信息：{json.dumps(asdict(meta), ensure_ascii=False)}

CIRO 已锁定判断：{ciro_text}

原始转写：
{transcript}

JSON 字段：cleaned_transcript_en, summary_zh, markets, horizon, direction, core_thesis, causal_chain, key_variables, risks, positioning, actions, triggers, falsifiers, forced_trading, liquidity_view, ciro_comparison, decision_chain_updates, confidence_notes
""".strip()
    resp = client.responses.create(model=OPENAI_TEXT_MODEL, input=prompt)
    text = resp.output_text.strip()
    text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.S)
    try:
        return json.loads(text)
    except Exception:
        return {"raw_model_output": text}


def save_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), "utf-8")


async def process_one(context: BrowserContext, video_id: str) -> dict:
    item_dir = DATA_DIR / video_id
    item_dir.mkdir(parents=True, exist_ok=True)
    meta_path = item_dir / "meta.json"
    transcript_path = item_dir / "transcript_raw.txt"
    if transcript_path.exists() and meta_path.exists():
        print(f"[skip] {video_id} already transcribed")
        return json.loads(meta_path.read_text("utf-8"))

    meta, media_candidates = await inspect_video(context, video_id)
    author_blob = f"{meta.author} {meta.title}".lower()
    if AUTHOR_KEYWORDS and not any(k.lower() in author_blob for k in AUTHOR_KEYWORDS):
        print(f"[skip] {video_id}: author not confirmed ({meta.author!r}, {meta.title!r})")
        meta.status = "author_unconfirmed"
        save_json(meta_path, asdict(meta))
        return asdict(meta)

    dt = date_parser.isoparse(meta.published_at) if meta.published_at else None
    if not is_recent(dt):
        meta.status = "outside_lookback"
        save_json(meta_path, asdict(meta))
        return asdict(meta)

    video_path = item_dir / "video.mp4"
    media_url = await download_media(context, meta, media_candidates, video_path)
    if not media_url:
        meta.status = "media_unavailable"
        save_json(meta_path, asdict(meta))
        save_json(item_dir / "media_candidates.json", media_candidates[:50])
        return asdict(meta)

    meta.media_url = media_url
    meta.status = "downloaded"
    save_json(meta_path, asdict(meta))

    wav = item_dir / "audio.wav"
    try:
        extract_audio(video_path, wav)
        transcript, segments = transcribe_audio(wav)
        transcript_path.write_text(transcript + "\n", "utf-8")
        save_json(item_dir / "segments.json", segments)
        meta.status = "transcribed"
        save_json(meta_path, asdict(meta))

        analysis = analyze_with_openai(meta, transcript, load_ciro_latest())
        if analysis:
            save_json(item_dir / "analysis.json", analysis)
            cleaned = analysis.get("cleaned_transcript_en") if isinstance(analysis, dict) else None
            if cleaned:
                (item_dir / "transcript_cleaned_en.txt").write_text(cleaned.strip() + "\n", "utf-8")
            meta.status = "analyzed"
            save_json(meta_path, asdict(meta))
    finally:
        wav.unlink(missing_ok=True)
        if os.getenv("KEEP_VIDEO", "0") != "1":
            video_path.unlink(missing_ok=True)
    return asdict(meta)


async def main() -> int:
    ensure_dirs()
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
        )
        context = await browser.new_context(
            locale="zh-CN",
            timezone_id="Asia/Shanghai",
            viewport={"width": 1440, "height": 1000},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
        )
        await apply_storage_state(context)
        try:
            ids = await discover_recent_video_ids(context)
            # Known IDs can be injected as a fallback/seed while discovery is being hardened.
            seeds = [x.strip() for x in os.getenv("NORDEN_SEED_VIDEO_IDS", "").split(",") if x.strip()]
            ids = list(dict.fromkeys(seeds + ids))[:MAX_VIDEOS]
            if not ids:
                print("[error] no Norden video ids discovered")
                return 2
            results = []
            for vid in ids:
                try:
                    results.append(await process_one(context, vid))
                except Exception as e:
                    print(f"[error] {vid}: {type(e).__name__}: {e}")
            index = {
                "updated_at": now_iso(),
                "profile_url": PROFILE_URL,
                "lookback_days": LOOKBACK_DAYS,
                "videos": results,
            }
            save_json(DATA_DIR / "index.json", index)
        finally:
            await context.close()
            await browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
