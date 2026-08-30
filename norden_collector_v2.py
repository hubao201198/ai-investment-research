from __future__ import annotations

import asyncio, base64, json, os, re, subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import unquote

import httpx
from faster_whisper import WhisperModel
from playwright.async_api import BrowserContext, async_playwright

TZ = timezone(timedelta(hours=8))
PROFILE_URL = os.getenv('DOUYIN_PROFILE_URL', 'https://v.douyin.com/JaidNObikLo/')
DATA_DIR = Path(os.getenv('NORDEN_DATA_DIR', 'data/norden'))
MAX_VIDEOS = int(os.getenv('NORDEN_MAX_VIDEOS', '12'))
WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'small.en')
OPENAI_TEXT_MODEL = os.getenv('OPENAI_TEXT_MODEL', 'gpt-5-mini')
TRUSTED_IDS = [x.strip() for x in os.getenv('NORDEN_SEED_VIDEO_IDS', '').split(',') if x.strip()]
AUTHOR_KEYWORDS = [x.strip().lower() for x in os.getenv('NORDEN_AUTHOR_KEYWORDS', 'Gary Norden,交易员Gary Norden').split(',') if x.strip()]
VIDEO_ID_RE = re.compile(r'(?:/video/|aweme_id[=\"\': ]+)(\d{15,22})')
MEDIA_HINTS = ('bytecdn', 'douyinvod', 'pstatp', 'snssdk', '.mp4', '/video/tos/', 'video_id=')
KNOWN = {
    '7679312806105156916': {'title': '沃什放鹰金银大跌，风险管理的价值再次验证', 'published_at': '2026-08-29T00:00+08:00'},
    '7678516691764071732': {'title': '贝森特把球踢给沃什，周五沃什会亮出底牌吗', 'published_at': None},
    '7678180835652029730': {'title': '沃什或成金银牛市最大考验，如何应对？', 'published_at': '2026-08-26T00:00+08:00'},
    '7677804082693672242': {'title': '三重因素影响科技股，英伟达能否对冲逆风？', 'published_at': '2026-08-25T00:00+08:00'},
}
_whisper = None

@dataclass
class VideoMeta:
    video_id: str
    page_url: str
    title: str = ''
    author: str = ''
    published_at: str | None = None
    discovered_at: str = ''
    source_profile: str = PROFILE_URL
    media_url: str | None = None
    status: str = 'discovered'
    trusted_source: bool = False


def now_iso(): return datetime.now(TZ).isoformat(timespec='seconds')
def save_json(path: Path, obj): path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), 'utf-8')
def norm_url(u: str): return unquote(u.replace('\\u002F','/').replace('\\/','/')).replace('&amp;','&')

def ids_from(text: str):
    out=[]
    for x in VIDEO_ID_RE.findall(text or ''):
        if x not in out: out.append(x)
    return out

async def add_optional_cookies(ctx: BrowserContext):
    raw=os.getenv('DOUYIN_COOKIES_JSON','').strip(); b64=os.getenv('DOUYIN_STORAGE_STATE_B64','').strip()
    try:
        if b64:
            raw=base64.b64decode(b64).decode('utf-8')
        if raw:
            obj=json.loads(raw); cookies=obj.get('cookies',[]) if isinstance(obj,dict) else obj
            if cookies: await ctx.add_cookies(cookies)
    except Exception as e: print('[warn] cookies:',e)

async def discover(ctx: BrowserContext):
    ids=list(TRUSTED_IDS)
    page=await ctx.new_page()
    try:
        print('[discover]',PROFILE_URL)
        await page.goto(PROFILE_URL,wait_until='domcontentloaded',timeout=45000)
        await page.wait_for_timeout(4000)
        for _ in range(6):
            html=await page.content()
            hrefs=await page.eval_on_selector_all('a[href]','els=>els.map(e=>e.href)')
            for src in [html,'\n'.join(hrefs),page.url]:
                for vid in ids_from(src):
                    if vid not in ids: ids.append(vid)
            await page.mouse.wheel(0,1600); await page.wait_for_timeout(1200)
    except Exception as e: print('[warn] profile discovery:',e)
    finally: await page.close()
    return ids[:MAX_VIDEOS]

async def inspect_page(ctx: BrowserContext, url: str, seen_urls: list[str]):
    page=await ctx.new_page(); page.on('request',lambda r: seen_urls.append(r.url))
    text=''; title=''; html=''
    try:
        await page.goto(url,wait_until='domcontentloaded',timeout=45000)
        await page.wait_for_timeout(5000)
        try: title=(await page.title()).strip()
        except: pass
        try: text=await page.locator('body').inner_text(timeout=4000)
        except: pass
        try: html=await page.content()
        except: pass
        try:
            src=await page.eval_on_selector('video','v=>v.currentSrc||v.src||""')
            if src: seen_urls.append(src)
        except: pass
        try:
            resources=await page.evaluate("performance.getEntriesByType('resource').map(e=>e.name)")
            seen_urls.extend(resources or [])
        except: pass
        for m in re.findall(r'https?:[^\"\'<> ]+',html):
            if any(h in m.lower() for h in MEDIA_HINTS): seen_urls.append(m)
    except Exception as e: print('[warn] page',url,e)
    finally: await page.close()
    return title,text

async def inspect_video(ctx: BrowserContext, vid: str):
    trusted=vid in TRUSTED_IDS
    known=KNOWN.get(vid,{})
    meta=VideoMeta(vid,f'https://www.douyin.com/video/{vid}',known.get('title',''),
                   '交易员 Gary Norden' if trusted else '',known.get('published_at'),now_iso(),trusted_source=trusted)
    seen=[]
    pages=[
        f'https://www.douyin.com/video/{vid}',
        f'https://jingxuan.douyin.com/m/video/{vid}',
        f'https://www.iesdouyin.com/share/video/{vid}/',
    ]
    for u in pages:
        title,text=await inspect_page(ctx,u,seen)
        if not meta.title and title and '记录美好生活' not in title: meta.title=title
        if not meta.author:
            for line in (text or '').splitlines()[:150]:
                if any(k in line.lower() for k in AUTHOR_KEYWORDS): meta.author=line.strip()[:160]; break
    candidates=[]
    for u in seen:
        u=norm_url(u); low=u.lower()
        if u.startswith('http') and any(h in low for h in MEDIA_HINTS) and u not in candidates:
            candidates.append(u)
    candidates.sort(key=lambda u: ('playwm' in u.lower(), 'image' in u.lower() or 'jpeg' in u.lower(), -len(u)))
    print(f'[inspect] {vid} trusted={trusted} media_candidates={len(candidates)} title={meta.title!r}')
    return meta,candidates

async def download(ctx: BrowserContext, meta: VideoMeta, urls: list[str], out: Path):
    cookies={c['name']:c['value'] for c in await ctx.cookies()}
    headers={'User-Agent':'Mozilla/5.0 (iPhone; CPU iPhone OS 17_6 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1','Referer':meta.page_url,'Accept':'*/*'}
    async with httpx.AsyncClient(headers=headers,cookies=cookies,follow_redirects=True,timeout=60) as client:
        for u in urls[:40]:
            tmp=out.with_suffix('.part'); total=0; ctype=''
            try:
                async with client.stream('GET',u) as r:
                    if r.status_code>=400: continue
                    ctype=r.headers.get('content-type','')
                    with tmp.open('wb') as f:
                        async for chunk in r.aiter_bytes(262144):
                            f.write(chunk); total+=len(chunk)
                            if total>250*1024*1024: raise RuntimeError('too large')
                if total>300000 and ('video' in ctype or 'octet' in ctype or total>1000000):
                    tmp.replace(out); print(f'[download] {meta.video_id} {total/1048576:.1f}MB'); return u
            except Exception as e: print('[warn] media candidate:',type(e).__name__,e)
            finally:
                if tmp.exists(): tmp.unlink(missing_ok=True)
    return None

def extract_audio(video: Path,wav: Path):
    subprocess.run(['ffmpeg','-y','-i',str(video),'-vn','-ac','1','-ar','16000','-c:a','pcm_s16le',str(wav)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

def transcribe(wav: Path):
    global _whisper
    if _whisper is None:
        print('[whisper] loading',WHISPER_MODEL); _whisper=WhisperModel(WHISPER_MODEL,device='cpu',compute_type='int8')
    initial='Gary Norden, Scott Bessent, Kevin Warsh, Federal Reserve, Treasury yields, bond yields, Jackson Hole, gold, silver, US dollar, yen, carry trade, Nvidia, Nasdaq, liquidity, leverage, put options, fiscal discipline, debasement trade.'
    segments,_=_whisper.transcribe(str(wav),language='en',vad_filter=True,beam_size=5,initial_prompt=initial)
    segs=[]; parts=[]
    for s in segments:
        t=s.text.strip()
        if t: parts.append(t); segs.append({'start':round(s.start,2),'end':round(s.end,2),'text':t})
    return ' '.join(parts),segs

def load_ciro():
    p=Path(os.getenv('CIRO_LATEST_PATH','data/ciro/latest.json'))
    try: return json.loads(p.read_text('utf-8')) if p.exists() else None
    except: return None

def analyze(meta,transcript):
    key=os.getenv('OPENAI_API_KEY','').strip()
    if not key: return None
    from openai import OpenAI
    ciro=load_ciro(); client=OpenAI(api_key=key)
    prompt=f'''你是 CIRO 的 Gary Norden 方法论校准器。只输出合法 JSON。\n先只纠正明显的人名/金融术语识别错误，给出 cleaned_transcript_en，不得改变观点。\n然后中文提取 markets,horizon,direction,core_thesis,causal_chain,key_variables,risks,positioning,actions,triggers,falsifiers,forced_trading,liquidity_view。\n若 CIRO 已锁定判断存在，再输出 ciro_comparison 与 decision_chain_updates；不要默认 Norden 一定正确。\n视频：{json.dumps(asdict(meta),ensure_ascii=False)}\nCIRO：{json.dumps(ciro,ensure_ascii=False) if ciro else 'null'}\n原始转写：{transcript}'''
    r=client.responses.create(model=OPENAI_TEXT_MODEL,input=prompt); txt=r.output_text.strip()
    txt=re.sub(r'^```json\s*|\s*```$','',txt,flags=re.S)
    try: return json.loads(txt)
    except: return {'raw_model_output':txt}

async def process(ctx,vid):
    d=DATA_DIR/vid; d.mkdir(parents=True,exist_ok=True)
    meta,cands=await inspect_video(ctx,vid); save_json(d/'media_candidates.json',cands[:80])
    trusted=meta.trusted_source
    author_ok=any(k in f'{meta.author} {meta.title}'.lower() for k in AUTHOR_KEYWORDS)
    if not trusted and not author_ok:
        meta.status='author_unconfirmed'; save_json(d/'meta.json',asdict(meta)); return asdict(meta)
    video=d/'video.mp4'; url=await download(ctx,meta,cands,video)
    if not url:
        meta.status='media_unavailable'; save_json(d/'meta.json',asdict(meta)); return asdict(meta)
    meta.media_url=url; meta.status='downloaded'; save_json(d/'meta.json',asdict(meta))
    wav=d/'audio.wav'
    try:
        extract_audio(video,wav); text,segs=transcribe(wav)
        (d/'transcript_raw.txt').write_text(text+'\n','utf-8'); save_json(d/'segments.json',segs)
        meta.status='transcribed'; save_json(d/'meta.json',asdict(meta))
        a=analyze(meta,text)
        if a:
            save_json(d/'analysis.json',a)
            if isinstance(a,dict) and a.get('cleaned_transcript_en'):
                (d/'transcript_cleaned_en.txt').write_text(a['cleaned_transcript_en'].strip()+'\n','utf-8')
            meta.status='analyzed'; save_json(d/'meta.json',asdict(meta))
    finally:
        wav.unlink(missing_ok=True)
        if os.getenv('KEEP_VIDEO','0')!='1': video.unlink(missing_ok=True)
    return asdict(meta)

async def main():
    DATA_DIR.mkdir(parents=True,exist_ok=True)
    async with async_playwright() as p:
        browser=await p.chromium.launch(headless=True,args=['--disable-blink-features=AutomationControlled','--no-sandbox'])
        ctx=await browser.new_context(locale='zh-CN',timezone_id='Asia/Shanghai',viewport={'width':1440,'height':1000},user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36')
        await add_optional_cookies(ctx)
        try:
            ids=await discover(ctx); results=[]
            for vid in ids:
                try: results.append(await process(ctx,vid))
                except Exception as e: print('[error]',vid,type(e).__name__,e)
            save_json(DATA_DIR/'index.json',{'updated_at':now_iso(),'profile_url':PROFILE_URL,'videos':results})
        finally:
            await ctx.close(); await browser.close()
    return 0

if __name__=='__main__': raise SystemExit(asyncio.run(main()))
