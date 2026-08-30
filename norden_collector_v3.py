from __future__ import annotations

import asyncio, base64, html as htmlmod, json, os, re, subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import unquote

import httpx
from faster_whisper import WhisperModel
from playwright.async_api import async_playwright

TZ=timezone(timedelta(hours=8))
PROFILE_URL=os.getenv('DOUYIN_PROFILE_URL','https://v.douyin.com/JaidNObikLo/')
DATA_DIR=Path(os.getenv('NORDEN_DATA_DIR','data/norden'))
MAX_VIDEOS=int(os.getenv('NORDEN_MAX_VIDEOS','10'))
WHISPER_MODEL=os.getenv('WHISPER_MODEL','small.en')
OPENAI_TEXT_MODEL=os.getenv('OPENAI_TEXT_MODEL','gpt-5-mini')
SEEDS=[x.strip() for x in os.getenv('NORDEN_SEED_VIDEO_IDS','').split(',') if x.strip()]
IOS_UA='Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1'
VIDEO_RE=re.compile(r'(?:/video/|aweme_id[=\"\': ]+)(\d{15,22})')
KNOWN={
 '7679312806105156916':('沃什放鹰金银大跌，风险管理的价值再次验证','2026-08-29'),
 '7678516691764071732':('贝森特把球踢给沃什，周五沃什会亮出底牌吗',None),
 '7678180835652029730':('沃什或成金银牛市最大考验，如何应对？','2026-08-26'),
 '7677804082693672242':('三重因素影响科技股，英伟达能否对冲逆风？','2026-08-25'),
}
_whisper=None

@dataclass
class Meta:
    video_id:str; title:str=''; published_at:str|None=None; page_url:str=''; share_url:str=''; media_url:str|None=None; status:str='discovered'; discovered_at:str=''

def now(): return datetime.now(TZ).isoformat(timespec='seconds')
def dump(p:Path,o): p.write_text(json.dumps(o,ensure_ascii=False,indent=2),'utf-8')

def normalize_url(u:str):
    return unquote(htmlmod.unescape(u.replace('\\u002F','/').replace('\\/','/')))

def parse_share_html(raw:str):
    title=''
    m=re.search(r'<title[^>]*>([^<]+)</title>',raw,re.I)
    if m: title=htmlmod.unescape(m.group(1)).strip()
    patterns=[r'"play_addr"\s*:\s*\{([^}]+)\}',r'\\"play_addr\\"\s*:\s*\{([^}]+)\}']
    block=''
    for pat in patterns:
        m=re.search(pat,raw,re.S)
        if m: block=m.group(1); break
    urls=[]
    if block:
        for u in re.findall(r'https?[^\"< ]+',block):
            u=normalize_url(u)
            if u.startswith('http') and u not in urls: urls.append(u)
    if not urls:
        for u in re.findall(r'https?[^\"\'< >]+',raw):
            u=normalize_url(u)
            if ('/play/' in u or '/playwm/' in u) and u not in urls: urls.append(u)
    return title,urls

def to_transcription_url(u:str):
    u=u.replace('playwm','play')
    if 'ratio=' in u: u=re.sub(r'ratio=[A-Za-z0-9]+','ratio=720p',u)
    else: u += ('&' if '?' in u else '?')+'ratio=720p'
    return u

async def ssr_media(vid:str):
    share=f'https://www.iesdouyin.com/share/video/{vid}/'
    async with httpx.AsyncClient(headers={'User-Agent':IOS_UA,'Accept':'text/html,*/*'},follow_redirects=True,timeout=45) as c:
        r=await c.get(share); print('[ssr]',vid,r.status_code,len(r.content))
        if r.status_code>=400 or not r.text: return '',[],share
        title,urls=parse_share_html(r.text)
        out=[]
        for u in urls:
            for x in (to_transcription_url(u),u):
                if x not in out: out.append(x)
        return title,out,share

async def download(meta:Meta,urls:list[str],out:Path):
    headers={'User-Agent':IOS_UA,'Referer':meta.share_url,'Accept':'*/*'}
    async with httpx.AsyncClient(headers=headers,follow_redirects=True,timeout=90) as c:
        for u in urls[:12]:
            tmp=out.with_suffix('.part'); total=0; ctype=''
            try:
                async with c.stream('GET',u) as r:
                    if r.status_code>=400: continue
                    ctype=r.headers.get('content-type','')
                    with tmp.open('wb') as f:
                        async for chunk in r.aiter_bytes(262144):
                            f.write(chunk); total+=len(chunk)
                            if total>150*1024*1024: raise RuntimeError('too large')
                if total>300000 and ('video' in ctype or 'octet' in ctype or total>1000000):
                    tmp.replace(out); print('[download]',meta.video_id,round(total/1048576,1),'MB'); return u
            except Exception as e: print('[warn] download',type(e).__name__,e)
            finally:
                if tmp.exists(): tmp.unlink(missing_ok=True)
    return None

async def discover_ids():
    ids=list(SEEDS)
    async with async_playwright() as p:
        b=await p.chromium.launch(headless=True,args=['--no-sandbox','--disable-blink-features=AutomationControlled'])
        page=await b.new_page(locale='zh-CN')
        try:
            await page.goto(PROFILE_URL,wait_until='domcontentloaded',timeout=45000); await page.wait_for_timeout(3500)
            for _ in range(5):
                sources=[await page.content(),page.url]
                try: sources.append('\n'.join(await page.eval_on_selector_all('a[href]','els=>els.map(e=>e.href)')))
                except: pass
                for s in sources:
                    for vid in VIDEO_RE.findall(s or ''):
                        if vid not in ids: ids.append(vid)
                await page.mouse.wheel(0,1800); await page.wait_for_timeout(1000)
        except Exception as e: print('[warn] discovery',e)
        finally: await b.close()
    return ids[:MAX_VIDEOS]

def audio(video:Path,wav:Path):
    subprocess.run(['ffmpeg','-y','-i',str(video),'-vn','-ac','1','-ar','16000','-c:a','pcm_s16le',str(wav)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

def transcribe(wav:Path):
    global _whisper
    if _whisper is None: _whisper=WhisperModel(WHISPER_MODEL,device='cpu',compute_type='int8')
    prompt='Gary Norden, Scott Bessent, Kevin Warsh, Federal Reserve, Treasury yields, bond yields, Jackson Hole, gold, silver, US dollar, yen, carry trade, Nvidia, Nasdaq, liquidity, leverage, put options, fiscal discipline, debasement trade.'
    segs,_=_whisper.transcribe(str(wav),language='en',vad_filter=True,beam_size=5,initial_prompt=prompt)
    rows=[]; parts=[]
    for s in segs:
        t=s.text.strip()
        if t: parts.append(t); rows.append({'start':round(s.start,2),'end':round(s.end,2),'text':t})
    return ' '.join(parts),rows

def analyze(meta:Meta,text:str):
    key=os.getenv('OPENAI_API_KEY','').strip()
    if not key: return None
    from openai import OpenAI
    ciro_path=Path(os.getenv('CIRO_LATEST_PATH','data/ciro/latest.json'))
    try: ciro=json.loads(ciro_path.read_text('utf-8')) if ciro_path.exists() else None
    except: ciro=None
    prompt=f'''你是 CIRO 的 Gary Norden 方法论校准器。只输出合法 JSON。先纠正明显的人名和金融术语识别错误，但不得改变观点。然后用中文提取：summary_zh,markets,horizon,direction,core_thesis,causal_chain,key_variables,risks,positioning,actions,triggers,falsifiers,forced_trading,liquidity_view。若有 CIRO 已锁定判断，再输出 ciro_comparison 和 decision_chain_updates，不要默认 Norden 一定正确。\n视频={json.dumps(asdict(meta),ensure_ascii=False)}\nCIRO={json.dumps(ciro,ensure_ascii=False) if ciro else 'null'}\n转写={text}'''
    r=OpenAI(api_key=key).responses.create(model=OPENAI_TEXT_MODEL,input=prompt)
    s=re.sub(r'^```json\s*|\s*```$','',r.output_text.strip(),flags=re.S)
    try: return json.loads(s)
    except: return {'raw_model_output':s}

async def process(vid:str):
    d=DATA_DIR/vid; d.mkdir(parents=True,exist_ok=True)
    known=KNOWN.get(vid,('',None)); meta=Meta(vid,known[0],known[1],f'https://www.douyin.com/video/{vid}',discovered_at=now())
    try:
        title,urls,share=await ssr_media(vid); meta.share_url=share
        if not meta.title: meta.title=title
        dump(d/'media_candidates.json',urls)
        if not urls: meta.status='media_unavailable'; dump(d/'meta.json',asdict(meta)); return asdict(meta)
        video=d/'video.mp4'; media=await download(meta,urls,video)
        if not media: meta.status='download_failed'; dump(d/'meta.json',asdict(meta)); return asdict(meta)
        meta.media_url=media; meta.status='downloaded'; dump(d/'meta.json',asdict(meta))
        wav=d/'audio.wav'
        try:
            audio(video,wav); text,rows=transcribe(wav)
            (d/'transcript_raw.txt').write_text(text+'\n','utf-8'); dump(d/'segments.json',rows)
            meta.status='transcribed'; dump(d/'meta.json',asdict(meta))
            a=analyze(meta,text)
            if a: dump(d/'analysis.json',a); meta.status='analyzed'; dump(d/'meta.json',asdict(meta))
        finally:
            wav.unlink(missing_ok=True); video.unlink(missing_ok=True)
    except Exception as e:
        meta.status='error'; dump(d/'meta.json',asdict(meta)); (d/'error.txt').write_text(f'{type(e).__name__}: {e}\n','utf-8'); print('[error]',vid,e)
    return asdict(meta)

async def main():
    DATA_DIR.mkdir(parents=True,exist_ok=True)
    ids=await discover_ids(); print('[ids]',ids); results=[]
    for vid in ids:
        results.append(await process(vid))
    dump(DATA_DIR/'index.json',{'updated_at':now(),'profile_url':PROFILE_URL,'videos':results})

if __name__=='__main__': asyncio.run(main())
