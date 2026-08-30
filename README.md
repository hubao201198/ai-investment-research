# AI Investment Research / CIRO Norden Collector

自动采集 Gary Norden 的抖音公开视频，提取视频音轨，生成英文逐字稿，并在配置模型 API 后生成结构化中文观点与 CIRO 对照结果。

## 当前链路

1. 打开 Norden 抖音个人名片：`https://v.douyin.com/JaidNObikLo/`
2. 自动发现最近作品 ID，并保留已确认的新视频 ID 作为首轮兜底。
3. 用 Playwright 打开本人作品页与移动分享页，监听播放器真实媒体请求。
4. 下载公开视频到临时目录。
5. 用 ffmpeg 提取 16k 单声道音频。
6. 用 faster-whisper `small.en` 本地转写，内置 Gary Norden / Scott Bessent / Kevin Warsh / Treasury yields / gold / silver 等金融词提示。
7. 保存：
   - `transcript_raw.txt`：原始英文转写
   - `segments.json`：带时间戳片段
   - `meta.json`：标题、时间、来源、状态
   - `analysis.json`：若配置 OPENAI_API_KEY，则生成 Norden 观点结构化分析和 CIRO 对照
   - `transcript_cleaned_en.txt`：模型只做术语纠错后的英文稿
8. GitHub Actions 每小时第 7 分钟执行一次并把新文稿提交回仓库。

## CIRO 对照规则

若仓库存在 `data/ciro/latest.json`，采集器会把它视为“在读取 Norden 观点之前已经锁定的 CIRO 独立判断”，随后才进行对照，避免答案污染。

分析字段包括：

- 市场与时间尺度
- 方向
- 核心主张
- 因果链
- 关键变量
- 风险事件
- 仓位/对冲动作
- 触发条件与反证
- 杠杆与被迫交易
- 流动性判断
- Norden 与 CIRO 的一致/分歧
- CIRO 决策链需要新增或修正的变量

## GitHub Secrets（可选）

公开页面可直接尝试抓取。若抖音要求登录态，可配置以下任意一种：

- `DOUYIN_COOKIES_JSON`：Playwright cookie 数组 JSON
- `DOUYIN_STORAGE_STATE_B64`：Playwright storage state JSON 的 base64

若希望自动做观点分析与 CIRO 对照：

- `OPENAI_API_KEY`

没有 API Key 时，视频抓取和本地 Whisper 转写仍可运行。

## 手工运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
python norden_collector.py
```

## 重要说明

2026 年的 yt-dlp Douyin extractor 仍有“web detail API 返回空内容”等公开失效报告，因此本项目不把 yt-dlp 作为主链路，而是优先通过真实浏览器打开公开页面并监听媒体资源。抖音风控会变化，因此采集器保留移动分享页、已确认作品 ID、登录 cookie 三层兜底。
