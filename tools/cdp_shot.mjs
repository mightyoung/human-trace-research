#!/usr/bin/env node
/**
 * 用 CDP 截页面（支持 <video> 与 canvas 叠加层）。
 *
 * 为什么不用 `chrome --screenshot`：
 *   1) 老 headless 不解码视频 → 六路全黑；
 *   2) 即使 `--headless=new` 出画，canvas 叠加层的像素也常被丢掉（截不到检测框）；
 *   3) `--virtual-time-budget` 会快进时钟，媒体 seek 到长视频中段时来不及取帧。
 * CDP 的 Page.captureScreenshot 走 compositor，且用真实时间等待，三个问题都避开。
 *
 * 用法：
 *   node tools/cdp_shot.mjs "<url>" "<out.png>" [waitMs]
 * 例：
 *   node tools/cdp_shot.mjs "http://127.0.0.1:8025/#/?t=1005&sel=305:140" docs/ops/shots/a.png 12000
 */
import { spawn } from 'node:child_process'
import { writeFileSync, mkdirSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { tmpdir } from 'node:os'

const [url, out, waitArg, evalJs] = process.argv.slice(2)
if (!url || !out) {
  console.error('用法: node tools/cdp_shot.mjs <url> <out.png> [waitMs]')
  process.exit(2)
}
const waitMs = Number(waitArg || 12000)
const PORT = 9333 + Math.floor(Math.random() * 200)
const CHROME = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe'
// SHOT_PROFILE=<path> 复用同一浏览器档案：用来测 localStorage 这类跨页面加载的状态
const profile = process.env.SHOT_PROFILE
  ? resolve(process.env.SHOT_PROFILE)
  : resolve(tmpdir(), `cdp_shot_${Date.now()}_${process.pid}`)

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

const proc = spawn(CHROME, [
  `--remote-debugging-port=${PORT}`,
  '--remote-allow-origins=*',
  '--headless=new',
  '--no-proxy-server',
  '--hide-scrollbars',
  `--window-size=${process.env.SHOT_W || 1600},${process.env.SHOT_H || 1000}`,
  '--autoplay-policy=no-user-gesture-required',
  '--disable-background-timer-throttling',
  '--disable-renderer-backgrounding',
  '--mute-audio',
  `--user-data-dir=${profile}`,
  'about:blank'
], { stdio: 'ignore' })

async function waitBrowser() {
  for (let i = 0; i < 120; i++) {
    try {
      const r = await fetch(`http://127.0.0.1:${PORT}/json/version`)
      if (r.ok) return await r.json()
    } catch { /* 还没起来 */ }
    await sleep(250)
  }
  throw new Error('Chrome 调试端口未就绪')
}

class Cdp {
  constructor(wsUrl) {
    this.ws = new WebSocket(wsUrl)
    this.id = 0
    this.pending = new Map()
    this.ready = new Promise((res, rej) => {
      this.ws.onopen = res
      this.ws.onerror = (e) => rej(new Error('ws error'))
    })
    this.ws.onmessage = (ev) => {
      let m
      try { m = JSON.parse(ev.data) } catch { return }
      if (m.id && this.pending.has(m.id)) {
        const { res, rej } = this.pending.get(m.id)
        this.pending.delete(m.id)
        m.error ? rej(new Error(`${m.error.message}`)) : res(m.result)
      }
    }
  }
  async send(method, params = {}) {
    await this.ready
    const id = ++this.id
    return new Promise((res, rej) => {
      this.pending.set(id, { res, rej })
      this.ws.send(JSON.stringify({ id, method, params }))
      setTimeout(() => {
        if (this.pending.has(id)) { this.pending.delete(id); rej(new Error(`${method} timeout`)) }
      }, 120000)
    })
  }
  close() { try { this.ws.close() } catch { /* ignore */ } }
}

let cdp
try {
  await waitBrowser()
  // 新建一个 page target（新版 Chrome 要求 PUT）
  const tr = await fetch(`http://127.0.0.1:${PORT}/json/new?${encodeURIComponent(url)}`, { method: 'PUT' })
  const target = tr.ok ? await tr.json() : (await (await fetch(`http://127.0.0.1:${PORT}/json/new?${encodeURIComponent(url)}`)).json())
  cdp = new Cdp(target.webSocketDebuggerUrl)
  await cdp.send('Page.enable')
  await cdp.send('Runtime.enable')
  // SHOT_MOBILE=1：真实设备视口（Chrome 的 --window-size 有 500px 最小宽度，测不了 390）
  if (process.env.SHOT_MOBILE) {
    const mw = Number(process.env.SHOT_W || 390)
    const mh = Number(process.env.SHOT_H || 844)
    await cdp.send('Emulation.setDeviceMetricsOverride', {
      width: mw, height: mh, deviceScaleFactor: 2, mobile: true
    })
    await cdp.send('Emulation.setTouchEmulationEnabled', { enabled: true, maxTouchPoints: 5 })
    await cdp.send('Page.reload')
    await sleep(800)
  }
  // 真实时间等待：视频 metadata + 首帧解码 + canvas 至少画过一帧
  // 若有 evalJs，则在前半段后执行（用于切单机位等），再等后半段
  if (evalJs) {
    await sleep(Math.round(waitMs * 0.6))
    const r = await cdp.send('Runtime.evaluate', { expression: evalJs, returnByValue: true, awaitPromise: true })
    console.log('[eval]', JSON.stringify(r.result?.value ?? r.exceptionDetails?.text ?? null))
    await sleep(Math.round(waitMs * 0.4))
  } else {
    await sleep(waitMs)
  }
  // 把页面里的诊断信息带回来（视频是否 ready、canvas 是否有非透明像素）
  const diag = await cdp.send('Runtime.evaluate', {
    expression: `(() => {
      const vs = [...document.querySelectorAll('video')].map(v => ({
        ready: v.readyState, w: v.videoWidth, h: v.videoHeight, t: +v.currentTime.toFixed(1)
      }));
      const cvs = [...document.querySelectorAll('canvas')].map(c => {
        const ctx = c.getContext('2d');
        let ink = 0, n = 0;
        if (ctx && c.width && c.height) {
          const d = ctx.getImageData(0, 0, c.width, c.height).data;
          for (let i = 3; i < d.length; i += 4 * 97) { n++; if (d[i]) ink++; }
        }
        return { w: c.width, h: c.height, inkRatio: n ? +(ink / n).toFixed(3) : 0 };
      });
      const boxes = [...document.querySelectorAll('.cam-body')].map(e => {
        const s = e.querySelector('.stage');
        return { cw: e.clientWidth, ch: e.clientHeight, stage: s ? [s.clientWidth, s.clientHeight] : null };
      });
      return JSON.stringify({ vp: [innerWidth, innerHeight], dpr: devicePixelRatio,
                              videos: vs, canvases: cvs, boxes });
    })()`,
    returnByValue: true
  })
  console.log('[page]', diag.result?.value || '(no diag)')
  const shot = await cdp.send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false })
  mkdirSync(dirname(resolve(out)), { recursive: true })
  writeFileSync(resolve(out), Buffer.from(shot.data, 'base64'))
  console.log('[ok]', resolve(out))
} catch (e) {
  console.error('[fail]', e?.message || e)
  process.exitCode = 1
} finally {
  try { await cdp?.send('Browser.close') } catch { /* ignore */ }
  cdp?.close()
  setTimeout(() => { try { proc.kill() } catch { /* ignore */ } process.exit(process.exitCode || 0) }, 300)
}
