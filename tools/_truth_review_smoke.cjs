// 复核页冒烟：渲染 / 标注 / 切换 / 导出（不实际下载）
const { chromium } = require('playwright')
const path = require('path')

const file = 'file:///' + path.resolve(__dirname, '..', 'output_crosscam', 'analysis', 'truth', '_review', 'review.html').replace(/\\/g, '/')

;(async () => {
  const errs = []
  const browser = await chromium.launch({ channel: 'msedge' })
  const page = await (await browser.newContext({ viewport: { width: 1500, height: 1000 } })).newPage()
  page.on('console', m => { if (m.type() === 'error') errs.push('console: ' + m.text()) })
  page.on('pageerror', e => errs.push('pageerror: ' + e.message))

  await page.goto(file, { waitUntil: 'load' })
  await page.waitForTimeout(400)

  const cards = await page.locator('#pairs .card').count()
  console.log('配对卡片数:', cards)
  const imgs = await page.locator('#pairs .card img').count()
  console.log('卡片内图片数:', imgs)

  // 标注第一张：改成"异人"（首卡 seed 已是"同人"，避免 no-op）
  await page.locator('#pairs .card').first().locator('input[value="0"]').check()
  await page.waitForTimeout(150)
  const st = await page.evaluate(() => JSON.parse(localStorage.getItem('truth_review_v1') || '{"pair":{}}'))
  const k0 = Object.keys(st.pair)[0]
  console.log('localStorage 配对键数:', Object.keys(st.pair).length, '首键:', k0, '值:', JSON.stringify(st.pair[k0]))
  console.log('互斥冲突角标数:', await page.locator('#pairs .b-x').count())

  // 过滤
  await page.selectOption('#f', 'relay')
  await page.waitForTimeout(200)
  console.log('relay 过滤后卡片数:', await page.locator('#pairs .card').count())
  await page.selectOption('#f', 'hi')
  await page.waitForTimeout(200)
  console.log('cos>=0.95 卡片数:', await page.locator('#pairs .card').count())
  await page.selectOption('#f', 'all')
  await page.waitForTimeout(150)

  // 切换到事件页
  await page.locator('#t2').click()
  await page.waitForTimeout(250)
  console.log('事件卡片数:', await page.locator('#events .ecard').count())
  await page.locator('#events .ecard').first().locator('input.pid').fill('7')
  await page.waitForTimeout(120)

  // 导出（拦截下载）
  const [dl] = await Promise.all([
    page.waitForEvent('download', { timeout: 5000 }).catch(() => null),
    page.evaluate(() => exp('pair'))
  ])
  console.log('导出文件名:', dl ? dl.suggestedFilename() : '未触发下载')

  await page.screenshot({ path: path.resolve(__dirname, '..', 'output_crosscam', 'analysis', 'truth', '_review', 'smoke_review.png'), fullPage: false })
  console.log('ERRORS:', errs.length ? JSON.stringify(errs) : 0)

  // 第二遍：模拟沙箱 iframe（localStorage 抛 SecurityError），验证容错后仍渲染
  const ctx2 = await browser.newContext({ viewport: { width: 1500, height: 1000 } })
  await ctx2.addInitScript(() => {
    Object.defineProperty(window, 'localStorage', {
      get() { throw new DOMException('The operation is insecure.', 'SecurityError') }
    })
  })
  const p2 = await ctx2.newPage()
  const errs2 = []
  p2.on('pageerror', e => errs2.push(e.message))
  await p2.goto(file, { waitUntil: 'load' })
  await p2.waitForTimeout(400)
  console.log('[沙箱模拟] 配对卡片数:', await p2.locator('#pairs .card').count(),
              '图片数:', await p2.locator('#pairs .card img').count(),
              '错误横幅:', await p2.locator('#err').isVisible().catch(() => false),
              '| pageerror:', errs2.length ? errs2 : 0)
  await browser.close()
})()
