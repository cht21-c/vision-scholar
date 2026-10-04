import { expect, test } from '@playwright/test'
import { mkdir, writeFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { resolve } from 'node:path'

test('V3 study → reusable result → verified download → real assistant → export', async ({ page }, info) => {
  const errors: string[] = []
  page.on('pageerror', e => errors.push(e.message))
  const directory = resolve('../data/v3/demo')
  await mkdir(directory, { recursive: true })
  await page.goto('/')
  await expect(page.getByLabel('运行版本')).toHaveValue('research')
  await page.locator('nav button').nth(2).click()
  const planned = page.waitForResponse(r => r.url().endsWith('/api/studies') && r.request().method() === 'POST')
  await page.getByRole('button', { name: '1. 创建固定方案' }).click()
  const study = await (await planned).json()
  expect(study.plan.config.seeds).toEqual([17, 43, 89])
  const executed = page.waitForResponse(r => r.url().endsWith(`/studies/${study.id}/run`))
  await page.locator('.study-config .primary').click()
  const completed = await (await executed).json()
  expect(completed.status).toBe('completed')
  const rows = page.locator('.study-current > .study-table tbody tr')
  await expect(rows).toHaveCount(6)
  for (let i = 0; i < 6; i++) {
    await expect(rows.nth(i)).toContainText((completed.result.summary[i].mean_accuracy * 100).toFixed(2) + '%')
  }
  await page.locator('.study-panel').screenshot({ path: resolve(directory, `${info.project.name}-study.png`) })
  await page.getByText('下载逐样本预测、拆分与干扰数据', { exact: true }).click()
  const downloaded = page.waitForEvent('download')
  await page.locator(`a[href$="/evidence/predictions-17.json"]`).click()
  const download = await downloaded
  const stream = await download.createReadStream()
  if (!stream) throw new Error('Missing downloaded bytes')
  const digest = createHash('sha256')
  for await (const chunk of stream) digest.update(chunk)
  expect(digest.digest('hex')).toBe(completed.result.evidence_files['predictions-17.json'])
  await page.getByRole('button', { name: '3. 让助手解读结果' }).click()
  await expect(page.getByLabel('运行版本')).toHaveValue('research')
  const launched = page.waitForResponse(r => r.url().endsWith('/api/runs') && r.request().method() === 'POST')
  await page.getByLabel('发送问题').click()
  const run = await (await launched).json()
  await expect(page.locator('.answer-actions')).toBeVisible({ timeout: 55000 })
  const detail = await (await page.request.get(`/api/runs/${run.id}`)).json()
  expect(detail.status).toBe('completed')
  const trace = await (await page.request.get(`/api/runs/${run.id}/trace`)).json()
  const names = trace.events.filter((e: { type: string }) => e.type === 'tool_end')
    .map((e: { data: { name: string } }) => e.data.name)
  expect(names).toContain('read_study')
  expect(names).not.toContain('run_study')
  const exportResponse = await page.request.get(`/api/sessions/${run.session_id}/export`)
  const exported = await exportResponse.text()
  expect(exported).toContain(detail.answer)
  expect(exported).toContain(study.id)
  await writeFile(resolve(directory, `${info.project.name}-answer.md`), exported)
  await writeFile(resolve(directory, `${info.project.name}-verification.json`), JSON.stringify({
    run_id: run.id, session_id: run.session_id, plan_id: study.id, status: detail.status,
    tools: names, download_sha_verified: true, numeric_ui_matches_result: true,
    provider: detail.usage.provider, answer: detail.answer,
  }, null, 2))
  await page.screenshot({ path: resolve(directory, `${info.project.name}-assistant.png`) })
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy()
  expect(errors).toEqual([])
})
