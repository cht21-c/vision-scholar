import { expect, test, type Page } from '@playwright/test'
import { mkdir } from 'node:fs/promises'
import { resolve } from 'node:path'

const screenshots = resolve('../data/screenshots')

async function layout(page: Page) {
  const size = await page.evaluate(() => ({
    width: innerWidth,
    document: document.documentElement.scrollWidth,
    workspace: document.querySelector('.workspace')?.getBoundingClientRect().right,
  }))
  expect(size.document).toBeLessThanOrEqual(size.width)
  expect(size.workspace || 0).toBeLessThanOrEqual(size.width + 1)
}

async function screenshot(page: Page, name: string) {
  await mkdir(screenshots, { recursive: true })
  await page.screenshot({ path: resolve(screenshots, name + '.png') })
}

test('library → real experiment → log → verification records', async ({ page }, info) => {
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto('/')
  await expect(page.getByRole('heading', { name: /从一个好问题/ })).toBeVisible()
  await layout(page)
  await screenshot(page, `${info.project.name}-welcome`)
  await page.locator('nav button').nth(1).click()
  await expect(page.locator('.paper-card')).toHaveCount(4)
  await screenshot(page, `${info.project.name}-library`)
  await page.getByLabel('检索论文').fill('symmetric cross entropy')
  await page.getByRole('button', { name: '搜索', exact: true }).click()
  await expect(page.locator('.search-results article').first()).toBeVisible()
  await expect(page.locator('.search-results')).toContainText('CLIP')
  await layout(page)
  await page.locator('nav button').nth(2).click()
  const resultResponse = page.waitForResponse(r => r.url().endsWith('/api/experiments') && r.request().method() === 'POST')
  await page.getByRole('button', { name: '运行实验', exact: true }).click()
  const response = await resultResponse
  expect(response.status()).toBe(200)
  const experiment = await response.json()
  expect(experiment.result.split).toEqual({ train: 1077, validation: 360, test: 360 })
  await expect(page.locator('.metric-card').first()).toContainText('97.78%')
  await page.locator('.metric-card').nth(1).click()
  await expect(page.locator('.metric-card').nth(1)).toHaveClass(/selected/)
  await page.locator('.matrix-details summary').click()
  await expect(page.locator('.confusion-matrix span')).toHaveCount(100)
  await page.getByRole('button', { name: '诊断本次 SGD 日志' }).click()
  await expect(page.locator('.log-result')).toContainText('最佳准确率 epoch')
  await layout(page)
  await page.locator('.result-card').first().scrollIntoViewIfNeeded()
  await screenshot(page, `${info.project.name}-lab`)
  await page.locator('nav button').nth(3).click()
  await expect(page.locator('.report-tabs button').first()).toBeVisible()
  await page.getByRole('button', { name: '检索回归 · 8 个固定问题' }).click()
  await expect(page.locator('.report-score')).toContainText('8 / 8')
  await page.locator('.report-cases summary').first().click()
  await expect(page.locator('.case-body').first()).toContainText('预设判据')
  await layout(page)
  await screenshot(page, `${info.project.name}-reports`)
  expect(errors).toEqual([])
})

test('restore answer → evidence → note/export → cancel and reload', async ({ page, request }, info) => {
  const reportResponse = await request.get('/api/reports/real-model.json')
  expect(reportResponse.ok()).toBeTruthy()
  const report = await reportResponse.json()
  const completed = report.cases.find((c: { id: string; passed: boolean }) => c.id === 'resnet' && c.passed)
  expect(completed).toBeTruthy()
  await page.addInitScript(id => {
    if (!localStorage.getItem('vs-session')) localStorage.setItem('vs-session', id)
  }, completed.run.session_id)
  await page.goto('/')
  await expect(page.locator('.answer-actions')).toBeVisible()
  await page.locator('.citation-list button').first().click()
  await expect(page.locator('.reader-text')).toBeVisible()
  await page.getByRole('button', { name: '整页原文', exact: true }).click()
  await expect(page.locator('.reader-pagination')).toContainText('/ 12 页')
  await layout(page)
  await screenshot(page, `${info.project.name}-evidence`)
  await page.getByLabel('关闭原文证据').click()
  await page.getByRole('button', { name: '保存笔记', exact: true }).click()
  await expect(page.getByRole('status')).toContainText('已保存到研究笔记')
  await page.locator('.note-nav').click()
  await expect(page.locator('.notes-modal article')).not.toHaveCount(0)
  await page.getByLabel('关闭笔记').click()
  const exported = await request.get(`/api/sessions/${completed.run.session_id}/export`)
  expect(await exported.text()).toContain(completed.run.prompt)
  await page.locator('.new-research').click()
  await expect(page.getByLabel('发送问题')).toBeDisabled()
  await page.getByLabel('研究问题').fill('请仔细比较四篇论文的研究方法和证据，逐篇检索原文。')
  await page.getByLabel('发送问题').click()
  await page.getByLabel('停止生成').click()
  await expect(page.locator('.run-error')).toContainText('已取消')
  await page.reload()
  await expect(page.locator('.run-error')).toContainText('已取消')
  await expect(page.locator('.progress-bar')).toHaveCount(0)
  await layout(page)
})
