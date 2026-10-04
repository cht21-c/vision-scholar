import { expect, test } from '@playwright/test'
import { mkdir } from 'node:fs/promises'
import { resolve } from 'node:path'

test('switch harness → V2 source citation → isolated history', async ({ page }, info) => {
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto('/')
  await expect(page.getByLabel('运行版本')).toBeVisible()
  await page.getByLabel('运行版本').selectOption('legacy')
  await expect(page.getByRole('heading', { name: /从一个好问题/ })).toBeVisible()
  await page.getByLabel('运行版本').selectOption('upgraded')
  await page.getByLabel('研究问题').fill('查看 patchify 源码，用两句话解释 reshape 和 transpose，引用对应的具体行号。')
  await page.getByLabel('发送问题').click()
  await expect(page.getByLabel('运行版本')).toBeDisabled()
  await expect(page.locator('.answer-actions')).toBeVisible({ timeout: 55000 })
  await expect(page.locator('.assistant-header')).toContainText('V2')
  await page.locator('.inline-citation').first().click()
  await expect(page.locator('.reader-text')).toContainText(/reshape|transpose/)
  await expect(page.locator('.source-id')).toContainText('code:examples/vision_ops.py:')
  await page.getByLabel('关闭原文证据').click()
  const box = await page.evaluate(() => ({
    width: innerWidth, document: document.documentElement.scrollWidth,
    composer: document.querySelector('.composer')?.getBoundingClientRect().right,
  }))
  expect(box.document).toBeLessThanOrEqual(box.width)
  expect(box.composer || 0).toBeLessThanOrEqual(box.width)
  const directory = resolve('../data/comparison/screenshots')
  await mkdir(directory, { recursive: true })
  await page.screenshot({ path: resolve(directory, `${info.project.name}-v2.png`) })
  await page.getByLabel('运行版本').selectOption('legacy')
  await expect(page.locator('.conversation-turn')).toHaveCount(0)
  expect(errors).toEqual([])
})
