import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  timeout: 60000,
  workers: 1,
  fullyParallel: false,
  reporter: [['list'], ['json', { outputFile: '../data/playwright-results.json' }]],
  outputDir: '../data/playwright-artifacts',
  use: {
    baseURL: process.env.VS_TEST_URL || 'http://127.0.0.1:8765',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    { name: 'desktop', use: { viewport: { width: 1440, height: 1000 } } },
    { name: 'mobile', use: { viewport: { width: 390, height: 844 }, isMobile: true, deviceScaleFactor: 1 } },
  ],
})
