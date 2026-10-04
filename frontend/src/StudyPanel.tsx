import { useState } from 'react'
import { ArrowRight, Check, FileText, FlaskConical, LoaderCircle, Sparkles } from 'lucide-react'
import { percent, post, type Study } from './api'

const conditions: Record<string, string> = { clean: '干净图像', noise: '高斯噪声', occlusion: '局部遮挡' }
const methods: Record<string, string> = { pixels_svm: '原始像素 + SVM', pca_svm: 'PCA + SVM' }
const statuses: Record<string, string> = {
  planned: '方案已固定', running: '正在运行', completed: '已完成', failed: '执行失败', interrupted: '已中断',
}

export default function StudyPanel({ studies, refresh, onError, onAsk }: {
  studies: Study[]; refresh: () => Promise<void>; onError: (s: string) => void
  onAsk: (s: string) => void
}) {
  const [selected, setSelected] = useState<Study | undefined>(studies[0])
  const [busy, setBusy] = useState('')
  const [seeds, setSeeds] = useState('17, 43, 89')
  const [dimensions, setDimensions] = useState(24)
  const [noise, setNoise] = useState(3)
  async function plan() {
    setBusy('正在固定研究方案')
    try {
      const parsed = seeds.split(/[,，\s]+/).filter(Boolean).map(Number)
      if (parsed.some(s => !Number.isInteger(s))) throw new Error('种子需要使用整数')
      setSelected(await post<Study>('/studies', {
        seeds: parsed, pca_components: dimensions, noise_sigma: noise,
      }))
      await refresh()
    } catch (error) { onError((error as Error).message) } finally { setBusy('') }
  }
  async function execute() {
    if (!selected) return
    setBusy('正在训练与评估全部种子')
    try {
      setSelected(await post<Study>(`/studies/${selected.id}/run`))
      await refresh()
    } catch (error) { onError((error as Error).message) } finally { setBusy('') }
  }
  const result = selected?.result
  return <section className="study-panel">
    <div className="section-heading"><div><span className="eyebrow">REPRODUCIBLE STUDY</span>
      <h2>一个方案，走完研究闭环。</h2><p>多种子 · 配对干扰 · 验证集选参 · 原始预测可下载</p>
    </div><FlaskConical size={26} /></div>
    <div className="study-config">
      <label>随机种子<input aria-label="研究种子" value={seeds} onChange={e => setSeeds(e.target.value)} /></label>
      <label>PCA 维度<input aria-label="研究PCA维度" type="number" min={2} max={48} value={dimensions} onChange={e => setDimensions(Number(e.target.value))} /></label>
      <label>噪声标准差<input aria-label="研究噪声标准差" type="number" min={0} max={8} step={0.5} value={noise} onChange={e => setNoise(Number(e.target.value))} /></label>
      <button className="secondary" disabled={!!busy} onClick={plan}>1. 创建固定方案</button>
      <button className="primary" disabled={!selected || !!busy} onClick={execute}>
        {busy ? <LoaderCircle size={16} className="spin" /> : <ArrowRight size={16} />}
        {selected?.status === 'completed' ? '核验并复用结果' : '2. 执行方案'}</button>
    </div>
    {busy && <div className="inline-notice"><LoaderCircle className="spin" size={16} />{busy}…</div>}
    {selected && <div className="study-current">
      <div className="experiment-meta"><span className="badge teal"><Check size={13} />{statuses[selected.status] || selected.status}</span>
        <span>种子 {selected.plan.config.seeds.join(' / ')}</span>
        <span>PCA {selected.plan.config.pca_components} 维</span>
        <span>σ {selected.plan.config.noise_sigma} · 遮挡 {selected.plan.config.occlusion_size} × {selected.plan.config.occlusion_size}</span>
        <a href={`/api/studies/${selected.id}`} target="_blank" rel="noreferrer">完整方案与结果</a>
      </div>
      <p className="study-id" title={selected.id}>方案 {selected.id}</p>
      <p className="small-muted">训练 / 验证 / 测试 = 60% / 20% / 20%；仅用验证集选 C；所有变换只拟合训练集。更改上方参数后需创建新方案。</p>
      {result && <>
        <div className="study-table"><table><thead><tr><th>测试条件</th><th>方法</th><th>平均准确率</th><th>跨种子标准差</th></tr></thead>
          <tbody>{result.summary.map(row => <tr key={`${row.condition}-${row.method}`}>
            <td>{conditions[row.condition]}</td><td>{methods[row.method]}</td>
            <td><strong>{percent(row.mean_accuracy)}</strong></td><td>{percent(row.std_accuracy)}</td>
          </tr>)}</tbody></table></div>
        <details className="study-details"><summary>查看配对差值与 95% bootstrap 区间</summary>
          <p>差值 = PCA − 原始像素，单位为百分点。区间针对每个固定拆分和已训练模型；不同 seed 测试集存在重叠。</p>
          <div className="study-table"><table><thead><tr><th>种子</th><th>条件</th><th>差值</th><th>95% 区间</th></tr></thead>
            <tbody>{result.paired_comparisons.map(row => <tr key={`${row.seed}-${row.condition}`}>
              <td>{row.seed}</td><td>{conditions[row.condition]}</td><td>{(row.pca_minus_pixels * 100).toFixed(2)}</td>
              <td>[{(row.paired_bootstrap_95_low * 100).toFixed(2)}, {(row.paired_bootstrap_95_high * 100).toFixed(2)}]</td>
            </tr>)}</tbody></table></div></details>
        <details className="study-details"><summary>下载逐样本预测、拆分与干扰数据</summary>
          <div className="study-files">{Object.entries(result.evidence_files).map(([name, hash]) =>
            <a key={name} href={`/api/studies/${selected.id}/evidence/${name}`} title={`SHA256 ${hash}`}>
              <FileText size={14} />{name}<small>{hash.slice(0, 12)}</small></a>)}</div></details>
        <div className="result-actions"><span className="small-muted">实际运行 {result.elapsed_seconds.toFixed(2)} 秒</span>
          <button className="secondary" onClick={() => onAsk(`读取研究方案 ${selected.id} 的实际结果，比较三种条件下 PCA 与原始像素的准确率。不要重新运行，说明统计边界。`)}>
            <Sparkles size={15} />3. 让助手解读结果</button></div>
        <p className="small-muted">digits 是小型真实视觉数据集；本实验不代表大型论文复现，或对所有数据集的泛化结论。</p>
      </>}
      {selected.error && <p className="run-error">{selected.error}</p>}
    </div>}
    {studies.length > 1 && <div className="study-history">{studies.map(study =>
      <button className="text-button" key={study.id} onClick={() => setSelected(study)}>
        {study.id.slice(0, 8)} · {statuses[study.status]}</button>)}</div>}
  </section>
}
