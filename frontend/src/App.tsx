import { useEffect, useRef, useState } from 'react'
import {
  ArrowDownToLine, ArrowRight, ArrowUp, BookOpen, Check, CheckCheck, ChevronDown,
  ChevronLeft, ChevronRight, CircleHelp, Code2, ExternalLink, FileText, FlaskConical,
  Layers3, LoaderCircle, MessageSquare, Microscope, PanelRightClose, Plus, Search,
  ShieldCheck, Sparkles, Square, Upload, X, NotebookPen, Activity, Focus, Library,
} from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import StudyPanel from './StudyPanel'
import {
  api, post, upload, percent, activeStatus,
  type Bootstrap, type Evidence, type Experiment, type LogResult, type Paper,
  type Report, type ResearchRun, type RunEvent, type Session, type Epoch,
} from './api'

type Tab = 'research' | 'library' | 'lab' | 'reports'
const tabs = [
  { id: 'research' as Tab, name: '研究工作台', icon: Microscope },
  { id: 'library' as Tab, name: '论文库', icon: Library },
  { id: 'lab' as Tab, name: '实验室', icon: FlaskConical },
  { id: 'reports' as Tab, name: '验证记录', icon: ShieldCheck },
]
const suggestions = [
  { icon: BookOpen, title: '读懂残差连接', subtitle: '从原文理解 ResNet 的核心设计', mode: 'paper',
    prompt: 'ResNet 为什么把目标写成 F(x)+x？请解释恒等快捷连接，并引用原文页码。' },
  { icon: Layers3, title: '比较两种研究思路', subtitle: 'ViT 与 CLIP 如何学习视觉表示', mode: 'paper',
    prompt: '比较 ViT 和 CLIP 的训练目标、监督信号和图像表示方式。分别检索两篇论文并给出引用。' },
  { icon: Code2, title: '连接论文与代码', subtitle: '看看图像如何被拆成 patch', mode: 'code',
    prompt: '查看 examples/vision_ops.py 中 patchify 的源码，解释每次 reshape 和 transpose 的作用，并标出真实代码行号。' },
  { icon: FlaskConical, title: '让想法接受实验', subtitle: '用真实数字数据比较两种特征', mode: 'experiment',
    prompt: '创建并运行默认 digits 多种子鲁棒性研究，再读取结果，比较原始像素与PCA在干净图像、噪声和遮挡下的表现，说明数据划分和实验局限。' },
]
const tag = (paper: Paper) => ['resnet', 'vit', 'clip', 'detr'].includes(paper.id)
  ? ({ resnet: 'ResNet', vit: 'ViT', clip: 'CLIP', detr: 'DETR' }[paper.id] ?? paper.id)
  : paper.title.slice(0, 12)
const date = (iso: string) => new Date(iso).toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' })
const statusLabel: Record<string, string> = {
  queued: '等待执行', running: '研究中', completed: '已完成', failed: '未完成',
  cancelled: '已取消', interrupted: '服务中断',
}
const harnessLabel: Record<string, string> = {
  legacy: 'V0 · LangGraph', openharness: 'V1 · OpenHarness', upgraded: 'V2 · 研究记忆增强',
  research: 'V3 · 独立研究引擎',
}

function Markdown({ text, onEvidence }: { text: string; onEvidence?: (id: string) => void }) {
  const mathText = text.replace(/\\\[([\s\S]*?)\\\]/g, (_, math) => `\n$$\n${math}\n$$\n`)
    .replace(/\\\((.*?)\\\)/g, (_, math) => `$${math}$`)
  const linked = mathText.replace(/\[((?:[a-zA-Z0-9_.-]+:p\d+:c\d+)|(?:code:[a-zA-Z0-9_./-]+:L\d+(?:-L\d+)?))\]/g,
    (_, id: string) => `[${id.startsWith('code:') ? '源码' : id.split(':').slice(0, 2).join(' · ')}](#evidence/${id})`)
  return <div className="markdown"><ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]} disallowedElements={['img']}
    components={{ a: ({ href, children }) => href?.startsWith('#evidence/')
      ? <button className="inline-citation" onClick={() => onEvidence?.(href.slice(10))}>{children}</button>
      : <a href={href} target="_blank" rel="noreferrer">{children}<ExternalLink size={12} /></a> }}>
    {linked}
  </ReactMarkdown></div>
}

export default function App() {
  const [data, setData] = useState<Bootstrap>()
  const [tab, setTab] = useState<Tab>('research')
  const [sessionId, setSessionId] = useState(localStorage.getItem('vs-session') || '')
  const [runs, setRuns] = useState<ResearchRun[]>([])
  const [events, setEvents] = useState<RunEvent[]>([])
  const [draft, setDraft] = useState('')
  const [question, setQuestion] = useState('')
  const [mode, setMode] = useState('auto')
  const [harness, setHarness] = useState('research')
  const [retriever, setRetriever] = useState('lsa')
  const [selectedPapers, setSelectedPapers] = useState<string[]>([])
  const [evidence, setEvidence] = useState<Evidence[]>([])
  const [selectedEvidence, setSelectedEvidence] = useState<Evidence>()
  const [toast, setToast] = useState('')
  const [bootError, setBootError] = useState('')
  const [sending, setSending] = useState(false)
  const [showTrace, setShowTrace] = useState(false)
  const [notesOpen, setNotesOpen] = useState(false)
  const [sessionLoading, setSessionLoading] = useState(false)
  const endRef = useRef<HTMLDivElement>(null)
  const activeRun = runs.find(run => activeStatus(run.status))

  async function refresh() { setData(await api<Bootstrap>('/bootstrap')) }
  useEffect(() => { refresh().catch(e => setBootError(e.message)) }, [])
  useEffect(() => {
    if (!toast) return
    const timer = setTimeout(() => setToast(''), 6000)
    return () => clearTimeout(timer)
  }, [toast])
  useEffect(() => {
    setEvents([]); setDraft(''); setEvidence([]); setSelectedEvidence(undefined)
    if (!sessionId) { setRuns([]); return }
    localStorage.setItem('vs-session', sessionId)
    let cancelled = false
    setSessionLoading(true)
    setRuns(previous => previous.filter(run => run.session_id === sessionId))
    api<Session & { runs: ResearchRun[] }>(`/sessions/${sessionId}`).then(result => {
      if (cancelled) return
      if (result.runs.length) {
        setHarness(result.harness || 'openharness')
        setRetriever(result.retriever || 'lsa')
      }
      setRuns(previous => [
        ...result.runs,
        ...previous.filter(run => run.session_id === sessionId && !result.runs.some(r => r.id === run.id)),
      ])
      setEvidence(result.runs.flatMap(r => r.citations))
    }).catch(e => { if (!cancelled) { setToast(e.message); setSessionId('') } })
      .finally(() => { if (!cancelled) setSessionLoading(false) })
    return () => { cancelled = true }
  }, [sessionId])
  useEffect(() => {
    if (!activeRun) return
    setEvents([]); setDraft('')
    const seen = new Set<number>()
    const source = new EventSource(`/api/runs/${activeRun.id}/events`)
    source.onmessage = async (message) => {
      const event: RunEvent = JSON.parse(message.data)
      if (seen.has(event.seq)) return
      seen.add(event.seq)
      setEvents(previous => previous.some(e => e.seq === event.seq) ? previous : [...previous, event])
      if (event.type === 'delta') setDraft(previous => previous + (event.data.text || ''))
      if (event.type === 'assistant_turn' && event.data.has_tools) setDraft('')
      if (event.type === 'evidence') setEvidence(event.data.items || [])
      if (event.type === 'terminal') {
        source.close()
        try {
          const final = await api<ResearchRun>(`/runs/${activeRun.id}`)
          setRuns(previous => previous.map(run => run.id === final.id ? final : run))
          setDraft(''); setShowTrace(false)
          await refresh()
        } catch (e) { setToast((e as Error).message) }
      }
    }
    source.onerror = () => {
      // EventSource resumes with Last-Event-ID. Also reconcile terminal state after a restart.
      api<ResearchRun>(`/runs/${activeRun.id}`).then(final => {
        if (!activeStatus(final.status)) {
          source.close()
          setRuns(previous => previous.map(r => r.id === final.id ? final : r))
          setDraft('')
        }
      }).catch(() => {})
    }
    return () => source.close()
  }, [activeRun?.id])
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }, [runs.length])

  async function newSession() {
    const session = await post<Session>('/sessions', {})
    setData(previous => previous ? { ...previous, sessions: [session, ...previous.sessions] } : previous)
    setRuns([]); setSessionId(session.id); setTab('research')
    return session.id
  }
  async function send(prompt = question, selectedMode = mode) {
    if (!prompt.trim() || sending || activeRun || sessionLoading) return
    setSending(true)
    try {
      const id = sessionId || await newSession()
      const run = await post<ResearchRun>('/runs', {
        session_id: id, prompt, paper_ids: selectedPapers, mode: selectedMode, harness, retriever,
      })
      setRuns(previous => [...previous, run]); setQuestion(''); setTab('research')
    } catch (e) { setToast((e as Error).message) }
    finally { setSending(false) }
  }
  async function openPaper(paper: Paper) {
    try {
      const result = await api<{ chunks: Evidence[] }>(`/papers/${paper.id}/pages/1`)
      if (result.chunks[0]) setSelectedEvidence({ ...result.chunks[0], title: paper.title })
    } catch (e) { setToast((e as Error).message) }
  }
  async function saveRun(run: ResearchRun) {
    try {
      await post('/notes', { session_id: run.session_id, title: run.prompt.slice(0, 50), content: run.answer })
      await refresh(); setToast('已保存到研究笔记')
    } catch (e) { setToast((e as Error).message) }
  }
  const findEvidence = (id: string) => {
    const item = [...evidence, ...runs.flatMap(r => r.citations)].find(e => e.id === id)
    if (item) setSelectedEvidence(item)
    else setToast('该引用未通过本轮核验')
  }
  const latest = [...events].reverse().find(e => ['status', 'route', 'tool_start', 'tool_end'].includes(e.type))
  const progress = latest?.data.message || latest?.data.label ||
    (latest?.data.name ? `${latest.type === 'tool_start' ? '正在调用' : '已完成'} · ${latest.data.name}` : '正在准备研究')

  if (!data) return <div className="boot-screen"><div className="brand-symbol"><Focus /></div>
    <h1>Vision Scholar</h1>{bootError ? <><p>{bootError}</p><button className="primary" onClick={() => location.reload()}>重试连接</button></>
      : <><LoaderCircle className="spin" /><p>正在打开你的研究工作空间</p></>}</div>

  return <div className="app">
    <aside className="sidebar">
      <button className="brand" onClick={() => setTab('research')} aria-label="Vision Scholar 首页">
        <span className="brand-symbol"><Focus size={25} /></span><span>Vision<span className="brand-light">Scholar</span><small>视觉研究工作台</small></span>
      </button>
      <button className="new-research" onClick={() => newSession().catch(e => setToast(e.message))}><Plus size={18} /> 开始新的研究</button>
      <div className="nav-label">工作空间</div>
      <nav>{tabs.map(({ id, name, icon: Icon }) => <button key={id} className={tab === id ? 'nav-item active' : 'nav-item'} onClick={() => setTab(id)}>
        <Icon size={19} /><span>{name}</span>{id === 'library' && <small>{data.stats.papers}</small>}
      </button>)}</nav>
      <div className="nav-label session-label">最近的研究 <MessageSquare size={13} /></div>
      <div className="session-list">{data.sessions.length ? data.sessions.slice(0, 16).map(session =>
        <button key={session.id} className={`session-item ${sessionId === session.id && tab === 'research' ? 'selected' : ''}`}
          onClick={() => { setSessionId(session.id); setTab('research') }}>
          <MessageSquare size={14} /><span>{session.title}</span>
        </button>) : <p className="sidebar-empty">你的第一个问题<br />将成为一段新的研究。</p>}</div>
      <button className="note-nav" onClick={() => setNotesOpen(true)}><NotebookPen size={17} />研究笔记 <span>{data.notes.length}</span></button>
      <div className="sidebar-footer"><span className={`status-dot ${data.model.available ? '' : 'offline'}`} /><div>本地工作空间<small>论文与研究记录保存在本机</small></div><ShieldCheck size={16} /></div>
    </aside>
    <div className="workspace">
      <header className="topbar">
        <div className="breadcrumb"><span>个人工作空间</span><ChevronRight size={14} /><strong>{tabs.find(t => t.id === tab)?.name}</strong></div>
        <div className="topbar-actions"><span className={`model-pill ${data.model.mock ? 'mock' : ''}`} title={data.model.model || '请在 .env 中配置模型'}>
          <span className={`status-dot ${data.model.available ? '' : 'offline'}`} />
          {data.model.mock ? '脚本 Mock' : data.model.available ? '模型已配置' : '离线模式'}
        </span><button className="icon-button" title="查看使用说明" onClick={() => setToast('选择论文后提问，点击引用查看原文；实验室可独立运行。模型设置见项目 README。')}><CircleHelp size={18} /></button><span className="avatar">研</span></div>
      </header>
      {tab === 'research' ? <div className={`research-layout ${selectedEvidence ? 'has-evidence' : ''}`}>
        <main className="research-main">
          <div className="research-scroll">
            {sessionLoading ? <div className="loading-inline"><LoaderCircle className="spin" />正在恢复研究记录</div>
              : runs.length === 0 ? <div className="welcome">
                <div className="eyebrow"><span /> RESEARCH, WITH EVIDENCE</div>
                <h1>从一个好问题，<br />走向有依据的发现<span>。</span></h1>
                <p className="welcome-subtitle">读论文、看代码、做实验。<br className="mobile-break" /> 把好奇心变成可以验证的下一步。</p>
                <div className="corpus-strip"><div className="mini-paper-stack"><FileText size={20} /></div>
                  <span><strong>{data.stats.papers} 篇论文已就绪</strong><small>{data.stats.pages} 页原文 · {data.stats.chunks} 个可追溯片段</small></span>
                  <button onClick={() => setTab('library')}>管理论文 <ArrowRight size={15} /></button>
                </div>
                <div className="suggestion-heading"><span>从这里开始探索</span><span>01 — 04</span></div>
                <div className="suggestion-grid">{suggestions.map(({ icon: Icon, title, subtitle, prompt, mode: nextMode }, i) =>
                  <button className="suggestion" key={title} onClick={() => { setQuestion(prompt); setMode(nextMode) }}>
                    <span className={`suggestion-icon tone-${i}`}><Icon size={21} /></span>
                    <span><strong>{title}</strong><small>{subtitle}</small></span><ArrowRight size={17} />
                  </button>)}</div>
              </div> : <div className="conversation">
                <div className="conversation-heading"><span>研究记录</span>{sessionId && <a href={`/api/sessions/${sessionId}/export`} className="text-button"><ArrowDownToLine size={15} />导出</a>}</div>
                {runs.map(run => <article className="conversation-turn" key={run.id}>
                  <div className="user-question"><span className="message-label">你的问题</span><p>{run.prompt}</p></div>
                  <div className="assistant-header"><span className="assistant-icon"><Focus size={18} /></span><strong>Vision Scholar</strong>
                    <span className="role-label">{{ paper: '论文研究', code: '代码分析', experiment: '实验分析' }[run.role] || '研究助手'}</span>
                    <span className="role-label">{harnessLabel[run.harness] || 'V1 · OpenHarness'}</span>
                    {run.usage.provider === 'mock' && <span className="badge amber">脚本 Mock</span>}
                  </div>
                  {run.answer ? <>
                    <Markdown text={run.answer} onEvidence={findEvidence} />
                    {run.citations.length > 0 && <div className="citation-list">{run.citations.map((citation, i) =>
                      <button key={citation.id} onClick={() => setSelectedEvidence(citation)}><span>{i + 1}</span><FileText size={13} />
                        {citation.kind === 'code' ? citation.title : `${citation.paper_id?.toUpperCase()} · p.${citation.page}`}</button>)}</div>}
                    <div className="answer-actions"><span title={run.verification.scope}><CheckCheck size={14} />
                      {run.citations.length ? `${run.citations.length} 处引用可回溯` : '回答已完成'}</span>
                      <button onClick={() => saveRun(run)}><NotebookPen size={14} />保存笔记</button>
                      <button onClick={async () => { try {
                        const trace = await api<{ events: RunEvent[] }>(`/runs/${run.id}/trace`)
                        setEvents(trace.events); setShowTrace(true)
                      } catch (e) { setToast((e as Error).message) } }}><Activity size={14} />执行记录</button>
                    </div>
                  </> : activeStatus(run.status) ? <div className="streaming-answer">
                    {draft ? <Markdown text={draft} onEvidence={findEvidence} /> : <p className="thinking"><span /><span /><span /> 正在查找证据、组织回答</p>}
                    {draft && <small className="draft-label">正在生成 · 引用将在完成后核验</small>}
                  </div> : <div className="run-error"><strong>{statusLabel[run.status]}</strong><p>{run.error}</p>
                    <button onClick={() => { setQuestion(run.prompt); setMode(run.mode) }}>重新编辑问题 <ArrowRight size={14} /></button></div>}
                </article>)}
                <div ref={endRef} />
              </div>}
          </div>
          <div className="composer-area">
            {showTrace && <Trace events={events} onClose={() => setShowTrace(false)} />}
            {activeRun && <button className="progress-bar" onClick={() => setShowTrace(!showTrace)}>
              <LoaderCircle size={15} className="spin" /><span>{progress}</span><small>{events.filter(e => e.type === 'tool_end').length} 次工具完成</small><ChevronDown size={15} /></button>}
            <div className="composer">
              <div className="composer-context"><details className="paper-select"><summary><BookOpen size={14} />
                {selectedPapers.length ? `已选 ${selectedPapers.length} 篇论文` : '全部论文'}<ChevronDown size={13} /></summary>
                <div className="paper-options"><p>限定本次问题的参考范围</p>{data.papers.map(paper =>
                  <label key={paper.id}><input type="checkbox" checked={selectedPapers.includes(paper.id)} onChange={e =>
                    setSelectedPapers(previous => e.target.checked ? [...previous, paper.id] : previous.filter(id => id !== paper.id))} />
                    <span>{tag(paper)}<small>{paper.title}</small></span></label>)}
                  <button onClick={() => setSelectedPapers([])}>使用全部论文</button></div></details>
                <span className="context-divider" /><Sparkles size={13} /><select aria-label="研究模式" value={mode} onChange={e => setMode(e.target.value)}>
                  <option value="auto">智能分流</option><option value="paper">论文研究</option><option value="code">代码分析</option><option value="experiment">实验分析</option>
                </select>
                <span className="context-divider" />
                <select aria-label="运行版本" value={harness} disabled={!!activeRun || sending || sessionLoading}
                  title="切换后开启新的研究，分别保留历史"
                  onChange={e => { setHarness(e.target.value); setSessionId(''); localStorage.removeItem('vs-session') }}>
                  {(data.model.harnesses || ['legacy', 'openharness']).map(h =>
                    <option key={h} value={h}>{harnessLabel[h]}</option>)}
                </select>
                <select aria-label="检索方案" value={retriever} disabled={!!activeRun || sending || sessionLoading}
                  title="两版均可使用同一检索器；神经检索首次加载较慢"
                  onChange={e => { setRetriever(e.target.value); setSessionId(''); localStorage.removeItem('vs-session') }}>
                  <option value="lsa">轻量检索</option><option value="neural">神经混合检索</option>
                </select>
              </div>
              <textarea aria-label="研究问题" value={question} onChange={e => setQuestion(e.target.value)} rows={2}
                placeholder="关于计算机视觉，你想研究什么？"
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); send() } }} />
              <div className="composer-bottom"><span><span className="keycap">↵</span> 发送 <span className="keycap">shift ↵</span> 换行</span>
                {activeRun ? <button className="send-button cancel" aria-label="停止生成" onClick={() => post(`/runs/${activeRun.id}/cancel`).catch(e => setToast(e.message))}><Square size={16} /></button>
                  : <button className="send-button" aria-label="发送问题" disabled={!question.trim() || sending || !data.model.available || sessionLoading} onClick={() => send()}>
                    {sending ? <LoaderCircle size={18} className="spin" /> : <ArrowUp size={21} />}</button>}
              </div>
            </div>
            <p className="composer-footnote">{data.model.mock ? '当前是固定脚本演示，不代表真实模型质量。' : !data.model.available ? '配置模型后可开始对话；论文库和实验室可正常使用。' : '让证据支撑结论。关键研究判断，请回到原文复核。'}</p>
          </div>
        </main>
        <aside className={`evidence-panel ${selectedEvidence ? 'open' : ''}`}>
          {selectedEvidence ? <EvidenceReader evidence={selectedEvidence} onClose={() => setSelectedEvidence(undefined)} onError={setToast} /> :
            <><div className="panel-heading"><span><Layers3 size={16} />研究上下文</span><span className="tiny-badge">LOCAL</span></div>
              <div className="context-intro"><div className="context-orbit"><Focus size={30} /></div><h3>好研究，始于好信源</h3><p>论文原文与引用证据<br />会在这里与你的思考同行。</p></div>
              <div className="panel-section-label">你的论文架 <span>{data.papers.length}</span></div>
              <div className="shelf">{data.papers.slice(0, 6).map((paper, i) => <button className="shelf-paper" key={paper.id} onClick={() => openPaper(paper)}>
                <span className={`paper-spine spine-${i % 4}`}><FileText size={19} /><small>{paper.year || 'PDF'}</small></span><span><strong>{tag(paper)}</strong><small>{paper.title}</small><em>{paper.page_count} 页 · 可检索</em></span><ChevronRight size={14} />
              </button>)}</div>
              {evidence.length > 0 && <div className="retrieved-evidence"><div className="panel-section-label">已读取的证据 <span>{evidence.length}</span></div>
                {evidence.slice(0, 8).map(item => <button key={item.id} onClick={() => setSelectedEvidence(item)}><FileText size={14} /><span>{item.title}<small>{item.paper_id ? `p.${item.page}` : item.path}</small></span><ChevronRight size={14} /></button>)}
              </div>}
              <div className="evidence-tip"><ShieldCheck size={19} /><p><strong>每一条引用都有来处</strong><span>点击回答中的引用，可以查看原文片段与真实页码。</span></p></div>
            </>}
        </aside>
      </div> : <main className="page-scroll">
        {tab === 'library' && <LibraryPage data={data} refresh={refresh} onError={setToast} onPaper={paper => { openPaper(paper); setTab('research') }} />}
        {tab === 'lab' && <LabPage data={data} refresh={refresh} onError={setToast} onAsk={prompt => { setQuestion(prompt); setMode('experiment'); setTab('research') }} />}
        {tab === 'reports' && <ReportsPage reports={data.reports} />}
      </main>}
    </div>
    {toast && <div className="toast" role="status"><span>{toast}</span><button onClick={() => setToast('')} aria-label="关闭提示"><X size={15} /></button></div>}
    {notesOpen && <div className="modal-backdrop" onClick={() => setNotesOpen(false)}><section className="notes-modal" onClick={e => e.stopPropagation()}>
      <div className="panel-heading"><h2>研究笔记</h2><button className="icon-button" onClick={() => setNotesOpen(false)} aria-label="关闭笔记"><X size={20} /></button></div>
      {data.notes.length ? data.notes.map(note => <article key={note.id}><h3>{note.title}</h3><Markdown text={note.content} onEvidence={id => { setNotesOpen(false); setTab('research'); findEvidence(id) }} /></article>)
        : <div className="empty-state"><NotebookPen size={36} /><h3>留下值得继续的思考</h3><p>点击回答下方的「保存笔记」，<br />或在对话中明确要求助手记住研究决策。</p></div>}
    </section></div>}
  </div>
}

function Trace({ events, onClose }: { events: RunEvent[]; onClose: () => void }) {
  return <div className="trace-panel"><div className="panel-heading"><span><Activity size={15} />执行记录</span><button className="icon-button" onClick={onClose} aria-label="关闭执行记录"><X size={15} /></button></div>
    {events.filter(e => !['delta', 'evidence', 'answer', 'assistant_turn'].includes(e.type)).map(event =>
      <div className={`trace-row ${event.data.is_error ? 'trace-error' : ''}`} key={event.seq}><span>{event.seq}</span>
        <div><strong>{event.data.name || event.data.label || event.data.message || event.type}</strong>
          {event.data.input && <details><summary>输入参数</summary><pre>{JSON.stringify(event.data.input, null, 2)}</pre></details>}
          {event.data.output && <details><summary>{event.data.is_error ? '工具返回错误' : '查看结果'}</summary><pre>{event.data.output}</pre></details>}
          {event.type === 'verification' && <small>检查引用是否存在，以及本轮是否读取</small>}
        </div><small>{new Date(event.created_at).toLocaleTimeString('zh-CN', { hour12: false })}</small>
      </div>)}
  </div>
}

function EvidenceReader({ evidence, onClose, onError }: { evidence: Evidence; onClose: () => void; onError: (s: string) => void }) {
  const [pageText, setPageText] = useState('')
  const [number, setNumber] = useState(evidence.page || 1)
  const [pageCount, setPageCount] = useState(0)
  const [fullPage, setFullPage] = useState(false)
  useEffect(() => { setNumber(evidence.page || 1); setFullPage(false); setPageText(''); setPageCount(0) }, [evidence.id])
  useEffect(() => {
    if (!evidence.paper_id) return
    let cancelled = false
    api<{ text: string; paper: Paper }>(`/papers/${evidence.paper_id}/pages/${number}`).then(result => {
      if (!cancelled) { setPageText(result.text); setPageCount(result.paper.page_count) }
    }).catch(e => onError(e.message))
    return () => { cancelled = true }
  }, [evidence.paper_id, number])
  return <><div className="panel-heading"><span><FileText size={16} />原文证据</span><button className="icon-button" onClick={onClose} aria-label="关闭原文证据"><PanelRightClose size={18} /></button></div>
    <div className="reader-meta"><span className="eyebrow">{evidence.kind === 'code' ? 'SOURCE CODE' : 'PAPER SOURCE'}</span><h3>{evidence.title}</h3>
      <span className="source-id">{evidence.id}</span>{evidence.section && <p>{evidence.section}</p>}
    </div>
    {evidence.paper_id && <div className="reader-toolbar"><button className={!fullPage ? 'selected' : ''} onClick={() => setFullPage(false)}>引用片段</button><button className={fullPage ? 'selected' : ''} onClick={() => setFullPage(true)}>整页原文</button>
      <a href={`/api/papers/${evidence.paper_id}/pdf#page=${number}`} target="_blank" rel="noreferrer" aria-label="打开原始 PDF"><ExternalLink size={15} /></a></div>}
    <div className={`reader-text ${evidence.kind === 'code' ? 'source-code' : ''}`}>{fullPage ? pageText || '正在读取原文…' : evidence.text}</div>
    {evidence.paper_id && <div className="reader-pagination"><button className="icon-button" disabled={number <= 1} onClick={() => { setNumber(n => n - 1); setFullPage(true) }} aria-label="上一页"><ChevronLeft size={18} /></button>
      <span>第 {number} / {pageCount || '—'} 页</span><button className="icon-button" disabled={!pageCount || number >= pageCount} onClick={() => { setNumber(n => n + 1); setFullPage(true) }} aria-label="下一页"><ChevronRight size={18} /></button></div>}
    <div className="reader-disclaimer"><ShieldCheck size={15} />页码以 PDF 物理页为准。引用可回溯不代表结论已通过语义评审。</div>
  </>
}

function PageHeading({ eyebrow, title, description, children }: { eyebrow: string; title: string; description: string; children?: React.ReactNode }) {
  return <div className="page-heading"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{children}</div>
}

function LibraryPage({ data, refresh, onError, onPaper }: {
  data: Bootstrap; refresh: () => Promise<void>; onError: (s: string) => void; onPaper: (p: Paper) => void
}) {
  const [query, setQuery] = useState('')
  const [searchMode, setSearchMode] = useState('local')
  const [hits, setHits] = useState<Evidence[] | null>(null)
  const [remoteHits, setRemoteHits] = useState<{ arxiv_id: string; title: string; summary: string; year: string }[]>([])
  const [arxivId, setArxivId] = useState('')
  const [busy, setBusy] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)
  async function work(label: string, task: () => Promise<unknown>) {
    setBusy(label)
    try { await task(); await refresh() } catch (e) { onError((e as Error).message) }
    finally { setBusy('') }
  }
  async function search() {
    if (!query.trim()) { setHits(null); setRemoteHits([]); return }
    await work('正在搜索', async () => {
      if (searchMode === 'local') setHits((await api<{ results: Evidence[] }>(`/search?q=${encodeURIComponent(query)}`)).results)
      else setRemoteHits((await api<{ results: typeof remoteHits }>(`/arxiv?q=${encodeURIComponent(query)}`)).results)
    })
  }
  return <div className="page-content"><PageHeading eyebrow="YOUR RESEARCH LIBRARY" title="让知识有迹可循。" description={`${data.stats.papers} 篇论文 · ${data.stats.pages} 页原文 · 已建立本地检索索引`}>
    <button className="primary" onClick={() => inputRef.current?.click()} disabled={!!busy}><Upload size={16} />上传 PDF</button>
    <input type="file" accept=".pdf,application/pdf" hidden ref={inputRef} onChange={e => { const file = e.target.files?.[0]; if (file) work('正在解析 PDF', () => upload('/papers/upload', file)); e.target.value = '' }} />
  </PageHeading>
    <div className="library-toolbar"><div className="search-box"><Search size={18} /><input aria-label="检索论文" value={query} onChange={e => setQuery(e.target.value)} onKeyDown={e => e.key === 'Enter' && search()} placeholder={searchMode === 'local' ? '在原文中搜索概念、方法或公式…' : '搜索 arXiv 标题与摘要（英文关键词）'} /></div>
      <select value={searchMode} onChange={e => { setSearchMode(e.target.value); setHits(null); setRemoteHits([]) }} aria-label="搜索范围"><option value="local">本地原文</option><option value="arxiv">arXiv</option></select>
      <button className="primary" disabled={!!busy} onClick={search}>搜索</button>
    </div>
    <div className="import-strip"><FileText size={18} /><span>添加一篇 arXiv 论文</span><input aria-label="arXiv ID" value={arxivId} onChange={e => setArxivId(e.target.value)} placeholder="例如 2010.11929" />
      <button disabled={!arxivId.trim() || !!busy} onClick={() => work('正在下载并解析', () => post('/papers/import', { arxiv_id: arxivId }))}>导入<ArrowRight size={14} /></button>
      <button className="seed-button" disabled={!!busy} onClick={() => work('正在准备经典论文', () => post('/papers/seed'))}>导入经典四篇</button>
    </div>
    {busy && <div className="inline-notice"><LoaderCircle className="spin" size={16} />{busy}…</div>}
    {searchMode === 'arxiv' && remoteHits.length > 0 ? <div className="remote-results">{remoteHits.map(hit => <article className="remote-paper" key={hit.arxiv_id}><span className="eyebrow">ARXIV · {hit.arxiv_id}</span><h3>{hit.title}</h3><p>{hit.summary}</p><button className="secondary" disabled={!!busy} onClick={() => work('正在导入论文', () => post('/papers/import', { arxiv_id: hit.arxiv_id }))}><Plus size={15} />导入论文</button></article>)}</div>
      : hits !== null ? <div className="search-results"><div className="section-heading"><h3>检索结果 · {hits.length}</h3><button className="text-button" onClick={() => setHits(null)}>返回论文库</button></div>
        {hits.length ? hits.map(hit => <article key={hit.id}><div><span className="badge teal">p.{hit.page}</span><strong>{hit.title}</strong></div><p>{hit.text}</p><a href={`/api/papers/${hit.paper_id}/pdf#page=${hit.page}`} target="_blank" rel="noreferrer">查看原文 <ExternalLink size={13} /></a></article>)
          : <div className="empty-state"><Search size={30} /><h3>没有找到相关片段</h3><p>试试英文术语，或先导入相关论文。</p></div>}</div>
      : <div className="paper-grid">{data.papers.map((paper, i) => <article className="paper-card" key={paper.id}><div className={`paper-card-top spine-${i % 4}`}><span>{tag(paper)}</span><FileText size={24} /><small>{paper.year || 'PDF'}</small></div>
        <div className="paper-card-body"><span className="paper-year">{paper.year || '本地文档'} · {paper.page_count} PAGES</span><h3>{paper.title}</h3><p>{paper.authors || '本地上传论文'}</p><div className="paper-card-meta"><span><CheckCheck size={14} />{paper.chunk_count} 个片段可检索</span><button onClick={() => onPaper(paper)}>阅读<ArrowRight size={15} /></button></div></div></article>)}</div>}
    <p className="page-caption">支持文字型 PDF（≤ 25 MB、100 页），扫描件请先 OCR。检索采用 BM25 + LSA + RRF 与词项重排。</p>
  </div>
}

function LearningChart({ records }: { records: Epoch[] }) {
  const [focus, setFocus] = useState('')
  const width = 620, height = 190, pad = 28
  const maxLoss = Math.max(...records.flatMap(r => [r.train_loss, r.val_loss])) * 1.15
  const points = (key: 'train_loss' | 'val_loss') => records.map((r, i) =>
    `${pad + i / Math.max(records.length - 1, 1) * (width - pad * 2)},${height - pad - r[key] / maxLoss * (height - pad * 2)}`).join(' ')
  return <div className="learning-chart"><div className="chart-legend"><strong>真实 SGD 学习曲线</strong>{[['train_loss', '训练 loss'], ['val_loss', '验证 loss']].map(([key, label]) =>
    <button key={key} className={key} onClick={() => setFocus(focus === key ? '' : key)} style={{ opacity: !focus || focus === key ? 1 : 0.35 }}><span />{label}</button>)}</div>
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="训练和验证损失曲线">
      {[0, 1, 2, 3].map(i => <g key={i}><line x1={pad} x2={width - pad} y1={pad + i * (height - pad * 2) / 3} y2={pad + i * (height - pad * 2) / 3} stroke="#e5e9e5" strokeDasharray="3 4" />
        <text x={pad - 5} y={pad + i * (height - pad * 2) / 3 + 4} textAnchor="end" fontSize="10" fill="#8b9591">{(maxLoss * (1 - i / 3)).toFixed(1)}</text></g>)}
      <polyline points={points('train_loss')} fill="none" stroke="#137c68" strokeWidth="2.5" opacity={!focus || focus === 'train_loss' ? 1 : 0.2} />
      <polyline points={points('val_loss')} fill="none" stroke="#a377de" strokeWidth="2.5" opacity={!focus || focus === 'val_loss' ? 1 : 0.2} />
      <text x={pad} y={height - 5} fontSize="10" fill="#8b9591">epoch {records[0].epoch}</text>
      <text x={width - pad} y={height - 5} textAnchor="end" fontSize="10" fill="#8b9591">{records.at(-1)?.epoch}</text>
    </svg></div>
}

function LabPage({ data, refresh, onError, onAsk }: {
  data: Bootstrap; refresh: () => Promise<void>; onError: (s: string) => void; onAsk: (s: string) => void
}) {
  const [kind, setKind] = useState('digits')
  const [seed, setSeed] = useState(42)
  const [components, setComponents] = useState(24)
  const [busy, setBusy] = useState(false)
  const [experiment, setExperiment] = useState<Experiment | undefined>(data.experiments[0])
  const [variant, setVariant] = useState(0)
  const [log, setLog] = useState<LogResult>()
  const logRef = useRef<HTMLInputElement>(null)
  async function run() {
    setBusy(true)
    try {
      setExperiment(await post<Experiment>('/experiments', { kind, seed, pca_components: components }))
      setVariant(0); setLog(undefined); await refresh()
    } catch (e) { onError((e as Error).message) } finally { setBusy(false) }
  }
  async function diagnose(content: string) {
    try { setLog(await post<LogResult>('/logs/analyze', { content })) } catch (e) { onError((e as Error).message) }
  }
  const result = experiment?.result
  const matrix = result?.results?.[variant]?.confusion_matrix
  return <div className="page-content"><PageHeading eyebrow="FROM IDEAS TO EXPERIMENTS" title="把想法，交给实验。" description="用真实数据和可复现的过程，回答一个具体的问题。" />
    <StudyPanel studies={data.studies || []} refresh={refresh} onError={onError} onAsk={onAsk} />
    <div className="lab-layout"><section className="lab-controls"><h3><FlaskConical size={18} />新建实验</h3><label>实验类型<select value={kind} onChange={e => setKind(e.target.value)}><option value="digits">手写数字 · 特征对比</option><option value="attention">Patch & Attention · 数学演示</option></select></label>
      <p>{kind === 'digits' ? '比较原始像素与 PCA 特征，使用验证集选择 SVM 参数，在独立测试集上评估。' : '生成随机图像，拆分 patch，计算 scaled dot-product attention，检查矩阵形状和归一化。'}</p>
      <label>随机种子<input type="number" min="0" max="100000" value={seed} onChange={e => setSeed(Number(e.target.value))} /></label>
      {kind === 'digits' && <label>PCA 维度 <span>{components}</span><input type="range" min="2" max="48" value={components} onChange={e => setComponents(Number(e.target.value))} /></label>}
      <button className="primary full-width" disabled={busy} onClick={run}>{busy ? <LoaderCircle size={17} className="spin" /> : <ArrowRight size={17} />}{busy ? '正在运行实验' : '运行实验'}</button>
      <div className="lab-control-note"><ShieldCheck size={16} /><span>固定 Python 实验<br />记录数据拆分、种子与代码哈希</span></div>
      <div className="section-divider" /><h3><FileText size={17} />训练日志诊断</h3><p>上传自己的 JSON / CSV 日志，计算最佳 epoch、损失变化和泛化差距。</p>
      <button className="secondary full-width" onClick={() => logRef.current?.click()}><Upload size={15} />上传训练日志</button>
      <input hidden type="file" accept=".json,.csv" ref={logRef} onChange={async e => { const file = e.target.files?.[0]; if (file) { try { setLog(await upload<LogResult>('/logs/upload', file)) } catch (error) { onError((error as Error).message) } } e.target.value = '' }} />
    </section>
    <div className="lab-results">{experiment && result ? <><section className="result-card"><div className="section-heading"><div><span className="eyebrow">EXPERIMENT RESULT</span><h2>{experiment.kind === 'digits' ? '每一个数字，都来自实际运行。' : '看见 Attention 的计算过程。'}</h2></div><span className="badge teal"><Check size={13} />已完成</span></div>
      <div className="experiment-meta"><span>Seed {result.seed}</span><span>{result.elapsed_seconds.toFixed(3)}s</span><span>ID {experiment.id.slice(0, 8)}</span><a href={`/api/experiments/${experiment.id}`} target="_blank" rel="noreferrer">完整 JSON<ExternalLink size={12} /></a></div>
      {result.results ? <><div className="metric-comparison">{result.results.map((r, i) => <button key={r.variant} className={variant === i ? 'metric-card selected' : 'metric-card'} onClick={() => setVariant(i)}><span>{r.variant === 'pixels' ? '原始像素 + SVM' : `PCA (${experiment.config.pca_components}维) + SVM`}</span><strong>{percent(r.test_accuracy)}</strong><small>测试集准确率</small><div>Macro F1 <b>{r.test_macro_f1.toFixed(4)}</b></div><div>验证集准确率 <b>{percent(r.validation_accuracy)}</b></div><div>验证集选择 C <b>{r.selected_C}</b></div></button>)}</div>
        <div className="split-bar"><span style={{ flex: result.split?.train }}>训练 {result.split?.train}</span><span style={{ flex: result.split?.validation }}>验证 {result.split?.validation}</span><span style={{ flex: result.split?.test }}>测试 {result.split?.test}</span></div>
        <p className="small-muted">分层划分 · 预处理仅拟合训练集 · 使用验证集选参</p>
        {result.training_log && <LearningChart records={result.training_log} />}
        {matrix && <details className="matrix-details"><summary>查看所选方案的混淆矩阵 <ChevronDown size={15} /></summary><p>行：真实标签 · 列：预测标签（0–9）</p><div className="confusion-matrix">{matrix.flatMap((row, i) => row.map((value, j) =>
          <span key={`${i}-${j}`} title={`真实 ${i} → 预测 ${j}：${value}`} style={{ backgroundColor: `rgba(19,124,104,${value / Math.max(...matrix.flat()) * 0.85 + 0.04})`, color: value > 10 ? 'white' : '#78877f' }}>{value}</span>))}</div></details>}
        <div className="result-actions"><button className="secondary" onClick={() => diagnose(JSON.stringify(result.training_log))}><Activity size={15} />诊断本次 SGD 日志</button><button className="secondary" onClick={() => onAsk(`请使用 analyze_log 分析实验 ${experiment.id} 的真实 SGD 训练日志，给出最佳 epoch、泛化差距及下一步建议。`)}><Sparkles size={15} />让助手分析</button></div>
      </> : <div className="attention-result"><div className="attention-shapes"><span>Patch<br /><strong>{result.patch_shape?.join(' × ')}</strong></span><ArrowRight /><span>Attention<br /><strong>{result.token_count} tokens</strong></span><ArrowRight /><span>Output<br /><strong>{result.output_shape?.join(' × ')}</strong></span></div>
        <div className="attention-grid" style={{ gridTemplateColumns: `repeat(${result.attention_preview?.[0]?.length || 1}, 1fr)` }}>{result.attention_preview?.flatMap((row, i) => row.map((value, j) => <span key={`${i}-${j}`} title={`[${i},${j}] ${value}`} style={{ background: `rgba(19,124,104,${0.06 + value * 0.94})` }} />))}</div>
        <p>注意力权重每行之和：{result.row_sums_min?.toFixed(8)} – {result.row_sums_max?.toFixed(8)}</p></div>}
      <div className="result-limitation"><CircleHelp size={16} /><p>{result.limitation}</p></div>
    </section></> : <section className="result-card empty-state"><FlaskConical size={42} /><h2>让第一场实验开始</h2><p>选择一个实验和随机种子，<br />这里会展示真实指标与计算过程。</p></section>}
      {log && <section className="result-card log-result"><div className="section-heading"><h3>日志诊断</h3><span className="badge teal">Python 计算</span></div><div className="log-metrics"><span>最佳准确率 epoch<strong>{log.best_accuracy_epoch}</strong></span><span>最佳验证准确率<strong>{percent(log.best_val_accuracy)}</strong></span><span>末轮泛化差距<strong>{percent(log.last_generalization_gap)}</strong></span></div><LearningChart records={log.records} />{log.signals.map(s => <p className="diagnosis" key={s}>{s}</p>)}</section>}
      {data.experiments.length > 0 && <section className="experiment-history"><h3>过去的实验</h3>{data.experiments.slice(0, 8).map(e => <button key={e.id} className={e.id === experiment?.id ? 'selected' : ''} onClick={() => { setExperiment(e); setVariant(0); setLog(undefined) }}><FlaskConical size={15} /><span>{e.kind === 'digits' ? '手写数字 · 特征对比' : 'Attention 数学演示'}<small>seed {e.config.seed} · {e.id.slice(0, 8)}</small></span><time>{date(e.created_at)}</time><ChevronRight size={16} /></button>)}</section>}
    </div></div>
  </div>
}

function ReportsPage({ reports }: { reports: Report[] }) {
  const [selected, setSelected] = useState('')
  const report = reports.find(r => r.file === selected) || reports[0]
  const cases = report?.cases || (report?.run ? [{ id: report.run.id, question: report.run.prompt, answer: report.run.answer,
    passed: report.run.status === 'completed', citations: report.run.citations.map(c => c.id), reason: report.run.error || '' }] : [])
  return <div className="page-content"><PageHeading eyebrow="VERIFY BEFORE YOU TRUST" title="让可靠性，有据可查。" description="查看真实模型、工具回归和检索测试的独立记录。" />
    <div className="evaluation-notice"><ShieldCheck size={23} /><p><strong>测试口径透明，结论才有边界。</strong><span>工程回归题由本项目编写，不是外部人评金标。引用存在与任务完成，并不等同于回答的语义质量。</span></p></div>
    {report ? <><div className="report-tabs">{reports.map(r => <button key={r.file} className={r.file === report.file ? 'selected' : ''} onClick={() => setSelected(r.file)}><span className={`status-dot ${r.kind === 'mock' ? 'offline' : ''}`} />{r.name || r.file}</button>)}</div>
      <section className="report-summary"><div><span className="eyebrow">{{ real_model: 'REAL MODEL', mock: 'SCRIPTED MOCK', retrieval: 'RETRIEVAL', engineering: 'ENGINEERING TESTS' }[report.kind] || report.kind}</span><h2>{report.name || report.file}</h2><p>{report.model || '本地确定性计算'} · {date(report.created_at)}</p></div>
        {report.summary && <div className="report-score"><strong>{report.summary.passed}<span> / {report.summary.total}</span></strong><small>通过预设工程判据</small></div>}
        <a className="secondary" href={`/api/reports/${report.file}`}><ArrowDownToLine size={16} />原始报告</a>
      </section>
      {report.summary?.description && <p className="report-description">{report.summary.description}</p>}
      <div className="report-cases">{cases.map((item, i) => <details key={item.id}><summary><span className="case-number">{String(i + 1).padStart(2, '0')}</span><span>{item.question || item.name || item.id}</span><span className={`badge ${item.passed ? 'teal' : 'amber'}`}>{item.passed ? '通过' : '待检查'}</span><ChevronDown size={15} /></summary>
        <div className="case-body">{item.expected && <p><strong>预设判据：</strong>{item.expected}</p>}{item.reason && <p><strong>记录：</strong>{item.reason}</p>}{item.elapsed_seconds !== undefined && <p>耗时：{item.elapsed_seconds}s</p>}
          {item.answer && <Markdown text={item.answer} />}{item.citations && <p className="case-citations">引用：{item.citations.join(' · ') || '无'}</p>}</div>
      </details>)}</div></> : <div className="empty-state"><ShieldCheck size={38} /><h3>还没有验证记录</h3><p>运行项目评测脚本后，报告会出现在这里。</p></div>}
  </div>
}
