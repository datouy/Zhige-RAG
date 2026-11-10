/**
 * 知阁 Web UI —— 本地知识库问答前端
 * 
 * 开箱即用设计：
 * - 无需登录：所有请求自动归属固定的本地用户
 * - 首次使用引导
 * - 模型状态可视化
 * 
 * 依赖后端 REST + WebSocket：
 *   GET/POST/PATCH/DELETE /api/v1/kb       知识库
 *   GET/POST/PATCH/DELETE /api/v1/sessions 会话
 *   POST /api/v1/ingest                    文件入库（multipart，可带 kb_id）
 *   POST /api/v1/ingest/url                网页入库
 *   WS   /ws/chat                          流式问答
 */
class ChatApp {
    constructor() {
        this.ws = null;
        this.connected = false;
        this.currentSources = [];
        this.topK = 8;

        this.knowledgeBases = [];
        this.currentKb = null;
        this.sessions = [];
        this.currentSession = null;
        
        // 默认直接进入聊天界面：欢迎引导只在用户主动"重置引导"后才出现
        // （旧逻辑 `!== 'false'` 会让新用户首访被引导弹窗挡住，看不到聊天界面）
        this.firstVisit = localStorage.getItem('zhige_first_visit') === 'true';

        this.el = {};

        this.init();
    }

    // ================= 基础设施 =================

    async init() {
        this.cacheElements();
        this.bindEvents();

        // 显示首次使用引导
        if (this.firstVisit) {
            this.showWelcome();
        } else {
            await this.bootstrap();
        }
    }
    
    showWelcome() {
        const overlay = document.getElementById('welcome-overlay');
        if (overlay) {
            overlay.classList.add('active');
            this.checkModelStatus();
        }
    }
    
    async checkModelStatus() {
        const box = document.getElementById('model-status-content');
        if (!box) return;
        
        try {
            const s = await this.api('/settings/model');
            const llmBackend = (s.llm && s.llm.backend) || 'extractive';
            const embBackend = (s.embedding && s.embedding.backend) || 'onnx';
            const backendLabel = {
                extractive: '原文摘录（无 LLM）',
                onnx: '本地 CPU 向量',
                local: '本机权重',
                ollama: '本机 Ollama',
                openai: '云端 API',
            };
            let html = '<ul class="status-list">';

            html += `<li class="ok">
                    <span class="status-dot green"></span>
                    <span>对话模型: ${llmBackend}（${backendLabel[llmBackend] || llmBackend}）</span>
                </li>`;
            html += `<li class="ok">
                    <span class="status-dot green"></span>
                    <span>向量模型: ${embBackend}（${backendLabel[embBackend] || embBackend}）</span>
                </li>`;

            html += '</ul>';

            // 添加帮助提示
            if (llmBackend === 'extractive' && embBackend === 'onnx') {
                html += `<div class="status-ready">
                    ✅ 无 LLM 模式：BM25 + 本地 ONNX 向量混合检索，答案直接引用原文
                </div>`;
            } else {
                html += `<div class="status-help">
                    <p>当前使用可选的自定义后端，可在「模型设置」里测试连通性。</p>
                </div>`;
            }

            box.innerHTML = html;
        } catch (e) {
            box.innerHTML = '<span class="warn">无法检测模型状态，请启动后端后重试</span>';
        }
    }

    hideWelcome() {
        const overlay = document.getElementById('welcome-overlay');
        if (overlay) {
            overlay.classList.remove('active');
            // 不再显示引导（用户可重置）
            localStorage.setItem('zhige_first_visit', 'false');
        }
    }
    
    resetWelcome() {
        localStorage.setItem('zhige_first_visit', 'true');
        this.firstVisit = true;
        this.showWelcome();
    }

    cacheElements() {
        const ids = [
            'app-layout', 'kb-list', 'session-list',
            'messages', 'chat-input', 'send-btn', 'status', 'current-kb-name',
            'kb-meta', 'file-input', 'modal', 'modal-title', 'modal-body',
            'welcome-state', 'welcome-overlay',
            // 模型设置
            'settings-modal', 'settings-btn', 'settings-result', 'settings-save', 'settings-reset',
            'llm-backend', 'llm-remote', 'llm-base-url', 'llm-model', 'llm-api-key', 'llm-test',
            'emb-backend', 'emb-remote', 'emb-base-url', 'emb-model', 'emb-api-key', 'emb-test',
            // 文档管理
            'docs-modal', 'docs-content', 'docs-kb-name', 'docs-refresh', 'docs-back', 'docs-btn',
        ];
        ids.forEach((id) => { this.el[id] = document.getElementById(id); });
    }

    bindEvents() {
        // ---- 首次使用引导 ----
        document.getElementById('welcome-start')?.addEventListener('click', () => {
            this.hideWelcome();
            this.bootstrap();
        });

        // ---- 侧边栏 ----
        document.getElementById('new-kb-btn')?.addEventListener('click', () => this.promptNewKb());
        document.getElementById('new-session-btn')?.addEventListener('click', () => this.newSession());
        document.getElementById('upload-btn')?.addEventListener('click', () => this.el['file-input'].click());
        document.getElementById('url-btn')?.addEventListener('click', () => this.promptImportUrl());
        document.getElementById('guide-btn')?.addEventListener('click', () => this.showGuide());
        this.el['file-input']?.addEventListener('change', (e) => this.uploadFiles(e.target.files));

        // ---- 对话 ----
        this.el['send-btn']?.addEventListener('click', () => this.sendMessage());
        this.el['chat-input']?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // ---- 模型设置 ----
        this.el['settings-btn']?.addEventListener('click', () => this.openSettings());
        ['llm', 'emb'].forEach((k) => {
            document.getElementById(`${k}-backend`)
                ?.addEventListener('change', () => this.syncRemoteFields(k));
            document.getElementById(`${k}-test`)
                ?.addEventListener('click', () => this.testModel(k));
        });
        this.el['settings-save']?.addEventListener('click', () => this.saveSettings());
        this.el['settings-reset']?.addEventListener('click', () => this.resetSettings());

        // ---- 文档管理 ----
        this.el['docs-btn']?.addEventListener('click', () => this.openDocs());
        this.el['docs-refresh']?.addEventListener('click', () => this.loadDocs());
        this.el['docs-back']?.addEventListener('click', () => { this.el['docs-back'].hidden = true; this.loadDocs(); });

        // ---- 弹窗关闭（通用 modal 与设置弹窗共用）----
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal') || e.target.classList.contains('modal-close')) {
                document.querySelectorAll('.modal.active').forEach((m) => m.classList.remove('active'));
            }
        });
    }

    /** 统一请求封装：统一错误处理 */
    async api(path, { method = 'GET', body = null, raw = false } = {}) {
        const headers = {};
        if (body && !raw) headers['Content-Type'] = 'application/json';

        const resp = await fetch(`/api/v1${path}`, {
            method,
            headers,
            body: body ? (raw ? body : JSON.stringify(body)) : undefined,
        });

        const text = await resp.text();
        let data = null;
        try { data = text ? JSON.parse(text) : null; } catch { data = text; }

        if (!resp.ok) {
            const msg = (data && (data.detail || data.message || data.error)) || text || `HTTP ${resp.status}`;
            throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
        }
        return data;
    }

    /** 进入主界面：拉知识库、会话，并建立 WS */
    async bootstrap() {
        this.el['app-layout'].hidden = false;
        await this.loadKnowledgeBases();
        await this.loadSessions();
        this.connectWebSocket();
    }

    // ================= 知识库 =================

    async loadKnowledgeBases() {
        const data = await this.api('/kb');
        this.knowledgeBases = data.knowledge_bases || [];
        // 始终换成列表里的新对象：否则上传/删除后侧边栏已是新计数，顶栏还
        // 拿着旧对象（表现为一直显示「0 文档 / 0 分块」）。
        const fresh = this.knowledgeBases.find((k) => this.currentKb && k.id === this.currentKb.id);
        this.currentKb = fresh
            || this.knowledgeBases.find((k) => k.is_default)
            || this.knowledgeBases[0]
            || null;
        this.renderKnowledgeBases();
        this.renderCurrentKb();
    }

    renderKnowledgeBases() {
        const list = this.el['kb-list'];
        if (!list) return;
        if (!this.knowledgeBases.length) {
            list.innerHTML = '<li class="list-empty">暂无知识库</li>';
            return;
        }
        list.innerHTML = this.knowledgeBases.map((kb) => `
            <li class="list-item ${this.currentKb && kb.id === this.currentKb.id ? 'active' : ''}"
                data-kb="${kb.id}">
                <span class="item-main" title="${this.esc(kb.description || kb.name)}">
                    ${kb.is_default ? '🏠' : '📁'} ${this.esc(kb.name)}
                </span>
                <span class="item-meta">${kb.doc_count}</span>
                ${kb.is_default ? '' : `<button class="item-del" data-del-kb="${kb.id}" title="删除知识库" type="button">×</button>`}
            </li>
        `).join('');

        list.querySelectorAll('[data-kb]').forEach((li) => {
            li.addEventListener('click', (e) => {
                if (e.target.dataset.delKb) return;
                this.selectKb(li.dataset.kb);
            });
        });
        list.querySelectorAll('[data-del-kb]').forEach((btn) => {
            btn.addEventListener('click', () => this.deleteKb(btn.dataset.delKb));
        });

    }

    renderCurrentKb() {
        if (!this.currentKb) return;
        this.el['current-kb-name'].textContent = this.currentKb.name;
        this.el['kb-meta'].textContent =
            `${this.currentKb.doc_count} 文档 / ${this.currentKb.chunk_count} 分块`;
    }

    async selectKb(kbId) {
        this.currentKb = this.knowledgeBases.find((k) => k.id === kbId) || this.currentKb;
        this.currentSession = null;
        this.renderKnowledgeBases();
        this.renderCurrentKb();
        this.clearMessages();
        await this.loadSessions();
    }

    async promptNewKb() {
        const name = prompt('新建知识库，输入名称：');
        if (!name || !name.trim()) return;
        try {
            await this.api('/kb', { method: 'POST', body: { name: name.trim(), description: '' } });
            this.showAlert('知识库已创建', 'success');
            await this.loadKnowledgeBases();
        } catch (e) {
            this.showAlert('创建失败：' + e.message, 'error');
        }
    }

    async deleteKb(kbId) {
        const kb = this.knowledgeBases.find((k) => k.id === kbId);
        if (!kb) return;
        if (!confirm(`删除知识库「${kb.name}」？\n其中的文档、向量数据与该库的会话都会被一并删除，且不可恢复。`)) return;
        try {
            await this.api(`/kb/${kbId}`, { method: 'DELETE' });
            this.showAlert('知识库已删除', 'success');
            if (this.currentKb && this.currentKb.id === kbId) this.currentKb = null;
            await this.loadKnowledgeBases();
            await this.loadSessions();
        } catch (e) {
            this.showAlert('删除失败：' + e.message, 'error');
        }
    }

    // ================= 会话 =================

    async loadSessions() {
        const q = this.currentKb ? `?kb_id=${encodeURIComponent(this.currentKb.id)}` : '';
        try {
            const data = await this.api(`/sessions${q}`);
            this.sessions = data.sessions || [];
            this.renderSessions();
        } catch (e) {
            this.sessions = [];
            this.renderSessions();
        }
    }

    renderSessions() {
        const list = this.el['session-list'];
        if (!list) return;
        if (!this.sessions.length) {
            list.innerHTML = '<li class="list-empty">暂无会话，点 ＋ 开始</li>';
            return;
        }
        list.innerHTML = this.sessions.map((s) => `
            <li class="list-item ${this.currentSession && s.id === this.currentSession.id ? 'active' : ''}"
                data-session="${s.id}">
                <span class="item-main" title="${this.esc(s.preview || s.title)}">
                    💬 ${this.esc(s.title || '新会话')}
                </span>
                <button class="item-ren" data-ren="${s.id}" title="重命名" type="button">✎</button>
                <button class="item-del" data-del-session="${s.id}" title="删除会话" type="button">×</button>
            </li>
        `).join('');

        list.querySelectorAll('[data-session]').forEach((li) => {
            li.addEventListener('click', (e) => {
                if (e.target.dataset.ren || e.target.dataset.delSession) return;
                this.openSession(li.dataset.session);
            });
        });
        list.querySelectorAll('[data-ren]').forEach((btn) => {
            btn.addEventListener('click', () => this.renameSession(btn.dataset.ren));
        });
        list.querySelectorAll('[data-del-session]').forEach((btn) => {
            btn.addEventListener('click', () => this.deleteSession(btn.dataset.delSession));
        });
    }

    async newSession() {
        try {
            const s = await this.api('/sessions', {
                method: 'POST',
                body: { kb_id: this.currentKb ? this.currentKb.id : null },
            });
            this.currentSession = s;
            this.clearMessages();
            await this.loadSessions();
            this.el['chat-input'].focus();
        } catch (e) {
            this.showAlert('新建会话失败：' + e.message, 'error');
        }
    }

    async openSession(sessionId) {
        this.currentSession = this.sessions.find((s) => s.id === sessionId) || null;
        this.renderSessions();
        this.clearMessages();
        try {
            const data = await this.api(`/sessions/${sessionId}/messages`);
            (data.messages || []).forEach((m) => {
                this.appendMessage(m.role, m.content);
                if (m.role === 'assistant' && m.sources && m.sources.length) {
                    this.showSources(m.sources);
                }
            });
        } catch (e) {
            this.showAlert('加载历史失败：' + e.message, 'error');
        }
    }

    async renameSession(sessionId) {
        const s = this.sessions.find((x) => x.id === sessionId);
        const title = prompt('重命名会话：', s ? s.title : '');
        if (!title || !title.trim()) return;
        try {
            await this.api(`/sessions/${sessionId}`, { method: 'PATCH', body: { title: title.trim() } });
            await this.loadSessions();
        } catch (e) {
            this.showAlert('重命名失败：' + e.message, 'error');
        }
    }

    async deleteSession(sessionId) {
        if (!confirm('删除该会话及其消息？')) return;
        try {
            await this.api(`/sessions/${sessionId}`, { method: 'DELETE' });
            if (this.currentSession && this.currentSession.id === sessionId) {
                this.currentSession = null;
                this.clearMessages();
            }
            await this.loadSessions();
        } catch (e) {
            this.showAlert('删除失败：' + e.message, 'error');
        }
    }

    // ================= WebSocket 问答 =================

    connectWebSocket() {
        const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = `${proto}//${window.location.host}/ws/chat`;

        try {
            this.ws = new WebSocket(url);
        } catch (e) {
            this.updateStatus('连接失败', false);
            return;
        }

        this.ws.onopen = () => {
            this.connected = true;
            this.updateStatus('已就绪', true);
        };
        this.ws.onmessage = (ev) => {
            try { this.handleEvent(JSON.parse(ev.data)); } catch (e) { console.error(e); }
        };
        this.ws.onclose = (ev) => {
            this.connected = false;
            this.updateStatus('已断开，重连中…', false);
            setTimeout(() => this.connectWebSocket(), 3000);
        };
        this.ws.onerror = () => this.updateStatus('连接错误', false);
    }

    updateStatus(text, ok) {
        const el = this.el.status;
        if (!el) return;
        el.textContent = text;
        el.className = ok ? 'status connected' : 'status disconnected';
    }

    sendMessage() {
        const input = this.el['chat-input'];
        const query = input.value.trim();
        if (!query) return;
        if (!this.connected) {
            this.showAlert('连接尚未就绪，请稍候…', 'error');
            return;
        }
        if (!this.currentSession) {
            // 没选会话就自动开一个，保证问答能被记录
            this.newSession().then(() => this.sendMessage());
            input.value = query;  // newSession 会清空输入，这里还原
            return;
        }

        this.appendMessage('user', query);
        input.value = '';
        input.disabled = true;
        this.el['send-btn'].disabled = true;
        this.currentSources = [];
        this.showTypingIndicator();

        this.ws.send(JSON.stringify({
            query,
            top_k: this.topK,
            session_id: this.currentSession.id,
            kb_id: this.currentKb ? this.currentKb.id : null,
        }));
    }

    handleEvent(evt) {
        const { event, data } = evt;
        if (event === 'hits') {
            this.currentSources = data || [];
        } else if (event === 'token') {
            this.hideTypingIndicator();
            this.appendToken(String(data ?? ''));
        } else if (event === 'done') {
            this.hideTypingIndicator();
            this.finalizeAssistant();
            this.showSources(this.currentSources);
            this.enableInput();
            this.loadSessions();   // 会话标题/预览可能已更新
            if (this.currentKb) { this.loadKnowledgeBases(); }
        } else if (event === 'error') {
            this.hideTypingIndicator();
            this.finalizeAssistant();
            this.showAlert('错误：' + data, 'error');
            this.enableInput();
        }
    }

    // ================= 消息渲染 =================

    clearMessages() {
        this.el['messages'].innerHTML = `
            <div class="empty-state">
                <div class="icon">💬</div>
                <p>向当前知识库提问</p>
                <p class="muted small">在左侧切换知识库与会话；先上传文档或导入网页</p>
            </div>`;
        this.currentSources = [];
    }

    appendMessage(role, content) {
        const wrap = document.createElement('div');
        wrap.className = `message ${role}`;
        const body = document.createElement('div');
        body.className = 'message-content';
        body.textContent = content;
        wrap.appendChild(body);
        this.el['messages'].appendChild(wrap);
        this.scrollBottom();
    }

    appendToken(token) {
        let last = this.el['messages'].querySelector('.message.assistant.streaming');
        if (!last) {
            last = document.createElement('div');
            last.className = 'message assistant streaming';
            const body = document.createElement('div');
            body.className = 'message-content';
            last.appendChild(body);
            this.el['messages'].appendChild(last);
        }
        last.querySelector('.message-content').textContent += token;
        this.scrollBottom();
    }

    finalizeAssistant() {
        // 回答结束/出错时移除 streaming 标记，避免下一次回答 token 继续追加到上一条回答里
        this.el['messages'].querySelectorAll('.message.assistant.streaming').forEach((el) => {
            el.classList.remove('streaming');
        });
    }

    showSources(sources) {
        if (!sources || !sources.length) return;
        const box = document.createElement('details');
        box.className = 'message sources';
        const title = document.createElement('summary');
        title.textContent = `📎 引用来源（${sources.length} 条）`;
        box.appendChild(title);

        const ul = document.createElement('ul');
        sources.forEach((s, i) => {
            const li = document.createElement('li');
            // 分数来源不同、量纲也不同：dense 是余弦相似度，rerank 是交叉编码器
            // 相关度，keyword 是 BM25 命中强度。混在一起叫「相似度」会误导。
            const has = typeof s.score === 'number';
            const label = s.score_kind === 'keyword' ? '关键词'
                : s.score_kind === 'rerank' ? '相关度' : '相似度';
            const head = has ? `${label} ${s.score.toFixed(3)}` : label;
            li.innerHTML = `<strong>[${i + 1}]</strong> ${this.esc(s.title || s.source || '未知')}
                <span class="score">${head}</span>`;
            li.addEventListener('click', () => this.showChunkDetail(s));
            ul.appendChild(li);
        });
        box.appendChild(ul);
        this.el['messages'].appendChild(box);
        this.scrollBottom();
    }

    showChunkDetail(src) {
        this.openModal('分块详情', src.content || src.snippet || '无内容', true);
    }

    showTypingIndicator() {
        this.hideTypingIndicator();
        const el = document.createElement('div');
        el.className = 'message assistant';
        el.id = 'typing-indicator';
        el.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
        this.el['messages'].appendChild(el);
        this.scrollBottom();
    }

    hideTypingIndicator() {
        document.getElementById('typing-indicator')?.remove();
    }

    enableInput() {
        this.el['chat-input'].disabled = false;
        this.el['send-btn'].disabled = false;
        this.el['chat-input'].focus();
    }

    scrollBottom() {
        this.el['messages'].scrollTop = this.el['messages'].scrollHeight;
    }

    // ================= 入库 =================

    async uploadFiles(files) {
        if (!files || !files.length) return;
        if (!this.currentKb) { this.showAlert('请先选择知识库', 'error'); return; }

        const fd = new FormData();
        for (const f of files) fd.append('file', f);
        fd.append('kb_id', this.currentKb.id);   // 上传到当前知识库

        const btn = document.getElementById('upload-btn');
        const old = btn.textContent;
        btn.disabled = true;
        btn.textContent = '上传中…';
        try {
            const r = await this.api('/ingest', { method: 'POST', body: fd, raw: true });
            this.showAlert(`已入库：${r.docs} 篇 / ${r.chunks} 个分块`, 'success');
            await this.loadKnowledgeBases();
        } catch (e) {
            this.showAlert('入库失败：' + e.message, 'error');
        } finally {
            btn.disabled = false;
            btn.textContent = old;
            this.el['file-input'].value = '';
        }
    }

    promptImportUrl() {
        const url = prompt('导入网页，输入 URL：\n（抓取正文并入库到当前知识库）');
        if (!url || !url.trim()) return;
        this.importUrl(url.trim());
    }

    async importUrl(url) {
        if (!this.currentKb) { this.showAlert('请先选择知识库', 'error'); return; }
        const btn = document.getElementById('url-btn');
        const old = btn.textContent;
        btn.disabled = true;
        btn.textContent = '抓取中…';
        try {
            const r = await this.api('/ingest/url', {
                method: 'POST',
                body: { url, kb_id: this.currentKb.id },
            });
            this.showAlert(`已导入：${r.title}（${r.chars} 字 / ${r.chunks} 分块）`, 'success');
            await this.loadKnowledgeBases();
        } catch (e) {
            this.showAlert('导入失败：' + e.message, 'error');
        } finally {
            btn.disabled = false;
            btn.textContent = old;
        }
    }

    // ================= 模型设置 =================

    async openSettings() {
        this.setSettingsResult('', '');
        try {
            const s = await this.api('/settings/model');
            this.fillSettings(s);
            this.el['settings-modal'].classList.add('active');
        } catch (e) {
            this.showAlert('读取模型设置失败：' + e.message, 'error');
        }
    }

    /** 用后端返回的配置填充表单 */
    fillSettings(s) {
        ['llm', 'emb'].forEach((k) => {
            const src = (k === 'llm' ? s.llm : s.embedding) || {};
            const backend = src.backend || (k === 'llm' ? 'extractive' : 'onnx');
            const select = document.getElementById(`${k}-backend`);
            if (select) select.value = backend;

            const sec = src[backend] || {};
            const set = (id, val) => { const el = document.getElementById(id); if (el) el.value = val || ''; };
            set(`${k}-base-url`, sec.base_url);
            set(`${k}-model`, sec.model);
            set(`${k}-api-key`, '');
            const keyEl = document.getElementById(`${k}-api-key`);
            if (keyEl) {
                keyEl.placeholder = sec.api_key
                    ? `已保存（${sec.api_key}），留空表示不修改`
                    : '本地服务可留空';
            }
            this.syncRemoteFields(k);
        });
    }

    /** 显示/隐藏远程字段，并给常见默认值降低填写成本 */
    syncRemoteFields(k) {
        const backend = document.getElementById(`${k}-backend`)?.value || (k === 'llm' ? 'extractive' : 'onnx');
        const box = document.getElementById(`${k}-remote`);
        if (box) box.hidden = !['ollama', 'openai'].includes(backend);

        const baseEl = document.getElementById(`${k}-base-url`);
        const modelEl = document.getElementById(`${k}-model`);
        if (backend === 'ollama') {
            if (baseEl && !baseEl.value) baseEl.value = 'http://localhost:11434/v1';
            if (modelEl && !modelEl.value) modelEl.placeholder = k === 'llm' ? 'qwen3:4b-instruct-2507-q4_K_M' : 'nomic-embed-text';
        } else if (backend === 'openai') {
            if (baseEl && (!baseEl.value || baseEl.value.includes('localhost'))) {
                baseEl.value = 'https://api.deepseek.com/v1';
            }
            if (modelEl && !modelEl.value) modelEl.placeholder = k === 'llm' ? 'deepseek-chat' : 'text-embedding-3-small';
        }
    }

    /** 收集某一类（llm / emb）的表单值 */
    collectKind(k) {
        const backend = document.getElementById(`${k}-backend`)?.value || (k === 'llm' ? 'extractive' : 'onnx');
        const out = { backend };
        if (backend === 'ollama' || backend === 'openai') {
            const sec = {
                base_url: (document.getElementById(`${k}-base-url`)?.value || '').trim(),
                model: (document.getElementById(`${k}-model`)?.value || '').trim(),
            };
            const key = (document.getElementById(`${k}-api-key`)?.value || '').trim();
            if (key) sec.api_key = key;   // 留空 = 不修改已保存的 Key
            out[backend] = sec;
        }
        return out;
    }

    async testModel(k) {
        const cfg = this.collectKind(k);
        const backend = cfg.backend;
        const payload = {
            kind: k === 'llm' ? 'llm' : 'embedding',
            backend,
            ...(cfg[backend] || {}),
        };
        this.setSettingsResult('测试中…', 'info');
        try {
            const r = await this.api('/settings/model/test', { method: 'POST', body: payload });
            this.setSettingsResult((r.ok ? '✅ ' : '❌ ') + r.detail, r.ok ? 'ok' : 'err');
        } catch (e) {
            this.setSettingsResult('❌ ' + e.message, 'err');
        }
    }

    async saveSettings() {
        const body = { llm: this.collectKind('llm'), embedding: this.collectKind('emb') };
        const btn = this.el['settings-save'];
        btn.disabled = true;
        this.setSettingsResult('保存中…', 'info');
        try {
            const r = await this.api('/settings/model', { method: 'PUT', body });
            const warns = (r.warnings || []).join('　');
            this.setSettingsResult('✅ 已保存并生效。' + (warns ? ' ' + warns : ''), 'ok');
            this.showAlert('模型设置已更新，立即生效', 'success');
            // 后端单例已重建；Embedding 切换可能换了 collection，刷新一下库
            await this.loadKnowledgeBases();
        } catch (e) {
            this.setSettingsResult('❌ ' + e.message, 'err');
        } finally {
            btn.disabled = false;
        }
    }

    async resetSettings() {
        if (!confirm('恢复为 config/config.yaml 中的模型配置？\n会删除界面写入的本地覆盖文件。')) return;
        try {
            const r = await this.api('/settings/model', { method: 'DELETE' });
            this.fillSettings(r);
            this.setSettingsResult('✅ 已恢复配置文件中的设置', 'ok');
            await this.loadKnowledgeBases();
        } catch (e) {
            this.setSettingsResult('❌ ' + e.message, 'err');
        }
    }

    setSettingsResult(text, kind) {
        const el = this.el['settings-result'];
        if (!el) return;
        el.textContent = text;
        el.className = 'settings-result' + (kind ? ` settings-${kind}` : '');
    }

    // ================= 文档管理（预览 / 分块编辑） =================

    async openDocs() {
        this.el['docs-modal'].classList.add('active');
        this.el['docs-back'].hidden = true;
        this.el['docs-kb-name'].textContent = this.currentKb
            ? ('当前知识库：' + this.currentKb.name) : '全部知识库';
        await this.loadDocs();
    }

    async loadDocs() {
        const box = this.el['docs-content'];
        box.innerHTML = '<p class="muted small">加载中…</p>';
        try {
            const data = await this.api('/documents');
            this.renderDocs(data.documents || []);
        } catch (e) {
            box.innerHTML = '<p class="warn">加载失败：' + this.esc(e.message) + '</p>';
        }
    }

    renderDocs(docs) {
        const box = this.el['docs-content'];
        if (!docs.length) {
            box.innerHTML = '<p class="muted small">还没有文档。先「上传文档」或「导入网页」。</p>';
            return;
        }
        // 列表接口返回全部知识库的文档，这里按当前知识库过滤
        const kbId = this.currentKb ? this.currentKb.id : null;
        const list = kbId ? docs.filter((d) => d.kb_id === kbId) : docs;
        if (!list.length) {
            box.innerHTML = '<p class="muted small">当前知识库还没有文档。</p>';
            return;
        }
        box.innerHTML = '<ul class="docs-list">' + list.map((d) => {
            const title = this.esc(d.title || d.source);
            const src = this.esc(d.source);
            const meta = d.chunks + ' 分块 · ' + this.esc(d.doc_status || 'active');
            return '<li class="docs-item" data-source="' + src + '">'
                + '<span class="item-main" title="' + src + '">📄 ' + title + '</span>'
                + '<span class="item-meta">' + meta + '</span>'
                + '</li>';
        }).join('') + '</ul>';
        box.querySelectorAll('.docs-item').forEach((li) => {
            li.addEventListener('click', () => this.openDocChunks(li.dataset.source));
        });
    }

    async openDocChunks(source) {
        const box = this.el['docs-content'];
        box.innerHTML = '<p class="muted small">加载分块…</p>';
        const params = new URLSearchParams();
        if (this.currentKb) params.set('kb_id', this.currentKb.id);
        params.set('limit', '200');
        const q = params.toString() ? ('?' + params.toString()) : '';
        try {
            const data = await this.api('/documents/' + encodeURIComponent(source) + '/chunks' + q);
            this.renderChunks(source, data.chunks || []);
            this.el['docs-back'].hidden = false;
        } catch (e) {
            box.innerHTML = '<p class="warn">加载分块失败：' + this.esc(e.message) + '</p>';
        }
    }

    renderChunks(source, chunks) {
        const box = this.el['docs-content'];
        if (!chunks.length) {
            box.innerHTML = '<p class="muted small">该文档没有分块。</p>';
            return;
        }
        // 按 chunk_index 排序，保持原文顺序（Chroma 返回顺序不保证）
        const sorted = chunks.slice().sort((a, b) => {
            const ai = parseInt((a.metadata && a.metadata.chunk_index) ?? 0);
            const bi = parseInt((b.metadata && b.metadata.chunk_index) ?? 0);
            return ai - bi;
        });
        box.innerHTML = '<h4 class="docs-doc-title">📄 ' + this.esc(source) + '</h4>'
            + sorted.map((c, i) => {
                const head = this.esc((c.metadata && (c.metadata.chunk_prefix || c.metadata.breadcrumb)) || '—');
                return '<div class="chunk-card" data-chunk="' + this.esc(c.id) + '">'
                    + '<div class="chunk-head">'
                    + '<span class="muted small">#' + (i + 1) + ' · ' + head + '</span>'
                    + '<button class="btn btn-primary btn-sm chunk-save" type="button">保存</button>'
                    + '</div>'
                    + '<textarea class="chunk-text" rows="4">' + this.esc(c.text || '') + '</textarea>'
                    + '</div>';
            }).join('');
        box.querySelectorAll('.chunk-card').forEach((card) => {
            const id = card.dataset.chunk;
            card.querySelector('.chunk-save').addEventListener('click', () => this.saveChunk(id, card));
        });
    }

    async saveChunk(chunkId, card) {
        const textarea = card.querySelector('.chunk-text');
        const text = textarea.value.trim();
        if (!text) { this.showAlert('分块内容不能为空', 'error'); return; }
        const params = new URLSearchParams();
        if (this.currentKb) params.set('kb_id', this.currentKb.id);
        const q = params.toString() ? ('?' + params.toString()) : '';
        const btn = card.querySelector('.chunk-save');
        btn.disabled = true;
        btn.textContent = '保存中…';
        try {
            await this.api('/documents/chunks/' + encodeURIComponent(chunkId) + q, {
                method: 'PATCH', body: { text },
            });
            this.showAlert('分块已更新并重新向量化', 'success');
        } catch (e) {
            this.showAlert('保存失败：' + e.message, 'error');
        } finally {
            btn.disabled = false;
            btn.textContent = '保存';
        }
    }

    // ================= 弹窗 / 提示 =================

    showGuide() {
        const content = `
            <div class="guide-content">
                <h4>快速开始</h4>
                <ol>
                    <li><strong>上传文档</strong> - 点击左侧「上传文档」按钮，选择 PDF、Word、Markdown 或 TXT 文件</li>
                    <li><strong>导入网页</strong> - 点击「导入网页」，粘贴网址即可自动抓取内容</li>
                    <li><strong>开始问答</strong> - 在下方输入框中输入问题，系统会基于已上传的文档给出答案</li>
                </ol>
                
                <h4>模型配置</h4>
                <p>点击「模型设置」可以切换不同的模型后端：</p>
                <ul>
                    <li><strong>extractive（当前）</strong> - 无 LLM，直接返回带引用的知识库原文</li>
                    <li><strong>ONNX 向量</strong> - 本地 CPU 运行中文 BGE 向量模型</li>
                    <li><strong>可选后端</strong> - 需要时再配置 Ollama、云端 API 或本机生成权重</li>
                </ul>
                
                <h4>数据安全</h4>
                <p>所有数据都保存在本地电脑上，不会上传到任何服务器。</p>
            </div>
        `;
        this.openModal('使用指南', content);
    }

    openModal(title, content, mono = false) {
        this.el['modal-title'].textContent = title;
        this.el['modal-body'].textContent = content;
        this.el['modal-body'].style.whiteSpace = 'pre-wrap';
        this.el['modal-body'].style.fontFamily = mono ? 'ui-monospace, Consolas, monospace' : '';
        this.el.modal.classList.add('active');
    }

    closeModal() {
        document.querySelectorAll('.modal').forEach((m) => m.classList.remove('active'));
    }

    showAlert(message, type = 'success') {
        const container = document.getElementById('alert-container');
        if (!container) return;
        const el = document.createElement('div');
        el.className = `alert alert-${type}`;
        el.textContent = message;
        container.innerHTML = '';
        container.appendChild(el);
        setTimeout(() => el.remove(), 5000);
    }

    /** HTML 转义，防 XSS（知识库名/文档名来自用户输入） */
    esc(text) {
        return String(text ?? '').replace(/[&<>"']/g, (c) => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
        }[c]));
    }
}

let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new ChatApp();
});
