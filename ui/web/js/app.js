/**
 * 知阁 Web UI —— 本地知识库问答前端
 *
 * 依赖后端 REST + WebSocket：
 *   POST /api/v1/auth/{register,login}     认证
 *   GET/POST/PATCH/DELETE /api/v1/kb       知识库
 *   GET/POST/PATCH/DELETE /api/v1/sessions 会话
 *   POST /api/v1/ingest                    文件入库（multipart，可带 kb_id）
 *   POST /api/v1/ingest/url                网页入库
 *   WS   /ws/chat?token=…                  流式问答
 */
class ChatApp {
    constructor() {
        this.token = localStorage.getItem('zhige_token') || '';
        this.user = null;
        this.ws = null;
        this.connected = false;
        this.currentSources = [];
        this.topK = 4;

        this.knowledgeBases = [];
        this.currentKb = null;
        this.sessions = [];
        this.currentSession = null;

        this.el = {};

        this.init();
    }

    // ================= 基础设施 =================

    async init() {
        this.cacheElements();
        this.bindEvents();

        // 先问后端"要不要登录"。默认的本地单用户模式会回答 false ——
        // 那就直接进主界面，不拿登录页挡人。
        try {
            const st = await this.probeAuthStatus();
            this.authRequired = !!st.auth_required;
            this.modeHint = st.hint || '';
        } catch (e) {
            // 探测失败也按本地模式走，宁可放行也不要卡住用户
            this.authRequired = false;
        }

        if (!this.authRequired) {
            await this.bootstrap();
            return;
        }
        if (this.token) {
            await this.bootstrap();
        } else {
            this.showAuth();
        }
    }

    async probeAuthStatus() {
        const resp = await fetch('/api/v1/auth/status');
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return resp.json();
    }

    cacheElements() {
        const ids = [
            'auth-overlay', 'auth-username', 'auth-email', 'auth-password',
            'auth-submit', 'auth-error', 'app-layout', 'kb-list', 'session-list',
            'messages', 'chat-input', 'send-btn', 'status', 'current-kb-name',
            'kb-meta', 'file-input', 'modal', 'modal-title', 'modal-body',
            'welcome-state',
            // 模型设置
            'settings-modal', 'settings-btn', 'settings-result', 'settings-save', 'settings-reset',
            'llm-backend', 'llm-remote', 'llm-base-url', 'llm-model', 'llm-api-key', 'llm-test',
            'emb-backend', 'emb-remote', 'emb-base-url', 'emb-model', 'emb-api-key', 'emb-test',
        ];
        ids.forEach((id) => { this.el[id] = document.getElementById(id); });
        this.el.authTabs = document.querySelectorAll('.auth-tab');
    }

    bindEvents() {
        // ---- 认证 ----
        this.el.authTabs.forEach((tab) => {
            tab.addEventListener('click', () => this.switchAuthMode(tab.dataset.mode));
        });
        this.el['auth-submit']?.addEventListener('click', () => this.submitAuth());
        this.el['auth-password']?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.submitAuth();
        });

        // ---- 侧边栏 ----
        document.getElementById('new-kb-btn')?.addEventListener('click', () => this.promptNewKb());
        document.getElementById('new-session-btn')?.addEventListener('click', () => this.newSession());
        document.getElementById('upload-btn')?.addEventListener('click', () => this.el['file-input'].click());
        document.getElementById('url-btn')?.addEventListener('click', () => this.promptImportUrl());
        document.getElementById('logout-btn')?.addEventListener('click', () => this.logout());
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

        // ---- 弹窗关闭（通用 modal 与设置弹窗共用）----
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal') || e.target.classList.contains('modal-close')) {
                document.querySelectorAll('.modal.active').forEach((m) => m.classList.remove('active'));
            }
        });
    }

    /** 统一请求封装：自动带 token、统一错误处理 */
    async api(path, { method = 'GET', body = null, raw = false } = {}) {
        const headers = {};
        if (this.token) headers['Authorization'] = `Bearer ${this.token}`;
        if (body && !raw) headers['Content-Type'] = 'application/json';

        const resp = await fetch(`/api/v1${path}`, {
            method,
            headers,
            body: body ? (raw ? body : JSON.stringify(body)) : undefined,
        });

        if (resp.status === 401) {
            this.token = '';
            localStorage.removeItem('zhige_token');
            // 本地模式本不该出现 401；真出现了说明是其它问题，不要误导用户去登录
            if (this.authRequired) {
                this.showAuth();
                throw new Error('登录已过期，请重新登录');
            }
            throw new Error('请求被拒绝（401），请检查后端日志');
        }

        const text = await resp.text();
        let data = null;
        try { data = text ? JSON.parse(text) : null; } catch { data = text; }

        if (!resp.ok) {
            const msg = (data && (data.detail || data.message || data.error)) || text || `HTTP ${resp.status}`;
            throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
        }
        return data;
    }

    // ================= 认证 =================

    switchAuthMode(mode) {
        this.authMode = mode;
        this.el.authTabs.forEach((t) => t.classList.toggle('active', t.dataset.mode === mode));
        this.el['auth-email'].hidden = mode !== 'register';
        this.el['auth-submit'].textContent = mode === 'register' ? '注册并登录' : '登录';
        this.el['auth-error'].textContent = '';
    }

    showAuth() {
        this.el['auth-overlay'].hidden = false;
        this.el['app-layout'].hidden = true;
        if (!this.authMode) this.switchAuthMode('login');
    }

    async submitAuth() {
        const username = this.el['auth-username'].value.trim();
        const email = this.el['auth-email'].value.trim();
        const password = this.el['auth-password'].value;
        const errEl = this.el['auth-error'];
        errEl.textContent = '';

        if (!username || !password) { errEl.textContent = '请填写用户名和密码'; return; }
        if (this.authMode === 'register' && !email) { errEl.textContent = '请填写邮箱'; return; }
        if (password.length < 8) { errEl.textContent = '密码至少 8 位'; return; }

        const btn = this.el['auth-submit'];
        btn.disabled = true;
        try {
            if (this.authMode === 'register') {
                await this.api('/auth/register', { method: 'POST', body: { username, email, password } });
            }
            const data = await this.api('/auth/login', {
                method: 'POST',
                body: { login: username, password },
            });
            this.token = data.access_token;
            localStorage.setItem('zhige_token', this.token);
            await this.bootstrap();
        } catch (e) {
            errEl.textContent = e.message;
        } finally {
            btn.disabled = false;
        }
    }

    logout() {
        this.token = '';
        localStorage.removeItem('zhige_token');
        if (this.ws) { try { this.ws.close(); } catch (_) {} this.ws = null; }
        this.knowledgeBases = [];
        this.sessions = [];
        this.currentSession = null;
        this.el['chat-input'].value = '';
        this.el['messages'].innerHTML = '';
        this.showAuth();
    }

    /** 进入主界面：拉用户信息、知识库、会话，并建立 WS */
    async bootstrap() {
        try {
            this.user = await this.api('/auth/me');
        } catch (e) {
            if (this.authRequired) {
                this.showAuth();
                return;
            }
            this.showAlert('无法连接后端：' + e.message, 'error');
            return;
        }
        this.el['auth-overlay'].hidden = true;
        this.el['app-layout'].hidden = false;

        // 本地单用户模式没有"登录"这回事，把退出按钮收起来，免得误导
        const logoutBtn = document.getElementById('logout-btn');
        if (logoutBtn) logoutBtn.hidden = !this.authRequired;

        this.updateStatus(this.authRequired ? this.user.username : '本地模式', true);

        await this.loadKnowledgeBases();
        await this.loadSessions();
        this.connectWebSocket();
    }

    // ================= 知识库 =================

    async loadKnowledgeBases() {
        const data = await this.api('/kb');
        this.knowledgeBases = data.knowledge_bases || [];
        if (!this.currentKb || !this.knowledgeBases.some((k) => k.id === this.currentKb.id)) {
            this.currentKb = this.knowledgeBases.find((k) => k.is_default) || this.knowledgeBases[0] || null;
        }
        this.renderKnowledgeBases(data.quota);
        this.renderCurrentKb();
    }

    renderKnowledgeBases(quota) {
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

        if (quota !== undefined && quota !== -1) {
            this.el['kb-meta'].textContent = `${this.knowledgeBases.length}/${quota} 个库`;
        }
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
        // 本地模式没有 token，就不带这个参数（后端也不会校验）
        const qs = this.token ? `?token=${encodeURIComponent(this.token)}` : '';
        const url = `${proto}//${window.location.host}/ws/chat${qs}`;

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
            // 4401 = 未认证（仅认证模式下会出现）
            if (ev.code === 4401) {
                this.updateStatus('认证失败', false);
                if (this.authRequired) this.logout();
                return;
            }
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
            this.showSources(this.currentSources);
            this.enableInput();
            this.loadSessions();   // 会话标题/预览可能已更新
            if (this.currentKb) { this.loadKnowledgeBases(); }
        } else if (event === 'error') {
            this.hideTypingIndicator();
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

    showSources(sources) {
        if (!sources || !sources.length) return;
        const box = document.createElement('div');
        box.className = 'message sources';
        const title = document.createElement('h4');
        title.textContent = `📎 引用来源（${sources.length} 条）`;
        box.appendChild(title);

        const ul = document.createElement('ul');
        sources.forEach((s, i) => {
            const li = document.createElement('li');
            const score = typeof s.score === 'number' ? s.score.toFixed(3) : '—';
            li.innerHTML = `<strong>[${i + 1}]</strong> ${this.esc(s.title || s.source || '未知')}
                <span class="score">相似度 ${score}</span>`;
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
            const backend = src.backend || 'local';
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
        const backend = document.getElementById(`${k}-backend`)?.value || 'local';
        const box = document.getElementById(`${k}-remote`);
        if (box) box.hidden = backend === 'local';

        const baseEl = document.getElementById(`${k}-base-url`);
        const modelEl = document.getElementById(`${k}-model`);
        if (backend === 'ollama') {
            if (baseEl && !baseEl.value) baseEl.value = 'http://localhost:11434/v1';
            if (modelEl && !modelEl.value) modelEl.placeholder = k === 'llm' ? 'qwen2.5:7b' : 'nomic-embed-text';
        } else if (backend === 'openai') {
            if (baseEl && (!baseEl.value || baseEl.value.includes('localhost'))) {
                baseEl.value = 'https://api.deepseek.com/v1';
            }
            if (modelEl && !modelEl.value) modelEl.placeholder = k === 'llm' ? 'deepseek-chat' : 'text-embedding-3-small';
        }
    }

    /** 收集某一类（llm / emb）的表单值 */
    collectKind(k) {
        const backend = document.getElementById(`${k}-backend`)?.value || 'local';
        const out = { backend };
        if (backend !== 'local') {
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

    // ================= 弹窗 / 提示 =================

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
