class ChatApp {
    constructor() {
        this.ws = null;
        this.agentWs = null;
        this.connected = false;
        this.isStreaming = false;
        this.activeRequestId = null;
        this.activeSocket = null;
        this.topK = 8;
        this.directMode = false;
        this.agentMode = false;
        this.currentKb = null;
        this.knowledgeBases = [];
        this.agents = [];
        this.activeAgent = null;
        this.mcpServers = [];
        this.pendingAgentPersist = null;
        this.currentSession = null;
        this.sessions = [];
        this.currentSources = [];
        this.lastUserQuery = "";
        this.citationSequence = 0;
        this.documents = [];
        this.ingestJobs = [];
        this.tasksTimer = null;
        this.searchHits = [];
        this.searchSequence = 0;
        this.searchTimer = null;
        this.reconnectTimer = null;
        this.el = {};
        this.init();
    }

    async init() {
        this.cacheElements();
        this.bindEvents();
        this.updateShortcutLabel();
        try {
            await this.bootstrap();
        } catch (error) {
            this.showAlert(`启动失败：${error.message}`, "error");
        }
    }

    cacheElements() {
        const ids = [
            "app-layout", "messages", "chat-input", "send-btn", "stop-btn", "composer", "status",
            "current-kb-name", "kb-meta", "kb-list", "session-list", "file-input",
            "mode-btn", "mode-menu", "mode-label", "search-modal", "search-input", "search-results",
            "export-btn",
            "import-modal", "import-form", "import-url-input", "import-url-submit",
            "modal", "modal-title", "modal-body", "docs-modal", "docs-content", "docs-filter",
            "docs-kb-name", "docs-refresh", "docs-back", "docs-btn", "docs-source-filter", "settings-modal",
            "settings-result", "settings-save", "settings-reset", "llm-backend", "llm-remote",
            "llm-base-url", "llm-model", "llm-api-key", "llm-test", "emb-backend", "emb-remote",
            "emb-base-url", "emb-model", "emb-api-key", "emb-test", "upload-btn", "url-btn",
            "settings-btn", "new-kb-btn", "new-session-btn", "search-btn", "sidebar",
            "sidebar-open", "sidebar-close", "sidebar-backdrop", "drop-overlay",
            "ingest-status", "ingest-status-text", "ingest-progress",
            "citation-panel", "citation-title", "citation-source", "citation-close", "citation-content",
            "tasks-btn", "tasks-modal", "tasks-content",
            "agents-modal", "agents-list-view", "agents-list", "agent-new", "agent-form",
            "agent-id", "agent-name", "agent-kb", "agent-instructions", "agent-tools",
            "agent-form-result", "agent-delete", "agent-cancel", "agent-save",
            "mcp-open", "mcp-modal", "mcp-servers-list", "mcp-form", "mcp-name",
            "mcp-command", "mcp-args", "mcp-form-result",
        ];
        ids.forEach((id) => { this.el[id] = document.getElementById(id); });
    }

    bindEvents() {
        document.querySelectorAll("[data-upload]").forEach((button) => {
            button.addEventListener("click", () => this.el["file-input"].click());
        });
        document.querySelectorAll("[data-import-url]").forEach((button) => {
            button.addEventListener("click", () => this.openImport());
        });
        document.querySelectorAll("[data-search]").forEach((button) => {
            button.addEventListener("click", () => this.openSearch());
        });

        this.el["file-input"]?.addEventListener("change", (event) => {
            this.uploadFiles(event.target.files);
            event.target.value = "";
        });
        this.el["upload-btn"]?.addEventListener("click", () => this.el["file-input"].click());
        this.el["url-btn"]?.addEventListener("click", () => this.openImport());
        this.el["new-kb-btn"]?.addEventListener("click", () => this.promptNewKb());
        this.el["new-session-btn"]?.addEventListener("click", () => this.newSession());
        this.el["export-btn"]?.addEventListener("click", () => this.exportSession());
        this.el["search-btn"]?.addEventListener("click", () => this.openSearch());
        this.el["docs-btn"]?.addEventListener("click", () => this.openDocs());
        this.el["tasks-btn"]?.addEventListener("click", () => this.openTasks());
        this.el["settings-btn"]?.addEventListener("click", () => this.openSettings());
        this.el["agent-new"]?.addEventListener("click", () => this.startAgentEditor());
        this.el["agent-cancel"]?.addEventListener("click", () => this.showAgentsList());
        this.el["agent-delete"]?.addEventListener("click", () => this.deleteAgent());
        this.el["agent-form"]?.addEventListener("submit", (event) => {
            event.preventDefault();
            this.saveAgent();
        });
        this.el["agent-kb"]?.addEventListener("change", () => this.loadAgentToolOptions());
        this.el["mcp-open"]?.addEventListener("click", () => this.openMcpServers());
        this.el["mcp-form"]?.addEventListener("submit", (event) => {
            event.preventDefault();
            this.createMcpServer();
        });

        this.el["composer"]?.addEventListener("submit", (event) => {
            event.preventDefault();
            this.sendMessage();
        });
        this.el["stop-btn"]?.addEventListener("click", () => this.stopStreaming());
        this.el["chat-input"]?.addEventListener("input", () => this.resizeComposer());
        this.el["chat-input"]?.addEventListener("keydown", (event) => {
            if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
                event.preventDefault();
                this.sendMessage();
            }
        });

        this.el["mode-btn"]?.addEventListener("click", () => this.toggleModeMenu());
        this.el["mode-menu"]?.querySelectorAll("[data-mode]").forEach((button) => {
            button.addEventListener("click", () => {
                this.chooseConversationMode(button.dataset.mode);
                this.closeModeMenu();
                this.el["mode-btn"].focus();
            });
        });

        this.el["search-input"]?.addEventListener("input", () => this.scheduleSearch());
        this.el["import-form"]?.addEventListener("submit", (event) => {
            event.preventDefault();
            this.importUrl(this.el["import-url-input"].value.trim());
        });
        this.el["docs-refresh"]?.addEventListener("click", () => this.loadDocs());
        this.el["docs-back"]?.addEventListener("click", () => this.loadDocs());
        this.el["docs-filter"]?.addEventListener("input", () => this.renderDocs(this.documents));
        this.el["docs-source-filter"]?.addEventListener("change", () => this.renderDocs(this.documents));
        this.el["citation-close"]?.addEventListener("click", () => this.closeCitation());

        ["llm", "emb"].forEach((kind) => {
            document.getElementById(`${kind}-backend`)?.addEventListener("change", () => this.syncRemoteFields(kind));
            document.getElementById(`${kind}-test`)?.addEventListener("click", () => this.testModel(kind));
        });
        this.el["settings-save"]?.addEventListener("click", () => this.saveSettings());
        this.el["settings-reset"]?.addEventListener("click", () => this.resetSettings());

        this.el["sidebar-open"]?.addEventListener("click", () => this.setSidebarOpen(true));
        this.el["sidebar-close"]?.addEventListener("click", () => this.setSidebarOpen(false));
        this.el["sidebar-backdrop"]?.addEventListener("click", () => this.setSidebarOpen(false));

        document.addEventListener("click", (event) => {
            if (event.target.classList.contains("modal")) this.closeAllModals();
            if (event.target.closest(".modal-close")) this.closeAllModals();
            if (!event.target.closest(".mode-control")) this.closeModeMenu();
        });

        document.addEventListener("keydown", (event) => {
            const modifier = event.metaKey || event.ctrlKey;
            if (modifier && event.key.toLowerCase() === "k") {
                event.preventDefault();
                this.openSearch();
            } else if (modifier && event.key.toLowerCase() === "n") {
                event.preventDefault();
                this.newSession();
            } else if (event.key === "Escape") {
                if (document.querySelector(".modal.active")) this.closeAllModals();
                else if (!this.el["citation-panel"].hidden) this.closeCitation();
                else this.setSidebarOpen(false);
            }
        });

        let dragDepth = 0;
        window.addEventListener("dragenter", (event) => {
            if (!this.hasFiles(event)) return;
            event.preventDefault();
            dragDepth += 1;
            document.body.classList.add("is-dragging");
        });
        window.addEventListener("dragover", (event) => {
            if (!this.hasFiles(event)) return;
            event.preventDefault();
            event.dataTransfer.dropEffect = "copy";
        });
        window.addEventListener("dragleave", (event) => {
            if (!this.hasFiles(event)) return;
            dragDepth = Math.max(0, dragDepth - 1);
            if (!dragDepth) document.body.classList.remove("is-dragging");
        });
        window.addEventListener("drop", (event) => {
            if (!this.hasFiles(event)) return;
            event.preventDefault();
            dragDepth = 0;
            document.body.classList.remove("is-dragging");
            this.uploadFiles(event.dataTransfer.files);
        });
    }

    hasFiles(event) {
        return Array.from(event.dataTransfer?.types || []).includes("Files");
    }

    updateShortcutLabel() {
        const label = this.el["search-btn"]?.querySelector("kbd");
        if (label && !/Mac|iPhone|iPad/.test(navigator.platform)) label.textContent = "Ctrl K";
    }

    async api(path, { method = "GET", body = null, raw = false } = {}) {
        const headers = {};
        if (body && !raw) headers["Content-Type"] = "application/json";
        const response = await fetch(`/api/v1${path}`, {
            method,
            headers,
            body: body ? (raw ? body : JSON.stringify(body)) : undefined,
        });
        const text = await response.text();
        let data = null;
        try { data = text ? JSON.parse(text) : null; } catch { data = text; }
        if (!response.ok) {
            const detail = data && (data.detail || data.message || data.error);
            throw new Error(typeof detail === "string" ? detail : (detail ? JSON.stringify(detail) : text || `HTTP ${response.status}`));
        }
        return data;
    }

    async bootstrap() {
        this.el["app-layout"].hidden = false;
        await this.loadDefaults();
        await this.loadKnowledgeBases();
        await this.loadAgents();
        await this.loadSessions();
        await this.restoreSessionFromUrl();
        this.connectWebSocket();
        this.el["chat-input"].focus();
    }

    async loadDefaults() {
        try {
            const defaults = await this.api("/config/defaults");
            if (Number.isFinite(defaults?.top_k) && defaults.top_k > 0) this.topK = defaults.top_k;
        } catch {
            // The configured frontend default remains usable when this endpoint is unavailable.
        }
    }

    urlState() {
        const params = new URLSearchParams(window.location.search);
        return { kbId: params.get("kb"), sessionId: params.get("session") };
    }

    syncUrl() {
        const url = new URL(window.location.href);
        if (this.currentKb?.id) url.searchParams.set("kb", this.currentKb.id);
        else url.searchParams.delete("kb");
        if (this.currentSession?.id) url.searchParams.set("session", this.currentSession.id);
        else url.searchParams.delete("session");
        window.history.replaceState(null, "", url);
    }

    async restoreSessionFromUrl() {
        const state = this.urlState();
        if (!state.sessionId) return;
        const session = this.sessions.find((item) => item.id === state.sessionId);
        if (session) await this.openSession(session.id, { syncUrl: false });
    }

    async loadKnowledgeBases() {
        const data = await this.api("/kb");
        this.knowledgeBases = data.knowledge_bases || [];
        const preferredId = this.currentKb?.id || this.urlState().kbId;
        this.currentKb = this.knowledgeBases.find((kb) => kb.id === preferredId)
            || this.knowledgeBases.find((kb) => kb.is_default)
            || this.knowledgeBases[0]
            || null;
        this.renderKnowledgeBases();
        this.renderCurrentKb();
    }

    renderKnowledgeBases() {
        const list = this.el["kb-list"];
        if (!list) return;
        if (!this.knowledgeBases.length) {
            list.innerHTML = '<li class="list-empty">暂无知识库</li>';
            return;
        }
        list.innerHTML = this.knowledgeBases.map((kb) => `
            <li class="list-item ${this.currentKb?.id === kb.id ? "active" : ""}" data-kb-row="${this.esc(kb.id)}">
                <button class="item-select" type="button" data-kb="${this.esc(kb.id)}"
                        ${this.currentKb?.id === kb.id ? 'aria-current="true"' : ""}>
                    <span class="item-main">${this.esc(kb.name)}</span>
                    <span class="item-meta">${Number(kb.doc_count || 0)}</span>
                </button>
                ${kb.is_default ? "" : `<button class="item-del" data-del-kb="${this.esc(kb.id)}" aria-label="删除知识库" type="button">×</button>`}
            </li>
        `).join("");

        list.querySelectorAll("[data-kb]").forEach((button) => {
            button.addEventListener("click", () => this.selectKb(button.dataset.kb));
        });
        list.querySelectorAll("[data-del-kb]").forEach((button) => {
            button.addEventListener("click", () => this.deleteKb(button.dataset.delKb));
        });
    }

    renderCurrentKb() {
        if (!this.currentKb) return;
        this.el["current-kb-name"].textContent = this.currentKb.name;
        this.el["kb-meta"].textContent = `${Number(this.currentKb.doc_count || 0)} 文档 · ${Number(this.currentKb.chunk_count || 0)} 分块`;
    }

    async selectKb(kbId) {
        const kb = this.knowledgeBases.find((item) => item.id === kbId);
        if (!kb) return;
        this.currentKb = kb;
        this.setConversationMode("knowledge");
        this.currentSession = null;
        this.renderKnowledgeBases();
        this.renderCurrentKb();
        this.clearMessages();
        this.syncUrl();
        this.setSidebarOpen(false);
        await this.loadSessions();
    }

    async promptNewKb() {
        const name = prompt("知识库名称");
        if (!name?.trim()) return;
        try {
            const kb = await this.api("/kb", { method: "POST", body: { name: name.trim(), description: "" } });
            await this.loadKnowledgeBases();
            if (kb?.id) await this.selectKb(kb.id);
        } catch (error) {
            this.showAlert(`创建失败：${error.message}`, "error");
        }
    }

    async deleteKb(kbId) {
        const kb = this.knowledgeBases.find((item) => item.id === kbId);
        if (!kb || !confirm(`删除「${kb.name}」及其中的资料？`)) return;
        try {
            await this.api(`/kb/${kbId}`, { method: "DELETE" });
            if (this.activeAgent?.kb_id === kbId) this.activeAgent = null;
            if (this.currentKb?.id === kbId) this.currentKb = null;
            this.currentSession = null;
            await this.loadKnowledgeBases();
            await this.loadAgents();
            await this.loadSessions();
            this.setConversationMode("knowledge");
            this.clearMessages();
            this.syncUrl();
        } catch (error) {
            this.showAlert(`删除失败：${error.message}`, "error");
        }
    }

    async loadSessions() {
        if (!this.currentKb) {
            this.sessions = [];
            this.renderSessions();
            return;
        }
        try {
            const data = await this.api(`/sessions?kb_id=${encodeURIComponent(this.currentKb.id)}`);
            this.sessions = data.sessions || [];
            if (this.currentSession) {
                this.currentSession = this.sessions.find((item) => item.id === this.currentSession.id) || this.currentSession;
            }
        } catch {
            this.sessions = [];
        }
        this.renderSessions();
    }

    renderSessions() {
        const list = this.el["session-list"];
        this.syncExportButton();
        if (!list) return;
        if (!this.sessions.length) {
            list.innerHTML = '<li class="list-empty">还没有对话</li>';
            return;
        }
        list.innerHTML = this.sessions.map((session) => `
            <li class="list-item ${this.currentSession?.id === session.id ? "active" : ""}" data-session-row="${this.esc(session.id)}">
                <button class="item-select" type="button" data-session="${this.esc(session.id)}"
                        title="${this.esc(session.preview || session.title)}"
                        aria-label="${this.esc(session.title || "新对话")}"
                        ${this.currentSession?.id === session.id ? 'aria-current="true"' : ""}>
                    <span class="item-main">${this.esc(session.title || "新对话")}</span>
                </button>
                <button class="item-ren" data-ren="${this.esc(session.id)}" aria-label="重命名" type="button">✎</button>
                <button class="item-del" data-del-session="${this.esc(session.id)}" aria-label="删除" type="button">×</button>
            </li>
        `).join("");

        list.querySelectorAll("[data-session]").forEach((button) => {
            button.addEventListener("click", () => this.openSession(button.dataset.session));
        });
        list.querySelectorAll("[data-ren]").forEach((button) => {
            button.addEventListener("click", () => this.renameSession(button.dataset.ren));
        });
        list.querySelectorAll("[data-del-session]").forEach((button) => {
            button.addEventListener("click", () => this.deleteSession(button.dataset.delSession));
        });
    }

    async newSession() {
        if (!this.currentKb) {
            this.showAlert("请先选择知识库", "error");
            return null;
        }
        try {
            const session = await this.api("/sessions", {
                method: "POST",
                body: { kb_id: this.currentKb.id },
            });
            this.currentSession = session;
            this.clearMessages();
            await this.loadSessions();
            this.syncUrl();
            this.setSidebarOpen(false);
            this.el["chat-input"].focus();
            return session;
        } catch (error) {
            this.showAlert(`新建对话失败：${error.message}`, "error");
            return null;
        }
    }

    async openSession(sessionId, { syncUrl = true } = {}) {
        this.currentSession = this.sessions.find((item) => item.id === sessionId) || null;
        if (!this.currentSession) return;
        this.renderSessions();
        this.clearMessages();
        if (syncUrl) this.syncUrl();
        this.setSidebarOpen(false);
        try {
            const data = await this.api(`/sessions/${sessionId}/messages`);
            let queryForAnswer = "";
            (data.messages || []).forEach((message) => {
                if (message.role === "user") queryForAnswer = message.content;
                const node = this.appendMessage(message.role, message.content);
                if (message.role === "user") this.addUserActions(node, message.content);
                if (message.role === "assistant") {
                    const sources = message.sources || [];
                    this.addMessageActions(node, queryForAnswer, sources);
                    if (sources.length) this.showSources(sources);
                }
            });
            this.scrollBottom();
        } catch (error) {
            this.showAlert(`加载历史失败：${error.message}`, "error");
        }
    }

    syncExportButton() {
        const button = this.el["export-btn"];
        if (button) button.disabled = !this.currentSession?.id;
    }

    exportSession() {
        if (!this.currentSession?.id) return;
        const link = document.createElement("a");
        link.href = `/api/v1/sessions/${encodeURIComponent(this.currentSession.id)}/export.md`;
        link.rel = "noopener";
        document.body.appendChild(link);
        link.click();
        link.remove();
    }

    async renameSession(sessionId) {
        const session = this.sessions.find((item) => item.id === sessionId);
        const title = prompt("对话名称", session?.title || "");
        if (!title?.trim()) return;
        try {
            await this.api(`/sessions/${sessionId}`, { method: "PATCH", body: { title: title.trim() } });
            await this.loadSessions();
        } catch (error) {
            this.showAlert(`重命名失败：${error.message}`, "error");
        }
    }

    async deleteSession(sessionId) {
        if (!confirm("删除这段对话？")) return;
        try {
            await this.api(`/sessions/${sessionId}`, { method: "DELETE" });
            if (this.currentSession?.id === sessionId) {
                this.currentSession = null;
                this.clearMessages();
            }
            await this.loadSessions();
            this.syncUrl();
        } catch (error) {
            this.showAlert(`删除失败：${error.message}`, "error");
        }
    }

    connectWebSocket() {
        if (this.ws && [WebSocket.OPEN, WebSocket.CONNECTING].includes(this.ws.readyState)) return;
        clearTimeout(this.reconnectTimer);
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        try {
            this.ws = new WebSocket(`${protocol}//${window.location.host}/ws/chat`);
        } catch {
            this.updateStatus("连接失败", false);
            this.scheduleReconnect();
            return;
        }
        this.ws.onopen = () => {
            this.connected = true;
            this.updateStatus("已连接", true);
        };
        this.ws.onmessage = (event) => {
            try {
                this.handleEvent(JSON.parse(event.data)).catch((error) => console.error(error));
            } catch (error) {
                console.error(error);
            }
        };
        this.ws.onclose = () => {
            this.connected = false;
            this.updateStatus("连接中断", false);
            if (this.isStreaming) {
                this.isStreaming = false;
                this.activeRequestId = null;
                this.hideTypingIndicator();
                this.finalizeAssistant();
                this.setInputBusy(false);
            }
            this.scheduleReconnect();
        };
        this.ws.onerror = () => this.updateStatus("连接错误", false);
    }

    async ensureAgentWebSocket() {
        if (this.agentWs?.readyState === WebSocket.OPEN) return this.agentWs;
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const socket = new WebSocket(`${protocol}//${window.location.host}/api/v1/ws/agent`);
        this.agentWs = socket;
        socket.onmessage = (event) => {
            try {
                this.handleEvent(JSON.parse(event.data)).catch((error) => console.error(error));
            } catch (error) {
                console.error(error);
            }
        };
        socket.onclose = () => {
            if (this.agentMode && this.isStreaming) {
                this.isStreaming = false;
                this.activeRequestId = null;
                this.activeSocket = null;
                this.hideTypingIndicator();
                this.finalizeAssistant();
                this.setInputBusy(false);
                this.persistAgentTurn().catch(() => {});
            }
        };
        await new Promise((resolve, reject) => {
            const timer = setTimeout(() => reject(new Error("Agent 连接超时")), 5000);
            socket.onopen = () => {
                clearTimeout(timer);
                resolve();
            };
            socket.onerror = () => {
                clearTimeout(timer);
                reject(new Error("Agent 连接失败"));
            };
        });
        return socket;
    }

    scheduleReconnect() {
        clearTimeout(this.reconnectTimer);
        this.reconnectTimer = setTimeout(() => this.connectWebSocket(), 3000);
    }

    updateStatus(text, connected) {
        if (!this.el.status) return;
        this.el.status.textContent = text;
        this.el.status.className = `status ${connected ? "connected" : "disconnected"}`;
    }

    async sendMessage(queryOverride = null) {
        if (this.isStreaming) return;
        const input = this.el["chat-input"];
        const query = String(queryOverride ?? input.value).trim();
        if (!query) return;
        if (!this.currentKb) {
            this.showAlert("请先选择知识库", "error");
            return;
        }
        if (this.agentMode && !this.activeAgent) {
            this.openAgents();
            return;
        }
        if (!this.agentMode && !this.connected) {
            this.showAlert("连接尚未就绪", "error");
            return;
        }
        if (!this.currentSession && !(await this.newSession())) return;

        let socket = this.ws;
        if (this.agentMode) {
            try {
                socket = await this.ensureAgentWebSocket();
            } catch (error) {
                this.showAlert(error.message, "error");
                return;
            }
        }

        const userMessage = this.appendMessage("user", query);
        this.addUserActions(userMessage, query);
        input.value = "";
        this.resizeComposer();
        this.isStreaming = true;
        this.activeRequestId = window.crypto?.randomUUID?.()
            || `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
        this.activeSocket = socket;
        this.lastUserQuery = query;
        this.currentSources = [];
        this.pendingAgentPersist = this.agentMode ? { query } : null;
        this.setInputBusy(true);
        this.showTypingIndicator();
        socket.send(JSON.stringify({
            query,
            top_k: this.topK,
            session_id: this.currentSession.id,
            kb_id: this.currentKb.id,
            direct: this.directMode,
            agent_id: this.agentMode ? this.activeAgent.id : null,
            request_id: this.activeRequestId,
        }));
    }

    async handleEvent(payload) {
        const { event, data } = payload;
        if (payload.request_id && payload.request_id !== this.activeRequestId) return;
        if (event === "hits") {
            this.currentSources = data || [];
        } else if (event === "token") {
            this.hideTypingIndicator();
            this.appendToken(String(data ?? ""));
        } else if (event === "done") {
            this.hideTypingIndicator();
            this.isStreaming = false;
            this.activeRequestId = null;
            this.finalizeAssistant();
            this.showSources(this.currentSources);
            await this.persistAgentTurn();
            this.setInputBusy(false);
            this.loadSessions();
            this.loadKnowledgeBases();
        } else if (event === "error") {
            this.hideTypingIndicator();
            this.isStreaming = false;
            this.activeRequestId = null;
            this.finalizeAssistant();
            this.setInputBusy(false);
            await this.persistAgentTurn();
            this.showAlert(`回答失败：${data}`, "error");
        } else if (event === "cancelled") {
            this.finishStoppedStream();
        } else if (event === "metadata" && Array.isArray(data?.sources)) {
            this.currentSources = data.sources;
        } else if (event === "action" && data?.tool) {
            const indicator = document.querySelector("#typing-indicator .typing-indicator");
            if (indicator) indicator.setAttribute("aria-label", `正在使用 ${data.tool}`);
        }
    }

    setInputBusy(busy) {
        this.el["chat-input"].disabled = busy;
        this.el["send-btn"].hidden = busy;
        this.el["stop-btn"].hidden = !busy;
        this.el["send-btn"].disabled = false;
        if (!busy) this.el["chat-input"].focus();
    }

    stopStreaming() {
        if (!this.isStreaming) return;
        const requestId = this.activeRequestId;
        const socket = this.activeSocket || (this.agentMode ? this.agentWs : this.ws);
        if (requestId && socket?.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ event: "cancel", request_id: requestId }));
        }
        this.activeRequestId = null;
        this.activeSocket = null;
        this.finishStoppedStream();
    }

    finishStoppedStream() {
        if (!this.isStreaming && this.activeRequestId == null) return;
        this.hideTypingIndicator();
        this.isStreaming = false;
        this.activeRequestId = null;
        this.finalizeAssistant();
        if (this.currentSources.length) this.showSources(this.currentSources);
        this.setInputBusy(false);
        this.persistAgentTurn().catch(() => {});
    }

    async persistAgentTurn() {
        const pending = this.pendingAgentPersist;
        this.pendingAgentPersist = null;
        if (!pending || !this.currentSession?.id) return;
        try {
            await this.api(`/sessions/${this.currentSession.id}/messages`, {
                method: "POST",
                body: { role: "user", content: pending.query },
            });
            const assistant = Array.from(this.el.messages.querySelectorAll(".message.assistant .message-content")).pop();
            const answer = assistant?.textContent?.trim() || "";
            if (answer) {
                await this.api(`/sessions/${this.currentSession.id}/messages`, {
                    method: "POST",
                    body: { role: "assistant", content: answer, sources: [] },
                });
            }
        } catch (error) {
            console.error("Agent 消息保存失败", error);
        }
    }

    resizeComposer() {
        const input = this.el["chat-input"];
        if (!input) return;
        input.style.height = "auto";
        input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
    }

    clearMessages() {
        this.el.messages.innerHTML = `
            <div class="empty-state" id="welcome-state">
                <h2>开始提问</h2>
                <div class="empty-actions">
                    <button type="button" data-upload>${this.icon("upload")}添加文档</button>
                    <button type="button" data-import-url>${this.icon("link")}导入网页</button>
                    <button type="button" data-search>${this.icon("search")}搜索资料</button>
                </div>
            </div>`;
        this.el.messages.querySelectorAll("[data-upload]").forEach((button) => {
            button.addEventListener("click", () => this.el["file-input"].click());
        });
        this.el.messages.querySelectorAll("[data-import-url]").forEach((button) => {
            button.addEventListener("click", () => this.openImport());
        });
        this.el.messages.querySelectorAll("[data-search]").forEach((button) => {
            button.addEventListener("click", () => this.openSearch());
        });
        this.currentSources = [];
    }

    appendMessage(role, content) {
        this.el.messages.querySelector(".empty-state")?.remove();
        const wrap = document.createElement("div");
        wrap.className = `message ${role}`;
        const body = document.createElement("div");
        body.className = "message-content";
        if (role === "assistant") this.setRichText(body, content);
        else body.textContent = content;
        wrap.appendChild(body);
        this.el.messages.appendChild(wrap);
        this.scrollBottom();
        return wrap;
    }

    appendToken(token) {
        let message = this.el.messages.querySelector(".message.assistant.streaming");
        if (!message) {
            message = document.createElement("div");
            message.className = "message assistant streaming";
            const body = document.createElement("div");
            body.className = "message-content";
            message.appendChild(body);
            this.el.messages.appendChild(message);
        }
        message.querySelector(".message-content").textContent += token;
        this.scrollBottom();
    }

    finalizeAssistant() {
        this.el.messages.querySelectorAll(".message.assistant.streaming").forEach((message) => {
            message.classList.remove("streaming");
            const body = message.querySelector(".message-content");
            this.setRichText(body, body.textContent);
            this.addMessageActions(message, this.lastUserQuery, this.currentSources);
        });
    }

    showTypingIndicator() {
        this.hideTypingIndicator();
        const message = document.createElement("div");
        message.className = "message assistant";
        message.id = "typing-indicator";
        message.innerHTML = '<div class="typing-indicator" aria-label="正在回答"><span></span><span></span><span></span></div>';
        this.el.messages.appendChild(message);
        this.scrollBottom();
    }

    hideTypingIndicator() {
        document.getElementById("typing-indicator")?.remove();
    }

    showSources(sources) {
        if (!sources?.length) return;
        const box = document.createElement("details");
        box.className = "message sources";
        const summary = document.createElement("summary");
        summary.textContent = `来源 ${sources.length}`;
        box.appendChild(summary);
        const list = document.createElement("ul");
        sources.forEach((source, index) => {
            const item = document.createElement("li");
            const button = document.createElement("button");
            button.type = "button";
            button.className = "source-link";
            const score = typeof source.score === "number" ? this.scoreLabel(source) : "";
            button.innerHTML = `<strong>${this.esc(source.index || index + 1)}</strong><span>${this.esc(source.title || source.source || "资料")}</span><span class="score">${this.esc(score)}</span>`;
            button.addEventListener("click", () => this.showChunkDetail(source));
            item.appendChild(button);
            list.appendChild(item);
        });
        box.appendChild(list);
        this.el.messages.appendChild(box);
        this.scrollBottom();
    }

    scoreLabel(source) {
        const label = source.score_kind === "keyword" ? "词面" : source.score_kind === "rerank" ? "相关" : "相近";
        return `${label} ${source.score.toFixed(2)}`;
    }

    async showChunkDetail(source) {
        const panel = this.el["citation-panel"];
        if (!panel) return;
        const sequence = ++this.citationSequence;
        const metadata = source.metadata || {};
        const sourcePath = source.source || source.filepath || metadata.source || metadata.filepath || "";
        const title = source.title || metadata.title || sourcePath || "引用";
        this.el["citation-title"].textContent = title;
        this.el["citation-source"].textContent = sourcePath;
        this.el["citation-source"].title = sourcePath;
        this.el["citation-content"].innerHTML = `<div class="citation-loading">${this.esc(source.snippet || source.text || "")}</div>`;
        panel.hidden = false;

        if (!sourcePath) {
            this.renderCitationFallback(source);
            return;
        }
        const params = new URLSearchParams({ limit: "200" });
        if (this.currentKb?.id) params.set("kb_id", this.currentKb.id);
        try {
            const data = await this.api(`/documents/${encodeURIComponent(sourcePath)}/chunks?${params}`);
            if (sequence !== this.citationSequence) return;
            this.renderCitationChunks(data.chunks || [], source);
        } catch (error) {
            if (sequence !== this.citationSequence) return;
            this.renderCitationFallback(source, error.message);
        }
    }

    renderCitationChunks(chunks, source) {
        const box = this.el["citation-content"];
        if (!chunks.length) {
            this.renderCitationFallback(source);
            return;
        }
        const metadata = source.metadata || {};
        const citedId = source.chunk_id || metadata.chunk_id || source.id || "";
        const citedIndex = source.chunk_index ?? metadata.chunk_index;
        const citedText = source.snippet || source.text || source.content || "";
        const sorted = chunks.slice().sort((a, b) => Number(a.metadata?.chunk_index || 0) - Number(b.metadata?.chunk_index || 0));
        let selected = citedId ? sorted.findIndex((chunk) => chunk.id === citedId) : -1;
        if (selected < 0 && citedIndex != null) {
            selected = sorted.findIndex((chunk) => Number(chunk.metadata?.chunk_index) === Number(citedIndex));
        }
        if (selected < 0 && citedText) {
            selected = sorted.findIndex((chunk) => chunk.text === citedText || String(chunk.text || "").startsWith(citedText));
        }
        box.innerHTML = sorted.map((chunk, index) => {
            const section = chunk.metadata?.chunk_prefix || chunk.metadata?.breadcrumb || (index === selected ? "引用原文" : "");
            return `<article class="citation-chunk ${index === selected ? "active" : ""}" data-citation-chunk="${this.esc(chunk.id)}"
                    ${index === selected ? 'aria-current="true"' : ""}>
                <div class="citation-chunk-head">
                    <span>${this.esc(section || `第 ${index + 1} 段`)}</span>
                    <span>${index + 1}/${sorted.length}</span>
                </div>
                <div class="citation-chunk-text">${this.esc(chunk.text || "")}</div>
            </article>`;
        }).join("");
        const active = box.querySelector(".citation-chunk.active");
        requestAnimationFrame(() => active?.scrollIntoView({ block: "center" }));
    }

    renderCitationFallback(source, error = "") {
        const text = source.snippet || source.content || source.text || "";
        this.el["citation-content"].innerHTML = text
            ? `<article class="citation-chunk active"><div class="citation-chunk-text">${this.esc(text)}</div></article>`
            : `<div class="citation-empty">${this.esc(error || "原文不可用")}</div>`;
    }

    closeCitation() {
        const panel = this.el["citation-panel"];
        if (!panel) return;
        this.citationSequence += 1;
        panel.hidden = true;
    }

    addUserActions(message, query) {
        if (!message || message.querySelector(".message-actions")) return;
        const actions = document.createElement("div");
        actions.className = "message-actions";
        const edit = this.actionButton("edit", "编辑并重发");
        edit.addEventListener("click", () => this.editQuery(query));
        const regenerate = this.actionButton("refresh", "重新生成");
        regenerate.addEventListener("click", () => this.regenerate(query));
        actions.append(edit, regenerate);
        message.appendChild(actions);
    }

    editQuery(query) {
        if (this.isStreaming) this.stopStreaming();
        const input = this.el["chat-input"];
        input.value = query;
        this.resizeComposer();
        input.focus();
        input.setSelectionRange(input.value.length, input.value.length);
    }

    async regenerate(query) {
        if (this.isStreaming) this.stopStreaming();
        const input = this.el["chat-input"];
        input.value = query;
        this.resizeComposer();
        await this.sendMessage();
    }

    addMessageActions(message, query, sources = []) {
        if (!message || message.querySelector(".message-actions")) return;
        const actions = document.createElement("div");
        actions.className = "message-actions";
        const answer = message.querySelector(".message-content")?.textContent || "";

        const copy = this.actionButton("copy", "复制");
        copy.addEventListener("click", async () => {
            try {
                await navigator.clipboard.writeText(answer);
                copy.setAttribute("aria-label", "已复制");
                setTimeout(() => copy.setAttribute("aria-label", "复制"), 1200);
            } catch {
                this.showAlert("复制失败", "error");
            }
        });

        const helpful = this.actionButton("up", "有帮助");
        const notHelpful = this.actionButton("down", "没帮助");
        helpful.addEventListener("click", () => this.submitFeedback(helpful, notHelpful, "helpful", query, answer, sources));
        notHelpful.addEventListener("click", () => this.submitFeedback(notHelpful, helpful, "not_helpful", query, answer, sources));

        const regenerate = this.actionButton("refresh", "重新生成");
        regenerate.addEventListener("click", () => this.regenerate(query));

        actions.append(copy, helpful, notHelpful, regenerate);
        message.appendChild(actions);
    }

    actionButton(icon, label) {
        const button = document.createElement("button");
        button.type = "button";
        button.setAttribute("aria-label", label);
        button.title = label;
        button.innerHTML = this.icon(icon);
        return button;
    }

    async submitFeedback(selected, other, rating, query, answer, sources) {
        if (!query) return;
        try {
            await this.api("/feedback", {
                method: "POST",
                body: {
                    query,
                    answer,
                    sources_json: JSON.stringify(sources || []),
                    rating,
                    correction: "",
                    question_type: "",
                },
            });
            selected.setAttribute("aria-pressed", "true");
            other.setAttribute("aria-pressed", "false");
            this.showAlert("已记录");
        } catch (error) {
            this.showAlert(`反馈失败：${error.message}`, "error");
        }
    }

    scrollBottom() {
        this.el.messages.scrollTop = this.el.messages.scrollHeight;
    }

    openSearch() {
        this.openModalElement(this.el["search-modal"]);
        this.el["search-input"].value = "";
        this.el["search-results"].innerHTML = "";
        requestAnimationFrame(() => this.el["search-input"].focus());
    }

    scheduleSearch() {
        clearTimeout(this.searchTimer);
        const query = this.el["search-input"].value.trim();
        if (query.length < 2) {
            this.searchHits = [];
            this.el["search-results"].innerHTML = "";
            return;
        }
        this.searchTimer = setTimeout(() => this.runSearch(query), 180);
    }

    async runSearch(query) {
        const sequence = ++this.searchSequence;
        this.el["search-results"].innerHTML = '<div class="command-empty">搜索中</div>';
        try {
            const data = await this.api("/search", {
                method: "POST",
                body: {
                    query,
                    top_k: Math.min(this.topK, 8),
                    kb_id: this.currentKb?.id || null,
                },
            });
            if (sequence !== this.searchSequence) return;
            this.searchHits = data.hits || [];
            this.renderSearchResults();
        } catch (error) {
            if (sequence !== this.searchSequence) return;
            this.el["search-results"].innerHTML = `<div class="command-empty">${this.esc(error.message)}</div>`;
        }
    }

    renderSearchResults() {
        const box = this.el["search-results"];
        if (!this.searchHits.length) {
            box.innerHTML = '<div class="command-empty">无结果</div>';
            return;
        }
        box.innerHTML = this.searchHits.map((hit, index) => `
            <button class="command-result" type="button" data-hit="${index}">
                <span class="command-result-head">
                    <span>${this.esc(hit.metadata?.title || hit.metadata?.source || "资料")}</span>
                    <span>${this.esc(hit.metadata?.breadcrumb || hit.metadata?.source || "")}</span>
                </span>
                <p>${this.esc(hit.text || "")}</p>
            </button>
        `).join("");
        box.querySelectorAll("[data-hit]").forEach((button) => {
            button.addEventListener("click", () => {
                const hit = this.searchHits[Number(button.dataset.hit)];
                this.closeAllModals();
                if (hit) this.showChunkDetail({ ...hit, title: hit.metadata?.title || hit.metadata?.source });
            });
        });
    }

    openImport() {
        if (!this.currentKb) {
            this.showAlert("请先选择知识库", "error");
            return;
        }
        this.openModalElement(this.el["import-modal"]);
        this.el["import-url-input"].value = "";
        requestAnimationFrame(() => this.el["import-url-input"].focus());
    }

    async importUrl(url, kbId = this.currentKb?.id) {
        if (!url || !kbId) return;
        const button = this.el["import-url-submit"];
        button.disabled = true;
        const oldLabel = button.textContent;
        button.textContent = "导入中";
        try {
            let result = await this.api("/ingest/url", {
                method: "POST",
                body: { url, kb_id: kbId, background: true },
            });
            if (result.job_id) {
                result = await this.waitForIngestJob(result.job_id, result.title || url);
            }
            this.closeAllModals();
            this.showAlert(`已导入 ${result.title || url}`);
            await this.loadKnowledgeBases();
        } catch (error) {
            this.showAlert(`导入失败：${error.message}`, "error");
        } finally {
            button.disabled = false;
            button.textContent = oldLabel;
        }
    }

    async uploadFiles(files) {
        if (!files?.length) return;
        if (!this.currentKb) {
            this.showAlert("请先选择知识库", "error");
            return;
        }
        const controls = [this.el["upload-btn"], ...document.querySelectorAll("[data-upload]")];
        controls.forEach((button) => { button.disabled = true; });
        try {
            const queued = [];
            const failures = [];
            for (const file of Array.from(files)) {
                const form = new FormData();
                form.append("file", file);
                form.append("kb_id", this.currentKb.id);
                form.append("background", "true");
                try {
                    const job = await this.api("/ingest", { method: "POST", body: form, raw: true });
                    queued.push({ file, job });
                } catch (error) {
                    failures.push(`${file.name}：${error.message}`);
                }
            }

            let docs = 0;
            let ingested = 0;
            let changed = 0;
            let unchanged = 0;
            for (let index = 0; index < queued.length; index += 1) {
                const { file, job } = queued[index];
                try {
                    const result = await this.waitForIngestJob(job.job_id, file.name, index + 1, queued.length);
                    docs += Number(result.docs || 0);
                    ingested += Number(result.ingested ?? result.chunks ?? 0);
                    changed += Number(result.changed || 0);
                    unchanged += Number(result.unchanged || 0);
                } catch (error) {
                    failures.push(`${file.name}：${error.message}`);
                }
            }
            if (failures.length) {
                const summary = this.ingestResultText({ docs, ingested, changed, unchanged });
                const prefix = summary ? `${summary}；` : "";
                this.showAlert(`${prefix}${failures.length} 个文件失败：${failures[0]}`, "error");
            } else if (docs || ingested || changed || unchanged) {
                const summary = this.ingestResultText({ docs, ingested, changed, unchanged });
                this.showAlert(summary);
            }
            await this.loadKnowledgeBases();
            if (this.el["docs-modal"].classList.contains("active")) await this.loadDocs();
        } catch (error) {
            this.showAlert(`添加失败：${error.message}`, "error");
        } finally {
            controls.forEach((button) => { button.disabled = false; });
            this.setIngestStatus("", 0, false);
        }
    }

    async waitForIngestJob(jobId, label, index = 1, total = 1) {
        if (!jobId) throw new Error("缺少入库任务");
        const stageLabels = { queued: "排队", started: "处理", parse: "解析", chunk: "分块", upsert: "写入", invalidate: "收尾", done: "完成" };
        while (true) {
            const job = await this.api(`/ingest/jobs/${encodeURIComponent(jobId)}`);
            const progress = Math.round(Number(job.progress || 0) * 100);
            const stage = stageLabels[job.stage] || job.stage || "处理";
            this.setIngestStatus(`${label} · ${stage}${total > 1 ? ` ${index}/${total}` : ""}`, progress, true);
            if (job.status === "done") {
                this.setIngestStatus(`${label} · 完成`, 100, true);
                return job.result || {};
            }
            if (["failed", "interrupted"].includes(job.status)) {
                throw new Error(job.error || "任务已中断");
            }
            await new Promise((resolve) => setTimeout(resolve, 700));
        }
    }

    setIngestStatus(text, progress, visible) {
        if (!this.el["ingest-status"]) return;
        this.el["ingest-status"].hidden = !visible;
        this.el["ingest-status-text"].textContent = text;
        this.el["ingest-progress"].value = progress;
    }

    ingestResultText(result) {
        const hasIncremental = result.changed != null || result.unchanged != null;
        const changed = Number(result.changed || 0);
        const unchanged = Number(result.unchanged || 0);
        const written = Number(result.ingested ?? result.chunks ?? 0);
        if (hasIncremental && !changed && unchanged) {
            return "资料未变化";
        }
        if (hasIncremental && changed) {
            const skipped = unchanged ? ` · ${unchanged} 个未变化` : "";
            return `已入库 ${changed} 个来源 · ${written} 个分块${skipped}`;
        }
        const docs = Number(result.docs || 0);
        return docs || written ? `${docs} 篇 · ${written} 个分块` : "";
    }

    toggleModeMenu() {
        const open = this.el["mode-menu"].hidden;
        this.el["mode-menu"].hidden = !open;
        this.el["mode-btn"].setAttribute("aria-expanded", String(open));
        if (open) this.el["mode-menu"].querySelector("button")?.focus();
    }

    closeModeMenu() {
        if (!this.el["mode-menu"] || this.el["mode-menu"].hidden) return;
        this.el["mode-menu"].hidden = true;
        this.el["mode-btn"].setAttribute("aria-expanded", "false");
    }

    chooseConversationMode(mode) {
        if (mode === "agent") {
            if (this.activeAgent && !this.agentMode) this.setConversationMode("agent");
            else this.openAgents();
            return;
        }
        this.setConversationMode(mode);
    }

    setConversationMode(mode) {
        this.directMode = mode === "direct";
        this.agentMode = mode === "agent" && !!this.activeAgent;
        this.el["mode-label"].textContent = this.agentMode
            ? this.activeAgent.name
            : (this.directMode ? "模型" : "知识库");
        this.el["mode-menu"].querySelectorAll("[data-mode]").forEach((button) => {
            button.setAttribute("aria-checked", String(button.dataset.mode === mode));
        });
        this.el["mode-btn"].title = this.agentMode ? `${this.activeAgent.name} · ${this.currentKb?.name || ""}` : "";
    }

    async loadAgents() {
        try {
            const data = await this.api("/agents");
            this.agents = data.agents || [];
            if (this.activeAgent) {
                this.activeAgent = this.agents.find((agent) => agent.id === this.activeAgent.id) || null;
            }
            if (!this.activeAgent && this.agentMode) this.setConversationMode("knowledge");
        } catch {
            this.agents = [];
        }
    }

    async openAgents() {
        this.showAgentsList();
        await this.loadAgents();
        this.renderAgents();
        this.openModalElement(this.el["agents-modal"]);
    }

    showAgentsList() {
        this.el["agents-list-view"].hidden = false;
        this.el["agent-form"].hidden = true;
        this.renderAgents();
    }

    renderAgents() {
        const box = this.el["agents-list"];
        if (!box) return;
        if (!this.agents.length) {
            box.innerHTML = '<div class="command-empty">暂无 Agent</div>';
            return;
        }
        box.innerHTML = this.agents.map((agent) => {
            const kbName = this.knowledgeBases.find((kb) => kb.id === agent.kb_id)?.name || "知识库";
            return `<div class="agent-row ${this.activeAgent?.id === agent.id ? "active" : ""}" data-agent-row="${this.esc(agent.id)}">
                <button class="agent-select" type="button" data-agent-use="${this.esc(agent.id)}">
                    <span class="agent-name">${this.esc(agent.name)}</span>
                    <span class="agent-meta">${this.esc(kbName)} · ${agent.allowed_tools.length} 工具</span>
                </button>
                <button class="icon-button agent-edit" type="button" data-agent-edit="${this.esc(agent.id)}" aria-label="编辑 ${this.esc(agent.name)}">
                    <svg viewBox="0 0 20 20" aria-hidden="true"><path d="m12.5 4 3.5 3.5-8 8H4.5V12z"/><path d="m11 5.5 3.5 3.5"/></svg>
                </button>
            </div>`;
        }).join("");
        box.querySelectorAll("[data-agent-use]").forEach((button) => {
            button.addEventListener("click", () => {
                const agent = this.agents.find((item) => item.id === button.dataset.agentUse);
                if (agent) this.activateAgent(agent);
            });
        });
        box.querySelectorAll("[data-agent-edit]").forEach((button) => {
            button.addEventListener("click", () => {
                const agent = this.agents.find((item) => item.id === button.dataset.agentEdit);
                if (agent) this.startAgentEditor(agent);
            });
        });
    }

    async activateAgent(agent) {
        this.activeAgent = agent;
        const kb = this.knowledgeBases.find((item) => item.id === agent.kb_id);
        if (kb) this.currentKb = kb;
        this.currentSession = null;
        this.renderKnowledgeBases();
        this.renderCurrentKb();
        this.clearMessages();
        this.syncUrl();
        await this.loadSessions();
        this.setConversationMode("agent");
        this.closeAllModals();
        this.setSidebarOpen(false);
        this.el["chat-input"].focus();
    }

    async startAgentEditor(agent = null) {
        this.el["agents-list-view"].hidden = true;
        this.el["agent-form"].hidden = false;
        this.el["agent-id"].value = agent?.id || "";
        this.el["agent-name"].value = agent?.name || "";
        this.el["agent-kb"].innerHTML = this.knowledgeBases.map((kb) =>
            `<option value="${this.esc(kb.id)}">${this.esc(kb.name)}</option>`
        ).join("");
        this.el["agent-kb"].value = agent?.kb_id || this.currentKb?.id || this.knowledgeBases[0]?.id || "";
        this.el["agent-instructions"].value = agent?.instructions || "";
        this.el["agent-delete"].hidden = !agent;
        this.el["agent-form-result"].textContent = "";
        this.agentMaxSteps = agent?.max_steps || 5;
        const defaults = ["search_documents", "list_documents", "get_document_chunks", "calculator", "get_current_time", "text_stats"];
        this.agentSelectedTools = new Set(agent?.allowed_tools || defaults);
        this.agentToolsRendered = false;
        await this.loadAgentToolOptions();
        requestAnimationFrame(() => this.el["agent-name"].focus());
    }

    async loadAgentToolOptions() {
        const selected = this.el["agent-tools"].querySelectorAll("input:checked");
        if (this.agentToolsRendered) {
            this.agentSelectedTools = new Set(Array.from(selected).map((input) => input.value));
        }
        const selectedTools = this.agentSelectedTools || new Set();
        const kbId = this.el["agent-kb"].value;
        this.el["agent-tools"].innerHTML = '<span class="muted small">加载中</span>';
        let builtinTools = [];
        try {
            const params = new URLSearchParams();
            if (kbId) params.set("kb_id", kbId);
            const data = await this.api(`/agent/tools?${params}`);
            builtinTools = data.tools || [];
        } catch (error) {
            this.el["agent-form-result"].textContent = error.message;
        }
        const mcpTools = await this.loadMcpToolCatalog();
        const tools = [...builtinTools, ...mcpTools];
        const known = new Set(tools.map((tool) => tool.name));
        selectedTools.forEach((name) => {
            if (!known.has(name)) tools.push({ name, description: "当前不可用" });
        });
        const labels = {
            search_documents: "搜索资料", list_documents: "资料列表",
            get_document_chunks: "查看原文", calculator: "计算",
            get_current_time: "时间", text_stats: "文本统计", python_eval: "代码",
        };
        this.el["agent-tools"].innerHTML = tools.map((tool) => `
            <label class="agent-tool" title="${this.esc(tool.description || "")}">
                <input type="checkbox" value="${this.esc(tool.name)}" ${selectedTools.has(tool.name) ? "checked" : ""}>
                <span>${this.esc(labels[tool.name] || tool.name)}</span>
            </label>`).join("");
        this.agentSelectedTools = selectedTools;
        this.agentToolsRendered = true;
    }

    async saveAgent() {
        const id = this.el["agent-id"].value;
        const name = this.el["agent-name"].value.trim();
        const kbId = this.el["agent-kb"].value;
        if (!name || !kbId) return;
        const allowedTools = Array.from(this.el["agent-tools"].querySelectorAll("input:checked")).map((input) => input.value);
        const payload = {
            name,
            kb_id: kbId,
            instructions: this.el["agent-instructions"].value.trim(),
            allowed_tools: allowedTools,
            max_steps: this.agentMaxSteps || 5,
        };
        const button = this.el["agent-save"];
        button.disabled = true;
        this.el["agent-form-result"].textContent = "保存中";
        try {
            const saved = await this.api(id ? `/agents/${id}` : "/agents", {
                method: id ? "PATCH" : "POST",
                body: payload,
            });
            await this.loadAgents();
            await this.activateAgent(saved);
        } catch (error) {
            this.el["agent-form-result"].textContent = error.message;
        } finally {
            button.disabled = false;
        }
    }

    async deleteAgent() {
        const id = this.el["agent-id"].value;
        if (!id || !confirm("删除这个 Agent？")) return;
        try {
            await this.api(`/agents/${id}`, { method: "DELETE" });
            if (this.activeAgent?.id === id) {
                this.activeAgent = null;
                this.setConversationMode("knowledge");
            }
            await this.loadAgents();
            this.showAgentsList();
        } catch (error) {
            this.el["agent-form-result"].textContent = error.message;
        }
    }

    async openMcpServers() {
        this.el["mcp-form-result"].textContent = "";
        this.openModalElement(this.el["mcp-modal"]);
        await this.loadMcpServers();
    }

    async loadMcpServers() {
        const box = this.el["mcp-servers-list"];
        try {
            const data = await this.api("/mcp/servers");
            this.mcpServers = data.servers || [];
        } catch (error) {
            box.innerHTML = `<div class="command-empty">${this.esc(error.message)}</div>`;
            return;
        }
        if (!this.mcpServers.length) {
            box.innerHTML = '<div class="command-empty">暂无连接</div>';
            return;
        }
        box.innerHTML = this.mcpServers.map((server) => `
            <div class="mcp-row" data-mcp-row="${this.esc(server.id)}">
                <div class="mcp-main">
                    <span class="mcp-name">${this.esc(server.name)}</span>
                    <span class="mcp-meta">${server.enabled ? `${server.tool_count} 工具` : "已停用"}${server.last_error ? ` · ${this.esc(server.last_error)}` : ""}</span>
                    <code class="mcp-command">${this.esc(server.command)}</code>
                </div>
                <button class="btn btn-ghost btn-sm" type="button" data-mcp-test="${this.esc(server.id)}">检查</button>
                <button class="icon-button" type="button" data-mcp-delete="${this.esc(server.id)}" aria-label="删除 ${this.esc(server.name)}">
                    <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 6h12M8 6V4h4v2M6 6l1 10h6l1-10"/></svg>
                </button>
            </div>`).join("");
        box.querySelectorAll("[data-mcp-test]").forEach((button) => {
            button.addEventListener("click", () => this.testMcpServer(button.dataset.mcpTest, button));
        });
        box.querySelectorAll("[data-mcp-delete]").forEach((button) => {
            button.addEventListener("click", () => this.deleteMcpServer(button.dataset.mcpDelete));
        });
    }

    async createMcpServer() {
        const argsText = this.el["mcp-args"].value.trim();
        let args = [];
        if (argsText) {
            try {
                const parsed = JSON.parse(argsText);
                if (!Array.isArray(parsed)) throw new Error("参数需要 JSON 数组");
                args = parsed.map(String);
            } catch {
                args = argsText.split(/\r?\n/).map((item) => item.trim()).filter(Boolean);
            }
        }
        const submit = this.el["mcp-form"].querySelector("button[type=submit]");
        submit.disabled = true;
        this.el["mcp-form-result"].textContent = "添加中";
        try {
            await this.api("/mcp/servers", {
                method: "POST",
                body: {
                    name: this.el["mcp-name"].value.trim(),
                    command: this.el["mcp-command"].value.trim(),
                    args,
                },
            });
            this.el["mcp-form"].reset();
            this.el["mcp-form-result"].textContent = "已添加";
            await this.loadMcpServers();
        } catch (error) {
            this.el["mcp-form-result"].textContent = error.message;
        } finally {
            submit.disabled = false;
        }
    }

    async testMcpServer(serverId, button) {
        button.disabled = true;
        button.textContent = "检查中";
        try {
            const result = await this.api(`/mcp/servers/${serverId}/test`, { method: "POST" });
            this.showAlert(`${result.tools.length} 个工具`);
            await this.loadMcpServers();
        } catch (error) {
            this.showAlert(error.message, "error");
            await this.loadMcpServers();
        }
    }

    async deleteMcpServer(serverId) {
        if (!confirm("删除这个连接？")) return;
        try {
            await this.api(`/mcp/servers/${serverId}`, { method: "DELETE" });
            await this.loadMcpServers();
        } catch (error) {
            this.showAlert(error.message, "error");
        }
    }

    async loadMcpToolCatalog() {
        const servers = this.mcpServers.length ? this.mcpServers : (await this.api("/mcp/servers").catch(() => ({ servers: [] }))).servers || [];
        const enabled = servers.filter((server) => server.enabled);
        const results = await Promise.all(enabled.map(async (server) => {
            try {
                const data = await this.api(`/mcp/servers/${server.id}/tools`);
                return (data.tools || []).map((tool) => ({
                    ...tool,
                    description: `${server.name}：${tool.description || tool.remote_name}`,
                }));
            } catch {
                return [];
            }
        }));
        return results.flat();
    }

    setSidebarOpen(open) {
        document.body.classList.toggle("sidebar-open", !!open);
        this.el["sidebar-open"]?.setAttribute("aria-expanded", String(!!open));
    }

    async openSettings() {
        this.setSettingsResult("");
        try {
            const settings = await this.api("/settings/model");
            this.fillSettings(settings);
            this.openModalElement(this.el["settings-modal"]);
        } catch (error) {
            this.showAlert(`读取设置失败：${error.message}`, "error");
        }
    }

    fillSettings(settings) {
        ["llm", "emb"].forEach((kind) => {
            const section = (kind === "llm" ? settings.llm : settings.embedding) || {};
            const backend = section.backend || (kind === "llm" ? "extractive" : "onnx");
            const select = document.getElementById(`${kind}-backend`);
            if (select) select.value = backend;
            const config = section[backend] || {};
            this.setField(`${kind}-base-url`, config.base_url || "");
            this.setField(`${kind}-model`, config.model || "");
            this.setField(`${kind}-api-key`, "");
            const key = document.getElementById(`${kind}-api-key`);
            if (key) key.placeholder = config.api_key ? "已保存，留空不修改" : "可选";
            this.syncRemoteFields(kind);
        });
    }

    setField(id, value) {
        const element = document.getElementById(id);
        if (element) element.value = value;
    }

    syncRemoteFields(kind) {
        const backend = document.getElementById(`${kind}-backend`)?.value || (kind === "llm" ? "extractive" : "onnx");
        const container = document.getElementById(`${kind}-remote`);
        if (container) container.hidden = !["ollama", "openai"].includes(backend);
        const base = document.getElementById(`${kind}-base-url`);
        const model = document.getElementById(`${kind}-model`);
        if (backend === "ollama") {
            if (base && !base.value) base.value = "http://localhost:11434/v1";
            if (model && !model.value) model.placeholder = kind === "llm" ? "qwen3:4b-instruct-2507-q4_K_M" : "nomic-embed-text";
        } else if (backend === "openai") {
            if (base && (!base.value || base.value.includes("localhost"))) base.value = "https://api.deepseek.com/v1";
            if (model && !model.value) model.placeholder = kind === "llm" ? "deepseek-chat" : "text-embedding-3-small";
        }
    }

    collectKind(kind) {
        const backend = document.getElementById(`${kind}-backend`)?.value || (kind === "llm" ? "extractive" : "onnx");
        const result = { backend };
        if (["ollama", "openai"].includes(backend)) {
            const config = {
                base_url: document.getElementById(`${kind}-base-url`)?.value.trim() || "",
                model: document.getElementById(`${kind}-model`)?.value.trim() || "",
            };
            const apiKey = document.getElementById(`${kind}-api-key`)?.value.trim() || "";
            if (apiKey) config.api_key = apiKey;
            result[backend] = config;
        }
        return result;
    }

    async testModel(kind) {
        const config = this.collectKind(kind);
        const backend = config.backend;
        this.setSettingsResult("测试中");
        try {
            const result = await this.api("/settings/model/test", {
                method: "POST",
                body: {
                    kind: kind === "llm" ? "llm" : "embedding",
                    backend,
                    ...(config[backend] || {}),
                },
            });
            this.setSettingsResult(result.detail, result.ok ? "ok" : "err");
        } catch (error) {
            this.setSettingsResult(error.message, "err");
        }
    }

    async saveSettings() {
        const button = this.el["settings-save"];
        button.disabled = true;
        this.setSettingsResult("保存中");
        try {
            const result = await this.api("/settings/model", {
                method: "PUT",
                body: { llm: this.collectKind("llm"), embedding: this.collectKind("emb") },
            });
            this.setSettingsResult(result.warnings?.join(" ") || "已保存", "ok");
            await this.loadKnowledgeBases();
        } catch (error) {
            this.setSettingsResult(error.message, "err");
        } finally {
            button.disabled = false;
        }
    }

    async resetSettings() {
        if (!confirm("恢复配置文件中的模型设置？")) return;
        try {
            const result = await this.api("/settings/model", { method: "DELETE" });
            this.fillSettings(result);
            this.setSettingsResult("已恢复", "ok");
            await this.loadKnowledgeBases();
        } catch (error) {
            this.setSettingsResult(error.message, "err");
        }
    }

    setSettingsResult(text, kind = "") {
        if (!this.el["settings-result"]) return;
        this.el["settings-result"].textContent = text;
        this.el["settings-result"].className = `settings-result${kind ? ` settings-${kind}` : ""}`;
    }

    async openTasks() {
        this.openModalElement(this.el["tasks-modal"]);
        await this.loadIngestJobs();
    }

    async loadIngestJobs() {
        clearTimeout(this.tasksTimer);
        try {
            const data = await this.api("/ingest/jobs?limit=50");
            this.ingestJobs = data.jobs || [];
            this.renderIngestJobs();
            const busy = this.ingestJobs.some((job) => ["queued", "running"].includes(job.status));
            if (busy && this.el["tasks-modal"].classList.contains("active")) {
                this.tasksTimer = setTimeout(() => this.loadIngestJobs(), 1800);
            }
        } catch (error) {
            this.el["tasks-content"].innerHTML = `<div class="command-empty">${this.esc(error.message)}</div>`;
        }
    }

    renderIngestJobs() {
        const box = this.el["tasks-content"];
        if (!this.ingestJobs.length) {
            box.innerHTML = '<div class="command-empty">暂无任务</div>';
            return;
        }
        const labels = {
            queued: "排队", running: "处理中", done: "成功",
            failed: "失败", interrupted: "已中断",
        };
        box.innerHTML = this.ingestJobs.map((job, index) => {
            const status = labels[job.status] || job.status;
            const canRetry = ["failed", "interrupted"].includes(job.status) && ["file", "url", "dir"].includes(job.kind);
            const result = job.status === "done" && job.result
                ? this.ingestResultText(job.result)
                : "";
            const detail = job.error || result;
            return `<article class="task-row" data-task="${index}">
                <span class="task-state task-${this.esc(job.status)}"><span class="task-status"></span>${this.esc(status)}</span>
                <div class="task-main">
                    <div class="task-title">${this.esc(job.label || job.kind)}</div>
                    ${detail ? `<div class="task-detail">${this.esc(detail)}</div>` : ""}
                </div>
                <span class="task-time">${this.esc(this.jobTime(job.created_at))}</span>
                ${canRetry ? `<button class="btn btn-ghost btn-sm task-retry" type="button" data-retry="${index}">${job.kind === "file" ? "重新上传" : "重新导入"}</button>` : ""}
            </article>`;
        }).join("");
        box.querySelectorAll("[data-retry]").forEach((button) => {
            button.addEventListener("click", () => this.retryIngestJob(this.ingestJobs[Number(button.dataset.retry)]));
        });
    }

    jobTime(value) {
        const date = new Date(Number(value || 0) * 1000);
        if (Number.isNaN(date.getTime())) return "";
        return new Intl.DateTimeFormat("zh-CN", {
            month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit",
        }).format(date);
    }

    async retryIngestJob(job) {
        if (!job) return;
        if (job.kind === "file") {
            this.closeAllModals();
            this.el["file-input"].click();
            return;
        }
        const kbId = job.meta?.kb_id || this.currentKb?.id;
        if (job.kind === "url") {
            const url = job.meta?.url || job.label;
            this.closeAllModals();
            await this.importUrl(url, kbId);
            return;
        }
        if (job.kind === "dir") {
            const directory = job.meta?.dir || job.label;
            if (!directory || !kbId) return;
            this.closeAllModals();
            try {
                let result = await this.api("/ingest/dir", {
                    method: "POST",
                    body: { dir_path: directory, kb_id: kbId, background: true },
                });
                if (result.job_id) result = await this.waitForIngestJob(result.job_id, directory);
                this.showAlert(`已导入 ${Number(result.docs || 0)} 篇`);
                await this.loadKnowledgeBases();
            } catch (error) {
                this.showAlert(`导入失败：${error.message}`, "error");
            }
        }
    }

    async openDocs() {
        this.el["docs-kb-name"].textContent = this.currentKb?.name || "全部知识库";
        this.el["docs-filter"].value = "";
        this.el["docs-source-filter"].value = "";
        this.el["docs-back"].hidden = true;
        this.el["docs-filter"].hidden = false;
        this.openModalElement(this.el["docs-modal"]);
        await this.loadDocs();
    }

    async loadDocs() {
        const box = this.el["docs-content"];
        box.innerHTML = '<div class="command-empty">加载中</div>';
        try {
            const params = new URLSearchParams({ sort: "recent" });
            if (this.currentKb?.id) params.set("kb_id", this.currentKb.id);
            const data = await this.api(`/documents?${params}`);
            this.documents = data.documents || [];
            this.el["docs-back"].hidden = true;
            this.el["docs-filter"].hidden = false;
            this.renderDocs(this.documents);
        } catch (error) {
            box.innerHTML = `<div class="command-empty">${this.esc(error.message)}</div>`;
        }
    }

    renderDocs(documents) {
        const box = this.el["docs-content"];
        const keyword = this.el["docs-filter"].value.trim().toLowerCase();
        const sourceType = this.el["docs-source-filter"].value;
        const list = documents.filter((document) => {
            if (this.currentKb?.id && document.kb_id !== this.currentKb.id) return false;
            if (sourceType && document.source_type !== sourceType) return false;
            const text = `${document.title || ""} ${document.source || ""}`.toLowerCase();
            return !keyword || text.includes(keyword);
        });
        if (!list.length) {
            box.innerHTML = '<div class="command-empty">没有资料</div>';
            return;
        }
        box.innerHTML = `<ul class="docs-list">${list.map((document) => `
            <li>
                <button class="docs-item" type="button" data-source="${this.esc(document.source)}">
                    <span class="item-main">${this.esc(document.title || document.source)}</span>
                    <span class="item-meta">${Number(document.chunks || 0)} 分块</span>
                </button>
            </li>
        `).join("")}</ul>`;
        box.querySelectorAll(".docs-item").forEach((item) => {
            item.addEventListener("click", () => this.openDocChunks(item.dataset.source));
        });
    }

    async openDocChunks(source) {
        const box = this.el["docs-content"];
        box.innerHTML = '<div class="command-empty">加载中</div>';
        const params = new URLSearchParams();
        if (this.currentKb?.id) params.set("kb_id", this.currentKb.id);
        params.set("limit", "200");
        try {
            const data = await this.api(`/documents/${encodeURIComponent(source)}/chunks?${params}`);
            this.el["docs-back"].hidden = false;
            this.el["docs-filter"].hidden = true;
            this.renderChunks(source, data.chunks || []);
        } catch (error) {
            box.innerHTML = `<div class="command-empty">${this.esc(error.message)}</div>`;
        }
    }

    renderChunks(source, chunks) {
        const box = this.el["docs-content"];
        if (!chunks.length) {
            box.innerHTML = '<div class="command-empty">暂无分块</div>';
            return;
        }
        const sorted = chunks.slice().sort((a, b) => Number(a.metadata?.chunk_index || 0) - Number(b.metadata?.chunk_index || 0));
        box.innerHTML = `<h3 class="docs-doc-title">${this.esc(source)}</h3>${sorted.map((chunk, index) => {
            const heading = chunk.metadata?.chunk_prefix || chunk.metadata?.breadcrumb || "";
            return `<article class="chunk-card" data-chunk="${this.esc(chunk.id)}">
                <div class="chunk-head">
                    <span class="muted small">${index + 1}${heading ? ` · ${this.esc(heading)}` : ""}</span>
                    <button class="btn btn-secondary btn-sm chunk-save" type="button">保存</button>
                </div>
                <textarea class="chunk-text" rows="4" aria-label="分块 ${index + 1}">${this.esc(chunk.text || "")}</textarea>
            </article>`;
        }).join("")}`;
        box.querySelectorAll(".chunk-card").forEach((card) => {
            card.querySelector(".chunk-save").addEventListener("click", () => this.saveChunk(card.dataset.chunk, card));
        });
    }

    async saveChunk(chunkId, card) {
        const text = card.querySelector(".chunk-text").value.trim();
        if (!text) {
            this.showAlert("内容不能为空", "error");
            return;
        }
        const params = new URLSearchParams();
        if (this.currentKb?.id) params.set("kb_id", this.currentKb.id);
        const button = card.querySelector(".chunk-save");
        button.disabled = true;
        try {
            await this.api(`/documents/chunks/${encodeURIComponent(chunkId)}?${params}`, { method: "PATCH", body: { text } });
            this.showAlert("已保存");
        } catch (error) {
            this.showAlert(`保存失败：${error.message}`, "error");
        } finally {
            button.disabled = false;
        }
    }

    openModal(title, content, mono = false) {
        this.el["modal-title"].textContent = title;
        this.el["modal-body"].textContent = content;
        this.el["modal-body"].style.fontFamily = mono ? 'ui-monospace, "SFMono-Regular", Consolas, monospace' : "";
        this.openModalElement(this.el.modal);
    }

    openModalElement(element) {
        if (!element) return;
        this.lastFocus = document.activeElement;
        document.querySelectorAll(".modal.active").forEach((modal) => modal.classList.remove("active"));
        this.el["app-layout"].inert = true;
        element.classList.add("active");
        const focusable = element.querySelector("input, textarea, select, button:not(.modal-close)");
        requestAnimationFrame(() => focusable?.focus());
    }

    closeAllModals() {
        clearTimeout(this.tasksTimer);
        const active = document.querySelectorAll(".modal.active");
        if (!active.length) return;
        active.forEach((modal) => modal.classList.remove("active"));
        this.el["app-layout"].inert = false;
        if (this.lastFocus && document.contains(this.lastFocus)) this.lastFocus.focus();
    }

    setRichText(element, value) {
        if (!element) return;
        element.innerHTML = this.renderMarkdown(String(value ?? ""));
    }

    renderMarkdown(value) {
        const codeBlocks = [];
        let source = String(value ?? "").replace(/```([^\n]*)\n([\s\S]*?)```/g, (_, language, code) => {
            const index = codeBlocks.length;
            codeBlocks.push(`<pre><code${language ? ` data-language="${this.esc(language.trim())}"` : ""}>${this.esc(code.replace(/\n$/, ""))}</code></pre>`);
            return `\u0001CODE${index}\u0002`;
        });
        source = this.esc(source);
        const blocks = source.split(/\n{2,}/).map((block) => block.trim()).filter(Boolean).map((block) => {
            const heading = block.match(/^(#{1,6})\s+(.+)$/);
            if (heading) return `<h3>${this.inlineMarkdown(heading[2])}</h3>`;
            const lines = block.split("\n");
            if (lines.length && lines.every((line) => /^(?:[-*]|\d+\.)\s+/.test(line))) {
                const ordered = /^\d+\.\s+/.test(lines[0]);
                const items = lines.map((line) => `<li>${this.inlineMarkdown(line.replace(/^(?:[-*]|\d+\.)\s+/, ""))}</li>`).join("");
                return ordered ? `<ol>${items}</ol>` : `<ul>${items}</ul>`;
            }
            return `<p>${lines.map((line) => this.inlineMarkdown(line)).join("<br>")}</p>`;
        });
        return blocks.join("").replace(/\u0001CODE(\d+)\u0002/g, (_, index) => codeBlocks[Number(index)] || "");
    }

    inlineMarkdown(value) {
        return String(value)
            .replace(/`([^`]+)`/g, "<code>$1</code>")
            .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    }

    showAlert(message, type = "success") {
        const container = document.getElementById("alert-container");
        if (!container) return;
        const alert = document.createElement("div");
        alert.className = `alert alert-${type}`;
        alert.textContent = message;
        container.replaceChildren(alert);
        setTimeout(() => alert.remove(), 4200);
    }

    icon(name) {
        const icons = {
            upload: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 13V3m0 0L6.5 6.5M10 3l3.5 3.5M4 12v4h12v-4"/></svg>',
            link: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M8.2 11.8a3.2 3.2 0 0 0 4.5 0l2-2a3.2 3.2 0 0 0-4.5-4.5l-1 1M11.8 8.2a3.2 3.2 0 0 0-4.5 0l-2 2a3.2 3.2 0 0 0 4.5 4.5l1-1"/></svg>',
            search: '<svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="9" cy="9" r="5.5"/><path d="m13 13 4 4"/></svg>',
            copy: '<svg viewBox="0 0 20 20" aria-hidden="true"><rect x="6" y="6" width="9" height="9" rx="2"/><path d="M12 6V5a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v4a2 2 0 0 0 2 2h1"/></svg>',
            edit: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="m12.5 4 3.5 3.5-8 8H4.5V12z"/><path d="m11 5.5 3.5 3.5"/></svg>',
            refresh: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M16 10a6 6 0 1 1-1.8-4.2M16 3v4h-4"/></svg>',
            up: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M6 9v7H3V9h3Zm0 6 2.5 1h4.2l2.1-5.2A1.7 1.7 0 0 0 13.2 8H10l.5-3.4A1.6 1.6 0 0 0 8.9 3L6 9Z"/></svg>',
            down: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M6 11V4H3v7h3Zm0-6 2.5-1h4.2l2.1 5.2A1.7 1.7 0 0 1 13.2 12H10l.5 3.4A1.6 1.6 0 0 1 8.9 17L6 11Z"/></svg>',
        };
        return icons[name] || "";
    }

    esc(value) {
        return String(value ?? "").replace(/[&<>"']/g, (character) => ({
            "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
        }[character]));
    }
}

document.addEventListener("DOMContentLoaded", () => {
    window.zhigeApp = new ChatApp();
});
