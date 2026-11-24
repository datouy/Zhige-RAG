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
        this.authStatus = null;
        this.currentUser = null;
        this.authGateActive = false;
        this.appBootstrapped = false;
        this.onboarding = null;
        this.currentVerification = null;
        this.currentTimings = {};
        this.completedRequestId = null;
        this.lastAssistantMessage = null;
        this.feedbackContext = null;
        this.agentTemplates = [];
        this.adminTab = "overview";
        this.adminTeamData = null;
        this.evaluationResult = null;
        this.el = {};
        this.init();
    }

    async init() {
        this.cacheElements();
        this.bindEvents();
        this.updateShortcutLabel();
        try {
            if (!(await this.ensureAuthenticated())) return;
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
            "auth-gate", "auth-login-form", "auth-login", "auth-password", "auth-login-result",
            "auth-bootstrap-form", "auth-bootstrap-name", "auth-bootstrap-email",
            "auth-bootstrap-password", "auth-bootstrap-token", "auth-bootstrap-token-row",
            "auth-bootstrap-result", "feedback-modal", "feedback-form", "feedback-type",
            "feedback-correction", "feedback-citation", "feedback-missing", "feedback-result",
            "feedback-submit", "admin-modal", "admin-tabs", "admin-content", "admin-open",
            "team-logout", "account-summary",
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
        this.el["auth-login-form"]?.addEventListener("submit", (event) => {
            event.preventDefault();
            this.loginTeam();
        });
        this.el["auth-bootstrap-form"]?.addEventListener("submit", (event) => {
            event.preventDefault();
            this.bootstrapTeam();
        });
        this.el["feedback-form"]?.addEventListener("submit", (event) => {
            event.preventDefault();
            this.submitNotHelpfulFeedback();
        });
        this.el["admin-open"]?.addEventListener("click", () => this.openAdmin());
        this.el["team-logout"]?.addEventListener("click", () => this.logoutTeam());
        this.el["admin-tabs"]?.querySelectorAll("[data-admin-tab]").forEach((button) => {
            button.addEventListener("click", () => this.openAdminTab(button.dataset.adminTab));
        });
        document.querySelectorAll("[data-modal-cancel]").forEach((button) => {
            button.addEventListener("click", () => this.closeAllModals());
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
            document.getElementById(`${kind}-fetch-models`)?.addEventListener("click", () => this.fetchModels(kind));
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
            if (response.status === 401 && this.authStatus?.team_mode && !String(path).startsWith("/team/")) {
                this.showAuthGate();
            }
            throw new Error(typeof detail === "string" ? detail : (detail ? JSON.stringify(detail) : text || `HTTP ${response.status}`));
        }
        return data;
    }

    async ensureAuthenticated() {
        this.authStatus = await this.api("/team/status");
        if (this.authStatus.team_mode && !this.authStatus.authenticated) {
            this.showAuthGate();
            return false;
        }
        try {
            const me = await this.api("/team/me");
            this.currentUser = me.user || null;
        } catch {
            this.currentUser = this.authStatus.user || null;
        }
        this.renderAccountState();
        return true;
    }

    showAuthGate() {
        if (!this.authStatus?.team_mode) return;
        this.authGateActive = true;
        clearTimeout(this.reconnectTimer);
        this.ws?.close();
        this.agentWs?.close();
        this.el["app-layout"].hidden = true;
        this.el["auth-gate"].hidden = false;
        const bootstrap = Boolean(this.authStatus.bootstrap_available);
        this.el["auth-login-form"].hidden = bootstrap;
        this.el["auth-bootstrap-form"].hidden = !bootstrap;
        this.el["auth-bootstrap-token-row"].hidden = !this.authStatus.bootstrap_token_required;
        requestAnimationFrame(() => {
            (bootstrap ? this.el["auth-bootstrap-name"] : this.el["auth-login"]).focus();
        });
    }

    async finishTeamLogin(user) {
        this.currentUser = user || null;
        this.authStatus = { ...(this.authStatus || {}), authenticated: true, bootstrap_available: false, user: user || null };
        this.authGateActive = false;
        this.el["auth-gate"].hidden = true;
        this.renderAccountState();
        this.appBootstrapped = false;
        this.currentKb = null;
        this.currentSession = null;
        this.knowledgeBases = [];
        this.sessions = [];
        this.agents = [];
        this.currentUser = user || null;
        await this.bootstrap();
    }

    async loginTeam() {
        this.el["auth-login-result"].textContent = "正在验证";
        try {
            const result = await this.api("/team/login", {
                method: "POST",
                body: { login: this.el["auth-login"].value.trim(), password: this.el["auth-password"].value },
            });
            this.el["auth-password"].value = "";
            await this.finishTeamLogin(result.user);
        } catch (error) {
            this.el["auth-login-result"].textContent = error.message;
        }
    }

    async bootstrapTeam() {
        this.el["auth-bootstrap-result"].textContent = "正在创建管理员";
        try {
            const result = await this.api("/team/bootstrap", {
                method: "POST",
                body: {
                    username: this.el["auth-bootstrap-name"].value.trim(),
                    email: this.el["auth-bootstrap-email"].value.trim(),
                    password: this.el["auth-bootstrap-password"].value,
                    bootstrap_token: this.el["auth-bootstrap-token"].value,
                },
            });
            this.el["auth-bootstrap-password"].value = "";
            await this.finishTeamLogin(result.user);
        } catch (error) {
            this.el["auth-bootstrap-result"].textContent = error.message;
        }
    }

    async logoutTeam() {
        try { await this.api("/team/logout", { method: "POST" }); } catch {}
        this.currentUser = null;
        this.authStatus = { ...(this.authStatus || {}), authenticated: false };
        this.showAuthGate();
    }

    renderAccountState() {
        const team = Boolean(this.authStatus?.team_mode);
        const role = this.currentUser?.role || (team ? "viewer" : "admin");
        const name = this.currentUser?.username || (team ? "未登录" : "本地工作空间");
        if (this.el["account-summary"]) this.el["account-summary"].textContent = `${name} · ${role === "admin" ? "管理员" : role === "editor" ? "编辑者" : "成员"}`;
        if (this.el["admin-open"]) this.el["admin-open"].hidden = role !== "admin";
        if (this.el["team-logout"]) this.el["team-logout"].hidden = !team;
    }

    async bootstrap() {
        this.el["app-layout"].hidden = false;
        this.appBootstrapped = true;
        await this.loadDefaults();
        await this.loadKnowledgeBases();
        await this.loadOnboarding();
        await this.loadAgents();
        await this.loadSessions();
        await this.restoreSessionFromUrl();
        this.connectWebSocket();
        this.el["chat-input"].focus();
        const params = new URLSearchParams(window.location.search);
        if (params.get("docs") === "1") this.openDocs();
    }

    async loadDefaults() {
        try {
            const defaults = await this.api("/config/defaults");
            if (Number.isFinite(defaults?.top_k) && defaults.top_k > 0) this.topK = defaults.top_k;
        } catch {
            // The configured frontend default remains usable when this endpoint is unavailable.
        }
    }

    async loadOnboarding() {
        try {
            this.onboarding = await this.api("/onboarding");
        } catch {
            this.onboarding = null;
        }
        if (!this.currentSession) this.clearMessages();
    }

    onboardingExamples() {
        if (Array.isArray(this.onboarding?.examples) && this.onboarding.examples.length) {
            return this.onboarding.examples;
        }
        return [
            {
                label: "制度与流程",
                title: "查清具体标准和时限",
                question: "2026 版制度下，一线城市住宿标准和报销时限分别是什么？",
                hint: "定位到制度条款并附原文",
            },
            {
                label: "合同与审批",
                title: "确认多级审批责任",
                question: "超过 50 万元的合同需要哪些角色审批？",
                hint: "汇总审批角色并标注依据",
            },
            {
                label: "产品与安全",
                title: "看懂系统如何保证可信",
                question: "知阁如何让回答可追溯？",
                hint: "从产品机制资料中提取答案",
            },
            {
                label: "可信边界",
                title: "没有资料时明确拒答",
                question: "公司年会的抽奖一等奖是什么？",
                hint: "验证系统不会编造答案",
                tone: "boundary",
            },
        ];
    }

    welcomeMarkup() {
        const ready = Boolean(this.onboarding?.ready);
        const examples = this.onboardingExamples();
        const cards = examples.map((item) => `
            <button class="welcome-question ${item.tone === "boundary" ? "is-boundary" : ""}"
                    type="button" data-sample-question="${this.esc(item.question)}">
                <span class="welcome-question-top">
                    <span class="welcome-question-label">${this.esc(item.label || "示例问题")}</span>
                    <span class="welcome-question-try">试问</span>
                </span>
                <strong>${this.esc(item.title || item.question)}</strong>
                <span class="welcome-question-prompt">${this.esc(item.question)}</span>
                <small>${this.esc(item.hint || "查看答案与引用依据")}</small>
            </button>`).join("");
        return `
            <div class="empty-state welcome-overview" id="welcome-state">
                <section class="welcome-hero" aria-labelledby="welcome-title">
                    <div class="welcome-copy">
                        <p class="welcome-eyebrow">企业知识库问答</p>
                        <h2 id="welcome-title">公司资料，问一句就有答案</h2>
                        <p class="welcome-lead">制度、合同、产品与运营资料都能问。关键结论附原文出处，资料不足时直接拒答。</p>
                        <div class="welcome-scopes" aria-label="适用场景">
                            <span>制度流程</span><span>合同审批</span><span>产品知识</span><span>团队协作</span>
                        </div>
                    </div>
                    <div class="welcome-start">
                        <button class="welcome-primary" type="button" data-demo-experience>
                            <span>${ready ? "继续体验示例问答" : "看一次真实问答"}</span>
                            <small>${ready ? "四篇资料已就绪，直接提问" : "自动载入四篇示例资料"}</small>
                        </button>
                        <button class="welcome-upload" type="button" data-upload>
                            <span>换成我的资料</span>
                            <small>PDF · Word · Excel · 网页</small>
                        </button>
                    </div>
                </section>
                <section class="welcome-question-section" aria-labelledby="welcome-examples-title">
                    <div class="welcome-section-head">
                        <div>
                            <h3 id="welcome-examples-title">可以这样问</h3>
                            <p>点一个问题，直接看回答和依据</p>
                        </div>
                        <span>${examples.length} 种能力</span>
                    </div>
                    <div class="welcome-question-grid">${cards}</div>
                </section>
                <div class="welcome-proof" aria-label="产品能力概览">
                    <span><strong>答案可核验</strong>点击引用回到原文</span>
                    <span><strong>边界可感知</strong>没有依据就明确拒答</span>
                    <span><strong>越用越准确</strong>差评进入评估与治理</span>
                </div>
            </div>`;
    }

    bindWelcomeActions() {
        this.el.messages.querySelectorAll("[data-upload]").forEach((button) => {
            button.addEventListener("click", () => this.el["file-input"].click());
        });
        this.el.messages.querySelectorAll("[data-import-url]").forEach((button) => {
            button.addEventListener("click", () => this.openImport());
        });
        this.el.messages.querySelectorAll("[data-search]").forEach((button) => {
            button.addEventListener("click", () => this.openSearch());
        });
        this.el.messages.querySelectorAll("[data-demo-experience]").forEach((button) => {
            button.addEventListener("click", () => {
                const first = this.onboardingExamples()[0]?.question;
                this.askSample(first);
            });
        });
        this.el.messages.querySelectorAll("[data-sample-question]").forEach((button) => {
            button.addEventListener("click", () => this.askSample(button.dataset.sampleQuestion));
        });
    }

    async loadDemo() {
        const buttons = this.el.messages.querySelectorAll("[data-demo], [data-demo-experience]");
        buttons.forEach((button) => { button.disabled = true; });
        try {
            const result = await this.api("/onboarding/demo", { method: "POST" });
            this.onboarding = result;
            await this.loadKnowledgeBases();
            const kbId = result.kb?.id || result.demo_kb_id;
            if (kbId) await this.selectKb(kbId);
            this.showAlert(`示例库已就绪 · ${result.chunks || 0} 个可检索分块`);
        } catch (error) {
            this.showAlert(`载入示例失败：${error.message}`, "error");
        } finally {
            buttons.forEach((button) => { button.disabled = false; });
        }
    }

    async askSample(question) {
        if (!question) return;
        if (!this.onboarding?.ready) {
            await this.loadDemo();
        } else if (this.onboarding?.demo_kb_id && this.currentKb?.id !== this.onboarding.demo_kb_id) {
            await this.selectKb(this.onboarding.demo_kb_id);
        }
        await this.sendMessage(question);
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
        const groups = new Map();
        this.sessions.forEach((session) => {
            const label = this.sessionGroup(session.last_message_at || session.created_at);
            if (!groups.has(label)) groups.set(label, []);
            groups.get(label).push(session);
        });
        list.innerHTML = Array.from(groups).map(([label, sessions]) => `
            <li class="session-group-label" aria-hidden="true">${this.esc(label)}</li>
            ${sessions.map((session) => {
                const timeLabel = this.sessionTime(session.last_message_at || session.created_at);
                return `
            <li class="list-item ${this.currentSession?.id === session.id ? "active" : ""}" data-session-row="${this.esc(session.id)}">
                <button class="item-select" type="button" data-session="${this.esc(session.id)}"
                        title="${this.esc(session.preview || session.title)}"
                        aria-label="${this.esc(session.title || "新对话")}"
                        ${this.currentSession?.id === session.id ? 'aria-current="true"' : ""}>
                    <span class="item-main">${this.esc(session.title || "新对话")}</span>
                    ${timeLabel ? `<span class="item-meta">${this.esc(timeLabel)}</span>` : ""}
                </button>
                <button class="item-ren" data-ren="${this.esc(session.id)}" aria-label="重命名" type="button">✎</button>
                <button class="item-del" data-del-session="${this.esc(session.id)}" aria-label="删除" type="button">×</button>
            </li>
            `;
            }).join("")}
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
                const node = this.appendMessage(message.role, message.content, message.created_at);
                if (message.role === "user") this.addUserActions(node, message.content);
                if (message.role === "assistant") {
                    const sources = message.sources || [];
                    this.addMessageActions(node, queryForAnswer, sources);
                    this.attachEvidence(
                        node,
                        message.verification || null,
                        message.timings || {},
                        sources,
                        queryForAnswer,
                    );
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
        if (this.authGateActive) return;
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
        this.currentVerification = null;
        this.currentTimings = {};
        this.completedRequestId = null;
        this.lastAssistantMessage = null;
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
        const requestId = payload.request_id || null;
        const requestMatches = !requestId
            || requestId === this.activeRequestId
            || requestId === this.completedRequestId;
        if (!requestMatches) return;
        if (event === "hits") {
            this.currentSources = data || [];
        } else if (event === "token") {
            this.hideTypingIndicator();
            this.appendToken(String(data ?? ""));
        } else if (event === "done") {
            this.hideTypingIndicator();
            this.completedRequestId = requestId || this.activeRequestId;
            if (Array.isArray(data)) this.currentSources = data;
            else if (Array.isArray(data?.sources)) this.currentSources = data.sources;
            if (payload.verification) this.currentVerification = payload.verification;
            this.isStreaming = false;
            this.activeRequestId = null;
            this.finalizeAssistant(this.completedRequestId);
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
        } else if (event === "metadata") {
            if (Array.isArray(data?.sources)) this.currentSources = data.sources;
            if (data?.verification) this.currentVerification = data.verification;
            if (data?.timings && typeof data.timings === "object") {
                this.currentTimings = { ...data.timings };
            }
            this.refreshCurrentEvidence();
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
                body: {
                    role: "assistant",
                    content: answer,
                    sources: this.currentSources,
                    verification: this.currentVerification,
                    timings: this.currentTimings,
                },
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
        this.el.messages.innerHTML = this.welcomeMarkup();
        this.bindWelcomeActions();
        this.currentSources = [];
        this.currentVerification = null;
        this.currentTimings = {};
        this.completedRequestId = null;
        this.lastAssistantMessage = null;
    }

    appendMessage(role, content, timestamp) {
        this.el.messages.querySelector(".empty-state")?.remove();
        const wrap = document.createElement("div");
        wrap.className = `message ${role}`;
        const body = document.createElement("div");
        body.className = "message-content";
        if (role === "assistant") this.setRichText(body, content);
        else body.textContent = content;
        wrap.appendChild(body);
        const time = document.createElement("div");
        time.className = "message-time";
        time.textContent = this.formatTime(timestamp) || this.formatTime(new Date().toISOString());
        wrap.appendChild(time);
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
            const time = document.createElement("div");
            time.className = "message-time";
            time.textContent = this.formatTime(new Date().toISOString());
            message.appendChild(time);
            this.el.messages.appendChild(message);
        }
        message.querySelector(".message-content").textContent += token;
        this.scrollBottom();
    }

    finalizeAssistant(requestId = null) {
        const streaming = Array.from(this.el.messages.querySelectorAll(".message.assistant.streaming"));
        streaming.forEach((message) => {
            message.classList.remove("streaming");
            const body = message.querySelector(".message-content");
            this.setRichText(body, body.textContent);
            this.addMessageActions(message, this.lastUserQuery, this.currentSources);
            this.lastAssistantMessage = message;
            this.attachEvidence(
                message,
                this.currentVerification,
                this.currentTimings,
                this.currentSources,
                this.lastUserQuery,
            );
        });
        if (streaming.length && requestId) this.completedRequestId = requestId;
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

    durationLabel(value) {
        const ms = Number(value);
        if (!Number.isFinite(ms) || ms < 0) return "—";
        return ms >= 1000 ? `${(ms / 1000).toFixed(ms >= 10000 ? 0 : 1)} 秒` : `${Math.round(ms)} ms`;
    }

    evidenceStatus(verification) {
        const status = verification?.refused ? "refused" : (verification?.status || "unverified");
        return {
            status,
            label: {
                verified: "依据充分",
                partial: "部分待核实",
                unverified: "未通过校验",
                refused: "无依据拒答",
            }[status] || "待校验",
            note: {
                verified: "关键事实均能找到对应原文。",
                partial: "结论有来源，但仍有事实或引用需要人工核对。",
                unverified: "引用覆盖或原文接地不足，请打开来源逐项核对。",
                refused: "当前知识库没有足够依据，系统已拒答并记录知识缺口。",
            }[status] || "系统尚未返回完整的可验证性报告。",
        };
    }

    attachEvidence(message, verification, timings, sources = [], query = "") {
        if (!message) return;
        message.querySelector(".evidence-card")?.remove();
        if (!verification && !sources?.length && !Object.keys(timings || {}).length) return;
        const state = this.evidenceStatus(verification);
        const ratio = Number(verification?.cited_sentence_ratio);
        const sourceList = Array.isArray(sources) ? sources : [];
        const issues = [];
        if (verification?.invalid_citations?.length) {
            issues.push(`无效引用：${verification.invalid_citations.map((item) => `[${item}]`).join("、")}`);
        }
        if (verification?.unsupported_facts?.length) {
            issues.push(`待核实事实：${verification.unsupported_facts.slice(0, 4).join("、")}`);
        }
        if (verification?.unsupported_sentences?.length) {
            issues.push(`${verification.unsupported_sentences.length} 句引用与所引原文贴合度不足`);
        }
        const card = document.createElement("section");
        card.className = "evidence-card";
        card.dataset.status = state.status;
        card.innerHTML = `
            <div class="evidence-head">
                <span class="evidence-badge">${this.esc(state.label)}</span>
                <span class="evidence-time">${this.esc(this.durationLabel(timings?.total_ms))}</span>
            </div>
            <div class="evidence-metrics">
                <span class="evidence-metric"><strong>${Number.isFinite(ratio) ? `${Math.round(ratio * 100)}%` : "—"}</strong> 句子引用率</span>
                <span class="evidence-metric"><strong>${sourceList.length}</strong> 个来源</span>
                <span class="evidence-metric"><strong>${this.esc(this.durationLabel(timings?.retrieval_ms))}</strong> 检索</span>
                <span class="evidence-metric"><strong>${this.esc(this.durationLabel(timings?.ttft_ms))}</strong> 首字</span>
            </div>
            <p class="evidence-note">${this.esc(state.note)}</p>
            ${issues.length ? `<p class="evidence-note evidence-warning">${this.esc(issues.join("；"))}</p>` : ""}
            ${sourceList.length ? `<div class="evidence-source-row">${sourceList.slice(0, 5).map((source, index) =>
                `<button class="evidence-source" type="button" data-evidence-source="${index}" title="${this.esc(source.title || source.source || "查看引用原文")}">[${this.esc(source.index || index + 1)}] ${this.esc(source.title || source.source || "来源")}</button>`
            ).join("")}</div>` : ""}
            ${state.status === "refused" ? '<div class="evidence-head"><span class="evidence-note">已进入知识缺口队列</span><button class="evidence-action" type="button" data-evidence-upload>补充资料</button></div>' : ""}`;
        card.querySelectorAll("[data-evidence-source]").forEach((button) => {
            button.addEventListener("click", () => {
                const source = sourceList[Number(button.dataset.evidenceSource)];
                if (source) this.showChunkDetail(source);
            });
        });
        card.querySelector("[data-evidence-upload]")?.addEventListener("click", () => {
            if (!this.currentKb) this.showAlert("请先选择要补充资料的知识库", "error");
            else this.el["file-input"].click();
        });
        const actions = message.querySelector(":scope > .message-actions");
        if (actions) message.insertBefore(card, actions);
        else message.appendChild(card);
    }

    refreshCurrentEvidence() {
        if (!this.lastAssistantMessage || !document.body.contains(this.lastAssistantMessage)) return;
        this.attachEvidence(
            this.lastAssistantMessage,
            this.currentVerification,
            this.currentTimings,
            this.currentSources,
            this.lastUserQuery,
        );
        this.scrollBottom();
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
        notHelpful.addEventListener("click", () => this.openFeedback(query, answer, sources));

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
                    citation_issue: false,
                    missing_source: false,
                    kb_id: this.currentKb?.id || null,
                    session_id: this.currentSession?.id || null,
                },
            });
            selected.setAttribute("aria-pressed", "true");
            other.setAttribute("aria-pressed", "false");
            this.showAlert("已记录");
        } catch (error) {
            this.showAlert(`反馈失败：${error.message}`, "error");
        }
    }

    openFeedback(query, answer, sources) {
        if (!query) return;
        this.feedbackContext = {
            query,
            answer: answer || "",
            sources: Array.isArray(sources) ? sources : [],
            kb_id: this.currentKb?.id || null,
            session_id: this.currentSession?.id || null,
        };
        this.el["feedback-form"].reset();
        this.el["feedback-result"].textContent = "";
        this.openModalElement(this.el["feedback-modal"]);
    }

    async submitNotHelpfulFeedback() {
        const context = this.feedbackContext;
        if (!context) {
            this.el["feedback-result"].textContent = "反馈上下文已失效，请重新点击没帮助";
            return;
        }
        const button = this.el["feedback-submit"];
        button.disabled = true;
        this.el["feedback-result"].textContent = "正在进入治理队列";
        try {
            const result = await this.api("/feedback", {
                method: "POST",
                body: {
                    query: context.query,
                    answer: context.answer,
                    sources_json: JSON.stringify(context.sources),
                    rating: "not_helpful",
                    correction: this.el["feedback-correction"].value.trim(),
                    question_type: this.el["feedback-type"].value,
                    citation_issue: this.el["feedback-citation"].checked || this.el["feedback-type"].value === "citation",
                    missing_source: this.el["feedback-missing"].checked || this.el["feedback-type"].value === "missing_source",
                    kb_id: context.kb_id,
                    session_id: context.session_id,
                },
            });
            this.el["feedback-result"].textContent = result.knowledge_gap_id
                ? `已进入知识缺口队列 · ${result.knowledge_gap_id.slice(0, 8)}`
                : "反馈已提交";
            this.showAlert(result.knowledge_gap_id ? "差评已进入质量治理队列" : "已记录反馈");
            setTimeout(() => this.closeAllModals(), 900);
        } catch (error) {
            this.el["feedback-result"].textContent = `提交失败：${error.message}`;
        } finally {
            button.disabled = false;
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
        const [agentResult, templateResult] = await Promise.allSettled([
            this.api("/agents"),
            this.api("/agent/templates"),
        ]);
        if (agentResult.status === "fulfilled") {
            this.agents = agentResult.value.agents || [];
        } else {
            this.agents = [];
        }
        this.agentTemplates = templateResult.status === "fulfilled" ? (templateResult.value.templates || []) : [];
        if (this.activeAgent) {
            this.activeAgent = this.agents.find((agent) => agent.id === this.activeAgent.id) || null;
        }
        if (!this.activeAgent && this.agentMode) this.setConversationMode("knowledge");
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
        const templates = this.agentTemplates.length ? `
            <section class="agent-templates">
                <div class="agent-template-grid">${this.agentTemplates.map((template) => `
                    <button class="agent-template" type="button" data-agent-template="${this.esc(template.id)}">
                        <strong>${this.esc(template.name)}</strong>
                        <small>${this.esc(template.description || "预填工作说明与工具")}</small>
                        <span class="status-pill info">使用模板</span>
                    </button>`).join("")}</div>
            </section>` : "";
        const rows = this.agents.length ? this.agents.map((agent) => {
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
        }).join("") : '<div class="command-empty">还没有自定义 Agent；可从上方模板开始。</div>';
        box.innerHTML = `${templates}${rows}`;
        box.querySelectorAll("[data-agent-template]").forEach((button) => {
            button.addEventListener("click", () => {
                const template = this.agentTemplates.find((item) => item.id === button.dataset.agentTemplate);
                if (template) this.startAgentEditor(null, template);
            });
        });
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

    async startAgentEditor(agent = null, template = null) {
        const source = agent || template || {};
        this.el["agents-list-view"].hidden = true;
        this.el["agent-form"].hidden = false;
        this.el["agent-id"].value = agent?.id || "";
        this.el["agent-name"].value = source.name || "";
        this.el["agent-kb"].innerHTML = this.knowledgeBases.map((kb) =>
            `<option value="${this.esc(kb.id)}">${this.esc(kb.name)}</option>`
        ).join("");
        this.el["agent-kb"].value = source.kb_id || this.currentKb?.id || this.knowledgeBases[0]?.id || "";
        this.el["agent-instructions"].value = source.instructions || "";
        this.el["agent-delete"].hidden = !agent;
        this.el["agent-form-result"].textContent = template && !agent ? "模板已填入；保存后会创建你自己的 Agent。" : "";
        this.agentMaxSteps = source.max_steps || 5;
        const defaults = ["search_documents", "list_documents", "get_document_chunks", "calculator", "get_current_time", "text_stats"];
        this.agentSelectedTools = new Set(source.allowed_tools || defaults);
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
        const approvalLabels = { pending: "等待管理员审批", approved: "已批准", rejected: "已拒绝" };
        const approvalClasses = { pending: "warn", approved: "ok", rejected: "bad" };
        box.innerHTML = this.mcpServers.map((server) => {
            const approval = server.approval_status || "pending";
            const canTest = server.enabled && approval === "approved";
            return `
            <div class="mcp-row" data-mcp-row="${this.esc(server.id)}">
                <div class="mcp-main">
                    <span class="mcp-name">${this.esc(server.name)} <span class="status-pill ${approvalClasses[approval] || ""}">${this.esc(approvalLabels[approval] || approval)}</span></span>
                    <span class="mcp-meta">${server.enabled ? `${server.tool_count || 0} 工具` : "已停用"}${server.last_error ? ` · ${this.esc(server.last_error)}` : ""}</span>
                    <code class="mcp-command">${this.esc(server.command)}</code>
                </div>
                <button class="btn btn-ghost btn-sm" type="button" data-mcp-test="${this.esc(server.id)}" ${canTest ? "" : 'disabled title="管理员批准后才能检查"'}>检查</button>
                <button class="icon-button" type="button" data-mcp-delete="${this.esc(server.id)}" aria-label="删除 ${this.esc(server.name)}">
                    <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 6h12M8 6V4h4v2M6 6l1 10h6l1-10"/></svg>
                </button>
            </div>`;
        }).join("");
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
        const enabled = servers.filter((server) => server.enabled && (server.approval_status || "approved") === "approved");
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

    async openAdmin() {
        const role = this.currentUser?.role || (this.authStatus?.team_mode ? "viewer" : "admin");
        if (role !== "admin") {
            this.showAlert("当前账号没有管理权限", "error");
            return;
        }
        this.openModalElement(this.el["admin-modal"]);
        await this.openAdminTab(this.adminTab || "overview");
    }

    async openAdminTab(tab) {
        const validTabs = new Set(["overview", "quality", "team", "audit", "eval", "backup"]);
        this.adminTab = validTabs.has(tab) ? tab : "overview";
        this.el["admin-tabs"]?.querySelectorAll("[data-admin-tab]").forEach((button) => {
            button.setAttribute("aria-selected", String(button.dataset.adminTab === this.adminTab));
        });
        const content = this.el["admin-content"];
        if (!content) return;
        content.innerHTML = '<div class="admin-empty">正在加载治理数据</div>';
        try {
            if (this.adminTab === "overview") {
                this.renderAdminOverview(await this.api("/admin/overview"));
            } else if (this.adminTab === "quality") {
                this.renderAdminQuality(await this.api("/admin/quality"));
            } else if (this.adminTab === "team") {
                await this.loadAdminTeam();
            } else if (this.adminTab === "audit") {
                await this.loadAdminAudit();
            } else if (this.adminTab === "eval") {
                this.renderAdminEvaluation(await this.api("/admin/evaluations"));
            } else if (this.adminTab === "backup") {
                this.renderAdminBackup(await this.api("/admin/backup"));
            }
        } catch (error) {
            content.innerHTML = `<div class="admin-empty">加载失败：${this.esc(error.message)}</div>`;
        }
    }

    adminHeading(title, description, action = "") {
        return `<div class="admin-section-head"><div><h3>${this.esc(title)}</h3><p>${this.esc(description)}</p></div>${action}</div>`;
    }

    adminMetric(label, value) {
        return `<div class="metric-card"><span>${this.esc(label)}</span><strong>${this.esc(value)}</strong></div>`;
    }

    formatAdminTime(value) {
        if (!value) return "—";
        const date = new Date(value);
        return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString("zh-CN", { hour12: false });
    }

    formatTime(value) {
        if (!value) return "";
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) return "";
        return new Intl.DateTimeFormat("zh-CN", {
            month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false,
        }).format(date);
    }

    sessionGroup(value) {
        if (!value) return "更早";
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) return "更早";
        const startOfDay = (item) => new Date(
            item.getFullYear(), item.getMonth(), item.getDate()
        ).getTime();
        const diffDays = Math.floor((startOfDay(new Date()) - startOfDay(date)) / 86400000);
        if (diffDays <= 0) return "今天";
        if (diffDays === 1) return "昨天";
        if (diffDays < 7) return "7 天内";
        if (diffDays < 30) return "30 天内";
        return "更早";
    }

    sessionTime(value) {
        if (!value) return "";
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) return "";
        const now = new Date();
        const sameDay = date.getFullYear() === now.getFullYear()
            && date.getMonth() === now.getMonth()
            && date.getDate() === now.getDate();
        if (sameDay) {
            return new Intl.DateTimeFormat("zh-CN", {
                hour: "2-digit", minute: "2-digit", hour12: false,
            }).format(date);
        }
        return new Intl.DateTimeFormat("zh-CN", {
            month: "numeric", day: "numeric",
        }).format(date);
    }

    renderAdminOverview(data) {
        const content = this.el["admin-content"];
        const ingest = data.ingest || {};
        const byStatus = ingest.by_status || {};
        content.innerHTML = `<section class="admin-section">
            ${this.adminHeading("工作区概览", "成员、资料、质量风险与恢复状态的统一入口")}
            <div class="metric-grid">
                ${this.adminMetric("活跃成员", data.users?.active ?? 0)}
                ${this.adminMetric("知识库", data.knowledge?.kbs ?? 0)}
                ${this.adminMetric("已登记资料", data.knowledge?.documents ?? 0)}
                ${this.adminMetric("可检索分块", data.knowledge?.chunks ?? 0)}
                ${this.adminMetric("未处理差评", data.quality?.unresolved_feedback ?? 0)}
                ${this.adminMetric("开放知识缺口", data.quality?.open_gaps ?? 0)}
                ${this.adminMetric("索引失败资料", data.knowledge?.failed_documents ?? 0)}
                ${this.adminMetric("待审批 MCP", data.security?.pending_mcp_approvals ?? 0)}
            </div>
            <div class="admin-row">
                <div class="admin-row-head">
                    <span class="admin-row-title">运行状态</span>
                    <span class="status-pill ${data.settings?.database_ok ? "ok" : "bad"}">${data.settings?.database_ok ? "数据库正常" : "数据库异常"}</span>
                </div>
                <p class="admin-row-text">当前操作者：${this.esc(data.settings?.authenticated_as || "-")} · 入库任务 ${Number(ingest.total || 0)} 个 · 处理中 ${Number(ingest.pending || 0)} 个</p>
                <div class="admin-row-actions">
                    ${Object.entries(byStatus).map(([status, count]) => `<span class="status-pill ${status === "failed" ? "bad" : status === "done" ? "ok" : "info"}">${this.esc(status)} ${Number(count || 0)}</span>`).join("") || '<span class="status-pill">暂无入库任务</span>'}
                    ${data.security?.restore_pending ? '<span class="status-pill warn">有待应用恢复包</span>' : '<span class="status-pill ok">无待恢复包</span>'}
                </div>
            </div>
        </section>`;
    }

    renderAdminQuality(data) {
        const content = this.el["admin-content"];
        const feedback = data.feedback || [];
        const gaps = data.knowledge_gaps || [];
        const stats = data.stats || {};
        content.innerHTML = `<section class="admin-section">
            ${this.adminHeading("质量治理", "逐条处理差评、修正知识缺口，并把问题转成可回归的资料")}
            <div class="metric-grid">
                ${this.adminMetric("差评率", `${Math.round(Number(stats.bad_rate || 0) * 100)}%`)}
                ${this.adminMetric("未处理差评", stats.unresolved_feedback ?? 0)}
                ${this.adminMetric("开放缺口", stats.open_gaps ?? 0)}
                ${this.adminMetric("反馈总量", stats.feedback_total ?? 0)}
            </div>
            <div>
                <div class="admin-section-head"><div><h3>待处理差评</h3><p>处理时可补充纠正结论；未解决的缺资料问题会同步进入知识缺口。</p></div></div>
                <div class="admin-list">
                    ${feedback.length ? feedback.map((item) => `<article class="admin-row" data-admin-feedback="${this.esc(item.id)}">
                        <div class="admin-row-head">
                            <span class="admin-row-title">${this.esc(item.query)}</span>
                            <span class="status-pill ${item.resolved ? "ok" : "warn"}">${item.resolved ? "已处理" : "待处理"}</span>
                        </div>
                        <p class="admin-row-text">${this.esc(item.answer || "无回答快照")}</p>
                        <div class="admin-row-actions">
                            <span class="status-pill">${this.esc(item.question_type || "未分类")}</span>
                            ${item.citation_issue ? '<span class="status-pill bad">引用问题</span>' : ""}
                            ${item.missing_source ? '<span class="status-pill warn">缺资料</span>' : ""}
                            <span class="admin-row-meta">${this.esc(this.formatAdminTime(item.created_at))}</span>
                        </div>
                        ${item.correction ? `<p class="admin-row-text">纠正：${this.esc(item.correction)}</p>` : ""}
                        <textarea class="form-control" rows="2" maxlength="2000" data-feedback-resolution="${this.esc(item.id)}" placeholder="处理说明">${this.esc(item.correction || "")}</textarea>
                        <div class="admin-row-actions">
                            <button class="btn btn-ghost btn-sm" type="button" data-feedback-resolve="true" data-id="${this.esc(item.id)}">标记已处理</button>
                            <button class="btn btn-ghost btn-sm" type="button" data-feedback-resolve="false" data-id="${this.esc(item.id)}">重新打开</button>
                            <button class="btn btn-secondary btn-sm" type="button" data-feedback-gap="${this.esc(item.id)}">转为知识缺口</button>
                        </div>
                    </article>`).join("") : '<div class="admin-empty">暂无差评，质量队列为空</div>'}
                </div>
            </div>
            <div>
                <div class="admin-section-head"><div><h3>知识缺口</h3><p>拒答和缺资料反馈会在这里沉淀，处理后才能形成闭环。</p></div></div>
                <div class="admin-list">
                    ${gaps.length ? gaps.map((item) => `<article class="admin-row" data-admin-gap="${this.esc(item.id)}">
                        <div class="admin-row-head">
                            <span class="admin-row-title">${this.esc(item.query)}</span>
                            <span class="status-pill warn">${this.esc(item.status)}</span>
                        </div>
                        <p class="admin-row-text">${this.esc(item.detail || item.answer || "等待补充依据")}</p>
                        <div class="admin-row-actions">
                            <span class="status-pill">${this.esc(item.reason || "refused")}</span>
                            <span class="admin-row-meta">${this.esc(this.formatAdminTime(item.created_at))}</span>
                        </div>
                        <div class="admin-form-grid">
                            <select class="form-control" data-gap-status="${this.esc(item.id)}" aria-label="缺口状态">
                                ${["open", "in_progress", "resolved", "dismissed"].map((status) => `<option value="${status}" ${item.status === status ? "selected" : ""}>${status}</option>`).join("")}
                            </select>
                            <input class="form-control" data-gap-resolution="${this.esc(item.id)}" maxlength="4000" value="${this.esc(item.resolution || "")}" placeholder="处理结论或补充资料说明">
                        </div>
                        <div class="admin-row-actions"><button class="btn btn-primary btn-sm" type="button" data-gap-save="${this.esc(item.id)}">保存处理结果</button></div>
                    </article>`).join("") : '<div class="admin-empty">暂无知识缺口</div>'}
                </div>
            </div>
        </section>`;
        content.querySelectorAll("[data-feedback-resolve]").forEach((button) => {
            button.addEventListener("click", () => this.resolveAdminFeedback(button.dataset.id, button.dataset.feedbackResolve === "true", false, button));
        });
        content.querySelectorAll("[data-feedback-gap]").forEach((button) => {
            button.addEventListener("click", () => this.resolveAdminFeedback(button.dataset.feedbackGap, true, true, button));
        });
        content.querySelectorAll("[data-gap-save]").forEach((button) => {
            button.addEventListener("click", () => this.saveAdminGap(button.dataset.gapSave, button));
        });
    }

    async resolveAdminFeedback(feedbackId, resolved, createGap, button) {
        button.disabled = true;
        try {
            const resolution = this.el["admin-content"].querySelector(`[data-feedback-resolution="${CSS.escape(feedbackId)}"]`)?.value.trim() || "";
            await this.api(`/feedback/${encodeURIComponent(feedbackId)}/resolve`, {
                method: "PATCH",
                body: { resolved, resolution, create_gap: createGap },
            });
            this.showAlert(createGap ? "已创建知识缺口" : "反馈状态已更新");
            await this.openAdminTab("quality");
        } catch (error) {
            this.showAlert(`处理失败：${error.message}`, "error");
        } finally {
            button.disabled = false;
        }
    }

    async saveAdminGap(gapId, button) {
        button.disabled = true;
        try {
            const status = this.el["admin-content"].querySelector(`[data-gap-status="${CSS.escape(gapId)}"]`)?.value || "open";
            const resolution = this.el["admin-content"].querySelector(`[data-gap-resolution="${CSS.escape(gapId)}"]`)?.value.trim() || "";
            await this.api(`/knowledge-gaps/${encodeURIComponent(gapId)}`, {
                method: "PATCH",
                body: { status, resolution },
            });
            this.showAlert("知识缺口已更新");
            await this.openAdminTab("quality");
        } catch (error) {
            this.showAlert(`更新失败：${error.message}`, "error");
        } finally {
            button.disabled = false;
        }
    }

    async loadAdminTeam() {
        const content = this.el["admin-content"];
        const [usersData, mcpData] = await Promise.all([
            this.api("/team/users"),
            this.api("/team/mcp-servers"),
        ]);
        let membersData = { members: [], error: "" };
        if (this.currentKb?.id) {
            try {
                membersData = await this.api(`/team/kbs/${encodeURIComponent(this.currentKb.id)}/members`);
            } catch (error) {
                membersData = { members: [], error: error.message };
            }
        }
        this.adminTeamData = {
            users: usersData.users || [],
            servers: mcpData.servers || [],
            members: membersData.members || [],
            membersError: membersData.error || "",
        };
        const teamMode = Boolean(this.authStatus?.team_mode);
        const memberIds = new Set(this.adminTeamData.members.map((member) => member.id));
        const shareableUsers = this.adminTeamData.users.filter((user) => user.is_active && !memberIds.has(user.id));
        content.innerHTML = `<section class="admin-section">
            ${this.adminHeading("团队与权限", "管理成员角色、知识库共享和 MCP 工具审批")}
            ${teamMode ? `<form id="admin-create-user" class="admin-row">
                <div class="admin-row-head"><span class="admin-row-title">新增成员</span><span class="status-pill info">团队模式</span></div>
                <div class="admin-form-grid">
                    <input class="form-control" name="username" minlength="2" maxlength="50" placeholder="用户名" required>
                    <input class="form-control" name="email" type="email" maxlength="255" placeholder="邮箱" required>
                    <input class="form-control" name="password" type="password" minlength="8" maxlength="128" placeholder="初始密码（至少 8 位）" required>
                    <input class="form-control" name="department" maxlength="100" placeholder="部门（可选）">
                    <select class="form-control" name="role"><option value="viewer">viewer · 只读</option><option value="editor">editor · 可编辑</option><option value="admin">admin · 管理员</option></select>
                    <button class="btn btn-primary btn-sm" type="submit">创建成员</button>
                </div>
                <div class="admin-row-meta" data-admin-team-result></div>
            </form>` : '<div class="admin-row"><p class="admin-row-text">当前是本地单用户模式。设置 <code>ZHIGE_AUTH_MODE=team</code> 并重启后，可创建成员和分配角色。</p></div>'}
            <div class="admin-row">
                <table class="admin-table">
                    <thead><tr><th>成员</th><th>部门</th><th>角色</th><th>启用</th><th></th></tr></thead>
                    <tbody>${this.adminTeamData.users.map((user) => `<tr data-admin-user="${this.esc(user.id)}">
                        <td><strong>${this.esc(user.username)}</strong><div class="admin-row-meta">${this.esc(user.email)}</div></td>
                        <td>${this.esc(user.department || "—")}</td>
                        <td><select class="form-control" data-user-role ${teamMode ? "" : "disabled"}><option value="viewer" ${user.role === "viewer" ? "selected" : ""}>viewer</option><option value="editor" ${user.role === "editor" ? "selected" : ""}>editor</option><option value="admin" ${user.role === "admin" ? "selected" : ""}>admin</option></select></td>
                        <td><input type="checkbox" data-user-active ${user.is_active ? "checked" : ""} ${teamMode ? "" : "disabled"} aria-label="启用 ${this.esc(user.username)}"></td>
                        <td><button class="btn btn-ghost btn-sm" type="button" data-user-save="${this.esc(user.id)}" ${teamMode ? "" : "disabled"}>保存</button></td>
                    </tr>`).join("")}</tbody>
                </table>
            </div>
            <div class="admin-row">
                <div class="admin-row-head"><span class="admin-row-title">知识库共享 · ${this.esc(this.currentKb?.name || "未选择知识库")}</span><span class="status-pill">${this.esc(this.currentKb?.access_role || "")}</span></div>
                ${membersData.error ? `<p class="admin-row-text">成员加载失败：${this.esc(membersData.error)}</p>` : ""}
                <div class="admin-row-actions">${this.adminTeamData.members.length ? this.adminTeamData.members.map((member) => `<span class="status-pill">${this.esc(member.username)} · ${this.esc(member.access_role)}</span>`).join("") : '<span class="admin-row-meta">暂无共享成员</span>'}</div>
                ${teamMode && this.currentKb ? `<form id="admin-share-form" class="admin-form-grid">
                    <select class="form-control" name="user_id"><option value="">选择成员</option>${shareableUsers.map((user) => `<option value="${this.esc(user.id)}">${this.esc(user.username)} · ${this.esc(user.email)}</option>`).join("")}</select>
                    <select class="form-control" name="role"><option value="viewer">只读</option><option value="editor">可编辑</option></select>
                    <button class="btn btn-secondary btn-sm" type="submit">共享当前知识库</button>
                </form>` : ""}
                <div class="admin-row-actions">${this.adminTeamData.members.filter((member) => member.member_id).map((member) => `<button class="btn btn-ghost btn-sm" type="button" data-member-remove="${this.esc(member.member_id)}">移除 ${this.esc(member.username)}</button>`).join("")}</div>
            </div>
            <div class="admin-row">
                <div class="admin-row-head"><span class="admin-row-title">MCP 工具审批</span><span class="status-pill warn">${this.adminTeamData.servers.filter((server) => server.approval_status === "pending").length} 个待审批</span></div>
                <div class="admin-list">${this.adminTeamData.servers.length ? this.adminTeamData.servers.map((server) => `<article class="admin-row" data-admin-mcp="${this.esc(server.id)}">
                    <div class="admin-row-head"><span class="admin-row-title">${this.esc(server.name)}</span><span class="status-pill ${server.approval_status === "approved" ? "ok" : server.approval_status === "rejected" ? "bad" : "warn"}">${this.esc(server.approval_status || "pending")}</span></div>
                    <code class="admin-row-meta">${this.esc(server.command)}</code>
                    <div class="admin-form-grid">
                        <input class="form-control span-2" data-mcp-reason="${this.esc(server.id)}" maxlength="1000" placeholder="审批备注或拒绝原因">
                        <div class="admin-row-actions span-2">
                            <button class="btn btn-primary btn-sm" type="button" data-mcp-decision="approved" data-id="${this.esc(server.id)}">批准</button>
                            <button class="btn btn-ghost btn-sm" type="button" data-mcp-decision="rejected" data-id="${this.esc(server.id)}">拒绝</button>
                        </div>
                    </div>
                </article>`).join("") : '<div class="admin-empty">暂无 MCP 连接</div>'}</div>
            </div>
        </section>`;
        content.querySelector("#admin-create-user")?.addEventListener("submit", (event) => {
            event.preventDefault();
            this.createAdminUser(event.currentTarget);
        });
        content.querySelectorAll("[data-user-save]").forEach((button) => {
            button.addEventListener("click", () => this.saveAdminUser(button.dataset.userSave, button));
        });
        content.querySelector("#admin-share-form")?.addEventListener("submit", (event) => {
            event.preventDefault();
            this.shareAdminKb(event.currentTarget);
        });
        content.querySelectorAll("[data-member-remove]").forEach((button) => {
            button.addEventListener("click", () => this.removeAdminKbMember(button.dataset.memberRemove, button));
        });
        content.querySelectorAll("[data-mcp-decision]").forEach((button) => {
            button.addEventListener("click", () => this.decideAdminMcp(button.dataset.id, button.dataset.mcpDecision, button));
        });
    }

    async createAdminUser(form) {
        if (!form.reportValidity()) return;
        const button = form.querySelector("button[type=submit]");
        const result = form.querySelector("[data-admin-team-result]");
        button.disabled = true;
        result.textContent = "正在创建成员";
        const data = new FormData(form);
        try {
            await this.api("/team/users", {
                method: "POST",
                body: {
                    username: String(data.get("username") || "").trim(),
                    email: String(data.get("email") || "").trim(),
                    password: String(data.get("password") || ""),
                    department: String(data.get("department") || "").trim(),
                    role: String(data.get("role") || "viewer"),
                },
            });
            this.showAlert("成员已创建");
            await this.loadAdminTeam();
        } catch (error) {
            result.textContent = `创建失败：${error.message}`;
        } finally {
            button.disabled = false;
        }
    }

    async saveAdminUser(userId, button) {
        const row = this.el["admin-content"].querySelector(`[data-admin-user="${CSS.escape(userId)}"]`);
        if (!row) return;
        button.disabled = true;
        try {
            await this.api(`/team/users/${encodeURIComponent(userId)}`, {
                method: "PATCH",
                body: {
                    role: row.querySelector("[data-user-role]").value,
                    is_active: row.querySelector("[data-user-active]").checked,
                },
            });
            this.showAlert("成员权限已更新");
            await this.loadAdminTeam();
        } catch (error) {
            this.showAlert(`更新成员失败：${error.message}`, "error");
        } finally {
            button.disabled = false;
        }
    }

    async shareAdminKb(form) {
        if (!this.currentKb?.id) return;
        const data = new FormData(form);
        const userId = String(data.get("user_id") || "");
        if (!userId) {
            this.showAlert("请选择要共享的成员", "error");
            return;
        }
        const button = form.querySelector("button[type=submit]");
        button.disabled = true;
        try {
            await this.api(`/team/kbs/${encodeURIComponent(this.currentKb.id)}/members`, {
                method: "POST",
                body: { user_id: userId, role: String(data.get("role") || "viewer") },
            });
            this.showAlert("知识库共享已更新");
            await this.loadAdminTeam();
        } catch (error) {
            this.showAlert(`共享失败：${error.message}`, "error");
        } finally {
            button.disabled = false;
        }
    }

    async removeAdminKbMember(memberId, button) {
        if (!this.currentKb?.id || !confirm("移除这个成员的知识库访问权限？")) return;
        button.disabled = true;
        try {
            await this.api(`/team/kbs/${encodeURIComponent(this.currentKb.id)}/members/${encodeURIComponent(memberId)}`, { method: "DELETE" });
            this.showAlert("已移除共享权限");
            await this.loadAdminTeam();
        } catch (error) {
            this.showAlert(`移除失败：${error.message}`, "error");
        } finally {
            button.disabled = false;
        }
    }

    async decideAdminMcp(serverId, decision, button) {
        button.disabled = true;
        try {
            const reason = this.el["admin-content"].querySelector(`[data-mcp-reason="${CSS.escape(serverId)}"]`)?.value.trim() || "";
            await this.api(`/team/mcp-servers/${encodeURIComponent(serverId)}/decision`, {
                method: "POST",
                body: { decision, reason },
            });
            this.showAlert(decision === "approved" ? "MCP 连接已批准" : "MCP 连接已拒绝");
            this.mcpServers = [];
            await this.loadAdminTeam();
        } catch (error) {
            this.showAlert(`审批失败：${error.message}`, "error");
        } finally {
            button.disabled = false;
        }
    }

    async loadAdminAudit() {
        const content = this.el["admin-content"];
        const data = await this.api("/admin/audit?limit=200");
        const labels = {
            document_delete: "删除资料", document_status: "资料状态", document_reindex: "重索引",
            kb_share: "知识库共享", feedback_resolve: "处理差评", knowledge_gap_update: "知识缺口",
            backup_export: "导出备份", backup_restore: "恢复包", mcp_approve: "MCP 审批",
            config_change: "配置变更", evaluation_run: "运行评估",
            user_disable: "停用成员", user_enable: "启用成员",
        };
        content.innerHTML = `<section class="admin-section">
            ${this.adminHeading("审计日志", "追踪资料、权限、审批、备份和质量治理操作", '<button class="btn btn-ghost btn-sm" id="admin-audit-refresh" type="button">刷新</button>')}
            <input class="form-control" id="admin-audit-filter" type="search" placeholder="按动作、操作者或资源筛选">
            <div class="admin-list">${data.items?.length ? data.items.map((item) => `<article class="admin-row" data-audit-row data-search="${this.esc(`${item.action} ${item.user_id || ""} ${item.resource || ""}`)}">
                <div class="admin-row-head"><span class="admin-row-title">${this.esc(labels[item.action] || item.action)}</span><span class="status-pill ${item.success ? "ok" : "bad"}">${item.success ? "成功" : "失败"}</span></div>
                <div class="admin-row-actions"><span class="status-pill">${this.esc(item.action)}</span><span class="admin-row-meta">${this.esc(this.formatAdminTime(item.created_at))}</span></div>
                <p class="admin-row-text">操作者 ${this.esc(item.user_id || "—")} · 资源 ${this.esc(item.resource || "—")}${item.error ? ` · ${this.esc(item.error)}` : ""}</p>
                ${Object.keys(item.details || {}).length ? `<pre class="admin-code">${this.esc(JSON.stringify(item.details, null, 2))}</pre>` : ""}
            </article>`).join("") : '<div class="admin-empty">暂无审计记录</div>'}</div>
        </section>`;
        content.querySelector("#admin-audit-refresh")?.addEventListener("click", () => this.openAdminTab("audit"));
        content.querySelector("#admin-audit-filter")?.addEventListener("input", (event) => {
            const keyword = event.target.value.trim().toLowerCase();
            content.querySelectorAll("[data-audit-row]").forEach((row) => {
                row.hidden = Boolean(keyword) && !String(row.dataset.search || "").toLowerCase().includes(keyword);
            });
        });
    }

    renderAdminEvaluation(data) {
        const content = this.el["admin-content"];
        const reports = data.reports || [];
        const percent = (value) => Number.isFinite(Number(value))
            ? `${(Number(value) * 100).toFixed(Number(value) === 1 ? 0 : 1)}%`
            : "—";
        const metricCard = (summary, title, meta, path = "") => {
            const verifiability = summary?.verifiability || {};
            const latency = Number(summary?.avg_latency_ms);
            return `<article class="admin-row">
                <div class="admin-row-head"><span class="admin-row-title">${this.esc(title)}</span><span class="status-pill ${Number(summary?.retrieval_hit_rate) >= 0.8 ? "ok" : "warn"}">${this.esc(String(summary?.samples || 0))} 题</span></div>
                <div class="admin-row-meta">${this.esc(meta || path || "评估报告")}</div>
                <div class="metric-grid">
                    <div class="metric-card"><span>检索命中</span><strong>${percent(summary?.retrieval_hit_rate)}</strong></div>
                    <div class="metric-card"><span>关键词覆盖</span><strong>${percent(summary?.avg_keyword_coverage)}</strong></div>
                    <div class="metric-card"><span>可验证回答</span><strong>${percent(verifiability.verified_rate)}</strong></div>
                    <div class="metric-card"><span>平均耗时</span><strong>${Number.isFinite(latency) ? `${Math.round(latency)} ms` : "—"}</strong></div>
                </div>
                <details><summary class="admin-row-meta">查看完整指标与逐题明细</summary><pre class="admin-code">${this.esc(JSON.stringify(summary, null, 2))}</pre></details>
            </article>`;
        };
        content.innerHTML = `<section class="admin-section">
            ${this.adminHeading("评估与回归", "把差评导出为评估集后，用同一套指标验证检索和回答是否退化", '<button class="btn btn-primary btn-sm" id="admin-eval-run" type="button">运行评估</button>')}
            ${this.evaluationResult ? metricCard(this.evaluationResult, "本次运行结果", `${this.evaluationResult.kb?.name || this.currentKb?.name || "默认知识库"} · ${this.formatAdminTime(new Date().toISOString())}`, this.evaluationResult.archive_path) : ""}
            <div class="admin-list">${reports.length ? reports.map((report) => metricCard(report.summary || { samples: 0 }, report.name, `${this.formatAdminTime(report.updated_at)} · ${Math.max(1, Math.round(Number(report.size || 0) / 1024))} KB`, report.path)).join("") : '<div class="admin-empty">还没有评估报告。准备当前知识库的评估集后运行。</div>'}</div>
        </section>`;
        content.querySelector("#admin-eval-run")?.addEventListener("click", (event) => this.runAdminEvaluation(event.currentTarget));
    }

    async runAdminEvaluation(button) {
        button.disabled = true;
        button.textContent = "评估运行中";
        const content = this.el["admin-content"];
        let status = content.querySelector("[data-eval-status]");
        if (!status) {
            status = document.createElement("div");
            status.dataset.evalStatus = "";
            status.className = "admin-empty";
            content.prepend(status);
        }
        status.textContent = "正在逐题运行检索与回答，可能需要几分钟。页面保持可用。";
        try {
            const query = this.currentKb?.id
                ? `?kb_id=${encodeURIComponent(this.currentKb.id)}`
                : "";
            this.evaluationResult = await this.api(`/eval/run${query}`, { method: "POST" });
            this.showAlert("评估已完成");
            this.renderAdminEvaluation(await this.api("/admin/evaluations"));
        } catch (error) {
            status.textContent = `评估失败：${error.message}`;
            button.disabled = false;
            button.textContent = "运行评估";
        }
    }

    renderAdminBackup(data) {
        const content = this.el["admin-content"];
        const pending = data.pending;
        const backups = data.backups || [];
        content.innerHTML = `<section class="admin-section">
            ${this.adminHeading("备份与恢复", "备份运行数据与原文，恢复包在服务重启前保持隔离", '<button class="btn btn-primary btn-sm" id="admin-backup-create" type="button">立即备份</button>')}
            <div class="admin-row">
                <div class="admin-row-head"><span class="admin-row-title">备份安全边界</span><span class="status-pill ok">秘密排除</span></div>
                <p class="admin-row-text">包含数据库、向量索引、原文、评估集和知识图谱；不打包 .env、API Key、Cookie 或浏览器数据。账号口令摘要随数据库用于恢复团队账号，备份文件应按敏感数据保管。</p>
            </div>
            ${pending ? `<div class="admin-row">
                <div class="admin-row-head"><span class="admin-row-title">待应用恢复包</span><span class="status-pill warn">等待重启</span></div>
                <p class="admin-row-text">${this.esc(pending.original_name || "backup.zip")} · ${this.esc(this.formatAdminTime(pending.created_at))} · ${this.esc(String(pending.token || "").slice(0, 10))}</p>
                <div class="admin-row-actions"><button class="btn btn-ghost btn-sm" id="admin-restore-cancel" type="button">取消待恢复包</button></div>
            </div>` : ""}
            <div class="admin-row">
                <div class="admin-row-head"><span class="admin-row-title">恢复工作区</span><span class="status-pill info">需二次确认</span></div>
                <form id="admin-restore-form" class="admin-form-grid">
                    <input class="form-control span-2" id="admin-restore-file" type="file" accept=".zip,application/zip" required>
                    <input class="form-control span-2" id="admin-restore-confirm" maxlength="7" placeholder="输入 RESTORE 确认" required>
                    <button class="btn btn-secondary btn-sm" type="submit">校验并暂存恢复包</button>
                </form>
                <div class="admin-row-meta" data-restore-result>恢复前会自动再生成一份当前数据备份。</div>
            </div>
            <div class="admin-row">
                <div class="admin-row-head"><span class="admin-row-title">最近备份</span><span class="status-pill">${backups.length} 个</span></div>
                <div class="admin-list">${backups.length ? backups.map((backup) => `<div class="admin-row-head">
                    <a href="${this.esc(backup.download_url)}" download>${this.esc(backup.name)}</a>
                    <span class="admin-row-meta">${Math.max(1, Math.round(Number(backup.size || 0) / 1024 / 1024))} MB · ${this.esc(this.formatAdminTime(backup.created_at))}</span>
                </div>`).join("") : '<div class="admin-empty">还没有备份</div>'}</div>
            </div>
            ${data.last ? `<div class="admin-row"><div class="admin-row-head"><span class="admin-row-title">上次恢复记录</span><span class="status-pill">${this.esc(data.last.status || "unknown")}</span></div><pre class="admin-code">${this.esc(JSON.stringify(data.last, null, 2))}</pre></div>` : ""}
        </section>`;
        content.querySelector("#admin-backup-create")?.addEventListener("click", (event) => this.createAdminBackup(event.currentTarget));
        content.querySelector("#admin-restore-form")?.addEventListener("submit", (event) => {
            event.preventDefault();
            this.stageAdminRestore(event.currentTarget);
        });
        content.querySelector("#admin-restore-cancel")?.addEventListener("click", () => this.cancelAdminRestore());
    }

    async createAdminBackup(button) {
        button.disabled = true;
        button.textContent = "正在备份";
        try {
            const result = await this.api("/admin/backup", { method: "POST" });
            this.showAlert(`备份完成 · ${Math.max(1, Math.round(Number(result.size || 0) / 1024 / 1024))} MB`);
            await this.openAdminTab("backup");
        } catch (error) {
            this.showAlert(`备份失败：${error.message}`, "error");
        } finally {
            button.disabled = false;
            button.textContent = "立即备份";
        }
    }

    async stageAdminRestore(form) {
        const file = this.el["admin-restore-file"].files?.[0];
        const confirmText = this.el["admin-restore-confirm"].value.trim();
        const resultBox = form.querySelector("[data-restore-result]");
        if (!file) {
            resultBox.textContent = "请选择 ZIP 恢复包";
            return;
        }
        if (confirmText.toUpperCase() !== "RESTORE") {
            resultBox.textContent = "请输入 RESTORE 确认恢复";
            return;
        }
        const body = new FormData();
        body.append("file", file);
        body.append("confirm", confirmText);
        const button = form.querySelector("button[type=submit]");
        button.disabled = true;
        resultBox.textContent = "正在校验恢复包";
        try {
            const result = await this.api("/admin/restore", { method: "POST", body, raw: true });
            resultBox.textContent = result.message || "恢复包已暂存，请重启服务";
            this.showAlert("恢复包校验通过，将在重启时应用");
            await this.openAdminTab("backup");
        } catch (error) {
            resultBox.textContent = `恢复包校验失败：${error.message}`;
        } finally {
            button.disabled = false;
        }
    }

    async cancelAdminRestore() {
        if (!confirm("取消待应用的恢复包？当前工作区不会改变。")) return;
        try {
            await this.api("/admin/restore", { method: "DELETE" });
            this.showAlert("待恢复包已取消");
            await this.openAdminTab("backup");
        } catch (error) {
            this.showAlert(`取消失败：${error.message}`, "error");
        }
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

    async fetchModels(kind) {
        const backend = document.getElementById(kind + "-backend")?.value || "openai";
        const baseUrl = document.getElementById(kind + "-base-url")?.value.trim() || "";
        const apiKey = document.getElementById(kind + "-api-key")?.value.trim() || "";
        if (!baseUrl) {
            this.setSettingsResult("请先填写服务地址", "err");
            return;
        }
        this.setSettingsResult("正在拉取模型…");
        try {
            const result = await this.api("/settings/models", {
                method: "POST",
                body: {
                    kind: kind === "llm" ? "llm" : "embedding",
                    backend,
                    base_url: baseUrl,
                    api_key: apiKey,
                },
            });
            if (!result.ok) {
                this.setSettingsResult(result.detail || "拉取失败", "err");
                return;
            }
            const datalist = document.getElementById(kind + "-model-list");
            if (datalist) {
                datalist.innerHTML = (result.models || []).map((m) => {
                    const id = m.id || "";
                    const name = m.name || id;
                    return '<option value="' + this.esc(id) + '">' + this.esc(name) + "</option>";
                }).join("");
            }
            this.setSettingsResult("已拉取 " + result.count + " 个模型，可下拉选择", "ok");
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
            month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false,
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
            <li class="docs-record">
                <button class="docs-item" type="button" data-doc-open="${this.esc(document.id)}">
                    <span class="item-main">${this.esc(document.title || document.source)}</span>
                    <span class="item-meta">${Number(document.chunks || 0)} 分块 · ${this.esc(document.doc_status || "active")} · ${this.esc(document.index_status || "indexed")}</span>
                </button>
                ${this.documentActionsMarkup(document)}
            </li>
        `).join("")}</ul>`;
        box.querySelectorAll("[data-doc-open]").forEach((item) => {
            item.addEventListener("click", () => {
                const document = this.documents.find((entry) => entry.id === item.dataset.docOpen);
                if (document) this.openDocChunks(document.source, document);
            });
        });
        box.querySelectorAll(".docs-record").forEach((row) => {
            const document = this.documents.find((entry) => entry.id === row.querySelector("[data-doc-open]")?.dataset.docOpen);
            if (document) this.bindDocumentActions(row, document);
        });
    }

    documentActionsMarkup(document) {
        const locked = document.access_role === "viewer";
        const status = document.doc_status || "active";
        const indexStatus = document.index_status || "indexed";
        const indexClass = indexStatus === "failed" ? "bad" : indexStatus === "indexed" ? "ok" : "warn";
        return `<div class="docs-row-actions">
            <span class="status-pill ${indexClass}" title="${this.esc(document.last_index_error || "")}">${this.esc(indexStatus)}</span>
            <select class="form-control docs-status-select" data-doc-status="${this.esc(document.id)}" aria-label="资料状态" ${locked ? "disabled" : ""}>
                <option value="active" ${status === "active" ? "selected" : ""}>生效</option>
                <option value="draft" ${status === "draft" ? "selected" : ""}>草稿</option>
                <option value="expired" ${status === "expired" ? "selected" : ""}>失效</option>
            </select>
            <button class="btn btn-ghost btn-sm" type="button" data-doc-reindex="${this.esc(document.id)}" ${locked || document.source_type === "web" ? "disabled" : ""}>重索引</button>
            <button class="btn btn-ghost btn-sm docs-delete" type="button" data-doc-delete="${this.esc(document.id)}" ${locked ? "disabled" : ""}>删除</button>
        </div>`;
    }

    bindDocumentActions(root, document) {
        root.querySelector(`[data-doc-status="${CSS.escape(document.id)}"]`)?.addEventListener("change", (event) => {
            this.updateDocumentStatus(document, event.target.value, event.target);
        });
        root.querySelector(`[data-doc-reindex="${CSS.escape(document.id)}"]`)?.addEventListener("click", (event) => {
            this.reindexDocument(document, event.currentTarget);
        });
        root.querySelector(`[data-doc-delete="${CSS.escape(document.id)}"]`)?.addEventListener("click", () => {
            this.deleteDocument(document);
        });
    }

    async updateDocumentStatus(document, status, select) {
        select.disabled = true;
        try {
            await this.api(`/documents/${encodeURIComponent(document.id)}/status`, {
                method: "PATCH",
                body: { status },
            });
            document.doc_status = status;
            this.showAlert(`资料状态已改为 ${status}`);
            if (this.el["docs-modal"].classList.contains("active")) await this.loadDocs();
        } catch (error) {
            select.value = document.doc_status || "active";
            this.showAlert(`状态修改失败：${error.message}`, "error");
        } finally {
            select.disabled = false;
        }
    }

    async reindexDocument(document, button) {
        button.disabled = true;
        const label = button.textContent;
        button.textContent = "索引中";
        try {
            const result = await this.api(`/documents/${encodeURIComponent(document.id)}/reindex`, { method: "POST" });
            this.showAlert(`重新索引完成 · ${result.ingested || result.chunks || 0} 个分块`);
            await this.loadDocs();
            await this.loadKnowledgeBases();
        } catch (error) {
            this.showAlert(`重索引失败：${error.message}`, "error");
            if (this.el["docs-modal"].classList.contains("active")) await this.loadDocs();
        } finally {
            button.disabled = false;
            button.textContent = label;
        }
    }

    async deleteDocument(document) {
        if (!confirm(`删除“${document.title || document.source}”的索引记录？`)) return;
        const purgeRaw = document.source_type !== "web"
            && confirm("是否同时删除本地原始文件？选择取消将保留原文件，之后可重新上传。");
        try {
            const params = new URLSearchParams({ purge_raw: String(purgeRaw) });
            await this.api(`/documents/${encodeURIComponent(document.id)}?${params}`, { method: "DELETE" });
            this.showAlert(`已删除 ${document.title || document.source}`);
            await this.loadDocs();
            await this.loadKnowledgeBases();
        } catch (error) {
            this.showAlert(`删除失败：${error.message}`, "error");
        }
    }

    async openDocChunks(source, document = null) {
        const box = this.el["docs-content"];
        box.innerHTML = '<div class="command-empty">加载中</div>';
        const params = new URLSearchParams();
        if (this.currentKb?.id) params.set("kb_id", this.currentKb.id);
        params.set("limit", "200");
        try {
            const data = await this.api(`/documents/${encodeURIComponent(source)}/chunks?${params}`);
            this.el["docs-back"].hidden = false;
            this.el["docs-filter"].hidden = true;
            this.renderChunks(source, data.chunks || [], document);
        } catch (error) {
            box.innerHTML = `<div class="command-empty">${this.esc(error.message)}</div>`;
        }
    }

    renderChunks(source, chunks, document = null) {
        const box = this.el["docs-content"];
        const detailHead = document ? `<header class="docs-detail-head">
            <div>
                <h3>${this.esc(document.title || source)}</h3>
                <div class="docs-detail-meta">${Number(document.chunks || chunks.length)} 分块 · ${this.esc(document.doc_status || "active")} · ${this.esc(document.index_status || "indexed")}</div>
            </div>
            ${this.documentActionsMarkup(document)}
        </header>` : `<h3 class="docs-doc-title">${this.esc(source)}</h3>`;
        if (!chunks.length) {
            box.innerHTML = `${detailHead}<div class="command-empty">暂无分块</div>`;
            if (document) this.bindDocumentActions(box, document);
            return;
        }
        const sorted = chunks.slice().sort((a, b) => Number(a.metadata?.chunk_index || 0) - Number(b.metadata?.chunk_index || 0));
        box.innerHTML = `${detailHead}${sorted.map((chunk, index) => {
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
        if (document) this.bindDocumentActions(box, document);
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
