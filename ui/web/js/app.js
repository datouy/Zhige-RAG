/**
 * ChineseRAGKB Web UI - Chat Application
 * 使用原生 fetch + WebSocket 实现流式问答
 */

class ChatApp {
    constructor() {
        this.ws = null;
        this.connected = false;
        this.currentSources = [];
        this.wsUrl = this.getWebSocketUrl();
        
        this.init();
    }

    getWebSocketUrl() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${protocol}//${window.location.host}/ws/chat`;
    }

    getApiUrl(path) {
        return `/api${path}`;
    }

    init() {
        this.bindEvents();
        this.loadDocuments();
        this.connectWebSocket();
    }

    bindEvents() {
        // 发送消息
        const input = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-btn');
        
        if (sendBtn) {
            sendBtn.addEventListener('click', () => this.sendMessage());
        }
        
        if (input) {
            input.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
        }

        // 文件上传
        const uploadBtn = document.getElementById('upload-btn');
        if (uploadBtn) {
            uploadBtn.addEventListener('click', () => this.uploadFiles());
        }

        // 清空对话
        const clearBtn = document.getElementById('clear-btn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearChat());
        }
    }

    connectWebSocket() {
        try {
            this.ws = new WebSocket(this.wsUrl);
            
            this.ws.onopen = () => {
                this.connected = true;
                this.updateStatus('已连接', true);
                console.log('WebSocket connected');
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                } catch (e) {
                    console.error('Failed to parse message:', e);
                }
            };
            
            this.ws.onclose = () => {
                this.connected = false;
                this.updateStatus('未连接', false);
                console.log('WebSocket disconnected');
                // 尝试重新连接
                setTimeout(() => this.connectWebSocket(), 3000);
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateStatus('连接错误', false);
            };
        } catch (e) {
            console.error('Failed to create WebSocket:', e);
            this.updateStatus('连接失败', false);
        }
    }

    updateStatus(text, connected) {
        const statusEl = document.getElementById('ws-status');
        if (statusEl) {
            statusEl.textContent = text;
            statusEl.className = connected ? 'status connected' : 'status disconnected';
        }
    }

    sendMessage() {
        const input = document.getElementById('chat-input');
        const query = input.value.trim();
        
        if (!query) return;
        if (!this.connected) {
            this.showAlert('WebSocket 未连接，请等待重连...', 'error');
            return;
        }

        // 显示用户消息
        this.appendMessage('user', query);
        input.value = '';

        // 禁用输入
        input.disabled = true;
        document.getElementById('send-btn').disabled = true;

        // 添加加载指示器
        this.showTypingIndicator();

        // 发送 WebSocket 消息
        const topK = parseInt(document.getElementById('top-k')?.value || '4');
        this.ws.send(JSON.stringify({
            query: query,
            top_k: topK
        }));
    }

    handleMessage(event) {
        this.hideTypingIndicator();
        
        const type = event.event;
        const data = event.data;

        switch (type) {
            case 'hits':
                this.currentSources = data;
                break;
            case 'token':
                this.appendToken(data);
                break;
            case 'done':
                this.showSources(this.currentSources);
                this.enableInput();
                break;
            case 'error':
                this.showAlert(`错误: ${data}`, 'error');
                this.enableInput();
                break;
        }
    }

    appendMessage(role, content) {
        const messages = document.getElementById('messages');
        if (!messages) return;

        const div = document.createElement('div');
        div.className = `message ${role}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;
        
        div.appendChild(contentDiv);
        messages.appendChild(div);
        messages.scrollTop = messages.scrollHeight;
    }

    appendToken(token) {
        const messages = document.getElementById('messages');
        if (!messages) return;

        // 找到最后一个 assistant 消息或创建新的
        let lastMsg = messages.querySelector('.message.assistant:last-child');
        if (!lastMsg) {
            lastMsg = document.createElement('div');
            lastMsg.className = 'message assistant';
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.id = 'current-answer';
            lastMsg.appendChild(contentDiv);
            messages.appendChild(lastMsg);
        }

        const contentDiv = lastMsg.querySelector('.message-content');
        const answerEl = document.getElementById('current-answer');
        if (answerEl) {
            answerEl.textContent += token;
            answerEl.id = null;
        } else {
            contentDiv.textContent += token;
        }
        
        messages.scrollTop = messages.scrollHeight;
    }

    showSources(sources) {
        if (!sources || sources.length === 0) return;

        const messages = document.getElementById('messages');
        if (!messages) return;

        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message sources';
        
        const title = document.createElement('h4');
        title.textContent = `📎 引用来源 (${sources.length} 条)`;
        sourcesDiv.appendChild(title);

        const ul = document.createElement('ul');
        sources.forEach((s, i) => {
            const li = document.createElement('li');
            const score = typeof s.score === 'number' ? s.score.toFixed(3) : '—';
            li.innerHTML = `<strong>[${i + 1}]</strong> ${s.cite || s.source || '未知'} <span class="score">(相似度: ${score})</span>`;
            
            // 点击显示详情
            li.addEventListener('click', () => this.showChunkDetail(s));
            
            ul.appendChild(li);
        });
        
        sourcesDiv.appendChild(ul);
        messages.appendChild(sourcesDiv);
        messages.scrollTop = messages.scrollHeight;
    }

    showChunkDetail(source) {
        const modal = document.getElementById('chunk-modal');
        const modalContent = document.getElementById('modal-text');
        
        if (modal && modalContent) {
            modalContent.textContent = source.snippet || source.content || '无内容';
            modal.classList.add('active');
        }
    }

    closeModal() {
        const modal = document.getElementById('chunk-modal');
        if (modal) {
            modal.classList.remove('active');
        }
    }

    showTypingIndicator() {
        const messages = document.getElementById('messages');
        if (!messages) return;

        const indicator = document.createElement('div');
        indicator.className = 'message assistant';
        indicator.id = 'typing-indicator';
        indicator.innerHTML = `
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
        `;
        messages.appendChild(indicator);
        messages.scrollTop = messages.scrollHeight;
    }

    hideTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    enableInput() {
        const input = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-btn');
        if (input) input.disabled = false;
        if (sendBtn) sendBtn.disabled = false;
        if (input) input.focus();
    }

    clearChat() {
        const messages = document.getElementById('messages');
        if (messages) {
            messages.innerHTML = '';
        }
        this.currentSources = [];
    }

    showAlert(message, type = 'success') {
        const container = document.getElementById('alert-container');
        if (!container) return;

        const alert = document.createElement('div');
        alert.className = `alert alert-${type}`;
        alert.textContent = message;
        
        container.innerHTML = '';
        container.appendChild(alert);
        
        setTimeout(() => alert.remove(), 5000);
    }

    async loadDocuments() {
        try {
            const resp = await fetch(this.getApiUrl('/documents'));
            const data = await resp.json();
            
            this.renderDocuments(data.documents || []);
            this.updateStats(data.total_chunks || 0);
        } catch (e) {
            console.error('Failed to load documents:', e);
        }
    }

    renderDocuments(docs) {
        const list = document.getElementById('doc-list');
        if (!list) return;

        if (docs.length === 0) {
            list.innerHTML = `
                <div class="empty-state">
                    <div class="icon">📭</div>
                    <p>知识库为空，请先上传文档</p>
                </div>
            `;
            return;
        }

        list.innerHTML = docs.map(doc => `
            <div class="doc-item">
                <div>
                    <div class="name">📄 ${doc.source}</div>
                    <div class="meta">${doc.chunks} 个分块</div>
                </div>
                <div class="actions">
                    <button class="btn btn-secondary" onclick="app.viewDocument('${doc.source}')">预览</button>
                    <button class="btn btn-danger" onclick="app.deleteDocument('${doc.source}')">删除</button>
                </div>
            </div>
        `).join('');
    }

    updateStats(chunks) {
        const statEl = document.getElementById('chunk-count');
        if (statEl) {
            statEl.textContent = chunks;
        }
    }

    async viewDocument(source) {
        try {
            const resp = await fetch(this.getApiUrl(`/documents/${encodeURIComponent(source)}/chunks`));
            const data = await resp.json();
            
            // 在模态框中显示
            const modal = document.getElementById('chunk-modal');
            const modalContent = document.getElementById('modal-text');
            
            if (modal && modalContent) {
                const chunksText = data.chunks.map((c, i) => 
                    `【分块 ${i + 1}】\n${c.text}\n`
                ).join('\n---\n');
                modalContent.textContent = chunksText || '无分块内容';
                modal.classList.add('active');
            }
        } catch (e) {
            this.showAlert('预览失败: ' + e.message, 'error');
        }
    }

    async deleteDocument(source) {
        if (!confirm(`确定要删除文档 "${source}" 吗？`)) return;
        
        try {
            // 通过 API 删除（需要实现 delete 端点）
            this.showAlert('删除功能需要后端支持', 'error');
            this.loadDocuments(); // 刷新列表
        } catch (e) {
            this.showAlert('删除失败: ' + e.message, 'error');
        }
    }

    async uploadFiles() {
        const fileInput = document.getElementById('file-input');
        const files = fileInput.files;
        
        if (!files || files.length === 0) {
            this.showAlert('请选择文件', 'error');
            return;
        }

        const formData = new FormData();
        for (const file of files) {
            formData.append('file', file);
        }

        const uploadBtn = document.getElementById('upload-btn');
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<span class="spinner"></span> 上传中...';

        try {
            const resp = await fetch(this.getApiUrl('/ingest'), {
                method: 'POST',
                body: formData
            });
            
            if (!resp.ok) throw new Error(await resp.text());
            
            const result = await resp.json();
            this.showAlert(`上传成功！新增 ${result.ingested} 个分块`, 'success');
            this.loadDocuments();
            fileInput.value = '';
        } catch (e) {
            this.showAlert('上传失败: ' + e.message, 'error');
        } finally {
            uploadBtn.disabled = false;
            uploadBtn.textContent = '上传并入库';
        }
    }
}

// 初始化应用
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new ChatApp();
});

// 关闭模态框
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal') || e.target.classList.contains('modal-close')) {
        const modal = document.getElementById('chunk-modal');
        if (modal) modal.classList.remove('active');
    }
});
