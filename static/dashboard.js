let configSnapshot = {};
let modelsSnapshot = {};
let lockStatuses = {};

async function fetchModels() {
    try {
        const resp = await fetch('/api/models');
        modelsSnapshot = await resp.json();
        
        // Update model count badges
        for (const [provider, list] of Object.entries(modelsSnapshot)) {
            const badge = document.getElementById('count-' + provider);
            if (badge) {
                badge.innerText = `${list.length} models fetched`;
            }
        }
    } catch (e) {
        console.error("Failed to fetch autocomplete list", e);
    }
}

async function loadConfigs() {
    try {
        const resp = await fetch('/api/config');
        const data = await resp.json();
        configSnapshot = data.configs;
        lockStatuses = data.key_statuses;
        
        for (const [key, value] of Object.entries(configSnapshot)) {
            const el = document.getElementById(key);
            if (!el) continue;

            if (el.type === 'checkbox') {
                el.checked = !!value;
            } else {
                el.value = value;
            }

            // Apply status badge & Lock check
            const badge = document.getElementById('badge-' + key);
            if (badge) {
                const status = lockStatuses[key] || "Not Set";
                badge.innerText = status;
                badge.className = 'badge-status ' + status.toLowerCase().replace(/[^a-z0-9]+/g, '-');
                
                if (status === "Env Locked" || status === "Locked") {
                    el.disabled = true;
                    // Add lock indicators
                    const parent = el.closest('.form-group');
                    if (parent) parent.style.opacity = '0.7';
                } else {
                    el.disabled = false;
                    const parent = el.closest('.form-group');
                    if (parent) parent.style.opacity = '1';
                }
            }
        }
        // Populate Fallback Chains
        if (data.fallbacks) {
            for (const [alias, entry] of Object.entries(data.fallbacks)) {
                const inputId = 'FALLBACK_ORDER_' + alias.toUpperCase();
                if (entry.fallback_order) {
                    tagState[inputId] = [...entry.fallback_order];
                    renderTags(inputId);
                }
            }
        }

        // Update dynamic guide tokens
        const tokenVal = configSnapshot["GATEWAY_AUTH_TOKEN"] || "freecc";
        document.querySelectorAll(".guide-token-display").forEach(el => {
            el.innerText = tokenVal;
        });
        const tokenState = document.getElementById("guide-token-state");
        if (tokenState) {
            if (configSnapshot["GATEWAY_AUTH_TOKEN"]) {
                tokenState.innerText = "ACTIVE (Value: " + configSnapshot["GATEWAY_AUTH_TOKEN"] + ")";
                tokenState.style.color = "#ffffff";
            } else {
                tokenState.innerText = "DISABLED (Any dummy token will be accepted)";
                tokenState.style.color = "var(--text-dim)";
            }
        }
    } catch (e) {
        console.error("Failed to load configs", e);
    }
}

async function revertConfigs() {
    await loadConfigs();
    showToast("Changes reverted.", "success");
}

async function saveConfigs(event) {
    event.preventDefault();
    const payload = {};
    for (const key of Object.keys(configSnapshot)) {
        const el = document.getElementById(key);
        if (!el) continue;

        // Check lock status
        if (lockStatuses[key] === "Env Locked" || lockStatuses[key] === "Locked") {
            payload[key] = configSnapshot[key];
            continue;
        }

        if (el.type === 'checkbox') {
            payload[key] = el.checked;
        } else if (el.type === 'number') {
            payload[key] = parseInt(el.value) || 0;
        } else {
            payload[key] = el.value;
        }
    }

    // Also collect Fallback Chains from tagState
    ['CLAUDE_DEFAULT', 'CLAUDE_OPUS', 'CLAUDE_SONNET', 'CLAUDE_SONNET_1M', 'CLAUDE_HAIKU'].forEach(alias => {
        const inputId = 'FALLBACK_ORDER_' + alias;
        // Add any remaining text typed in input before saving
        const inputEl = document.getElementById(inputId);
        if (inputEl && inputEl.value.trim()) {
            addTag(inputId, inputEl.value);
        }
        payload[inputId] = (tagState[inputId] || []).join(', ');
    });


    try {
        const resp = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ configs: payload })
        });
        const res = await resp.json();
        if (res.status === 'success') {
            showToast("Configuration saved & reloaded.", "success");
            configSnapshot = payload;
            await loadConfigs();
            fetchModels();
        } else {
            showToast("Commit failed: " + res.message, "error");
        }
    } catch (e) {
        showToast("Network error saving configs.", "error");
    }
}

function showToast(message, type = "success") {
    const toast = document.getElementById('saveToast');
    if (!toast) return;
    toast.innerText = message;
    if (type === "error") {
        toast.style.background = "#1a0f0f";
        toast.style.borderColor = "#f87171";
        toast.style.color = "#f87171";
    } else {
        toast.style.background = "#0c0d11";
        toast.style.borderColor = "#ffffff";
        toast.style.color = "#ffffff";
    }
    toast.classList.add('show');
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

function switchPane(paneId, btnElement = null) {
    if (!paneId) return;

    document.querySelectorAll('.control-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.panel-pane').forEach(pane => pane.classList.remove('active'));

    const targetPane = document.getElementById(paneId);
    if (targetPane) {
        targetPane.classList.add('active');
    } else {
        console.error("Pane target not found:", paneId);
        return;
    }

    if (btnElement && btnElement.classList) {
        btnElement.classList.add('active');
    } else {
        const btn = document.querySelector(`.control-btn[data-pane="${paneId}"]`) || document.querySelector(`.control-btn[onclick*="${paneId}"]`);
        if (btn) btn.classList.add('active');
    }

    const actionButtons = document.getElementById('actionButtons');
    if (actionButtons) {
        const readOnlyPanes = ['pane-doctor', 'pane-logs', 'pane-guide', 'pane-router', 'pane-dev'];
        actionButtons.style.display = readOnlyPanes.includes(paneId) ? 'none' : 'flex';
    }

    if (paneId === 'pane-router') {
        fetchRouterStatus();
    } else if (paneId === 'pane-dev') {
        fetchDevPayloads();
    }
}

async function fetchRouterStatus() {
    try {
        const resp = await fetch('/api/router-status');
        const data = await resp.json();
        
        document.getElementById('router-total-models').innerText = data.summary.total_models || 0;
        document.getElementById('router-healthy-models').innerText = data.summary.healthy || 0;
        document.getElementById('router-open-circuits').innerText = data.summary.circuit_open || 0;

        // Render Client Request Resolution Mappings
        const mappingsGrid = document.getElementById('router-client-mappings-grid');
        if (mappingsGrid && data.client_mappings) {
            mappingsGrid.innerHTML = '';
            data.client_mappings.forEach(m => {
                const card = document.createElement('div');
                card.style.background = '#050608';
                card.style.border = m.is_fallback ? '1px solid #eab308' : '1px solid var(--border-subtle)';
                card.style.padding = '1rem';
                card.style.borderRadius = '8px';
                card.style.display = 'flex';
                card.style.flexDirection = 'column';
                card.style.gap = '6px';

                const headerRow = document.createElement('div');
                headerRow.style.display = 'flex';
                headerRow.style.justifyContent = 'space-between';
                headerRow.style.alignItems = 'center';

                const titleSpan = document.createElement('span');
                titleSpan.style.fontSize = '0.8rem';
                titleSpan.style.fontWeight = '700';
                titleSpan.style.color = '#ffffff';
                titleSpan.innerText = m.label;

                const stepBadge = document.createElement('span');
                stepBadge.style.fontSize = '0.65rem';
                stepBadge.style.fontWeight = '800';
                stepBadge.style.padding = '2px 6px';
                stepBadge.style.borderRadius = '4px';
                if (m.is_fallback) {
                    stepBadge.style.background = 'rgba(234, 179, 8, 0.2)';
                    stepBadge.style.color = '#fef08a';
                    stepBadge.style.border = '1px solid #eab308';
                    stepBadge.innerText = m.step_name;
                } else {
                    stepBadge.style.background = 'rgba(34, 197, 94, 0.15)';
                    stepBadge.style.color = '#4ade80';
                    stepBadge.style.border = '1px solid #22c55e';
                    stepBadge.innerText = 'PRIMARY DIRECT';
                }

                headerRow.appendChild(titleSpan);
                headerRow.appendChild(stepBadge);
                card.appendChild(headerRow);

                const activeTarget = document.createElement('div');
                activeTarget.style.fontSize = '0.75rem';
                activeTarget.style.fontFamily = "'JetBrains Mono', monospace";
                activeTarget.style.color = m.is_fallback ? '#fef08a' : '#ffffff';
                activeTarget.innerHTML = `<strong>Active Target:</strong> ${m.resolved_target}`;
                card.appendChild(activeTarget);

                const primaryTargetVal = m.primary || (m.chain && m.chain[0]) || '-';
                const primaryInfo = document.createElement('div');
                primaryInfo.style.fontSize = '0.7rem';
                primaryInfo.style.color = 'var(--text-muted)';
                primaryInfo.innerText = `Primary Target: ${primaryTargetVal}`;
                card.appendChild(primaryInfo);

                mappingsGrid.appendChild(card);
            });
        }

        const tbody = document.getElementById('router-table-body');
        tbody.innerHTML = '';

        const entries = Object.entries(data.models);
        if (entries.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" style="padding: 1rem; text-align: center; color: var(--text-muted);">No models routed yet. Try sending a request through the proxy.</td></tr>';
            return;
        }

        for (const [modelId, status] of entries) {
            const tr = document.createElement('tr');
            tr.style.borderBottom = '1px solid var(--border-subtle)';

            const cb = status.circuit_breaker || {};
            const rl = status.rate_limit || {};

            const stateColor = cb.state === 'closed' ? '#4ade80' : (cb.state === 'half_open' ? '#facc15' : '#f87171');
            const headroomColor = rl.has_headroom ? '#4ade80' : '#f87171';

            const reqRem = rl.req_remaining !== null && rl.req_remaining !== undefined ? rl.req_remaining : '∞';
            const tokRem = rl.tok_remaining !== null && rl.tok_remaining !== undefined ? rl.tok_remaining : '∞';

            let stateDisplay = `● ${cb.state ? cb.state.toUpperCase() : 'CLOSED'}`;
            if (cb.state === 'open') {
                if (cb.recovery_remaining_s !== null && cb.recovery_remaining_s !== undefined) {
                    stateDisplay += ` (${cb.recovery_remaining_s}s remaining, reopens at ${cb.reopens_at_wall || ''})`;
                } else if (cb.opened_at_wall) {
                    stateDisplay += ` (Tripped at ${cb.opened_at_wall})`;
                }
            } else if (cb.state === 'half_open') {
                stateDisplay += ' (Probe Ready)';
            }

            const causeText = cb.last_failure_reason || 'None (Operational)';

            tr.innerHTML = `
                <td style="padding: 0.75rem 1rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">${modelId}</td>
                <td style="padding: 0.75rem 1rem;"><span style="color: ${stateColor}; font-weight: 700; font-size: 0.7rem;">${stateDisplay}</span></td>
                <td style="padding: 0.75rem 1rem; color: ${cb.state === 'open' ? '#f87171' : 'var(--text-muted)'}; font-size: 0.72rem;">${causeText}</td>
                <td style="padding: 0.75rem 1rem; color: var(--text-muted);">${cb.failure_count || 0}</td>
                <td style="padding: 0.75rem 1rem;"><span style="color: ${headroomColor}; font-weight: 700;">${rl.has_headroom ? 'YES (≥10%)' : 'NO (LIMITED)'}</span></td>
                <td style="padding: 0.75rem 1rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace;">${reqRem} req / ${tokRem} tok</td>
            `;
            tbody.appendChild(tr);
        }
    } catch (e) {
        console.error("Failed to fetch router status:", e);
    }
}

// DEV MODE: JSON Payload Inspector & Beautifier
let activeModalPayload = null;
let currentModalTab = 'request';

function syntaxHighlightJson(json) {
    if (json === undefined || json === null) return '<span class="json-null">null (empty payload)</span>';
    let str = typeof json !== 'string' ? JSON.stringify(json, null, 2) : json;
    try {
        if (typeof json === 'string') {
            const parsed = JSON.parse(json);
            str = JSON.stringify(parsed, null, 2);
        }
    } catch (e) {
        // preserve unparseable raw string
    }

    str = str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return str.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
        let cls = 'json-number';
        if (/^"/.test(match)) {
            if (/:$/.test(match)) {
                cls = 'json-key';
            } else {
                cls = 'json-string';
            }
        } else if (/true|false/.test(match)) {
            cls = 'json-boolean';
        } else if (/null/.test(match)) {
            cls = 'json-null';
        }
        return '<span class="' + cls + '">' + match + '</span>';
    });
}

class PayloadDataTable {
    constructor(tableBodyId) {
        this.tbodyId = tableBodyId;
        this.rawPayloads = [];
        this.filterQuery = '';
        this.cache = new Map();
    }

    setPayloads(payloads) {
        this.rawPayloads = payloads || [];
        this.cache.clear();
        this.rawPayloads.forEach(p => {
            if (p.id) this.cache.set(p.id, p);
        });
        this.render();
    }

    getPayload(reqId) {
        return this.cache.get(reqId) || null;
    }

    setFilter(query) {
        this.filterQuery = (query || '').toLowerCase().trim();
        this.render();
    }

    getFilteredData() {
        if (!this.filterQuery) return this.rawPayloads;
        return this.rawPayloads.filter(req => {
            const clientModel = (req.client_model || '').toLowerCase();
            const mappedModel = (req.mapped_model || '').toLowerCase();
            const path = (req.path || '').toLowerCase();
            const method = (req.method || '').toLowerCase();
            const reqBody = typeof req.request_body === 'string' ? req.request_body.toLowerCase() : JSON.stringify(req.request_body || {}).toLowerCase();
            const respBody = typeof req.response_body === 'string' ? req.response_body.toLowerCase() : JSON.stringify(req.response_body || {}).toLowerCase();

            return clientModel.includes(this.filterQuery) ||
                   mappedModel.includes(this.filterQuery) ||
                   path.includes(this.filterQuery) ||
                   method.includes(this.filterQuery) ||
                   reqBody.includes(this.filterQuery) ||
                   respBody.includes(this.filterQuery);
        });
    }

    render() {
        const tbody = document.getElementById(this.tbodyId);
        if (!tbody) return;
        const filtered = this.getFilteredData();
        tbody.innerHTML = '';

        if (filtered.length === 0) {
            tbody.innerHTML = `<tr><td colspan="7" style="padding: 1.5rem; text-align: center; color: var(--text-muted);">
                ${this.rawPayloads.length === 0 ? 'No Claude Code requests captured yet. Send a request to see JSON payloads here.' : 'No payloads match your search query.'}
            </td></tr>`;
            return;
        }

        filtered.forEach(req => {
            const tr = document.createElement('tr');
            tr.style.borderBottom = '1px solid var(--border-subtle)';

            const statusClass = req.status_code === 200 ? 'color: #4ade80;' : 'color: #f87171;';
            const inTok = req.input_tokens || 0;
            const outTok = req.output_tokens || 0;
            const tokBadge = `<span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #60a5fa;">📥 ${inTok.toLocaleString()} / 📤 ${outTok.toLocaleString()} tok</span>`;

            tr.innerHTML = `
                <td style="padding: 0.75rem 1rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace;">${req.timestamp}</td>
                <td style="padding: 0.75rem 1rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">${req.method} ${req.path}</td>
                <td style="padding: 0.75rem 1rem; color: white;">${req.client_model || '-'}</td>
                <td style="padding: 0.75rem 1rem; color: #fef08a; font-family: 'JetBrains Mono', monospace;">${req.mapped_model || '-'}</td>
                <td style="padding: 0.75rem 1rem;">${tokBadge}</td>
                <td style="padding: 0.75rem 1rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace;">${req.duration_ms} ms</td>
                <td style="padding: 0.75rem 1rem; ${statusClass} font-weight: 700;">${req.status_code}</td>
                <td style="padding: 0.75rem 1rem; text-align: right;">
                    <button type="button" class="btn-alt" style="padding: 0.35rem 0.75rem; font-size: 0.7rem; border-color: rgba(234, 179, 8, 0.5); color: #fef08a;" onclick="openJsonModal('${req.id}')">Inspect { JSON }</button>
                </td>
            `;
            tbody.appendChild(tr);
        });
    }
}

const devDataTable = new PayloadDataTable('dev-payloads-table-body');

function onDevSearchInput(val) {
    devDataTable.setFilter(val);
}

async function fetchDevPayloads() {
    try {
        const resp = await fetch('/api/dev/payloads');
        const data = await resp.json();
        if (data && data.payloads) {
            devDataTable.setPayloads(data.payloads);
        }
    } catch (e) {
        console.error("Failed to fetch dev payloads:", e);
    }
}

let isSplitView = true;

async function openJsonModal(reqId) {
    try {
        let payload = devDataTable.getPayload(reqId);
        if (!payload) {
            const resp = await fetch(`/api/dev/payloads/${reqId}`);
            payload = await resp.json();
        }
        activeModalPayload = payload;
        
        document.getElementById('modal-meta-client').innerText = payload.client_model || payload.path;
        document.getElementById('modal-meta-mapped').innerText = payload.mapped_model || '-';

        const inT = (payload.input_tokens || 0).toLocaleString();
        const outT = (payload.output_tokens || 0).toLocaleString();
        const totalT = ((payload.input_tokens || 0) + (payload.output_tokens || 0)).toLocaleString();

        document.getElementById('modal-meta-info').innerText = `${payload.method} ${payload.path} | Status: ${payload.status_code} | Latency: ${payload.duration_ms} ms | 📥 In: ${inT} | 📤 Out: ${outT} (Total: ${totalT} tok)`;


        renderModalSideBySide();

        const modal = document.getElementById('json-inspector-modal');
        if (modal) modal.classList.add('show');
    } catch (e) {
        showToast("Failed to load payload details.", "error");
    }
}

function renderModalSideBySide() {
    if (!activeModalPayload) return;
    const reqBox = document.getElementById('modal-json-request');
    const respBox = document.getElementById('modal-json-response');
    if (reqBox) reqBox.innerHTML = syntaxHighlightJson(activeModalPayload.request_body);
    if (respBox) respBox.innerHTML = syntaxHighlightJson(activeModalPayload.response_body);
}

function toggleModalLayout() {
    const grid = document.getElementById('modal-body-grid');
    if (!grid) return;
    isSplitView = !isSplitView;
    if (isSplitView) {
        grid.style.gridTemplateColumns = '1fr 1fr';
        showToast("Layout set to Side-by-Side Dual View", "success");
    } else {
        grid.style.gridTemplateColumns = '1fr';
        showToast("Layout set to Single Column Stacked View", "success");
    }
}

function copyRequestJson() {
    if (!activeModalPayload) return;
    const obj = activeModalPayload.request_body;
    const str = typeof obj === 'string' ? obj : JSON.stringify(obj, null, 2);
    navigator.clipboard.writeText(str).then(() => {
        showToast("Request JSON copied to clipboard!", "success");
    });
}

function copyResponseJson() {
    if (!activeModalPayload) return;
    const obj = activeModalPayload.response_body;
    const str = typeof obj === 'string' ? obj : JSON.stringify(obj, null, 2);
    navigator.clipboard.writeText(str).then(() => {
        showToast("Response JSON copied to clipboard!", "success");
    });
}

function copyAllModalJson() {
    if (!activeModalPayload) return;
    const fullTransaction = {
        meta: {
            id: activeModalPayload.id,
            timestamp: activeModalPayload.timestamp,
            method: activeModalPayload.method,
            path: activeModalPayload.path,
            client_model: activeModalPayload.client_model,
            mapped_model: activeModalPayload.mapped_model,
            status_code: activeModalPayload.status_code,
            duration_ms: activeModalPayload.duration_ms,
        },
        request: activeModalPayload.request_body,
        response: activeModalPayload.response_body,
    };
    navigator.clipboard.writeText(JSON.stringify(fullTransaction, null, 2)).then(() => {
        showToast("Full Request & Response Transaction JSON copied!", "success");
    });
}

function closeJsonModal(event) {
    if (event.target.id === 'json-inspector-modal') {
        closeJsonModalDirect();
    }
}

function closeJsonModalDirect() {
    const modal = document.getElementById('json-inspector-modal');
    if (modal) modal.classList.remove('show');
}

function toggleAccordion() {
    const content = document.getElementById('accordionContent');
    const arrow = document.getElementById('accordionArrow');
    const isOpen = content.classList.toggle('open');
    arrow.innerText = isOpen ? '▼' : '▶';
}

function toggleMask(inputId) {
    const el = document.getElementById(inputId);
    const wrapper = el.closest('.input-wrapper');
    const btn = wrapper.querySelector('.input-icon-btn');
    if (el.type === 'password') {
        el.type = 'text';
        btn.innerText = 'Hide';
    } else {
        el.type = 'password';
        btn.innerText = 'Show';
    }
}

async function runDiagnostics() {
    showToast("Running system diagnostics...", "success");
    const consoleEl = document.getElementById('terminalConsole');
    consoleEl.innerHTML = '<div class="terminal-line">> Initializing diagnostic check...</div>';
    try {
        const resp = await fetch('/api/diagnostics');
        const data = await resp.json();

        consoleEl.innerHTML = '';

        let i = 0;
        function printLine() {
            if (i < data.logs.length) {
                const div = document.createElement('div');
                div.className = 'terminal-line';
                div.innerText = data.logs[i];
                consoleEl.appendChild(div);
                consoleEl.scrollTop = consoleEl.scrollHeight;
                i++;
                setTimeout(printLine, 150);
            } else {
                const finalDiv = document.createElement('div');
                finalDiv.className = 'terminal-line';
                if (data.has_errors) {
                    finalDiv.style.color = '#f87171';
                    finalDiv.innerText = 'DIAGNOSTICS COMPLETE - ADVISORIES FOUND.';
                    showToast("Diagnostics completed with advisories.", "error");
                } else {
                    finalDiv.style.color = '#ffffff';
                    finalDiv.innerText = 'ALL SYSTEMS FUNCTIONAL. READY TO PROCESS.';
                    showToast("System diagnostics completed.", "success");
                }
                consoleEl.appendChild(finalDiv);
                consoleEl.scrollTop = consoleEl.scrollHeight;
            }
        }
        printLine();
    } catch (e) {
        consoleEl.innerHTML += '<div class="terminal-line" style="color: #f87171">> System connection failed: ' + e + '</div>';
        showToast("Diagnostics check failed.", "error");
    }
}

// Live Telemetry statistics poller
async function pollTelemetry() {
    try {
        const resp = await fetch('/api/stats');
        const data = await resp.json();

        // Update request counters
        document.getElementById('telemetry-requests').innerHTML = `${data.total_requests} <span>reqs</span>`;

        if (data.total_requests > 0) {
            const ratio = Math.round((data.mocked_requests / data.total_requests) * 100);
            document.getElementById('telemetry-savings').innerText = `${ratio}%`;
        } else {
            document.getElementById('telemetry-savings').innerText = `0%`;
        }
        document.getElementById('telemetry-mock-badge').innerText = `${data.mocked_requests} / ${data.total_requests} SAVES`;

        // Update Bot connectivity
        const dsEl = document.getElementById('telemetry-discord');
        if (dsEl) {
            dsEl.innerText = data.ds_bot_status;
            dsEl.style.color = data.ds_bot_status === 'Online' ? '#ffffff' : 'var(--text-dim)';
        }

        const tgEl = document.getElementById('telemetry-telegram');
        if (tgEl) {
            tgEl.innerText = data.tg_bot_status;
            tgEl.style.color = data.tg_bot_status === 'Online' ? '#ffffff' : 'var(--text-dim)';
        }

        // Update local services online health badges
        for (const [service, status] of Object.entries(data.endpoints)) {
            const statusBadge = document.getElementById('status-' + service);
            if (statusBadge) {
                statusBadge.innerText = status;
                statusBadge.style.color = status === 'Online' ? '#ffffff' : 'var(--text-dim)';
                statusBadge.style.borderColor = status === 'Online' ? '#ffffff' : 'var(--border-subtle)';
            }
        }

        // Render Live Request Trace log feed
        const feedBody = document.getElementById('live-request-feed-body');
        if (feedBody) {
            if (data.recent_requests && data.recent_requests.length > 0) {
                feedBody.innerHTML = '';
                // Display in reverse order (newest first)
                const sorted = [...data.recent_requests].reverse();
                sorted.forEach(req => {
                    const tr = document.createElement('tr');
                    
                    const tdTime = document.createElement('td');
                    tdTime.innerText = req.timestamp;
                    tr.appendChild(tdTime);

                    const tdPath = document.createElement('td');
                    tdPath.innerText = `${req.method} ${req.path}`;
                    tr.appendChild(tdPath);

                    const tdClient = document.createElement('td');
                    tdClient.innerText = req.client_model;
                    tr.appendChild(tdClient);

                    const tdUpstream = document.createElement('td');
                    tdUpstream.innerText = req.mapped_model || req.target_model || '-';
                    tr.appendChild(tdUpstream);

                    const tdLatency = document.createElement('td');
                    const dur = req.duration_ms !== undefined ? req.duration_ms : (req.latency_ms || 0);
                    tdLatency.innerText = `${dur} ms`;
                    tr.appendChild(tdLatency);

                    const tdStatus = document.createElement('td');
                    const spanStatus = document.createElement('span');
                    spanStatus.className = 'feed-badge ' + (req.status_code === 200 ? 'status-200' : 'status-error');
                    spanStatus.innerText = req.status_code;
                    tdStatus.appendChild(spanStatus);

                    if (req.mocked) {
                        const spanMock = document.createElement('span');
                        spanMock.className = 'feed-badge mock';
                        spanMock.style.marginLeft = '6px';
                        spanMock.innerText = 'MOCK';
                        tdStatus.appendChild(spanMock);
                    }

                    if (req.fallbacks_used && req.fallbacks_used.length > 0) {
                        const spanFb = document.createElement('span');
                        spanFb.className = 'feed-badge mock';
                        spanFb.style.marginLeft = '6px';
                        spanFb.style.background = 'rgba(234, 179, 8, 0.2)';
                        spanFb.style.color = '#fef08a';
                        spanFb.innerText = `FALLBACK (${req.fallbacks_used.length})`;
                        tdStatus.appendChild(spanFb);
                    }

                    if (req.id) {
                        const btnInspect = document.createElement('button');
                        btnInspect.type = 'button';
                        btnInspect.className = 'btn-alt';
                        btnInspect.style.marginLeft = '8px';
                        btnInspect.style.padding = '2px 6px';
                        btnInspect.style.fontSize = '0.65rem';
                        btnInspect.style.borderColor = 'rgba(234, 179, 8, 0.4)';
                        btnInspect.style.color = '#fef08a';
                        btnInspect.innerText = '{ JSON }';
                        btnInspect.onclick = () => openJsonModal(req.id);
                        tdStatus.appendChild(btnInspect);
                    }

                    tr.appendChild(tdStatus);
                    feedBody.appendChild(tr);
                });
            } else {
                feedBody.innerHTML = `<tr>
                    <td colspan="6" style="text-align: center; color: var(--text-dim); padding: 20px;">No operational traffic logged. Awaiting API events...</td>
                </tr>`;
            }
        }

        // Automatically refresh Dev Mode payloads on every telemetry poll
        await fetchDevPayloads();

    } catch (e) {
        console.error("Telemetry link lost", e);
    }
}

let activeSuggestionIndex = -1;
let currentVisibleItems = [];

async function refreshModelsList() {
    showToast("🔄 Refreshing available models list...", "success");
    await fetchModels();
    showToast("✅ Models list refreshed successfully.", "success");
}

// Search Autocomplete Suggestion Logic
function handleAutocomplete(inputId) {
    const inputEl = document.getElementById(inputId);
    const dropdownEl = document.getElementById(inputId + '-dropdown');
    if (!inputEl || !dropdownEl) return;
    
    dropdownEl.innerHTML = '';
    let count = 0;
    currentVisibleItems = [];
    activeSuggestionIndex = -1;

    for (const [provider, list] of Object.entries(modelsSnapshot)) {
        if (list.length === 0) continue;

        const textVal = inputEl.value.toLowerCase();
        const providerPrefix = provider + '/';
        const filtered = list.filter(m => {
            const full = providerPrefix + m;
            return !textVal || full.toLowerCase().includes(textVal);
        });

        if (filtered.length > 0) {
            const header = document.createElement('div');
            header.className = 'autocomplete-group';
            header.innerText = provider.replace('_', ' ');
            dropdownEl.appendChild(header);

            filtered.forEach(m => {
                count++;
                const fullStr = providerPrefix + m;
                const item = document.createElement('div');
                item.className = 'autocomplete-item';
                item.innerText = fullStr;
                item.onmousedown = () => {
                    selectSuggestion(inputId, fullStr);
                };
                dropdownEl.appendChild(item);
                currentVisibleItems.push(item);
            });
        }
    }

    if (count > 0) {
        dropdownEl.style.display = 'block';
    } else {
        dropdownEl.style.display = 'none';
    }
}

// Tag State Management
const tagState = {
    FALLBACK_ORDER_CLAUDE_DEFAULT: [],
    FALLBACK_ORDER_CLAUDE_OPUS: [],
    FALLBACK_ORDER_CLAUDE_SONNET: [],
    FALLBACK_ORDER_CLAUDE_SONNET_1M: [],
    FALLBACK_ORDER_CLAUDE_HAIKU: [],
};


function renderTags(inputId) {
    const container = document.getElementById('tag-container-' + inputId);
    if (!container) return;

    // Clear existing chips
    container.querySelectorAll('.tag-chip').forEach(chip => chip.remove());

    const tags = tagState[inputId] || [];
    const inputEl = document.getElementById(inputId);

    tags.forEach((tag, idx) => {
        const chip = document.createElement('div');
        chip.className = 'tag-chip';
        
        const spanText = document.createElement('span');
        spanText.innerText = tag;
        chip.appendChild(spanText);

        const removeBtn = document.createElement('span');
        removeBtn.className = 'tag-remove';
        removeBtn.innerHTML = '&times;';
        removeBtn.onclick = (e) => {
            e.stopPropagation();
            removeTag(inputId, idx);
        };
        chip.appendChild(removeBtn);

        container.insertBefore(chip, inputEl);
    });
}

function addTag(inputId, val) {
    const trimmed = val.trim().replace(/,/g, '');
    if (!trimmed) return;

    if (!tagState[inputId]) tagState[inputId] = [];
    if (!tagState[inputId].includes(trimmed)) {
        tagState[inputId].push(trimmed);
        renderTags(inputId);
    }
    const inputEl = document.getElementById(inputId);
    if (inputEl) inputEl.value = '';
}

function removeTag(inputId, index) {
    if (tagState[inputId]) {
        tagState[inputId].splice(index, 1);
        renderTags(inputId);
    }
}

function focusTagInput(inputId) {
    const inputEl = document.getElementById(inputId);
    if (inputEl) inputEl.focus();
}

function closeAllAutocompleteDropdowns() {
    document.querySelectorAll('.autocomplete-dropdown').forEach(dropdown => {
        dropdown.style.display = 'none';
    });
    activeSuggestionIndex = -1;
    currentVisibleItems = [];
}

function handleTagInputKeyDown(event, inputId) {
    const inputEl = document.getElementById(inputId);
    const dropdownEl = document.getElementById(inputId + '-dropdown');

    if (event.key === 'Escape') {
        event.preventDefault();
        closeAllAutocompleteDropdowns();
        return;
    }

    // If dropdown active with selection
    if (dropdownEl && dropdownEl.style.display !== 'none' && currentVisibleItems.length > 0) {
        if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
            handleInputKeyDown(event, inputId);
            return;
        }
        if (event.key === 'Enter') {
            if (activeSuggestionIndex >= 0 && activeSuggestionIndex < currentVisibleItems.length) {
                event.preventDefault();
                const selectedText = currentVisibleItems[activeSuggestionIndex].innerText;
                selectSuggestion(inputId, selectedText);
                return;
            }
        }
    }

    if (event.key === 'Enter' || event.key === ',') {
        event.preventDefault();
        addTag(inputId, inputEl.value);
        if (dropdownEl) dropdownEl.style.display = 'none';
    } else if (event.key === 'Backspace' && !inputEl.value) {
        if (tagState[inputId] && tagState[inputId].length > 0) {
            removeTag(inputId, tagState[inputId].length - 1);
        }
    }
}

function selectSuggestion(inputId, value) {
    if (inputId.startsWith('FALLBACK_ORDER_')) {
        addTag(inputId, value);
    } else {
        const inputEl = document.getElementById(inputId);
        if (inputEl) inputEl.value = value;
    }
    closeAllAutocompleteDropdowns();
}

// Keyboard navigation in autocomplete
function handleInputKeyDown(event, inputId) {
    const dropdownEl = document.getElementById(inputId + '-dropdown');
    if (!dropdownEl || dropdownEl.style.display === 'none' || currentVisibleItems.length === 0) return;

    if (event.key === 'ArrowDown') {
        event.preventDefault();
        activeSuggestionIndex = (activeSuggestionIndex + 1) % currentVisibleItems.length;
        updateSuggestionHighlight();
    } else if (event.key === 'ArrowUp') {
        event.preventDefault();
        activeSuggestionIndex = (activeSuggestionIndex - 1 + currentVisibleItems.length) % currentVisibleItems.length;
        updateSuggestionHighlight();
    } else if (event.key === 'Enter') {
        if (activeSuggestionIndex >= 0 && activeSuggestionIndex < currentVisibleItems.length) {
            event.preventDefault();
            selectSuggestion(inputId, currentVisibleItems[activeSuggestionIndex].innerText);
        }
    }
}

function updateSuggestionHighlight() {
    currentVisibleItems.forEach((item, index) => {
        if (index === activeSuggestionIndex) {
            item.classList.add('highlighted');
            item.scrollIntoView({ block: 'nearest' });
        } else {
            item.classList.remove('highlighted');
        }
    });
}

// Close dropdowns when clicking outside
document.addEventListener('click', (event) => {
    const isClickInsideGroup = event.target.closest('.form-group');
    if (!isClickInsideGroup) {
        closeAllAutocompleteDropdowns();
    }
});

// Wire event listeners after DOM load
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('input[type="text"]').forEach(input => {
        if (['MODEL_OPUS', 'MODEL_SONNET', 'MODEL_HAIKU', 'MODEL'].includes(input.id)) {
            input.addEventListener('input', () => handleAutocomplete(input.id));
            input.addEventListener('keydown', (e) => handleInputKeyDown(e, input.id));
        }
    });

    document.querySelectorAll('.control-btn[data-pane]').forEach(btn => {
        btn.addEventListener('click', function(e) {
            e.preventDefault();
            const paneId = this.getAttribute('data-pane');
            switchPane(paneId, this);
        });
    });

    const presetSelect = document.getElementById('provider-preset-select');
    if (presetSelect) {
        presetSelect.addEventListener('change', function() {
            const preset = PROVIDER_PRESETS[this.value];
            const descEl = document.getElementById('preset-desc');
            if (descEl) descEl.innerText = preset ? preset.label : '';
        });
    }

    // Initial load calls
    fetchModels();
    loadConfigs();
    pollTelemetry();
    setInterval(pollTelemetry, 3000);
});

// Provider default preset configs
const PROVIDER_PRESETS = {
    nvidia_nim: {
        label: 'NVIDIA NIM — high-throughput, optimised for long generation timeouts. Rate: 60 req/min, Concurrency: 10, Read: 300s.',
        PROVIDER_RATE_LIMIT: 60,
        PROVIDER_RATE_WINDOW: 60,
        PROVIDER_MAX_CONCURRENCY: 10,
        HTTP_READ_TIMEOUT: 300,
        HTTP_WRITE_TIMEOUT: 30,
        HTTP_CONNECT_TIMEOUT: 5,
    },
    openrouter: {
        label: 'OpenRouter — balanced defaults for free-tier cloud routing. Rate: 40 req/min, Concurrency: 5, Read: 180s.',
        PROVIDER_RATE_LIMIT: 40,
        PROVIDER_RATE_WINDOW: 60,
        PROVIDER_MAX_CONCURRENCY: 5,
        HTTP_READ_TIMEOUT: 180,
        HTTP_WRITE_TIMEOUT: 20,
        HTTP_CONNECT_TIMEOUT: 5,
    },
    groq: {
        label: 'Groq — ultra-fast inference, conservative rate limits. Rate: 30 req/min, Concurrency: 4, Read: 60s.',
        PROVIDER_RATE_LIMIT: 30,
        PROVIDER_RATE_WINDOW: 60,
        PROVIDER_MAX_CONCURRENCY: 4,
        HTTP_READ_TIMEOUT: 60,
        HTTP_WRITE_TIMEOUT: 10,
        HTTP_CONNECT_TIMEOUT: 5,
    },
    deepseek: {
        label: 'DeepSeek — moderate throughput with generous timeouts. Rate: 50 req/min, Concurrency: 6, Read: 180s.',
        PROVIDER_RATE_LIMIT: 50,
        PROVIDER_RATE_WINDOW: 60,
        PROVIDER_MAX_CONCURRENCY: 6,
        HTTP_READ_TIMEOUT: 180,
        HTTP_WRITE_TIMEOUT: 20,
        HTTP_CONNECT_TIMEOUT: 5,
    },
    mistral: {
        label: 'Mistral — standard cloud inference defaults. Rate: 40 req/min, Concurrency: 5, Read: 120s.',
        PROVIDER_RATE_LIMIT: 40,
        PROVIDER_RATE_WINDOW: 60,
        PROVIDER_MAX_CONCURRENCY: 5,
        HTTP_READ_TIMEOUT: 120,
        HTTP_WRITE_TIMEOUT: 15,
        HTTP_CONNECT_TIMEOUT: 5,
    },
    cerebras: {
        label: 'Cerebras — wafer-scale fast inference. Rate: 60 req/min, Concurrency: 8, Read: 60s.',
        PROVIDER_RATE_LIMIT: 60,
        PROVIDER_RATE_WINDOW: 60,
        PROVIDER_MAX_CONCURRENCY: 8,
        HTTP_READ_TIMEOUT: 60,
        HTTP_WRITE_TIMEOUT: 10,
        HTTP_CONNECT_TIMEOUT: 5,
    },
};

function applySelectedPreset() {
    const sel = document.getElementById('provider-preset-select');
    const providerKey = sel ? sel.value : '';
    if (!providerKey) {
        showToast('Select a provider from the dropdown first.', 'error');
        return;
    }
    applyProviderPreset(providerKey);
}

function applyProviderPreset(providerKey) {
    const preset = PROVIDER_PRESETS[providerKey];
    if (!preset) return;
    const fields = ['PROVIDER_RATE_LIMIT', 'PROVIDER_RATE_WINDOW', 'PROVIDER_MAX_CONCURRENCY',
                    'HTTP_READ_TIMEOUT', 'HTTP_WRITE_TIMEOUT', 'HTTP_CONNECT_TIMEOUT'];
    fields.forEach(field => {
        const el = document.getElementById(field);
        if (el) el.value = preset[field];
    });
    const descEl = document.getElementById('preset-desc');
    if (descEl) descEl.innerText = preset.label;
    showToast('Preset applied. Click Commit to save.', 'success');
}

// Explicitly bind all event handlers to window object for HTML inline access
window.switchPane = switchPane;
window.fetchModels = fetchModels;
window.refreshModelsList = refreshModelsList;
window.loadConfigs = loadConfigs;
window.revertConfigs = revertConfigs;
window.saveConfigs = saveConfigs;
window.fetchRouterStatus = fetchRouterStatus;
window.fetchDevPayloads = fetchDevPayloads;
window.openJsonModal = openJsonModal;
window.closeJsonModal = closeJsonModal;
window.closeJsonModalDirect = closeJsonModalDirect;
window.toggleModalLayout = toggleModalLayout;
window.copyRequestJson = copyRequestJson;
window.copyResponseJson = copyResponseJson;
window.copyAllModalJson = copyAllModalJson;
window.onDevSearchInput = onDevSearchInput;
window.toggleMask = toggleMask;
window.applySelectedPreset = applySelectedPreset;
