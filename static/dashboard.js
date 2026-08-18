let configSnapshot = {};
let modelsSnapshot = {};
let lockStatuses = {};
let liveRefreshTimer = null;
let currentRefreshIntervalSeconds = 4;

function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}


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

        startLiveAutoRefresh();
        await fetchSubagentEmergencyStatus();
    } catch (e) {
        console.error("Failed to load configs", e);
    }
}

function switchOsTab(osId) {
    try { localStorage.setItem('active_os_tab', osId); } catch(e){}
    document.querySelectorAll('.os-tab-btn').forEach(btn => {
        btn.classList.remove('active');
        btn.style.borderColor = 'var(--border-subtle)';
        btn.style.background = 'transparent';
        btn.style.color = 'var(--text-muted)';
        btn.style.fontWeight = 'normal';
    });
    document.querySelectorAll('.os-guide-card').forEach(card => {
        card.style.display = 'none';
    });

    const activeBtn = document.getElementById('os-btn-' + osId);
    const activeCard = document.getElementById('os-card-' + osId);

    if (activeBtn) {
        activeBtn.classList.add('active');
        activeBtn.style.borderColor = '#eab308';
        activeBtn.style.background = 'rgba(234, 179, 8, 0.15)';
        activeBtn.style.color = '#ffffff';
        activeBtn.style.fontWeight = '700';
    }
    if (activeCard) {
        activeCard.style.display = 'block';
    }
}

function startLiveAutoRefresh() {
    if (liveRefreshTimer) clearInterval(liveRefreshTimer);
    
    let intervalSec = 4;
    try {
        const savedRate = localStorage.getItem('dashboard_refresh_rate');
        if (savedRate !== null && !isNaN(parseInt(savedRate))) {
            intervalSec = parseInt(savedRate);
        } else if (configSnapshot["REFRESH_TIME"] !== undefined) {
            intervalSec = parseInt(configSnapshot["REFRESH_TIME"]);
        }
    } catch(e) {
        if (configSnapshot["REFRESH_TIME"] !== undefined) {
            intervalSec = parseInt(configSnapshot["REFRESH_TIME"]);
        }
    }
    
    const headerSel = document.getElementById('header-refresh-select');
    if (headerSel) headerSel.value = String(intervalSec);
    const inputEl = document.getElementById('REFRESH_TIME');
    if (inputEl) inputEl.value = intervalSec;

    if (isNaN(intervalSec) || intervalSec <= 0) {
        currentRefreshIntervalSeconds = 0;
        return;
    }

    currentRefreshIntervalSeconds = intervalSec;
    const ms = currentRefreshIntervalSeconds * 1000;

    pollTelemetry();
    fetchRouterStatus();

    liveRefreshTimer = setInterval(() => {
        pollTelemetry();
        fetchRouterStatus();
    }, ms);
}

function onHeaderRefreshRateChange(val) {
    const sec = parseInt(val);
    configSnapshot["REFRESH_TIME"] = sec;
    try {
        localStorage.setItem('dashboard_refresh_rate', sec);
    } catch(e) {}
    const inputEl = document.getElementById('REFRESH_TIME');
    if (inputEl) inputEl.value = sec;
    
    startLiveAutoRefresh();
    if (sec > 0) {
        showToast(`Live auto-refresh rate set to ${sec}s`, 'success');
    } else {
        showToast('Live auto-refresh paused', 'warning');
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


    // Save Subagent Emergency Switch status if present
    const subagentEl = document.getElementById('SUBAGENTS_ENABLED');
    if (subagentEl) {
        try {
            await fetch('/api/settings/subagents', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enabled: subagentEl.checked })
            });
            subagentsEnabledState = subagentEl.checked;
        } catch (e) {
            console.error("Failed to save subagent switch", e);
        }
    }

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
    try { localStorage.setItem('active_pane', paneId); } catch(e){}

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

            const escapedModelId = modelId.replace(/'/g, "\\'").replace(/"/g, '&quot;');
            const escapedCause = causeText.replace(/'/g, "\\'").replace(/"/g, '&quot;');

            tr.innerHTML = `
                <td style="padding: 0.75rem 1rem; max-width: 220px;">
                    <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px; width: 100%;">
                        <span style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: #ffffff;" title="${escapedModelId}">${modelId}</span>
                        <button type="button" class="btn-copy-icon" onclick="copyTextToClipboard('${escapedModelId}', this)" title="Copy full Model Target ID">Copy</button>
                    </div>
                </td>
                <td style="padding: 0.75rem 1rem; min-width: 140px;"><span style="color: ${stateColor}; font-weight: 700; font-size: 0.7rem;">${stateDisplay}</span></td>
                <td style="padding: 0.75rem 1rem; max-width: 260px;">
                    <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px; width: 100%;">
                        <span style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: ${cb.state === 'open' ? '#f87171' : 'var(--text-muted)'}; font-size: 0.72rem;" title="${escapedCause}">${causeText}</span>
                        <button type="button" class="btn-copy-icon" onclick="copyTextToClipboard('${escapedCause}', this)" title="Copy full Failure Reason">Copy</button>
                    </div>
                </td>
                <td style="padding: 0.75rem 1rem; color: var(--text-muted); text-align: center;">${cb.failure_count || 0}</td>
                <td style="padding: 0.75rem 1rem; white-space: nowrap;"><span style="color: ${headroomColor}; font-weight: 700;">${rl.has_headroom ? 'YES (≥10%)' : 'NO (LIMITED)'}</span></td>
                <td style="padding: 0.75rem 1rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; white-space: nowrap;">${reqRem} req / ${tokRem} tok</td>
                <td style="padding: 0.5rem 1rem; text-align: right; width: 110px;">
                    <div style="display: flex; flex-direction: column; gap: 4px; align-items: flex-end;">
                        <button type="button" class="btn-alt" style="padding: 0.25rem 0.6rem; font-size: 0.68rem; border-color: rgba(34, 197, 94, 0.4); color: #4ade80; width: 95px; text-align: center;" onclick="handleCircuitAction('${modelId}', 'reset')" title="Clear Timeout & Open Traffic (Reset to CLOSED)">Open (Reset)</button>
                        <button type="button" class="btn-alt" style="padding: 0.25rem 0.6rem; font-size: 0.68rem; border-color: rgba(248, 113, 113, 0.4); color: #f87171; width: 95px; text-align: center;" onclick="handleCircuitAction('${modelId}', 'trip')" title="Block Model & Extend Timeout (1m -> 2m -> 5m -> 10m -> 30m -> 60m -> 120m -> 240m -> 480m -> 1440m)">Close (+Timeout)</button>
                    </div>
                </td>
            `;
            tbody.appendChild(tr);
        }
    } catch (e) {
        console.error("Failed to fetch router status:", e);
    }
}

function copyTextToClipboard(text, btnEl) {
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
        if (btnEl) {
            const origText = btnEl.innerText;
            btnEl.innerText = 'Copied!';
            btnEl.style.borderColor = '#4ade80';
            btnEl.style.color = '#4ade80';
            setTimeout(() => {
                btnEl.innerText = origText;
                btnEl.style.borderColor = 'var(--border-subtle)';
                btnEl.style.color = 'var(--text-muted)';
            }, 1200);
        }
        showToast("Copied to clipboard!", "success");
    }).catch(err => {
        showToast("Failed to copy", "error");
    });
}
window.copyTextToClipboard = copyTextToClipboard;

async function handleCircuitAction(modelId, action) {
    try {
        const resp = await fetch('/api/circuit-breaker/action', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: modelId, action: action })
        });
        const res = await resp.json();
        if (res.status === 'success') {
            showToast(res.message, "success");
            await fetchRouterStatus();
        } else {
            showToast("Circuit action failed: " + res.message, "error");
        }
    } catch (e) {
        showToast("Error updating circuit breaker state.", "error");
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
    constructor(tbodyId) {
        this.tbodyId = tbodyId;
        this.rawPayloads = [];
        this.filterQuery = '';
        this.cache = new Map();
        this.currentPage = 1;
        this.pageSize = 20;
        this.totalItems = 0;
        this.totalPages = 1;
    }

    setServerData(data) {
        this.rawPayloads = data.payloads || [];
        this.totalItems = data.total !== undefined ? data.total : this.rawPayloads.length;
        this.currentPage = data.page || 1;
        this.totalPages = data.total_pages || 1;
        this.pageSize = data.limit || this.pageSize;

        this.cache.clear();
        this.rawPayloads.forEach(p => {
            if (p.id) this.cache.set(p.id, p);
        });
        this.render();
    }

    getPayload(reqId) {
        return this.cache.get(reqId) || null;
    }

    setPageSize(size) {
        this.pageSize = size;
        this.currentPage = 1;
        fetchDevPayloads();
    }

    setFilter(query) {
        this.filterQuery = (query || '').trim();
        this.currentPage = 1;
        fetchDevPayloads();
    }

    changePage(delta) {
        const nextP = this.currentPage + delta;
        if (nextP >= 1 && nextP <= this.totalPages) {
            this.currentPage = nextP;
            fetchDevPayloads();
        }
    }

    render() {
        const tbody = document.getElementById(this.tbodyId);
        if (!tbody) return;

        tbody.innerHTML = '';

        if (this.rawPayloads.length === 0) {
            tbody.innerHTML = `<tr><td colspan="8" style="padding: 1.5rem; text-align: center; color: var(--text-muted);">
                ${this.filterQuery ? 'No payloads match your search query.' : 'No Claude Code requests captured yet. Send a request to see JSON payloads here.'}
            </td></tr>`;
            this.updatePaginationUI(0, 0, 0, 1, 1);
            return;
        }

        const startIdx = (this.currentPage - 1) * (this.pageSize === 999999 ? this.totalItems : this.pageSize);
        const endIdx = startIdx + this.rawPayloads.length;

        this.rawPayloads.forEach(req => {
            const tr = document.createElement('tr');
            tr.style.borderBottom = '1px solid var(--border-subtle)';

            const statusClass = req.status_code === 200 ? 'color: #4ade80;' : 'color: #f87171;';
            const inTok = req.input_tokens || 0;
            const outTok = req.output_tokens || 0;
            const tokBadge = `<span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #60a5fa;">${inTok.toLocaleString()} / ${outTok.toLocaleString()} tok</span>`;
            const escapedPath = `${req.method} ${req.path}`.replace(/"/g, '&quot;');
            const escapedClient = (req.client_model || '').replace(/"/g, '&quot;');
            const escapedMapped = (req.mapped_model || '').replace(/"/g, '&quot;');

            tr.innerHTML = `
                <td style="padding: 0.75rem 1rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; white-space: nowrap; user-select: text;">${req.timestamp}</td>
                <td style="padding: 0.75rem 1rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; word-break: break-word; max-width: 180px; user-select: text;" title="${escapedPath}">${req.method} ${req.path}</td>
                <td style="padding: 0.75rem 1rem; color: white; word-break: break-word; max-width: 160px; user-select: text;" title="${escapedClient}">${req.client_model || '-'}</td>
                <td style="padding: 0.75rem 1rem; color: #fef08a; font-family: 'JetBrains Mono', monospace; word-break: break-word; max-width: 200px; user-select: text;" title="${escapedMapped}">${req.mapped_model || '-'}</td>
                <td style="padding: 0.75rem 1rem; white-space: nowrap;">${tokBadge}</td>
                <td style="padding: 0.75rem 1rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; white-space: nowrap;">${req.duration_ms} ms</td>
                <td style="padding: 0.75rem 1rem; ${statusClass} font-weight: 700; white-space: nowrap;">${req.status_code}</td>
                <td style="padding: 0.75rem 1rem; text-align: right; white-space: nowrap;">
                    <button type="button" class="btn-alt" style="padding: 0.35rem 0.75rem; font-size: 0.7rem; border-color: #eab308; background: rgba(234, 179, 8, 0.12); color: #ffffff;" onclick="openJsonModal('${req.id}')">Inspect { JSON }</button>
                </td>
            `;
            tbody.appendChild(tr);
        });

        this.updatePaginationUI(startIdx + 1, endIdx, this.totalItems, this.currentPage, this.totalPages);
    }

    updatePaginationUI(from, to, total, page, maxPage) {
        const infoEl = document.getElementById('inspector-page-info');
        const numEl = document.getElementById('inspector-page-number');
        const btnPrev = document.getElementById('inspector-btn-prev');
        const btnNext = document.getElementById('inspector-btn-next');

        if (infoEl) infoEl.innerText = total > 0 ? `Showing ${from}-${to} of ${total} requests` : 'Showing 0-0 of 0 requests';
        if (numEl) numEl.innerText = `Page ${page} of ${maxPage}`;
        if (btnPrev) btnPrev.disabled = page <= 1;
        if (btnNext) btnNext.disabled = page >= maxPage;
    }
}

const devDataTable = new PayloadDataTable('dev-payloads-table-body');
let devSearchDebounceTimer = null;

function onDevSearchInput(val) {
    if (devSearchDebounceTimer) clearTimeout(devSearchDebounceTimer);
    devSearchDebounceTimer = setTimeout(() => {
        devDataTable.setFilter(val);
    }, 250);
}

function changeInspectorPage(delta) {
    devDataTable.changePage(delta);
}

function changeInspectorPageSize(val) {
    try { localStorage.setItem('inspector_page_size', val); } catch(e){}
    const size = val === 'all' ? 999999 : (parseInt(val) || 20);
    devDataTable.setPageSize(size);
}

function changeTracePageSize(val) {
    try { localStorage.setItem('trace_page_size', val); } catch(e){}
    tracePageSize = val === 'all' ? 999999 : (parseInt(val) || 20);
    traceCurrentPage = 1;
    renderTraceFeed();
}

function switchProviderTab(providerId) {
    try { localStorage.setItem('active_provider_tab', providerId); } catch(e){}
    document.querySelectorAll('.provider-tab-btn').forEach(btn => {
        btn.classList.remove('active');
        btn.style.borderColor = 'var(--border-subtle)';
        btn.style.background = 'transparent';
        btn.style.color = 'var(--text-muted)';
        btn.style.fontWeight = 'normal';
    });
    document.querySelectorAll('.provider-tab-card').forEach(card => {
        card.style.display = 'none';
    });

    const activeBtn = document.getElementById('ptab-btn-' + providerId);
    const activeCard = document.getElementById('ptab-card-' + providerId);

    if (activeBtn) {
        activeBtn.classList.add('active');
        activeBtn.style.borderColor = '#eab308';
        activeBtn.style.background = 'rgba(234, 179, 8, 0.15)';
        activeBtn.style.color = '#ffffff';
        activeBtn.style.fontWeight = '700';
    }
    if (activeCard) {
        activeCard.style.display = 'block';
    }
}

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

    const pUpper = providerKey.toUpperCase();
    const pFields = [
        ['RPM', preset.PROVIDER_RATE_LIMIT || 30],
        ['TPM', 200000],
        ['RPD', preset.PROVIDER_RPD || 1000],
        ['RATE_WINDOW', preset.PROVIDER_RATE_WINDOW || 60],
        ['MAX_CONCURRENCY', preset.PROVIDER_MAX_CONCURRENCY || 5],
        ['CONTEXT', 128000],
        ['MAX_OUTPUT', 16384],
        ['HTTP_READ_TIMEOUT', preset.HTTP_READ_TIMEOUT || 120],
        ['HTTP_WRITE_TIMEOUT', preset.HTTP_WRITE_TIMEOUT || 10],
        ['HTTP_CONNECT_TIMEOUT', preset.HTTP_CONNECT_TIMEOUT || 2],
    ];

    pFields.forEach(([fSuffix, defaultVal]) => {
        const el = document.getElementById(`PROVIDER_${pUpper}_${fSuffix}`);
        if (el && (!el.value || el.value === "0")) {
            el.value = defaultVal;
        }
    });

    switchProviderTab(providerKey);

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
window.handleAutocomplete = handleAutocomplete;
window.blurAutocomplete = blurAutocomplete;
window.selectSuggestion = selectSuggestion;
window.focusTagInput = focusTagInput;
window.handleTagInputKeyDown = handleTagInputKeyDown;
window.handleInputKeyDown = handleInputKeyDown;
window.switchDevSubTab = switchDevSubTab;
window.switchOsTab = switchOsTab;
window.handleCircuitAction = handleCircuitAction;
window.switchProviderTab = switchProviderTab;
window.startLiveAutoRefresh = startLiveAutoRefresh;
window.onHeaderRefreshRateChange = onHeaderRefreshRateChange;
window.changeInspectorPage = changeInspectorPage;
window.changeTracePage = changeTracePage;
window.changeInspectorPageSize = changeInspectorPageSize;
window.changeTracePageSize = changeTracePageSize;
window.fetchSubagentEmergencyStatus = fetchSubagentEmergencyStatus;
window.toggleSubagentEmergencySwitch = toggleSubagentEmergencySwitch;

document.addEventListener('DOMContentLoaded', () => {
    // Restore Saved UI Preferences from localStorage
    try {
        const savedInspSize = localStorage.getItem('inspector_page_size');
        if (savedInspSize) {
            const inspSel = document.getElementById('inspector-page-size');
            if (inspSel) inspSel.value = savedInspSize;
            devDataTable.pageSize = savedInspSize === 'all' ? 999999 : (parseInt(savedInspSize) || 20);
        }

        const savedTraceSize = localStorage.getItem('trace_page_size');
        if (savedTraceSize) {
            const traceSel = document.getElementById('trace-page-size');
            if (traceSel) traceSel.value = savedTraceSize;
            tracePageSize = savedTraceSize === 'all' ? 999999 : (parseInt(savedTraceSize) || 20);
        }

        const savedDevSubTab = localStorage.getItem('active_dev_subtab');
        if (savedDevSubTab) {
            switchDevSubTab(savedDevSubTab);
        }

        const savedOsTab = localStorage.getItem('active_os_tab');
        if (savedOsTab && document.getElementById('os-card-' + savedOsTab)) {
            switchOsTab(savedOsTab);
        }

        const savedProviderTab = localStorage.getItem('active_provider_tab');
        if (savedProviderTab && document.getElementById('ptab-card-' + savedProviderTab)) {
            switchProviderTab(savedProviderTab);
        }

        const savedPane = localStorage.getItem('active_pane');
        if (savedPane && document.getElementById(savedPane)) {
            switchPane(savedPane);
        }
    } catch(e) {
        console.error("Failed to restore preferences from localStorage:", e);
    }

    fetchModels();
    loadConfigs();
    fetchSubagentEmergencyStatus();
    startLiveAutoRefresh();
    fetchDevPayloads();
});

async function fetchDevPayloads() {
    try {
        const url = `/api/dev/payloads?limit=${devDataTable.pageSize}&page=${devDataTable.currentPage}&query=${encodeURIComponent(devDataTable.filterQuery)}`;
        const resp = await fetch(url);
        const data = await resp.json();
        if (data) {
            devDataTable.setServerData(data);
        }
    } catch (e) {
        console.error("Failed to fetch dev payloads:", e);
    }
}

function switchDevSubTab(tab) {
    try { localStorage.setItem('active_dev_subtab', tab); } catch(e){}
    const inspectorEl = document.getElementById('dev-subtab-inspector');
    const traceEl = document.getElementById('dev-subtab-trace');
    const metricsEl = document.getElementById('dev-subtab-metrics');
    const btnInspector = document.getElementById('dev-subtab-btn-inspector');
    const btnTrace = document.getElementById('dev-subtab-btn-trace');
    const btnMetrics = document.getElementById('dev-subtab-btn-metrics');
    if (!inspectorEl || !traceEl || !metricsEl) return;

    [inspectorEl, traceEl, metricsEl].forEach(el => el.style.display = 'none');
    [btnInspector, btnTrace, btnMetrics].forEach(btn => {
        if (!btn) return;
        btn.classList.remove('active');
        btn.style.borderColor = 'var(--border-subtle)';
        btn.style.background = 'transparent';
        btn.style.color = 'var(--text-muted)';
        btn.style.fontWeight = 'normal';
    });

    if (tab === 'trace') {
        traceEl.style.display = 'block';
        if (btnTrace) {
            btnTrace.classList.add('active');
            btnTrace.style.borderColor = '#eab308';
            btnTrace.style.background = 'rgba(234, 179, 8, 0.15)';
            btnTrace.style.color = '#ffffff';
            btnTrace.style.fontWeight = '700';
        }
    } else if (tab === 'metrics') {
        metricsEl.style.display = 'block';
        if (btnMetrics) {
            btnMetrics.classList.add('active');
            btnMetrics.style.borderColor = '#eab308';
            btnMetrics.style.background = 'rgba(234, 179, 8, 0.15)';
            btnMetrics.style.color = '#ffffff';
            btnMetrics.style.fontWeight = '700';
        }
        fetchDevMetrics();
    } else {
        inspectorEl.style.display = 'block';
        if (btnInspector) {
            btnInspector.classList.add('active');
            btnInspector.style.borderColor = '#eab308';
            btnInspector.style.background = 'rgba(234, 179, 8, 0.15)';
            btnInspector.style.color = '#ffffff';
            btnInspector.style.fontWeight = '700';
        }
    }
}

async function fetchDevMetrics() {
    try {
        const resp = await fetch('/api/dev/metrics');
        const data = await resp.json();
        if (!data) return;

        const gm = data.global_metrics || {};
        const elRpm = document.getElementById('metrics-global-rpm');
        if (elRpm) elRpm.innerHTML = `${gm.global_rpm || 0} <span style="font-size: 0.7rem; color: var(--text-dim);">req/min</span>`;

        const elConc = document.getElementById('metrics-active-concurrency');
        if (elConc) elConc.innerHTML = `${gm.active_concurrency || 0} <span style="font-size: 0.7rem; color: var(--text-dim);">workers</span>`;

        const el429 = document.getElementById('metrics-global-429');
        if (el429) {
            const total429 = gm.global_429_count || 0;
            const r60 = gm.global_429_60s || 0;
            if (total429 > 0) {
                el429.innerHTML = `<span style="color: #f87171;">${total429}</span> <span style="font-size: 0.7rem; color: var(--text-dim);">(last 60s: ${r60})</span>`;
            } else {
                el429.innerHTML = `<span style="color: #4ade80;">0</span> <span style="font-size: 0.7rem; color: var(--text-dim);">events</span>`;
            }
        }

        const elLat = document.getElementById('metrics-avg-latency');
        if (elLat) elLat.innerHTML = `${gm.global_avg_latency_ms || 0} <span style="font-size: 0.7rem; color: var(--text-dim);">ms</span>`;

        const nim = data.nim_telemetry || {};
        const elNimSum = document.getElementById('metrics-nim-summary');
        if (elNimSum) {
            elNimSum.innerHTML = `${nim.active_keys || 0}/${nim.total_keys || 0} Keys | <span style="font-size: 0.85rem; color: white;">${nim.current_rpm || 0}/${nim.rpm_limit || 38} RPM</span>`;
        }

        const elWaitBadge = document.getElementById('metrics-nim-wait-badge');
        if (elWaitBadge) {
            if ((nim.estimated_delay_s || 0) > 0) {
                elWaitBadge.innerText = `QUEUED DELAY (${nim.estimated_delay_s}s)`;
                elWaitBadge.style.color = '#f87171';
                elWaitBadge.style.background = 'rgba(248, 113, 113, 0.1)';
                elWaitBadge.style.borderColor = 'rgba(248, 113, 113, 0.3)';
            } else {
                elWaitBadge.innerText = 'NO DELAY (0.0s)';
                elWaitBadge.style.color = '#4ade80';
                elWaitBadge.style.background = 'rgba(74, 222, 128, 0.1)';
                elWaitBadge.style.borderColor = 'rgba(74, 222, 128, 0.3)';
            }
        }

        const elNimRpmText = document.getElementById('metrics-nim-rpm-text');
        if (elNimRpmText) elNimRpmText.innerText = `${nim.current_rpm || 0} / ${nim.rpm_limit || 38} RPM`;

        const elNimBar = document.getElementById('metrics-nim-progress-bar');
        if (elNimBar) {
            const pct = Math.min(100, Math.round(((nim.current_rpm || 0) / (nim.rpm_limit || 38)) * 100));
            elNimBar.style.width = `${pct}%`;
            elNimBar.style.background = pct > 85 ? '#f87171' : (pct > 60 ? '#facc15' : '#4ade80');
        }

        const elKeyList = document.getElementById('metrics-key-pool-list');
        if (elKeyList && Array.isArray(nim.key_details)) {
            if (nim.key_details.length === 0) {
                elKeyList.innerHTML = `<span style="font-size: 0.7rem; color: var(--text-dim);">No NIM keys loaded</span>`;
            } else {
                elKeyList.innerHTML = nim.key_details.map(k => {
                    const isPassive = k.cooldown_s > 0;
                    const col = isPassive ? '#f87171' : '#4ade80';
                    const bg = isPassive ? 'rgba(248,113,113,0.1)' : 'rgba(74,222,128,0.1)';
                    return `<span style="font-size: 0.68rem; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: ${col}; background: ${bg}; border: 1px solid ${col}44; padding: 2px 6px; border-radius: 4px;">🔑 ${escapeHtml(k.key_masked)} (${isPassive ? k.cooldown_s + 's cooldown' : 'Active'})</span>`;
                }).join('');
            }
        }

        const th = data.throttle_telemetry || {};
        const elThBadge = document.getElementById('metrics-throttle-active-badge');
        if (elThBadge) {
            const activeCount = th.active_sleep_count || 0;
            if (activeCount > 0) {
                elThBadge.innerText = `${activeCount} ACTIVE SLEEPING REQUESTS`;
                elThBadge.style.color = '#f87171';
                elThBadge.style.background = 'rgba(248, 113, 113, 0.15)';
                elThBadge.style.borderColor = 'rgba(248, 113, 113, 0.3)';
            } else {
                elThBadge.innerText = '0 ACTIVE SLEEPS';
                elThBadge.style.color = '#4ade80';
                elThBadge.style.background = 'rgba(74, 222, 128, 0.1)';
                elThBadge.style.borderColor = 'rgba(74, 222, 128, 0.3)';
            }
        }

        const elThTotalReqs = document.getElementById('throttle-total-reqs');
        if (elThTotalReqs) elThTotalReqs.innerText = (th.total_throttled_requests || 0).toLocaleString();

        const elThTotalSec = document.getElementById('throttle-total-seconds');
        if (elThTotalSec) elThTotalSec.innerText = `${(th.total_sleep_time_seconds || 0).toFixed(1)}s`;

        const elThMaxThresh = document.getElementById('throttle-max-threshold');
        if (elThMaxThresh) elThMaxThresh.innerText = `${(th.max_sleep_threshold || 3.0).toFixed(1)}s`;

        const elThLastEv = document.getElementById('throttle-last-event');
        if (elThLastEv) {
            if (th.last_sleep_event) {
                const le = th.last_sleep_event;
                elThLastEv.innerHTML = `<span style="color: #fbbf24;">${escapeHtml(le.timestamp || '')}</span> <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;">(${le.sleep_seconds}s)</span>`;
            } else {
                elThLastEv.innerText = 'None';
            }
        }

        const elThActiveContainer = document.getElementById('throttle-active-sleeps-container');
        if (elThActiveContainer) {
            const activeSleeps = th.active_sleeps || [];
            if (activeSleeps.length === 0) {
                elThActiveContainer.innerHTML = `<div style="color: var(--text-dim); font-size: 0.72rem;">No requests currently throttled or sleeping. Fast fallback threshold is active at <strong style="color: #34d399;">${th.max_sleep_threshold || 3.0}s</strong>.</div>`;
            } else {
                elThActiveContainer.innerHTML = activeSleeps.map(s => {
                    return `<div style="background: rgba(248,113,113,0.08); border: 1px solid rgba(248,113,113,0.3); border-radius: 4px; padding: 0.5rem 0.75rem; margin-top: 0.4rem; display: flex; justify-content: space-between; align-items: center;">
                        <span><strong style="color: #f87171;">⏳ Sleeping:</strong> <span style="color: white; font-family: 'JetBrains Mono', monospace;">${escapeHtml(s.model_name)}</span></span>
                        <span style="font-family: 'JetBrains Mono', monospace; color: #fbbf24;">Elapsed: ${s.elapsed_seconds}s / Needed: ${s.sleep_needed}s (Rem: ${s.remaining_seconds}s)</span>
                    </div>`;
                }).join('');
            }
        }

        const pGrid = document.getElementById('provider-metrics-grid');
        if (pGrid && Array.isArray(data.provider_metrics)) {
            const providerLabels = {
                nvidia_nim: "NVIDIA NIM",
                open_router: "OpenRouter",
                gemini: "Google Gemini",
                groq: "Groq",
                deepseek: "DeepSeek",
                mistral: "Mistral / Codestral",
                cerebras: "Cerebras",
                fireworks: "Fireworks AI",
                kimi: "Kimi (Moonshot AI)",
                lmstudio: "LM Studio (Local)",
                ollama: "Ollama (Local)",
                llama_cpp: "llama.cpp (Local)",
            };

            pGrid.innerHTML = data.provider_metrics.map(pm => {
                const label = providerLabels[pm.provider] || pm.provider;
                const rpmBarCol = pm.rpm_usage_pct > 85 ? '#f87171' : (pm.rpm_usage_pct > 60 ? '#facc15' : '#4ade80');
                const tpmBarCol = pm.tpm_usage_pct > 85 ? '#f87171' : (pm.tpm_usage_pct > 60 ? '#facc15' : '#60a5fa');
                
                const statusBg = pm.rpd_exceeded ? 'rgba(248, 113, 113, 0.15)' : (pm.rpm_usage_pct > 85 ? 'rgba(250, 204, 21, 0.15)' : 'rgba(74, 222, 128, 0.1)');
                const statusCol = pm.rpd_exceeded ? '#f87171' : (pm.rpm_usage_pct > 85 ? '#facc15' : '#4ade80');

                return `
                <div style="background: rgba(255,255,255,0.02); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 0.85rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <strong style="color: white; font-size: 0.8rem;">${escapeHtml(label)}</strong>
                        <span style="font-size: 0.65rem; font-weight: 700; color: ${statusCol}; background: ${statusBg}; border: 1px solid ${statusCol}44; padding: 1px 6px; border-radius: 3px;">${pm.status}</span>
                    </div>

                    <!-- Fundamental RPM Gauge -->
                    <div style="margin-bottom: 0.4rem;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.7rem; color: var(--text-muted); margin-bottom: 2px;">
                            <span>RPM (Req/Min):</span>
                            <strong style="color: white; font-family: 'JetBrains Mono', monospace;">${pm.rpm_60s} / ${pm.rpm_limit} (${pm.rpm_usage_pct}%)</strong>
                        </div>
                        <div style="background: rgba(255,255,255,0.06); height: 5px; border-radius: 3px; overflow: hidden;">
                            <div style="background: ${rpmBarCol}; width: ${Math.min(100, pm.rpm_usage_pct)}%; height: 100%; transition: width 0.3s;"></div>
                        </div>
                    </div>

                    <!-- Fundamental TPM Gauge -->
                    <div style="margin-bottom: 0.5rem;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.7rem; color: var(--text-muted); margin-bottom: 2px;">
                            <span>TPM (Tokens/Min):</span>
                            <strong style="color: white; font-family: 'JetBrains Mono', monospace;">${pm.tpm_60s.toLocaleString()} / ${pm.tpm_limit.toLocaleString()} (${pm.tpm_usage_pct}%)</strong>
                        </div>
                        <div style="background: rgba(255,255,255,0.06); height: 5px; border-radius: 3px; overflow: hidden;">
                            <div style="background: ${tpmBarCol}; width: ${Math.min(100, pm.tpm_usage_pct)}%; height: 100%; transition: width 0.3s;"></div>
                        </div>
                    </div>

                    <div style="display: flex; justify-content: space-between; font-size: 0.68rem; color: var(--text-dim); border-top: 1px dashed rgba(255,255,255,0.08); padding-top: 0.35rem;">
                        <span>Today RPD: <strong style="color: white;">${pm.rpd_count}/${pm.rpd_limit}</strong></span>
                        <span>Timeout: <strong style="color: white;">${pm.http_read_timeout}s</strong></span>
                        <span>Max Conn: <strong style="color: white;">${pm.max_concurrency}</strong></span>
                    </div>
                </div>`;
            }).join('');
        }

        const tbody = document.getElementById('metrics-table-body');
        if (!tbody) return;

        const models = data.model_metrics || [];
        if (models.length === 0) {
            tbody.innerHTML = `<tr><td colspan="8" style="padding: 1.5rem; text-align: center; color: var(--text-muted);">No models tracked yet.</td></tr>`;
            return;
        }

        tbody.innerHTML = models.map(m => {
            const isCbOpen = m.circuit_state === 'open';
            const cbBadge = isCbOpen
                ? `<span style="color: #f87171; font-weight: 800; font-size: 0.7rem;">● CIRCUIT OPEN (${m.recovery_remaining_s || 0}s recovery)</span>`
                : (m.circuit_state === 'half_open'
                    ? `<span style="color: #facc15; font-weight: 800; font-size: 0.7rem;">● HALF OPEN</span>`
                    : `<span style="color: #4ade80; font-weight: 700; font-size: 0.7rem;">● CLOSED (HEALTHY)</span>`);

            const waitStr = (m.estimated_wait_s || 0) > 0
                ? `<span style="color: #f87171; font-weight: 800; font-family: 'JetBrains Mono', monospace;">⏳ ${m.estimated_wait_s}s wait</span>`
                : `<span style="color: #4ade80; font-family: 'JetBrains Mono', monospace;">0.0s</span>`;

            const rlBadge = (m.rate_limit_429_count || 0) > 0
                ? `<span style="color: #f87171; font-weight: 800; font-family: 'JetBrains Mono', monospace;">⚠️ ${m.rate_limit_429_count} hits (${m.rate_limit_429_60s || 0} in 60s)</span>`
                : `<span style="color: #4ade80; font-family: 'JetBrains Mono', monospace;">0 hits</span>`;

            const succColor = m.success_rate >= 90 ? '#4ade80' : (m.success_rate >= 70 ? '#facc15' : '#f87171');
            const rpmBadge = m.rpm_60s > 0
                ? `<span style="color: #ffffff; font-weight: 800; font-family: 'JetBrains Mono', monospace;">${m.rpm_60s} req/min</span>`
                : `<span style="color: var(--text-dim); font-family: 'JetBrains Mono', monospace;">0 req/min</span>`;

            const headroomBadge = m.has_headroom
                ? (m.req_remaining !== null
                    ? `<span style="color: white; font-family: 'JetBrains Mono', monospace;">${m.req_remaining} req left</span>`
                    : `<span style="color: #4ade80;">OK (100%)</span>`)
                : `<span style="color: #f87171; font-weight: 700;">TIGHT / LIMITED</span>`;

            return `<tr style="border-bottom: 1px solid var(--border-subtle);">
                <td style="padding: 0.75rem 1rem; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: white;">${escapeHtml(m.model_id)}</td>
                <td style="padding: 0.75rem 1rem; text-align: center;">${rpmBadge}</td>
                <td style="padding: 0.75rem 1rem; text-align: center;">${waitStr}</td>
                <td style="padding: 0.75rem 1rem; text-align: center;">${rlBadge}</td>
                <td style="padding: 0.75rem 1rem; text-align: center; font-family: 'JetBrains Mono', monospace;">${m.avg_latency_ms} ms</td>
                <td style="padding: 0.75rem 1rem; text-align: center; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: ${succColor};">${m.success_rate}%</td>
                <td style="padding: 0.75rem 1rem;">${cbBadge}</td>
                <td style="padding: 0.75rem 1rem; text-align: right;">${headroomBadge}</td>
            </tr>`;
        }).join('');

    } catch (e) {
        console.error("Failed to fetch dev metrics", e);
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

        const errBox = document.getElementById('modal-error-trace-container');
        const errContent = document.getElementById('modal-error-trace-content');
        if (errBox && errContent) {
            if (payload.error_details || (payload.attempt_history && payload.attempt_history.length > 0)) {
                errBox.style.display = 'block';
                let html = '';
                if (payload.error_details && payload.error_details.last_error) {
                    html += `<div style="margin-bottom: 6px;"><strong>Last Upstream Exception:</strong> ${payload.error_details.last_error}</div>`;
                }
                if (payload.attempt_history && payload.attempt_history.length > 0) {
                    html += `<div style="font-weight: 700; margin-top: 4px; color: #fca5a5;">Attempted Candidate History:</div>`;
                    payload.attempt_history.forEach((att, idx) => {
                        html += `<div style="padding-left: 8px;">├─ #${idx + 1} [${att.model}]: status ${att.status_code || 500} - ${att.failure_reason || att.error_message || 'Failed'}</div>`;
                    });
                }
                errContent.innerHTML = html;
            } else {
                errBox.style.display = 'none';
            }
        }

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
    if (consoleEl) {
        consoleEl.innerHTML = '<div class="terminal-line">> Initializing diagnostic check...</div>';
    }
    try {
        const resp = await fetch('/api/diagnostics');
        const data = await resp.json();

        const logs = (data && Array.isArray(data.logs)) ? data.logs : ["System diagnostic data structure unexpected."];
        if (consoleEl) consoleEl.innerHTML = '';

        let i = 0;
        function printLine() {
            if (!consoleEl) return;
            if (i < logs.length) {
                const div = document.createElement('div');
                div.className = 'terminal-line';
                div.innerText = logs[i];
                consoleEl.appendChild(div);
                consoleEl.scrollTop = consoleEl.scrollHeight;
                i++;
                setTimeout(printLine, 80);
            } else {
                const finalDiv = document.createElement('div');
                finalDiv.className = 'terminal-line';
                if (data && data.has_errors) {
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
        if (consoleEl) consoleEl.innerHTML += '<div class="terminal-line" style="color: #f87171">> System connection failed: ' + e + '</div>';
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

        // Update Subagent Status minimal badge in Gateway Port dial card
        const subBadge = document.getElementById('telemetry-subagent-badge');
        if (subBadge && data.subagents_enabled !== undefined) {
            if (data.subagents_enabled) {
                subBadge.innerText = 'SUBAGENTS: ON';
                subBadge.style.color = '#4ade80';
                subBadge.style.background = 'rgba(74, 222, 128, 0.08)';
                subBadge.style.borderColor = 'rgba(74, 222, 128, 0.3)';
            } else {
                subBadge.innerText = 'SUBAGENTS: OFF';
                subBadge.style.color = '#f87171';
                subBadge.style.background = 'rgba(248, 113, 113, 0.08)';
                subBadge.style.borderColor = 'rgba(248, 113, 113, 0.3)';
            }
        }

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

        // Render Live Request Trace log feed with pagination
        traceRawRequests = data.recent_requests || [];
        renderTraceFeed();

        // Automatically refresh Dev Mode payloads on every telemetry poll
        await fetchDevPayloads();
        await fetchDevMetrics();

    } catch (e) {
        console.error("Telemetry polling failed", e);
    }
}

let traceCurrentPage = 1;
let tracePageSize = 20;
let traceRawRequests = [];

function changeTracePage(delta) {
    const maxPage = Math.max(1, Math.ceil(traceRawRequests.length / tracePageSize));
    traceCurrentPage = Math.min(maxPage, Math.max(1, traceCurrentPage + delta));
    renderTraceFeed();
}

function changeTracePageSize(val) {
    tracePageSize = val === 'all' ? 999999 : (parseInt(val) || 20);
    traceCurrentPage = 1;
    renderTraceFeed();
}

function renderTraceFeed() {
    const feedBody = document.getElementById('live-request-feed-body');
    if (!feedBody) return;

    if (!traceRawRequests || traceRawRequests.length === 0) {
        feedBody.innerHTML = `<tr>
            <td colspan="6" style="text-align: center; color: var(--text-dim); padding: 20px;">No operational traffic logged. Awaiting API events...</td>
        </tr>`;
        updateTracePaginationUI(0, 0, 0, 1, 1);
        return;
    }

    const sorted = [...traceRawRequests].reverse();
    const totalItems = sorted.length;
    const maxPage = Math.max(1, Math.ceil(totalItems / tracePageSize));
    traceCurrentPage = Math.min(maxPage, Math.max(1, traceCurrentPage));

    const startIdx = (traceCurrentPage - 1) * tracePageSize;
    const endIdx = Math.min(totalItems, startIdx + tracePageSize);
    const pageItems = sorted.slice(startIdx, endIdx);

    feedBody.innerHTML = '';
    pageItems.forEach(req => {
        const tr = document.createElement('tr');
        
        const tdTime = document.createElement('td');
        tdTime.innerText = req.timestamp;
        tdTime.style.whiteSpace = 'nowrap';
        tr.appendChild(tdTime);

        const tdPath = document.createElement('td');
        tdPath.innerText = `${req.method} ${req.path}`;
        tdPath.style.wordBreak = 'break-all';
        tdPath.style.maxWidth = '160px';
        tr.appendChild(tdPath);

        const tdClient = document.createElement('td');
        tdClient.innerText = req.client_model;
        tdClient.style.wordBreak = 'break-all';
        tdClient.style.maxWidth = '160px';
        tr.appendChild(tdClient);

        const tdUpstream = document.createElement('td');
        tdUpstream.innerText = req.mapped_model || req.target_model || '-';
        tdUpstream.style.wordBreak = 'break-all';
        tdUpstream.style.maxWidth = '200px';
        tr.appendChild(tdUpstream);

        const tdLatency = document.createElement('td');
        const dur = req.duration_ms !== undefined ? req.duration_ms : (req.latency_ms || 0);
        tdLatency.innerText = `${dur} ms`;
        tdLatency.style.whiteSpace = 'nowrap';
        tr.appendChild(tdLatency);

        const tdStatus = document.createElement('td');
        tdStatus.style.wordBreak = 'break-word';
        tdStatus.style.maxWidth = '260px';
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

    updateTracePaginationUI(startIdx + 1, endIdx, totalItems, traceCurrentPage, maxPage);
}

function updateTracePaginationUI(from, to, total, page, maxPage) {
    const infoEl = document.getElementById('trace-page-info');
    const numEl = document.getElementById('trace-page-number');
    const btnPrev = document.getElementById('trace-btn-prev');
    const btnNext = document.getElementById('trace-btn-next');

    if (infoEl) infoEl.innerText = total > 0 ? `Showing ${from}-${to} of ${total} events` : 'Showing 0-0 of 0 events';
    if (numEl) numEl.innerText = `Page ${page} of ${maxPage}`;
    if (btnPrev) btnPrev.disabled = page <= 1;
    if (btnNext) btnNext.disabled = page >= maxPage;
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
            const full = m.startsWith(providerPrefix) ? m : providerPrefix + m;
            return !textVal || full.toLowerCase().includes(textVal);
        });

        if (filtered.length > 0) {
            const header = document.createElement('div');
            header.className = 'autocomplete-group';
            header.innerText = provider.replace('_', ' ');
            dropdownEl.appendChild(header);

            filtered.forEach(m => {
                count++;
                const fullStr = m.startsWith(providerPrefix) ? m : providerPrefix + m;
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

function blurAutocomplete(inputId) {
    setTimeout(() => {
        const dropdownEl = document.getElementById(inputId + '-dropdown');
        if (dropdownEl) dropdownEl.style.display = 'none';
    }, 200);
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

// Provider default preset configs (including RPM, TPM, Concurrency, and Timeouts)
const PROVIDER_PRESETS = {
    nvidia_nim: {
        label: 'NVIDIA NIM — Single-Lane Guard (1 concurrency limit), Rate: 38 req/min (capped buffer), TPM: 200k, Read: 120s.',
        PROVIDER_RATE_LIMIT: 38,
        PROVIDER_RATE_WINDOW: 60,
        PROVIDER_MAX_CONCURRENCY: 1,
        HTTP_READ_TIMEOUT: 120,
        HTTP_WRITE_TIMEOUT: 15,
        HTTP_CONNECT_TIMEOUT: 5,
    },
    openrouter: {
        label: 'OpenRouter — Balanced multi-model cloud aggregator. Rate: 40 req/min, TPM: 500k, Concurrency: 5, Read: 180s.',
        PROVIDER_RATE_LIMIT: 40,
        PROVIDER_RATE_WINDOW: 60,
        PROVIDER_MAX_CONCURRENCY: 5,
        HTTP_READ_TIMEOUT: 180,
        HTTP_WRITE_TIMEOUT: 20,
        HTTP_CONNECT_TIMEOUT: 5,
    },
    gemini: {
        label: 'Google Gemini — OpenAI compatible endpoint (v1beta). Rate: 30 req/min (Free) / 360 (Paid), TPM: 1,000k, Concurrency: 5, Read: 120s.',
        PROVIDER_RATE_LIMIT: 30,
        PROVIDER_RATE_WINDOW: 60,
        PROVIDER_MAX_CONCURRENCY: 5,
        HTTP_READ_TIMEOUT: 120,
        HTTP_WRITE_TIMEOUT: 15,
        HTTP_CONNECT_TIMEOUT: 5,
    },
    groq: {
        label: 'Groq — Ultra-fast LPU inference, strict rate caps. Rate: 30 req/min, TPM: 14k, Concurrency: 4, Read: 60s.',
        PROVIDER_RATE_LIMIT: 30,
        PROVIDER_RATE_WINDOW: 60,
        PROVIDER_MAX_CONCURRENCY: 4,
        HTTP_READ_TIMEOUT: 60,
        HTTP_WRITE_TIMEOUT: 10,
        HTTP_CONNECT_TIMEOUT: 5,
    },
    deepseek: {
        label: 'DeepSeek — High context reasoning & coding. Rate: 50 req/min, TPM: 100k, Concurrency: 6, Read: 180s.',
        PROVIDER_RATE_LIMIT: 50,
        PROVIDER_RATE_WINDOW: 60,
        PROVIDER_MAX_CONCURRENCY: 6,
        HTTP_READ_TIMEOUT: 180,
        HTTP_WRITE_TIMEOUT: 20,
        HTTP_CONNECT_TIMEOUT: 5,
    },
    mistral: {
        label: 'Mistral / Codestral — Enterprise coding models. Rate: 40 req/min, TPM: 100k, Concurrency: 5, Read: 120s.',
        PROVIDER_RATE_LIMIT: 40,
        PROVIDER_RATE_WINDOW: 60,
        PROVIDER_MAX_CONCURRENCY: 5,
        HTTP_READ_TIMEOUT: 120,
        HTTP_WRITE_TIMEOUT: 15,
        HTTP_CONNECT_TIMEOUT: 5,
    },
    cerebras: {
        label: 'Cerebras — Wafer-scale fast inference. Rate: 60 req/min, TPM: 60k, Concurrency: 8, Read: 60s.',
        PROVIDER_RATE_LIMIT: 60,
        PROVIDER_RATE_WINDOW: 60,
        PROVIDER_MAX_CONCURRENCY: 8,
        HTTP_READ_TIMEOUT: 60,
        HTTP_WRITE_TIMEOUT: 10,
        HTTP_CONNECT_TIMEOUT: 5,
    },
    fireworks: {
        label: 'Fireworks AI — Fast open-weights inference. Rate: 60 req/min, TPM: 200k, Concurrency: 8, Read: 120s.',
        PROVIDER_RATE_LIMIT: 60,
        PROVIDER_RATE_WINDOW: 60,
        PROVIDER_MAX_CONCURRENCY: 8,
        HTTP_READ_TIMEOUT: 120,
        HTTP_WRITE_TIMEOUT: 15,
        HTTP_CONNECT_TIMEOUT: 5,
    },
    kimi: {
        label: 'Kimi (Moonshot AI) — Long-context Chinese & English models. Rate: 20 req/min, TPM: 100k, Concurrency: 3, Read: 120s.',
        PROVIDER_RATE_LIMIT: 20,
        PROVIDER_RATE_WINDOW: 60,
        PROVIDER_MAX_CONCURRENCY: 3,
        HTTP_READ_TIMEOUT: 120,
        HTTP_WRITE_TIMEOUT: 15,
        HTTP_CONNECT_TIMEOUT: 5,
    },
    lmstudio: {
        label: 'LM Studio — Local desktop server. Rate: 100 req/min (Local), Concurrency: 2, Read: 300s.',
        PROVIDER_RATE_LIMIT: 100,
        PROVIDER_RATE_WINDOW: 60,
        PROVIDER_MAX_CONCURRENCY: 2,
        HTTP_READ_TIMEOUT: 300,
        HTTP_WRITE_TIMEOUT: 30,
        HTTP_CONNECT_TIMEOUT: 5,
    },
    ollama: {
        label: 'Ollama — Local CLI inference runner. Rate: 100 req/min (Local), Concurrency: 2, Read: 300s.',
        PROVIDER_RATE_LIMIT: 100,
        PROVIDER_RATE_WINDOW: 60,
        PROVIDER_MAX_CONCURRENCY: 2,
        HTTP_READ_TIMEOUT: 300,
        HTTP_WRITE_TIMEOUT: 30,
        HTTP_CONNECT_TIMEOUT: 5,
    },
    llama_cpp: {
        label: 'llama.cpp — Local server daemon. Rate: 100 req/min (Local), Concurrency: 1, Read: 300s.',
        PROVIDER_RATE_LIMIT: 100,
        PROVIDER_RATE_WINDOW: 60,
        PROVIDER_MAX_CONCURRENCY: 1,
        HTTP_READ_TIMEOUT: 300,
        HTTP_WRITE_TIMEOUT: 30,
        HTTP_CONNECT_TIMEOUT: 5,
    },
};


let subagentsEnabledState = true;

async function fetchSubagentEmergencyStatus() {
    try {
        const resp = await fetch('/api/settings/subagents');
        const data = await resp.json();
        subagentsEnabledState = !!data.subagents_enabled;
        updateSubagentUI(subagentsEnabledState);
    } catch (e) {
        console.error("Failed to fetch subagents status", e);
    }
}

function updateSubagentUI(enabled) {
    const el = document.getElementById('SUBAGENTS_ENABLED');
    if (el) {
        el.checked = !!enabled;
    }
}

async function toggleSubagentEmergencySwitch(val) {
    const nextState = val !== undefined ? !!val : !subagentsEnabledState;
    try {
        const resp = await fetch('/api/settings/subagents', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled: nextState })
        });
        const data = await resp.json();
        if (data.status === 'success') {
            subagentsEnabledState = data.subagents_enabled;
            updateSubagentUI(subagentsEnabledState);
            showToast(data.message, subagentsEnabledState ? 'success' : 'warning');
        } else {
            showToast("Failed to update switch: " + data.message, "error");
            updateSubagentUI(subagentsEnabledState);
        }
    } catch (e) {
        showToast("Error updating Subagent Emergency Switch", "error");
        updateSubagentUI(subagentsEnabledState);
    }
}




