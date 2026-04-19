/**
 * FinMEM Dashboard — Frontend Logic
 * API client and UI interactions for the Agentic FinMEM dashboard.
 */

const API_BASE = window.location.origin;

// ── Tab Navigation ──────────────────────────────────────────────────────

document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        // Deactivate all
        document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));

        // Activate clicked
        tab.classList.add('active');
        const panelId = 'panel-' + tab.dataset.tab;
        document.getElementById(panelId).classList.add('active');
    });
});

// ── API Helper ──────────────────────────────────────────────────────────

async function apiCall(endpoint, method = 'GET', body = null) {
    const opts = {
        method,
        headers: { 'Content-Type': 'application/json' },
    };
    if (body) opts.body = JSON.stringify(body);

    const response = await fetch(`${API_BASE}${endpoint}`, opts);
    return response.json();
}

function setLoading(btnId, loading) {
    const btn = document.getElementById(btnId);
    if (loading) {
        btn.classList.add('loading');
        btn.disabled = true;
    } else {
        btn.classList.remove('loading');
        btn.disabled = false;
    }
}

// ── System Status ───────────────────────────────────────────────────────

async function loadStatus() {
    try {
        const data = await apiCall('/api/status');
        if (!data || !data.config) return;

        const cfg = data.config;

        // Header badges
        const updateBadge = (id, isActive) => {
            const badge = document.getElementById(id);
            badge.className = `status-badge ${isActive ? 'active' : 'inactive'}`;
        };

        const obj1Active = cfg.adaptive_q === 'true';
        const obj2Active = cfg.learned_importance === 'true';
        const obj3Active = cfg.cross_ticker === 'true';

        updateBadge('badge-obj1', obj1Active);
        updateBadge('badge-obj2', obj2Active);
        updateBadge('badge-obj3', obj3Active);

        // Dashboard stats
        document.getElementById('stat-provider').textContent = cfg.llm_provider?.toUpperCase() || '—';
        document.getElementById('stat-model').textContent = cfg.bedrock_model?.split('.').pop() || '—';
        document.getElementById('stat-model').style.fontSize = '18px';

        const activeCount = [obj1Active, obj2Active, obj3Active].filter(Boolean).length;
        document.getElementById('stat-objectives').textContent = `${activeCount}/3`;

        // Importance model
        if (data.importance_model?.loaded) {
            document.getElementById('stat-importance').textContent =
                `${(data.importance_model.accuracy * 100).toFixed(0)}%`;
        } else {
            document.getElementById('stat-importance').textContent = 'Random';
        }

        // Config details
        document.getElementById('cfg-provider').textContent = cfg.llm_provider?.toUpperCase() || '—';
        document.getElementById('cfg-model').textContent = cfg.bedrock_model || '—';
        document.getElementById('cfg-tickers').textContent = cfg.portfolio_tickers || '—';

        // Objective statuses
        const setObjStatus = (id, active) => {
            const el = document.getElementById(id);
            el.textContent = active ? 'ON' : 'OFF';
            el.className = `action-badge ${active ? 'buy' : 'hold'}`;
        };
        setObjStatus('obj1-status', obj1Active);
        setObjStatus('obj2-status', obj2Active);
        setObjStatus('obj3-status', obj3Active);

    } catch (err) {
        console.error('Status fetch failed:', err);
    }
}

// ── Objective 2 ─────────────────────────────────────────────────────────

async function testObj2() {
    setLoading('btn-test-obj2', true);
    const output = document.getElementById('output-obj2');
    output.textContent = 'Running Objective 2 test...\n';
    output.classList.remove('error');

    try {
        const data = await apiCall('/api/test-obj2', 'POST');
        if (data.success) {
            output.textContent = data.output;
        } else {
            output.textContent = `❌ Error: ${data.error}\n\n${data.traceback || ''}`;
            output.classList.add('error');
        }
    } catch (err) {
        output.textContent = `❌ Connection error: ${err.message}`;
        output.classList.add('error');
    }

    setLoading('btn-test-obj2', false);
}

async function calcImportance() {
    try {
        const data = await apiCall('/api/importance-score', 'POST', {
            layer: document.getElementById('calc-layer').value,
            age_days: parseInt(document.getElementById('calc-age').value),
            access_count: parseInt(document.getElementById('calc-access').value),
            text_length: parseInt(document.getElementById('calc-textlen').value),
            sentiment_score: parseInt(document.getElementById('calc-sentiment').value) / 100,
        });

        const resultDiv = document.getElementById('calc-result');
        if (data.success) {
            const score = data.score.toFixed(1);
            const pct = ((data.score - 40) / 40 * 100).toFixed(0);
            resultDiv.innerHTML = `
                <div class="stat-card" style="margin-top: 8px;">
                    <div class="stat-value success">${score}</div>
                    <div class="stat-label">v_E (range: 40-80) · ${pct}% importance</div>
                </div>
            `;
        } else {
            resultDiv.innerHTML = `<p style="color: var(--accent-rose);">Error: ${data.error}</p>`;
        }
    } catch (err) {
        document.getElementById('calc-result').innerHTML =
            `<p style="color: var(--accent-rose);">Connection error: ${err.message}</p>`;
    }
}

async function loadReflections() {
    const output = document.getElementById('output-reflections');
    output.textContent = 'Loading...';

    try {
        const data = await apiCall('/api/reflections');
        if (data.success) {
            if (data.count === 0) {
                output.textContent = 'No reflection logs found. Run FinMEM in train mode first.';
            } else {
                output.textContent = `${data.count} total reflections:\n\n` +
                    data.reflections.map(r =>
                        `[${r.date}] ${r.ticker} → ${r.decision} | IDs: ${(r.memory_ids_used || []).join(',')} | ${(r.rationale || '').substring(0, 80)}...`
                    ).join('\n');
            }
        } else {
            output.textContent = `Error: ${data.error}`;
            output.classList.add('error');
        }
    } catch (err) {
        output.textContent = `Connection error: ${err.message}`;
        output.classList.add('error');
    }
}

// ── Objective 3 ─────────────────────────────────────────────────────────

async function testObj3() {
    setLoading('btn-test-obj3', true);
    const output = document.getElementById('output-obj3');
    output.textContent = 'Running Objective 3 test...\n';
    output.classList.remove('error');

    try {
        const data = await apiCall('/api/test-obj3', 'POST');
        if (data.success) {
            output.textContent = data.output;
        } else {
            output.textContent = `❌ Error: ${data.error}\n\n${data.traceback || ''}`;
            output.classList.add('error');
        }
    } catch (err) {
        output.textContent = `❌ Connection error: ${err.message}`;
        output.classList.add('error');
    }

    setLoading('btn-test-obj3', false);
}

async function loadCorrelation() {
    setLoading('btn-load-corr', true);
    const container = document.getElementById('heatmap-container');
    container.innerHTML = '<p style="color: var(--text-muted); text-align: center; padding: 20px;">Computing correlation matrix via yfinance...</p>';

    try {
        const data = await apiCall('/api/correlation-matrix', 'POST');

        if (data.success) {
            const tickers = data.tickers;
            const matrix = data.matrix;

            let html = '<table class="heatmap-table"><thead><tr><th></th>';
            tickers.forEach(t => html += `<th>${t}</th>`);
            html += '</tr></thead><tbody>';

            tickers.forEach(t1 => {
                html += `<tr><th>${t1}</th>`;
                tickers.forEach(t2 => {
                    const val = matrix[t1]?.[t2] ?? 0;
                    const color = getHeatmapColor(val);
                    html += `<td style="background:${color}; color: ${Math.abs(val) > 0.5 ? 'white' : 'var(--text-primary)'};">${val.toFixed(2)}</td>`;
                });
                html += '</tr>';
            });

            html += '</tbody></table>';
            container.innerHTML = html;
        } else {
            container.innerHTML = `<p style="color: var(--accent-rose); padding: 20px;">Error: ${data.error}</p>`;
        }
    } catch (err) {
        container.innerHTML = `<p style="color: var(--accent-rose); padding: 20px;">Connection error: ${err.message}</p>`;
    }

    setLoading('btn-load-corr', false);
}

function getHeatmapColor(val) {
    // Red-Yellow-Green gradient
    const normalized = (val + 1) / 2; // 0 to 1
    const r = Math.round(255 * (1 - normalized));
    const g = Math.round(200 * normalized);
    const b = Math.round(50 * normalized);
    const a = 0.3 + Math.abs(val) * 0.5;
    return `rgba(${val > 0 ? 50 : r}, ${g}, ${val > 0 ? 200 : b}, ${a})`;
}

async function testGuard() {
    setLoading('btn-test-guard', true);
    const container = document.getElementById('guard-results');
    container.innerHTML = '<p style="color: var(--text-muted);">Testing concentration guard...</p>';

    try {
        const data = await apiCall('/api/test-guard', 'POST');

        if (data.success) {
            let html = `<p style="color: var(--accent-amber); font-weight: 600; margin-bottom: 12px;">
                🛡️ Guard triggered ${data.trigger_count} time(s)</p>`;

            html += '<table class="decision-table"><thead><tr>';
            html += '<th>Ticker</th><th>Action</th><th>Confidence</th><th>Note</th>';
            html += '</tr></thead><tbody>';

            for (const [ticker, dec] of Object.entries(data.decisions)) {
                const actionClass = dec.action?.toLowerCase() || 'hold';
                const override = dec.override_reason || '';
                html += `<tr>
                    <td style="font-weight: 600; color: var(--text-primary);">${ticker}</td>
                    <td><span class="action-badge ${actionClass}">${dec.action}</span></td>
                    <td style="font-family: 'JetBrains Mono'; color: var(--accent-cyan);">${(dec.confidence || 0).toFixed(2)}</td>
                    <td class="override-badge">${override}</td>
                </tr>`;
            }

            html += '</tbody></table>';
            container.innerHTML = html;
        } else {
            container.innerHTML = `<p style="color: var(--accent-rose);">Error: ${data.error}</p>`;
        }
    } catch (err) {
        container.innerHTML = `<p style="color: var(--accent-rose);">Connection error: ${err.message}</p>`;
    }

    setLoading('btn-test-guard', false);
}

// ── Simulation ──────────────────────────────────────────────────────────

async function runSimulation() {
    setLoading('btn-run-sim', true);
    const resultsDiv = document.getElementById('sim-results');
    resultsDiv.innerHTML = '<p style="color: var(--accent-cyan); text-align: center; padding: 40px;">🚀 Running simulation... This may take a few minutes.</p>';

    try {
        const data = await apiCall('/api/run-simulation', 'POST', {
            ticker: document.getElementById('sim-ticker').value,
            mode: document.getElementById('sim-mode').value,
            start_date: document.getElementById('sim-start').value,
            end_date: document.getElementById('sim-end').value,
            capital: parseFloat(document.getElementById('sim-capital').value),
            adaptive_q: document.getElementById('sim-obj1').checked,
            learned_importance: document.getElementById('sim-obj2').checked,
            cross_ticker: document.getElementById('sim-obj3').checked,
        });

        if (data.success) {
            const m = data.metrics || {};
            const bh = data.bh_metrics || {};
            const returnColor = data.total_return >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)';

            let html = `
                <div class="grid-2" style="gap: 12px; margin-bottom: 16px;">
                    <div class="stat-card">
                        <div class="stat-value" style="color: ${returnColor}; -webkit-text-fill-color: ${returnColor};">
                            ${data.total_return >= 0 ? '+' : ''}$${data.total_return?.toFixed(2) || '0'}
                        </div>
                        <div class="stat-label">Total Return</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" style="color: ${returnColor}; -webkit-text-fill-color: ${returnColor};">
                            ${data.total_return_pct >= 0 ? '+' : ''}${data.total_return_pct?.toFixed(2) || '0'}%
                        </div>
                        <div class="stat-label">Return %</div>
                    </div>
                </div>

                <table class="decision-table" style="margin-top: 12px;">
                    <thead><tr><th>Metric</th><th>FinMEM</th><th>B&H</th></tr></thead>
                    <tbody>
                        <tr><td>Cum. Return (%)</td>
                            <td style="color: var(--accent-cyan);">${(m.cumulative_return_pct || 0).toFixed(2)}%</td>
                            <td style="color: var(--text-muted);">${(bh.cumulative_return_pct || 0).toFixed(2)}%</td></tr>
                        <tr><td>Sharpe Ratio</td>
                            <td style="color: var(--accent-cyan);">${(m.sharpe_ratio || 0).toFixed(4)}</td>
                            <td style="color: var(--text-muted);">${(bh.sharpe_ratio || 0).toFixed(4)}</td></tr>
                        <tr><td>Max Drawdown (%)</td>
                            <td style="color: var(--accent-rose);">${(m.max_drawdown_pct || 0).toFixed(2)}%</td>
                            <td style="color: var(--text-muted);">${(bh.max_drawdown_pct || 0).toFixed(2)}%</td></tr>
                        <tr><td>Days Processed</td>
                            <td style="color: var(--accent-cyan);">${data.days_processed}</td>
                            <td>—</td></tr>
                        <tr><td>Trades</td>
                            <td style="color: var(--accent-cyan);">${data.trade_count}</td>
                            <td>—</td></tr>
                    </tbody>
                </table>

                <div style="margin-top: 12px; padding: 12px; background: rgba(0,0,0,0.2); border-radius: 8px; font-size: 12px; color: var(--text-muted);">
                    Memory Stats: ${JSON.stringify(data.memory_stats || {})}
                </div>
            `;

            resultsDiv.innerHTML = html;
        } else {
            resultsDiv.innerHTML = `
                <div class="output-console error" style="margin: 0;">
                    ❌ Simulation failed:\n${data.error}\n\n${data.traceback || ''}
                </div>
            `;
        }
    } catch (err) {
        resultsDiv.innerHTML = `<p style="color: var(--accent-rose); text-align: center; padding: 40px;">Connection error: ${err.message}</p>`;
    }

    setLoading('btn-run-sim', false);
}

// ── Ablation Run ────────────────────────────────────────────────────────

let ablationDataStore = {};

async function loadAblationResults() {
    setLoading('btn-load-ablation', true);
    const select = document.getElementById('ablation-run-select');
    
    try {
        const data = await apiCall('/api/ablation-results', 'GET');
        
        if (data.success) {
            ablationDataStore = data.results;
            const labels = Object.keys(data.results).sort((a,b) => b.localeCompare(a));
            
            if (labels.length === 0) {
                select.innerHTML = '<option value="">No completed runs found</option>';
            } else {
                select.innerHTML = labels.map(L => `<option value="${L}">${L === 'default' ? 'Default Run' : L}</option>`).join('');
                renderAblationTable();
            }
        } else {
            console.error(data.error);
        }
    } catch (err) {
        console.error('Failed to load ablation results:', err);
    }
    
    setLoading('btn-load-ablation', false);
}

function renderAblationTable() {
    const select = document.getElementById('ablation-run-select');
    const label = select.value;
    const tbody = document.getElementById('ablation-table-body');
    
    if (!label || !ablationDataStore[label]) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; padding: 20px; color: var(--text-muted);">Select a run context.</td></tr>';
        return;
    }
    
    const rows = ablationDataStore[label];
    
    if (rows.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; padding: 20px; color: var(--text-muted);">No data in this run.</td></tr>';
        return;
    }
    
    // Sort array by name: Base -> Obj1 -> Obj1+2 -> Obj1+2+3 -> Obj1+2+3+4
    const orderScore = (name) => {
        if (!name) return 100;
        const n = name.toLowerCase();
        if (n.includes('base')) return 0;
        if (n.includes('obj1+2+3+4')) return 4;
        if (n.includes('obj1+2+3')) return 3;
        if (n.includes('obj1+2')) return 2;
        if (n.includes('obj1')) return 1;
        return 10;
    };
    
    const sorted = [...rows].sort((a,b) => orderScore(a.Configuration) - orderScore(b.Configuration));
    
    let html = '';
    
    // Check if we should render B&H baseline (assumes it's present in the first row's columns)
    const firstObj = sorted[0];
    if (firstObj["BH Cum. Return (%)"]) {
        const bhRet = parseFloat(firstObj["BH Cum. Return (%)"] || 0);
        html += `
            <tr style="background: rgba(255,255,255,0.02); border-bottom: 1px solid var(--border);">
                <td style="padding: 12px; font-weight: bold;">Buy & Hold (Baseline)</td>
                <td style="padding: 12px; font-family: 'JetBrains Mono'; color: ${bhRet >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)'};">${bhRet > 0 ? '+' : ''}${bhRet.toFixed(2)}%</td>
                <td style="padding: 12px; font-family: 'JetBrains Mono';">${parseFloat(firstObj["BH Sharpe Ratio"] || 0).toFixed(4)}</td>
                <td style="padding: 12px; font-family: 'JetBrains Mono';">${parseFloat(firstObj["BH Max Drawdown (%)"] || 0).toFixed(2)}%</td>
                <td style="padding: 12px; font-family: 'JetBrains Mono'; color: var(--text-muted);">—</td>
            </tr>
        `;
    }
    
    sorted.forEach(r => {
        const conf = r.Configuration || 'Unknown';
        const ret = parseFloat(r["Total Return (%)"] || r["Cumulative Return (%)"] || 0);
        const sharpe = parseFloat(r["Sharpe Ratio"] || 0);
        const maxdd = parseFloat(r["Max Drawdown (%)"] || 0);
        const annVol = parseFloat(r["Ann. Volatility"] || 0);
        
        const retColor = ret >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)';
        
        let labelColor = 'var(--text-primary)';
        if (conf.includes('Obj1+2+3+4')) labelColor = 'var(--accent-purple)';
        else if (conf.includes('Obj1+2+3')) labelColor = 'var(--accent-cyan)';
        
        html += `
            <tr style="border-bottom: 1px solid var(--border);">
                <td style="padding: 12px; font-weight: 500; color: ${labelColor};">${conf}</td>
                <td style="padding: 12px; font-family: 'JetBrains Mono'; color: ${retColor};">${ret > 0 ? '+' : ''}${ret.toFixed(2)}%</td>
                <td style="padding: 12px; font-family: 'JetBrains Mono';">${sharpe.toFixed(4)}</td>
                <td style="padding: 12px; font-family: 'JetBrains Mono'; color: var(--accent-amber);">${maxdd.toFixed(2)}%</td>
                <td style="padding: 12px; font-family: 'JetBrains Mono';">${annVol.toFixed(4)}</td>
            </tr>
        `;
    });
    
    tbody.innerHTML = html;
}

// ── Init ─────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    loadStatus();
    loadAblationResults(); // Auto-load results
    // Refresh status every 30s
    setInterval(loadStatus, 30000);
});
