"""Simple human-readable admin page (no JS frameworks)."""

ADMIN_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>YOLO_DRT · Админ</title>
  <style>
    :root {
      --bg: #12141a; --card: #1c2029; --text: #e8eaef; --muted: #9aa3b5;
      --ok: #3dbe7a; --warn: #e0a84a; --bad: #e15a5a; --accent: #5b8def;
      --line: #2a3140;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; font-family: "Segoe UI", system-ui, sans-serif;
      background: radial-gradient(1200px 600px at 10% -10%, #1e2a44, transparent),
                  var(--bg); color: var(--text); line-height: 1.45;
    }
    header {
      padding: 1.25rem 1.5rem; border-bottom: 1px solid var(--line);
      display: flex; flex-wrap: wrap; gap: 1rem; align-items: baseline;
      justify-content: space-between;
    }
    h1 { margin: 0; font-size: 1.35rem; font-weight: 650; letter-spacing: .02em; }
    .muted { color: var(--muted); font-size: .92rem; }
    main { padding: 1.25rem 1.5rem 3rem; max-width: 1100px; margin: 0 auto; }
    .grid { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
    .card {
      background: var(--card); border: 1px solid var(--line); border-radius: 12px;
      padding: 1rem 1.1rem;
    }
    .card h2 { margin: 0 0 .75rem; font-size: 1rem; font-weight: 600; }
    .row { display: flex; justify-content: space-between; gap: .75rem; margin: .35rem 0; }
    .label { color: var(--muted); }
    .val { text-align: right; font-variant-numeric: tabular-nums; }
    .badge {
      display: inline-block; padding: .15rem .55rem; border-radius: 999px;
      font-size: .8rem; font-weight: 600;
    }
    .b-ok { background: rgba(61,190,122,.18); color: var(--ok); }
    .b-run { background: rgba(91,141,239,.18); color: var(--accent); }
    .b-bad { background: rgba(225,90,90,.18); color: var(--bad); }
    .b-warn { background: rgba(224,168,74,.18); color: var(--warn); }
    .actions { display: flex; flex-wrap: wrap; gap: .55rem; margin-top: .9rem; }
    button, .btn {
      appearance: none; border: 1px solid var(--line); background: #262c3a;
      color: var(--text); border-radius: 8px; padding: .55rem .9rem;
      font: inherit; cursor: pointer; text-decoration: none;
    }
    button:hover, .btn:hover { border-color: var(--accent); }
    button.danger { background: rgba(225,90,90,.15); border-color: rgba(225,90,90,.45); }
    button.primary { background: rgba(91,141,239,.2); border-color: rgba(91,141,239,.55); }
    table { width: 100%; border-collapse: collapse; font-size: .88rem; }
    th, td { padding: .4rem .35rem; border-bottom: 1px solid var(--line); text-align: left; }
    th { color: var(--muted); font-weight: 550; }
    .mono { font-family: ui-monospace, Consolas, monospace; font-size: .82rem; word-break: break-all; }
    .tip { margin-top: .8rem; color: var(--muted); font-size: .88rem; }
    #msg { margin: .75rem 0 0; min-height: 1.2em; }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>YOLO_DRT · панель API</h1>
      <div class="muted">Статус сервиса · последняя job · рестарт контейнера · тайминги запросов</div>
    </div>
    <div class="actions">
      <a class="btn" href="/docs">Swagger</a>
      <a class="btn" href="/health">/health</a>
      <button onclick="refresh()">Обновить</button>
    </div>
  </header>
  <main>
    <div id="msg" class="muted"></div>
    <div class="grid" style="margin-top:.5rem">
      <section class="card" id="svc"></section>
      <section class="card" id="job"></section>
    </div>
    <section class="card" style="margin-top:1rem">
      <h2>Действия</h2>
      <div class="actions">
        <button class="danger" onclick="cancelLatest()">Принудительно завершить последнюю / активную job</button>
        <a class="btn primary" href="/v1/admin/logs?tail=3000&download=true">Скачать логи контейнера (.txt)</a>
        <button class="primary" onclick="restart('docker')">Рестарт контейнера (Docker)</button>
        <button onclick="restart('exit')">Рестарт через exit (fallback)</button>
      </div>
      <p class="tip">
        Docker-рестарт и логи контейнера нужен volume <code>/var/run/docker.sock</code>.
        Логи: <code>GET /v1/admin/logs?tail=5000</code> → файл <code>.txt</code>.
        Длинные ролики клади в <code>./videos</code> и шли path
        <code>/data/videos/…</code> — без копирования в volume (иначе 5ч видео «не грузится»).
      </p>
    </section>
    <section class="card" style="margin-top:1rem">
      <h2>Последние запросы API</h2>
      <div id="reqs"></div>
    </section>
    <section class="card" style="margin-top:1rem">
      <h2>Недавние jobs</h2>
      <div id="jobs"></div>
    </section>
  </main>
  <script>
    function badge(status) {
      const s = (status || '').toLowerCase();
      let cls = 'b-warn';
      if (s === 'ready' || s === 'done') cls = 'b-ok';
      else if (s === 'running' || s === 'queued' || s === 'building_engines') cls = 'b-run';
      else if (s === 'error' || s === 'gpu_missing' || s === 'cancelled') cls = 'b-bad';
      return `<span class="badge ${cls}">${status}</span>`;
    }
    function row(k, v) {
      return `<div class="row"><span class="label">${k}</span><span class="val">${v}</span></div>`;
    }
    async function refresh() {
      const r = await fetch('/v1/admin/status');
      const d = await r.json();
      document.getElementById('svc').innerHTML = `
        <h2>Сервис</h2>
        ${row('Статус', badge(d.service_status_ru || d.service_status))}
        ${row('Контейнер', `<span class="mono">${d.container_name || '—'}</span>`)}
        ${row('Docker sock', d.docker_sock ? 'есть' : 'нет (только exit-рестарт)')}
        ${row('Uptime', d.uptime_human || '—')}
        ${(d.disks||[]).map(x => row('Диск '+x.path, (x.free_human||'?')+' свободно / '+(x.total_human||'?'))).join('')}
        <p class="tip">${d.tip || ''}</p>
      `;
      const j = d.active_job || d.last_job;
      if (!j) {
        document.getElementById('job').innerHTML = '<h2>Job</h2><p class="muted">Пока нет задач</p>';
      } else {
        const title = d.active_job ? 'Активная job' : 'Последняя job';
        document.getElementById('job').innerHTML = `
          <h2>${title}</h2>
          ${row('ID', `<span class="mono">${j.job_id}</span>`)}
          ${row('Статус', badge(j.status_ru || j.status))}
          ${row('Фаза', j.phase_ru || j.phase || '—')}
          ${row('Прогресс', (j.percent||0).toFixed(1) + '%')}
          ${row('Время', j.elapsed_human || '—')}
          ${row('ETA', j.eta_human || '—')}
          ${row('Файл', `<span class="mono">${j.input_path||'—'}</span>`)}
          ${row('Размер', j.input_size_human || '—')}
          ${row('Ошибка', j.error ? `<span style="color:var(--bad)">${j.error}</span>` : '—')}
        `;
      }
      const reqs = d.recent_requests || [];
      document.getElementById('reqs').innerHTML = reqs.length ? `
        <table><thead><tr><th>Время UTC</th><th>Метод</th><th>Путь</th><th>Код</th><th>Длительность</th></tr></thead>
        <tbody>${reqs.slice(0,40).map(x => `<tr>
          <td>${x.time_utc||''}</td><td>${x.method}</td>
          <td class="mono">${x.path}</td><td>${x.status_code}</td>
          <td>${x.duration_human}</td></tr>`).join('')}</tbody></table>` : '<p class="muted">Пока пусто</p>';
      const jobs = d.recent_jobs || [];
      document.getElementById('jobs').innerHTML = jobs.length ? `
        <table><thead><tr><th>ID</th><th>Статус</th><th>%</th><th>Файл</th><th>Время</th></tr></thead>
        <tbody>${jobs.map(x => `<tr>
          <td class="mono">${x.job_id.slice(0,10)}…</td>
          <td>${x.status_ru||x.status}</td><td>${(x.percent||0).toFixed(0)}</td>
          <td class="mono">${(x.input_path||'').split('/').pop()}</td>
          <td>${x.elapsed_human||'—'}</td></tr>`).join('')}</tbody></table>` : '<p class="muted">Нет</p>';
    }
    async function cancelLatest() {
      document.getElementById('msg').textContent = 'Отмена…';
      const r = await fetch('/v1/admin/jobs/latest/cancel', {method:'POST'});
      const d = await r.json().catch(()=>({}));
      document.getElementById('msg').textContent = r.ok
        ? ('Отмена: ' + (d.status_ru||d.status||d.job_id||'ok'))
        : ('Ошибка: ' + (d.detail||r.status));
      refresh();
    }
    async function restart(mode) {
      if (!confirm(mode==='docker'
        ? 'Перезапустить Docker-контейнер целиком?'
        : 'Завершить процесс (Docker поднимет контейнер заново)?')) return;
      document.getElementById('msg').textContent = 'Рестарт…';
      const r = await fetch('/v1/admin/restart', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({mode})
      });
      const d = await r.json().catch(()=>({}));
      document.getElementById('msg').textContent = d.message || JSON.stringify(d);
      setTimeout(refresh, 4000);
    }
    refresh();
    setInterval(refresh, 3000);
  </script>
</body>
</html>
"""
