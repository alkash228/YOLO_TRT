(() => {
  const $ = (sel) => document.querySelector(sel);

  const statusPill = $("#status-pill");
  const statusText = $("#status-text");
  const dropzone = $("#dropzone");
  const fileInput = $("#file-input");
  const pickFile = $("#pick-file");
  const fileName = $("#file-name");
  const btnRun = $("#btn-run");
  const jobProgressWrap = $("#job-progress-wrap");
  const jobProgress = $("#job-progress");
  const jobStatus = $("#job-status");
  const cardVideo = $("#card-video");
  const runMeta = $("#run-meta");
  const crossWarn = $("#cross-warn");
  const btnBuild = $("#btn-build");
  const sourceVideoInput = $("#source-video-input");
  const buildProgressWrap = $("#build-progress-wrap");
  const buildProgress = $("#build-progress");
  const buildStatus = $("#build-status");
  const videoList = $("#video-list");
  const toastStack = $("#toast-stack");
  const reportCompany = $("#report-company");
  const reportOrganization = $("#report-organization");
  const reportDate = $("#report-date");
  const reportTime = $("#report-time");
  const reportViolatorsWrap = $("#report-violators-wrap");
  const reportViolatorId = $("#report-violator-id");
  const btnReportOne = $("#btn-report-one");
  const btnReportAll = $("#btn-report-all");
  const playerPanel = $("#player-panel");
  const playerHint = $("#player-hint");
  const playerStage = $("#player-stage");
  const playerVideo = $("#player-video");
  const playerCanvas = $("#player-canvas");
  const btnPlayViol = $("#btn-play-viol");
  const btnPlayStart = $("#btn-play-start");
  const btnPlayStop = $("#btn-play-stop");
  const jobIdInput = $("#job-id-input");
  const btnResumeJob = $("#btn-resume-job");
  const jobIdHint = $("#job-id-hint");
  const apiUrlInput = $("#api-url-input");
  const btnApiUrlApply = $("#btn-api-url-apply");
  const btnApiUrlAuto = $("#btn-api-url-auto");
  const runFolderSelect = $("#run-folder-select");
  const btnRefreshRuns = $("#btn-refresh-runs");
  const runDirInput = $("#run-dir-input");
  const btnOpenRunDir = $("#btn-open-run-dir");
  const outputDirHint = $("#output-dir-hint");

  let selectedFile = null;
  let currentJob = null;
  let currentRunDir = null;
  let currentRunId = null;
  let currentViolators = [];
  let overlayHls = null;
  let overlayTimeline = null;
  let overlayEventIdx = 0;
  let overlayHlsOffset = 0;
  let overlayRaf = 0;
  let overlayUseRvfc = false;
  let overlayLastEvent = null;
  let overlayComposite = false;
  let pollTimer = null;
  let pollJobId = null;
  let buildPollTimer = null;
  let apiReady = false;
  let currentSettings = null;

  const JOB_STORAGE_KEY = "yolo_drt_web_last_job_id";
  const API_URL_STORAGE_KEY = "yolo_drt_web_api_url";

  function storedApiUrl() {
    try {
      return (localStorage.getItem(API_URL_STORAGE_KEY) || "").trim();
    } catch (_) {
      return "";
    }
  }

  function rememberApiUrl(url) {
    const v = String(url || "").trim();
    try {
      if (v) localStorage.setItem(API_URL_STORAGE_KEY, v);
      else localStorage.removeItem(API_URL_STORAGE_KEY);
    } catch (_) {
      /* ignore */
    }
    if (apiUrlInput && v) apiUrlInput.value = v;
  }

  async function applyApiUrl(raw, { auto = false } = {}) {
    const body = { api_url: auto ? "" : String(raw || "").trim() };
    const r = await fetch("/proxy/api-url", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    const url = String(data.api_url || "");
    if (data.mode === "manual") rememberApiUrl(url);
    else {
      rememberApiUrl("");
      if (apiUrlInput) apiUrlInput.value = url;
    }
    const reach = data.reachable ? (data.status || "ok") : "не отвечает";
    showToast(
      data.mode === "manual" ? "API задан вручную" : "API: авто",
      `${url} · ${reach}`,
      data.reachable ? "success" : "error",
      5000
    );
    await checkHealth();
    return data;
  }

  async function refreshLocalRuns() {
    try {
      const r = await fetch("/local/runs?limit=100");
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      if (outputDirHint) {
        outputDirHint.textContent = `output: ${data.output_dir || "—"}`;
      }
      if (!runFolderSelect) return;
      const cur = runFolderSelect.value;
      runFolderSelect.innerHTML =
        '<option value="">— выберите папку —</option>' +
        (data.runs || [])
          .map((row) => {
            const label = `${row.name || row.run_id} (${row.run_id})`;
            const val = row.run_dir;
            return `<option value="${encodeURIComponent(JSON.stringify(row))}">${label}</option>`;
          })
          .join("");
      if (cur) {
        for (const opt of runFolderSelect.options) {
          if (opt.value === cur) {
            runFolderSelect.value = cur;
            break;
          }
        }
      }
    } catch (e) {
      if (outputDirHint) outputDirHint.textContent = `output: ошибка — ${e}`;
    }
  }

  async function openRunFolder(runDir, runId) {
    const r = await fetch("/local/select-run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_dir: runDir, run_id: runId || null }),
    });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    currentRunDir = data.run_dir;
    currentRunId = data.run_id;
    if (runDirInput) runDirInput.value = data.run_dir;
    cardVideo.classList.remove("hidden");
    initReportDefaults();
    runMeta.innerHTML = `
      <dt>Run ID</dt><dd><code>${currentRunId || "—"}</code></dd>
      <dt>Папка</dt><dd><code>${currentRunDir || "—"}</code></dd>
    `;
    crossWarn.classList.add("hidden");
    btnBuild.onclick = () => startBuild(currentRunDir, currentRunId);
    await loadReportViolators(currentRunDir, currentRunId);
    await preparePlayer();
    showToast("Прогон открыт", currentRunId || currentRunDir);
    cardVideo?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function syncApiUrlFromServer() {
    try {
      const r = await fetch("/proxy/api-url");
      if (!r.ok) return;
      const data = await r.json();
      if (apiUrlInput) {
        apiUrlInput.value = data.override || data.api_url || storedApiUrl() || "http://127.0.0.1:8080";
      }
    } catch (_) {
      if (apiUrlInput && !apiUrlInput.value) {
        apiUrlInput.value = storedApiUrl() || "http://127.0.0.1:8080";
      }
    }
  }

  function rememberJobId(jobId) {
    const id = String(jobId || "").trim();
    if (!id) return;
    try {
      localStorage.setItem(JOB_STORAGE_KEY, id);
    } catch (_) {
      /* ignore quota / private mode */
    }
    if (jobIdInput) jobIdInput.value = id;
    if (jobIdHint) jobIdHint.textContent = `Текущая задача: ${id}`;
    try {
      const url = new URL(window.location.href);
      url.searchParams.set("job", id);
      window.history.replaceState({}, "", url);
    } catch (_) {
      /* ignore */
    }
  }

  function storedJobId() {
    try {
      return (localStorage.getItem(JOB_STORAGE_KEY) || "").trim();
    } catch (_) {
      return "";
    }
  }

  function jobIdFromUrl() {
    try {
      return (new URL(window.location.href).searchParams.get("job") || "").trim();
    } catch (_) {
      return "";
    }
  }

  async function resumeJobById(rawId, { quiet = false } = {}) {
    const jobId = String(rawId || "").trim();
    if (!jobId) {
      if (!quiet) showToast("Укажите job ID", "", "error", 4000);
      return;
    }
    await checkHealth();
    if (!apiReady) {
      if (!quiet) showToast("API недоступен", "Сначала поднимите Docker API", "error", 6000);
      return;
    }
    stopJobPoll();
    rememberJobId(jobId);
    pollJobId = jobId;
    jobProgressWrap.classList.remove("hidden");
    jobProgress.classList.add("indeterminate");
    jobProgress.style.width = "30%";
    jobStatus.textContent = `Загрузка задачи ${jobId}…`;
    btnRun.disabled = true;
    btnRun.textContent = "Ожидание…";
    setStatus("busy", "Открытие задачи…");
    if (!quiet) showToast("Открываю задачу", jobId);
    await pollJob(jobId);
  }

  /** Baked fast profile — same keys as app/config/ui_fast_profile.json (person∩helmet). */
  const FAST_PROFILE = {
    use_reid: false,
    use_sam_identity: true,
    use_offline_tracklet_link: true,
    tracklet_link_use_reid: true,
    gpu_full_batch: false,
    preload_video: false,
    frame_source_mode: "windowed",
    infer_batch_size: 64,
    realtime_mode: true,
    use_seg: false,
    sam_osnet_reentry: false,
    encode_mode: "manual",
    cross_check_enabled: true,
    cross_check_object_prompt: "helmet",
    cross_check_warning_text: "NO HELMET",
    cross_check_conf: 0.35,
    cross_check_min_intersection_px: 20,
    cross_check_min_iou: 0.03,
    cross_check_helmet_min_conf: 0.30,
    cross_check_min_violation_streak: 2,
    cross_check_verdict_history: 5,
    cross_check_draw_head_box: true,
    cross_check_draw_boxes: true,
    draw_boxes: true,
    draw_masks: false,
    draw_centers: false,
    // Overlay only — WEB must not force pose skeletons / pose model UX.
    draw_pose: false,
    pose_kpt_conf: 0.30,
    default_prompt: "person",
  };

  /** Poll interval — progress reads throttled slot; 2s avoids GIL noise with infer. */
  const JOB_POLL_MS = 2000;

  function initReportDefaults() {
    const now = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    if (reportDate && !reportDate.value) {
      reportDate.value = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
    }
    if (reportTime && !reportTime.value) {
      reportTime.value = `${pad(now.getHours())}:${pad(now.getMinutes())}`;
    }
    if (reportOrganization && !reportOrganization.value && reportCompany) {
      reportOrganization.value = reportCompany.value;
    }
  }

  function getReportDatetimeIso() {
    const d = reportDate?.value || "";
    const t = reportTime?.value || "12:00";
    return d ? `${d}T${t}` : new Date().toISOString();
  }

  function getReportPayload(stableId) {
    const sid = Number(stableId);
    const row = currentViolators.find((v) => Number(v.stable_id) === sid) || {};
    return {
      run_dir: currentRunDir,
      run_id: currentRunId,
      stable_id: sid,
      company: reportCompany?.value || "",
      organization: (reportOrganization?.value || reportCompany?.value || "").trim(),
      incident_datetime: getReportDatetimeIso(),
      source_video: (sourceVideoInput?.value || "").trim() || null,
      violation_count: Number(row.violation_count) || null,
      presence_frames: Number(row.presence_frames) || null,
    };
  }

  async function downloadWordReport(stableId, btn, { quiet } = {}) {
    if (!currentRunDir || !currentRunId) {
      showToast("Ошибка", "Сначала выполните анализ видео", "error");
      return false;
    }
    const prevText = btn?.textContent;
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Формирование…";
    }
    try {
      const r = await fetch("/report/word", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(getReportPayload(stableId)),
      });
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      const a = document.createElement("a");
      a.href = data.download_url;
      a.download = data.filename || "report.docx";
      document.body.appendChild(a);
      a.click();
      a.remove();
      if (!quiet) showToast("Отчёт готов", `Word-акт для нарушителя №${stableId}`);
      return true;
    } catch (e) {
      showToast("Ошибка отчёта", String(e.message || e), "error", 8000);
      return false;
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.textContent = prevText || "Скачать отчёт Word";
      }
    }
  }

  if (reportCompany && reportOrganization) {
    reportCompany.addEventListener("change", () => {
      if (!reportOrganization.value.trim() || reportOrganization.dataset.auto === "1") {
        reportOrganization.value = reportCompany.value;
        reportOrganization.dataset.auto = "1";
      }
    });
    reportOrganization.addEventListener("input", () => {
      reportOrganization.dataset.auto = "0";
    });
  }

  if (btnReportOne) {
    btnReportOne.addEventListener("click", () => {
      const sid = reportViolatorId?.value;
      if (!sid) {
        showToast("Выберите нарушителя", "Список ID появится после анализа", "error");
        return;
      }
      downloadWordReport(sid, btnReportOne);
    });
  }

  if (btnReportAll) {
    btnReportAll.addEventListener("click", () => downloadAllWordReports(btnReportAll));
  }

  async function downloadAllWordReports(btn) {
    const ids = [...(reportViolatorId?.options || [])]
      .map((o) => o.value)
      .filter(Boolean);
    if (!ids.length) {
      showToast("Нарушителей нет", "Сначала дождитесь анализа", "error");
      return;
    }
    const prevText = btn?.textContent;
    if (btn) btn.disabled = true;
    let ok = 0;
    try {
      for (let i = 0; i < ids.length; i++) {
        if (btn) btn.textContent = `Отчёт ${i + 1}/${ids.length}…`;
        if (await downloadWordReport(ids[i], null, { quiet: true })) ok += 1;
      }
      showToast("Отчёты готовы", `Скачано ${ok} из ${ids.length}`);
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.textContent = prevText || "Все нарушители";
      }
    }
  }

  function overlayQuery() {
    const q = new URLSearchParams({
      run_dir: currentRunDir || "",
      run_id: currentRunId || "",
    });
    const src = (sourceVideoInput?.value || "").trim();
    if (src) q.set("source_video", src);
    return q;
  }

  async function clientHevcOk() {
    try {
      if (window.VideoDecoder) {
        const r = await VideoDecoder.isConfigSupported({
          codec: "hvc1.1.6.L93.B0",
          codedWidth: 1280,
          codedHeight: 720,
        });
        if (r && r.supported) return true;
      }
    } catch (_) {
      /* ignore */
    }
    const probe = document.createElement("video");
    return probe.canPlayType('video/mp4; codecs="hvc1.1.6.L93.B0"') === "probably";
  }

  function stopOverlayLoop() {
    if (!overlayRaf) return;
    if (overlayUseRvfc && playerVideo?.cancelVideoFrameCallback) {
      try {
        playerVideo.cancelVideoFrameCallback(overlayRaf);
      } catch (_) {
        /* ignore */
      }
    } else {
      cancelAnimationFrame(overlayRaf);
    }
    overlayRaf = 0;
  }

  function destroyOverlayHls() {
    stopOverlayLoop();
    overlayComposite = false;
    overlayLastEvent = null;
    playerStage?.classList.remove("is-composite");
    if (overlayHls) {
      try {
        overlayHls.destroy();
      } catch (_) {
        /* ignore */
      }
      overlayHls = null;
    }
    if (playerCanvas) {
      const ctx = playerCanvas.getContext("2d");
      if (ctx) ctx.clearRect(0, 0, playerCanvas.width, playerCanvas.height);
    }
    if (playerVideo) {
      playerVideo.pause();
      playerVideo.removeAttribute("src");
      try {
        playerVideo.load();
      } catch (_) {
        /* ignore */
      }
    }
  }

  function mediaAbsTime() {
    return (overlayHlsOffset || 0) + (playerVideo?.currentTime || 0);
  }

  function findOverlayEvent(t) {
    const events = overlayTimeline?.events || [];
    if (!events.length) return overlayLastEvent;
    while (overlayEventIdx + 1 < events.length && events[overlayEventIdx + 1].t0 <= t) {
      overlayEventIdx += 1;
    }
    while (overlayEventIdx > 0 && events[overlayEventIdx].t0 > t) {
      overlayEventIdx -= 1;
    }
    const ev = events[overlayEventIdx];
    if (!ev) return overlayLastEvent;
    if (t < ev.t0 - 0.25) return overlayLastEvent;
    if (ev.t1 != null && t > ev.t1 + 1.5) return overlayLastEvent;
    overlayLastEvent = ev;
    return ev;
  }

  function drawOverlayBoxes() {
    if (!playerVideo || !playerCanvas) return;
    const ctx = playerCanvas.getContext("2d", { alpha: true });
    if (!ctx) return;
    const w = playerVideo.videoWidth || overlayTimeline?.width || 0;
    const h = playerVideo.videoHeight || overlayTimeline?.height || 0;
    if (!w || !h) return;
    if (playerCanvas.width !== w) playerCanvas.width = w;
    if (playerCanvas.height !== h) playerCanvas.height = h;
    ctx.clearRect(0, 0, w, h);
    if (playerVideo.readyState >= 2) {
      try {
        ctx.drawImage(playerVideo, 0, 0, w, h);
        if (!overlayComposite) {
          overlayComposite = true;
          playerStage?.classList.add("is-composite");
        }
      } catch (_) {
        overlayComposite = false;
        playerStage?.classList.remove("is-composite");
      }
    }
    const ev = findOverlayEvent(mediaAbsTime());
    if (!ev) return;
    const sx = w / Math.max(1, overlayTimeline?.width || w);
    const sy = h / Math.max(1, overlayTimeline?.height || h);
    ctx.lineJoin = "round";
    ctx.font = "bold 18px Segoe UI, system-ui, sans-serif";
    for (const b of ev.boxes || []) {
      const x = b.x * sx;
      const y = b.y * sy;
      const bw = b.w * sx;
      const bh = b.h * sy;
      const helmet = b.k === "h";
      const color = helmet ? "#38bdf8" : b.v ? "#ef4444" : "#22c55e";
      ctx.lineWidth = helmet ? 2 : b.v ? 5 : 3;
      ctx.strokeStyle = color;
      ctx.strokeRect(x, y, bw, bh);
      ctx.fillStyle = color;
      const label = helmet ? "helmet" : b.v ? `NO HELMET ID ${b.id}` : `ID ${b.id}`;
      ctx.fillText(label, x, Math.max(18, y - 6));
    }
  }

  function startOverlayLoop() {
    const loop = () => {
      drawOverlayBoxes();
      if (!playerVideo || playerVideo.paused || playerVideo.ended) {
        overlayRaf = 0;
        return;
      }
      if (playerVideo.requestVideoFrameCallback) {
        overlayUseRvfc = true;
        overlayRaf = playerVideo.requestVideoFrameCallback(loop);
      } else {
        overlayUseRvfc = false;
        overlayRaf = requestAnimationFrame(loop);
      }
    };
    stopOverlayLoop();
    overlayRaf = requestAnimationFrame(loop);
  }

  async function loadOverlayWindow(startSec, durationSec) {
    overlayTimeline = null;
    overlayEventIdx = 0;
    overlayLastEvent = null;
    const q = overlayQuery();
    const t0 = Math.max(0, Number(startSec) || 0);
    const span = Math.max(8, Number(durationSec) || 90);
    q.set("t0", String(Math.max(0, t0 - 2)));
    q.set("t1", String(t0 + span + 4));
    const r = await fetch(`/overlay/timeline?${q}`);
    if (!r.ok) throw new Error(await r.text());
    overlayTimeline = await r.json();
    overlayEventIdx = 0;
    const n = overlayTimeline?.events?.length || 0;
    if (!n) {
      showToast("Оверлей", "В этом окне нет пакетов детекции", "error", 8000);
    }
    drawOverlayBoxes();
    return n;
  }

  async function preparePlayer() {
    if (!playerPanel || !currentRunDir || !currentRunId) return;
    playerPanel.classList.remove("hidden");
    try {
      const r = await fetch(`/overlay/info?${overlayQuery()}`);
      if (!r.ok) throw new Error(await r.text());
      const info = await r.json();
      if (!info.has_source) {
        playerHint.textContent =
          "Исходник не найден. Положи RUNID_source.mp4 в папку прогона или укажи путь ниже.";
        return;
      }
      if (info.hevc) {
        const nativeHevc = await clientHevcOk();
        playerHint.textContent = nativeHevc
          ? "HEVC: этот браузер умеет декодировать сам (GPU клиента, не сервера). Если чёрный экран — превью на CPU."
          : "HEVC: на веб-сервере нет GPU. «Смотреть» — короткое CPU-превью H.264 (~90 с, 480p), без NVENC.";
      } else {
        playerHint.textContent = `Кодек ${info.codec || "h264"} — играем исходник в Chrome без перекодирования.`;
      }
    } catch (_) {
      playerHint.textContent = "Плеер: не удалось проверить исходник.";
    }
  }

  function attachHls(playlistUrl) {
    destroyOverlayHls();
    playerStage?.classList.remove("hidden");
    const HlsCtor = window.Hls;
    if (HlsCtor && HlsCtor.isSupported()) {
      overlayHls = new HlsCtor({
        enableWorker: true,
        lowLatencyMode: false,
        liveDurationInfinity: true,
      });
      overlayHls.loadSource(playlistUrl);
      overlayHls.attachMedia(playerVideo);
      overlayHls.on(HlsCtor.Events.MANIFEST_PARSED, () => {
        playerVideo?.play()?.catch(() => {});
        startOverlayLoop();
      });
      overlayHls.on(HlsCtor.Events.ERROR, (_ev, data) => {
        if (data?.fatal) {
          showToast("Плеер", data.details || "HLS error", "error", 8000);
        }
      });
      return;
    }
    if (playerVideo?.canPlayType("application/vnd.apple.mpegurl")) {
      playerVideo.src = playlistUrl;
      playerVideo.play()?.catch(() => {});
    } else {
      showToast("Плеер", "Нужен Chrome/Edge с hls.js", "error");
    }
  }

  async function playNativeSource(startSec) {
    destroyOverlayHls();
    overlayHlsOffset = 0;
    playerStage?.classList.remove("hidden");
    playerVideo.src = `/overlay/source?${overlayQuery()}`;
    await new Promise((resolve, reject) => {
      const onMeta = () => {
        playerVideo.removeEventListener("loadedmetadata", onMeta);
        playerVideo.removeEventListener("error", onErr);
        resolve();
      };
      const onErr = () => {
        playerVideo.removeEventListener("loadedmetadata", onMeta);
        playerVideo.removeEventListener("error", onErr);
        reject(new Error("native"));
      };
      playerVideo.addEventListener("loadedmetadata", onMeta);
      playerVideo.addEventListener("error", onErr);
      playerVideo.load();
    });
    if (startSec > 0.2 && Number.isFinite(playerVideo.duration)) {
      const target = Math.min(startSec, Math.max(0, playerVideo.duration - 0.2));
      await new Promise((resolve) => {
        let settled = false;
        const done = () => {
          if (settled) return;
          settled = true;
          playerVideo.removeEventListener("seeked", done);
          resolve();
        };
        playerVideo.addEventListener("seeked", done);
        try {
          playerVideo.currentTime = target;
        } catch (_) {
          done();
          return;
        }
        setTimeout(done, 1800);
      });
    }
    await playerVideo.play();
  }

  async function playInBrowser(btn, { fromStart } = {}) {
    if (!currentRunDir || !currentRunId) {
      showToast("Ошибка", "Сначала выполните анализ", "error");
      return;
    }
    const prevText = btn?.textContent;
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Старт…";
    }
    try {
      const infoR = await fetch(`/overlay/info?${overlayQuery()}`);
      const info = infoR.ok ? await infoR.json() : {};
      let sid = fromStart ? null : reportViolatorId?.value;
      if (!fromStart && !sid && currentViolators.length) {
        sid = String(currentViolators[0].stable_id);
      }
      let startSec = fromStart ? 0 : null;
      if (startSec == null && sid) {
        const s = await fetch(`/overlay/seek?${overlayQuery()}&stable_id=${encodeURIComponent(sid)}`);
        if (s.ok) {
          const body = await s.json();
          startSec = Number(body.start_sec) || 0;
        } else {
          startSec = 0;
        }
      }
      startSec = Number(startSec) || 0;
      const nativeOk = !info.hevc || (await clientHevcOk());
      if (nativeOk && info.has_source) {
        try {
          await playNativeSource(startSec);
          overlayHlsOffset = 0;
          let nEv = 0;
          try {
            nEv = await loadOverlayWindow(startSec, 120);
          } catch (err) {
            showToast("Оверлей", String(err.message || err), "error", 8000);
          }
          startOverlayLoop();
          showToast("Плеер", "Исходник в браузере, рамки рисуются на кадре");
          playerHint.textContent = `Исходник с ${startSec.toFixed(1)}с. Рамки: ${nEv || 0} ключ.кадров.`;
          return;
        } catch (_) {
          if (!info.hevc) throw new Error("Не удалось открыть исходник");
        }
      }
      const r = await fetch("/overlay/hls/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_dir: currentRunDir,
          run_id: currentRunId,
          source_video: (sourceVideoInput?.value || "").trim() || null,
          stable_id: sid ? Number(sid) : null,
          start_sec: fromStart ? 0 : startSec,
          duration_sec: 90,
        }),
      });
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      overlayHlsOffset = Number(data.start_sec) || 0;
      attachHls(data.playlist_url);
      let nEv = 0;
      try {
        nEv = await loadOverlayWindow(overlayHlsOffset, Number(data.duration_sec) || 90);
      } catch (err) {
        showToast("Оверлей", String(err.message || err), "error", 8000);
      }
      playerHint.textContent = `CPU-превью ~${Number(data.duration_sec || 90)}с с ${overlayHlsOffset.toFixed(1)}с. Рамки: ${nEv || 0} ключ.кадров.`;
      showToast("Плеер", "Превью на CPU, детекции рисуются на кадре");
    } catch (e) {
      showToast("Плеер", String(e.message || e), "error", 10000);
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.textContent = prevText;
      }
    }
  }

  async function stopPlayer() {
    destroyOverlayHls();
    playerStage?.classList.add("hidden");
    try {
      await fetch("/overlay/hls/stop", { method: "POST" });
    } catch (_) {
      /* ignore */
    }
  }

  async function loadReportViolators(runDir, runId) {
    if (!reportViolatorsWrap || !reportViolatorId) return;
    try {
      const q = new URLSearchParams({ run_dir: runDir, run_id: runId });
      const r = await fetch(`/report/violators?${q}`);
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      const list = data.violators || [];
      currentViolators = list;
      if (!list.length) {
        reportViolatorsWrap.classList.add("hidden");
        showToast("Нарушителей нет", "В этом прогоне нет NO HELMET (или порог отсёк всех)", "error", 7000);
        return;
      }
      reportViolatorId.innerHTML = list
        .map(
          (v) =>
            `<option value="${v.stable_id}">№ ${v.stable_id} — ${v.violation_count} без каски, ${v.presence_frames} кадр.</option>`
        )
        .join("");
      reportViolatorsWrap.classList.remove("hidden");
    } catch {
      reportViolatorsWrap.classList.add("hidden");
    }
  }

  async function fetchSettings() {
    const r = await fetch("/proxy/settings");
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    currentSettings = { ...FAST_PROFILE, ...(data.settings || {}) };
    return currentSettings;
  }

  function fastProfilePatch(base) {
    const out = { ...FAST_PROFILE, ...(base || {}) };
    // Pin person-detect + helmet cross-check; never force pose overlay from WEB.
    out.draw_pose = false;
    out.cross_check_enabled = true;
    out.default_prompt = (out.default_prompt || "person").trim() || "person";
    if (out.cross_check_object_prompt == null || out.cross_check_object_prompt === "") {
      out.cross_check_object_prompt = "helmet";
    }
    return out;
  }

  async function applyFastProfile() {
    // Docker API has no /v1/settings — use baked FAST_PROFILE and continue to upload.
    try {
      if (!currentSettings) await fetchSettings();
    } catch {
      currentSettings = { ...FAST_PROFILE };
    }
    const patch = fastProfilePatch(currentSettings);
    try {
      const r = await fetch("/proxy/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          settings: patch,
          reload_processor: false,
          ui_equivalent: true,
        }),
      });
      if (r.ok) {
        const data = await r.json();
        currentSettings = data.settings || patch;
        return currentSettings;
      }
    } catch {
      /* ignore — Docker / offline settings */
    }
    currentSettings = patch;
    return currentSettings;
  }

  async function ensureProcessorReady() {
    // Desktop: POST bootstrap. Docker: endpoint missing — just re-check /health.
    try {
      const r = await fetch("/proxy/admin/bootstrap", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ force: false }),
      });
      if (r.ok) return r.json();
    } catch {
      /* fall through */
    }
    return null;
  }

  function formatDuration(sec) {
    const s = Math.max(0, Number(sec) || 0);
    if (s <= 0) return "";
    if (s < 60) return `${Math.round(s)} с`;
    const m = Math.floor(s / 60);
    const r = Math.round(s % 60);
    return r > 0 ? `${m} мин ${r} с` : `${m} мин`;
  }

  function formatEta(sec) {
    const text = formatDuration(sec);
    return text ? `осталось ~${text}` : "";
  }

  function showToast(title, detail, type = "success", durationMs = 6000) {
    if (!toastStack) return;
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.innerHTML = `
      <div class="toast-icon" aria-hidden="true">${type === "success" ? "✓" : "!"}</div>
      <div class="toast-body">
        <strong>${title}</strong>
        ${detail ? `<span>${detail}</span>` : ""}
      </div>`;
    toastStack.appendChild(toast);
    setTimeout(() => {
      toast.classList.add("toast-hide");
      toast.addEventListener("animationend", () => toast.remove(), { once: true });
    }, durationMs);
  }

  function notifyBrowser(title, body) {
    if (!("Notification" in window) || Notification.permission !== "granted") return;
    try {
      new Notification(title, { body, icon: "/static/favicon.ico" });
    } catch {
      /* ignore */
    }
  }

  async function requestNotifyPermission() {
    if (!("Notification" in window) || Notification.permission !== "default") return;
    try {
      await Notification.requestPermission();
    } catch {
      /* ignore */
    }
  }

  function formatJobStatusLine(job) {
    const p = job.progress || {};
    const status = job.status || "?";
    const total = Number(p.total) || 0;
    const current = Number(p.current) || 0;
    const pct =
      p.percent != null && Number(p.percent) > 0
        ? Number(p.percent)
        : total > 0
          ? (100 * current) / total
          : 0;
    const elapsed = Number(p.elapsed_sec) || 0;
    const fps = Number(p.fps) || 0;

    if (status === "queued") return "В очереди…";
    if (status === "error" || status === "cancelled") {
      return job.result?.error || job.error || "Ошибка обработки";
    }
    if (status === "done") return formatJobDoneLine(job);

    const parts = [];
    if (total > 0) {
      parts.push(`${current} / ${total} кадров`);
      if (pct > 0) parts.push(`${pct.toFixed(0)}%`);
    } else {
      parts.push("Подготовка…");
    }
    if (elapsed > 0) parts.push(`прошло ${formatDuration(elapsed)}`);
    const etaSec = Number(p.eta_seconds) || 0;
    if (etaSec > 0) parts.push(`осталось ~${formatDuration(etaSec)}`);
    if (fps > 0) parts.push(`${fps.toFixed(1)} кадр/с`);
    return parts.join(" · ");
  }

  function formatJobDoneLine(job) {
    const p = job.progress || {};
    const res = job.result || {};
    const rec = res.record || {};
    const st = rec.stats_summary || {};
    const inferSec =
      Number(st.elapsed_infer_sec) ||
      Number(st.stage_gpu_infer_sec) ||
      Number(res.elapsed_sec) ||
      Number(p.elapsed_sec) ||
      0;
    const pass2Sec = Number(st.elapsed_pass2_sec) || 0;
    const wallSec = Number(res.elapsed_sec) || Number(rec.elapsed_sec) || Number(p.elapsed_sec) || 0;
    const frames = st.processed_frame_count || st.source_frame_count || res.frames;
    const parts = ["Готово"];
    if (frames) parts.push(`${frames} кадров`);
    if (inferSec > 0) parts.push(`анализ ${formatDuration(inferSec)}`);
    if (pass2Sec > 0.05) parts.push(`склейка ${formatDuration(pass2Sec)}`);
    if (wallSec > inferSec + 0.3) parts.push(`всего ${formatDuration(wallSec)}`);
    const fpsVid = rec.fps || st.video_fps;
    if (inferSec > 0 && fpsVid > 0 && st.source_frame_count) {
      const vidSec = st.source_frame_count / fpsVid;
      parts.push(`${(inferSec / vidSec).toFixed(1)}× от видео`);
    }
    const violations = st.cross_check_violations;
    if (violations != null && Number(violations) >= 0) {
      parts.push(`${violations} нарушений`);
    }
    return parts.join(" · ");
  }

  function setJobProgressBar(p, status) {
    const total = Number(p.total) || 0;
    const current = Number(p.current) || 0;
    const pct =
      p.percent != null && Number(p.percent) > 0
        ? Number(p.percent)
        : total > 0
          ? (100 * current) / total
          : 0;
    const phase = String(p.phase || "").toLowerCase();
    const indeterminate =
      status === "queued" ||
      status === "running" &&
        (total <= 0 || phase === "staging" || phase === "warmup" || (current <= 0 && pct <= 0));

    jobProgressWrap.classList.remove("hidden");
    jobProgress.classList.toggle("indeterminate", Boolean(indeterminate && status === "running" && pct < 1));
    if (!indeterminate || pct > 0) {
      jobProgress.style.width = `${Math.min(100, Math.max(0, pct))}%`;
    } else {
      jobProgress.style.width = "30%";
    }
    return { total, current, pct, phase };
  }

  function stopJobPoll() {
    if (pollTimer) {
      clearTimeout(pollTimer);
      pollTimer = null;
    }
    pollJobId = null;
    jobProgress.classList.remove("indeterminate");
  }

  function scheduleJobPoll(jobId) {
    pollJobId = jobId;
    if (pollTimer) clearTimeout(pollTimer);
    pollTimer = setTimeout(() => pollJob(jobId), JOB_POLL_MS);
  }

  function setStatus(state, text) {
    if (!statusPill || !statusText) return;
    statusPill.classList.remove("ready", "busy", "error");
    if (state) statusPill.classList.add(state);
    statusText.textContent = text;
  }

  function setRunEnabled(on) {
    btnRun.disabled = !on || !selectedFile;
  }

  async function checkHealth() {
    setStatus("busy", "Проверка…");
    try {
      let r = await fetch("/proxy/health");
      if (!r.ok) throw new Error(await r.text());
      let data = await r.json();
      if (data.status !== "ready") {
        setStatus("busy", "Загрузка моделей…");
        await ensureProcessorReady();
        // Docker may stay in building_engines a while — poll health a few times.
        for (let i = 0; i < 15 && data.status !== "ready"; i++) {
          await new Promise((res) => setTimeout(res, 2000));
          r = await fetch("/proxy/health");
          if (!r.ok) throw new Error(await r.text());
          data = await r.json();
          if (data.status === "building_engines" || data.status === "starting") {
            setStatus("busy", "Сборка TensorRT / загрузка…");
          }
        }
      }
      const ok = data.status === "ready";
      apiReady = ok;
      const target = data.api_proxy_target ? ` · ${data.api_proxy_target}` : "";
      setStatus(
        ok ? "ready" : "error",
        ok ? `Готово к работе${target}` : `Система недоступна (${data.status || "?"})`
      );
      setRunEnabled(ok);
      if (ok && !currentSettings) {
        try {
          await fetchSettings();
        } catch {
          currentSettings = { ...FAST_PROFILE };
        }
      }
    } catch (e) {
      apiReady = false;
      setStatus("error", "Нет подключения к серверу");
      setRunEnabled(false);
      console.warn("checkHealth failed:", e);
    }
  }

  function onFile(f) {
    if (!f) return;
    selectedFile = f;
    fileName.textContent = f.name;
    btnRun.disabled = !apiReady;
    if (!apiReady) checkHealth();
  }

  pickFile.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => onFile(fileInput.files[0]));

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
    const f = e.dataTransfer.files[0];
    if (f) onFile(f);
  });

  async function pollJob(jobId) {
    if (pollJobId && pollJobId !== jobId) return;
    try {
      const r = await fetch(`/proxy/jobs/${jobId}`);
      if (!r.ok) throw new Error(await r.text());
      const job = await r.json();
      const p = job.progress || {};
      setJobProgressBar(p, job.status);
      jobStatus.textContent = formatJobStatusLine(job);

      if (job.status === "running") {
        const elapsed = Number(p.elapsed_sec) || 0;
        const eta = Number(p.eta_seconds) || 0;
        let pill = "Анализ видео…";
        if (elapsed > 0 && eta > 0) {
          pill = `${formatDuration(elapsed)} · ${formatEta(eta)}`;
        } else if (elapsed > 0) {
          pill = `прошло ${formatDuration(elapsed)}`;
        }
        setStatus("busy", pill);
      }

      if (job.status === "done") {
        stopJobPoll();
        jobProgress.style.width = "100%";
        jobStatus.textContent = formatJobDoneLine(job);
        currentJob = job;
        rememberJobId(jobId);
        showVideoSection(job);
        btnRun.disabled = false;
        btnRun.textContent = "Начать анализ";
        setStatus("ready", "Готово");
        const doneDetail = formatJobDoneLine(job).replace(/^Готово · /, "");
        showToast("Анализ завершён", doneDetail || "Можно скачать отчёты Word");
        notifyBrowser("Анализ завершён", doneDetail || "Можно скачать отчёты Word");
        cardVideo?.scrollIntoView({ behavior: "smooth", block: "start" });
        return;
      }
      if (job.status === "error" || job.status === "cancelled") {
        stopJobPoll();
        jobStatus.textContent = job.result?.error || job.error || "Ошибка обработки";
        btnRun.disabled = false;
        btnRun.textContent = "Начать анализ";
        setStatus("error", "Ошибка при обработке");
        showToast("Ошибка обработки", job.result?.error || job.error || "", "error", 8000);
        return;
      }
      rememberJobId(jobId);
      scheduleJobPoll(jobId);
    } catch (e) {
      const msg = String(e);
      jobStatus.textContent = msg;
      if (/404|not found|Job not found/i.test(msg)) {
        stopJobPoll();
        btnRun.disabled = false;
        btnRun.textContent = "Начать анализ";
        setStatus("error", "Задача не найдена");
        showToast(
          "Задача не найдена",
          "API мог перезапуститься — job ID живёт в памяти контейнера. Запустите анализ снова.",
          "error",
          9000
        );
        return;
      }
      scheduleJobPoll(jobId);
    }
  }

  async function resolveHostRun(runDir, runId, jobId) {
    const r = await fetch("/local/select-run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        run_dir: runDir,
        run_id: runId || null,
        job_id: jobId || null,
      }),
    });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  }

  async function showVideoSection(job) {
    const res = job.result || {};
    const record = res.record || {};
    const stats = record.stats_summary || {};
    const pipeline = record.pipeline || {};
    const inferSec =
      Number(stats.elapsed_infer_sec) ||
      Number(res.elapsed_sec) ||
      0;
    const pass2Sec = Number(stats.elapsed_pass2_sec) || 0;
    const violations = stats.cross_check_violations ?? pipeline.cross_check_violations ?? "—";
    const jobId = job.job_id || currentJob?.job_id || pollJobId || null;

    cardVideo.classList.remove("hidden");
    initReportDefaults();

    let runDir = res.out_dir || null;
    let runId = res.run_id || null;
    let resolveErr = "";
    if (runDir) {
      try {
        const mapped = await resolveHostRun(runDir, runId, jobId);
        runDir = mapped.run_dir || runDir;
        runId = mapped.run_id || runId;
      } catch (e) {
        resolveErr = String(e);
        showToast(
          "Папка прогона не найдена на диске",
          "WEB попробует скачать артефакты с API по job id. Проверь, что Docker запущен и job ещё в памяти.",
          "error",
          12000
        );
      }
    }

    currentRunDir = runDir;
    currentRunId = runId;
    if (runDirInput && runDir) runDirInput.value = runDir;
    try {
      await refreshLocalRuns();
      if (runFolderSelect && (runId || runDir)) {
        for (const opt of runFolderSelect.options) {
          if (!opt.value) continue;
          try {
            const row = JSON.parse(decodeURIComponent(opt.value));
            if (
              (runId && row.run_id === runId) ||
              (runDir && row.run_dir === runDir) ||
              (runId && String(row.name || "") === String(runId))
            ) {
              runFolderSelect.value = opt.value;
              break;
            }
          } catch (_) {
            /* ignore bad option */
          }
        }
      }
    } catch (_) {
      /* refresh optional */
    }
    if (currentRunDir && currentRunId && !resolveErr) {
      loadReportViolators(currentRunDir, currentRunId);
      preparePlayer();
    }
    runMeta.innerHTML = `
      <dt>Job ID</dt><dd><code>${jobId || "—"}</code></dd>
      <dt>Run ID</dt><dd><code>${currentRunId || "—"}</code></dd>
      <dt>Папка на диске</dt><dd><code>${currentRunDir || "—"}</code></dd>
      <dt>Кадров обработано</dt><dd>${res.frames ?? stats.processed_frame_count ?? "—"}</dd>
      <dt>Нарушений (без каски)</dt><dd>${violations}</dd>
      <dt>Время анализа (YOLO)</dt><dd>${inferSec > 0 ? formatDuration(inferSec) : "—"}</dd>
      <dt>Склейка ID (Pass2)</dt><dd>${pass2Sec > 0.05 ? formatDuration(pass2Sec) : "—"}</dd>
      <dt>Файл</dt><dd>${record.input_path ? record.input_path.split(/[/\\]/).pop() : selectedFile?.name || "—"}</dd>
    `;

    const cc = pipeline.cross_check_enabled;
    const overlayCc = (record.pipeline || {}).cross_check_enabled;
    const hasCc = cc || overlayCc;
    if (resolveErr) {
      crossWarn.textContent = resolveErr;
      crossWarn.classList.remove("hidden");
    } else if (!hasCc) {
      crossWarn.textContent =
        "Проверка касок была отключена в этом прогоне. Запустите анализ заново.";
      crossWarn.classList.remove("hidden");
    } else {
      crossWarn.classList.add("hidden");
    }

    btnBuild.onclick = () => startBuild(currentRunDir, currentRunId, jobId);
    cardVideo?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  btnRun.addEventListener("click", async () => {
    if (!selectedFile) return;
    requestNotifyPermission();
    stopJobPoll();
    btnRun.disabled = true;
    btnRun.textContent = "Подготовка…";
    setStatus("busy", "Подготовка…");
    jobProgressWrap.classList.remove("hidden");
    jobProgress.classList.remove("indeterminate");
    jobProgress.style.width = "0%";
    jobStatus.textContent = "Настройка профиля…";

    try {
      await applyFastProfile();
      await checkHealth();
      if (!apiReady) throw new Error("Сервер недоступен");

      btnRun.textContent = "Загрузка…";
      jobStatus.textContent = "Загрузка видео…";

      const patch = fastProfilePatch(currentSettings);
      const prompt = (patch.default_prompt || "person").trim() || "person";
      const fd = new FormData();
      fd.append("file", selectedFile);
      fd.append("prompt", prompt);
      const maxSec = patch.max_duration_seconds;
      if (maxSec != null && Number(maxSec) > 0) {
        fd.append("max_duration_seconds", String(maxSec));
      }

      const r = await fetch("/proxy/jobs/upload", { method: "POST", body: fd });
      if (!r.ok) throw new Error(await r.text());
      const { job_id } = await r.json();
      jobStatus.textContent = "Анализ видео…";
      jobProgress.classList.add("indeterminate");
      jobProgress.style.width = "30%";
      rememberJobId(job_id);
      pollJobId = job_id;
      pollJob(job_id);
    } catch (e) {
      stopJobPoll();
      jobStatus.textContent = String(e);
      btnRun.disabled = false;
      btnRun.textContent = "Начать анализ";
    }
  });

  function renderVideoPlayers(videos) {
    if (!videos || !videos.length) {
      videoList.classList.add("hidden");
      videoList.innerHTML = "";
      return;
    }
    videoList.classList.remove("hidden");
    videoList.innerHTML = videos
      .map(
        (v) => `
      <article class="video-card" data-stable-id="${v.stable_id}">
        <h3>Нарушитель #${v.stable_id}${v.violation_count != null ? ` · ${v.violation_count} без каски` : ""}${v.presence_frames != null ? ` · ${v.presence_frames} кадр. в кадре` : ""}</h3>
        <video controls preload="metadata" playsinline src="${encodeURI(v.video_url)}"></video>
        <a class="btn btn-download" href="${encodeURI(v.video_url)}" download="${v.video_name}">Скачать MP4</a>
        <button type="button" class="btn btn-report" data-report-id="${v.stable_id}">Скачать отчёт Word</button>
      </article>`
      )
      .join("");
    videoList.querySelectorAll(".btn-report").forEach((btn) => {
      btn.addEventListener("click", () => downloadWordReport(btn.dataset.reportId, btn));
    });
  }

  function finishBuildPoll(data, ok, message) {
    clearInterval(buildPollTimer);
    buildPollTimer = null;
    buildStatus.textContent = message;
    if (ok && (data.videos || []).length) {
      renderVideoPlayers(data.videos);
      showToast("Клипы готовы", `Собрано ${data.videos.length} видео`);
      notifyBrowser("Клипы готовы", `Собрано ${data.videos.length} видео`);
    }
    btnBuild.disabled = false;
    btnBuild.textContent = "Собрать клипы NO HELMET";
  }

  async function pollBuild(buildId) {
    try {
      const r = await fetch(`/build-video/${buildId}`);
      const data = await r.json();
      const prog = data.progress || {};
      const logs = data.logs || [];

      if (data.status === "running") {
        if (prog.total > 0) {
          buildProgress.style.width = `${Math.min(100, (100 * prog.done) / prog.total)}%`;
          buildStatus.textContent = `Encode ${prog.done}/${prog.total}`;
        } else if (logs.length) {
          buildStatus.textContent = logs[logs.length - 1];
        }
        return;
      }

      if (data.status === "done" && (data.videos || []).length) {
        finishBuildPoll(
          data,
          true,
          `Готово: ${data.videos.length} клип(ов)`
        );
        return;
      }

      if (data.status === "done") {
        finishBuildPoll(
          data,
          false,
          "Нарушений не найдено — клипы не созданы"
        );
        return;
      }

      if (data.status === "error") {
        const err = data.error || logs[logs.length - 1] || "Encode error";
        finishBuildPoll(data, false, err);
      }
    } catch (e) {
      finishBuildPoll({}, false, String(e));
    }
  }

  async function startBuild(runDir, runId, jobId) {
    if (!runDir || !runId) return;
    btnBuild.disabled = true;
    btnBuild.textContent = "Сборка…";
    buildProgressWrap.classList.remove("hidden");
    videoList.classList.add("hidden");
    videoList.innerHTML = "";
    buildProgress.style.width = "0%";
    buildStatus.textContent = "Старт…";

    try {
      const r = await fetch("/build-video", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_dir: runDir,
          run_id: runId,
          source_video: (sourceVideoInput?.value || "").trim() || null,
          job_id: jobId || currentJob?.job_id || pollJobId || null,
        }),
      });
      if (!r.ok) throw new Error(await r.text());
      const { build_id } = await r.json();
      buildPollTimer = setInterval(() => pollBuild(build_id), 1200);
      pollBuild(build_id);
    } catch (e) {
      buildStatus.textContent = String(e);
      btnBuild.disabled = false;
      btnBuild.textContent = "Собрать клипы NO HELMET";
    }
  }

  btnResumeJob?.addEventListener("click", () => {
    resumeJobById(jobIdInput?.value || storedJobId());
  });
  jobIdInput?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      resumeJobById(jobIdInput.value);
    }
  });

  btnApiUrlApply?.addEventListener("click", async () => {
    try {
      await applyApiUrl(apiUrlInput?.value || "");
    } catch (e) {
      showToast("API URL", String(e), "error", 7000);
    }
  });
  btnApiUrlAuto?.addEventListener("click", async () => {
    try {
      await applyApiUrl("", { auto: true });
    } catch (e) {
      showToast("API URL", String(e), "error", 7000);
    }
  });
  apiUrlInput?.addEventListener("keydown", async (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      try {
        await applyApiUrl(apiUrlInput.value);
      } catch (err) {
        showToast("API URL", String(err), "error", 7000);
      }
    }
  });

  btnRefreshRuns?.addEventListener("click", () => refreshLocalRuns());
  runFolderSelect?.addEventListener("change", async () => {
    const raw = runFolderSelect.value;
    if (!raw) return;
    try {
      const row = JSON.parse(decodeURIComponent(raw));
      await openRunFolder(row.run_dir, row.run_id);
    } catch (e) {
      showToast("Прогон", String(e), "error", 7000);
    }
  });
  btnOpenRunDir?.addEventListener("click", async () => {
    try {
      await openRunFolder(runDirInput?.value || "", null);
    } catch (e) {
      showToast("Прогон", String(e), "error", 7000);
    }
  });
  runDirInput?.addEventListener("keydown", async (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      try {
        await openRunFolder(runDirInput.value, null);
      } catch (err) {
        showToast("Прогон", String(err), "error", 7000);
      }
    }
  });

  async function bootstrapResume() {
    const fromUrl = jobIdFromUrl();
    const fromStore = storedJobId();
    const id = fromUrl || fromStore;
    if (jobIdInput && id) jobIdInput.value = id;
    if (jobIdHint && id) jobIdHint.textContent = `Сохранена задача: ${id}`;
    if (!id) return;
    await resumeJobById(id, { quiet: true });
  }

  async function bootstrap() {
    initReportDefaults();
    const saved = storedApiUrl();
    if (apiUrlInput) {
      apiUrlInput.value = saved || "http://127.0.0.1:8080";
    }
    try {
      if (saved) {
        await applyApiUrl(saved);
      } else {
        await syncApiUrlFromServer();
        await checkHealth();
      }
    } catch (e) {
      console.warn("API bootstrap failed:", e);
      await checkHealth();
    }
    await refreshLocalRuns();
    await bootstrapResume();
  }

  btnPlayViol?.addEventListener("click", () => playInBrowser(btnPlayViol));
  btnPlayStart?.addEventListener("click", () => playInBrowser(btnPlayStart, { fromStart: true }));
  btnPlayStop?.addEventListener("click", () => stopPlayer());
  playerVideo?.addEventListener("timeupdate", drawOverlayBoxes);
  playerVideo?.addEventListener("seeked", drawOverlayBoxes);
  playerVideo?.addEventListener("play", startOverlayLoop);
  sourceVideoInput?.addEventListener("change", () => {
    if (currentRunDir && currentRunId) preparePlayer();
  });

  bootstrap();
})();
