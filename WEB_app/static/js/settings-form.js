/**
 * PipelineSettings form — same groups as desktop Pipeline tab + Advanced.
 */
(() => {
  const SECTIONS = [
    {
      id: "models",
      title: "Models & output",
      open: true,
      fields: [
        { key: "detect_model", label: "Detect model", type: "model", group: "detect" },
        { key: "seg_model", label: "Seg model", type: "model", group: "seg" },
        { key: "reid_model", label: "ReID model", type: "model", group: "reid" },
        { key: "cross_check_model", label: "Cross-check model", type: "model", group: "detect" },
        { key: "use_seg", label: "Use segmentation", type: "bool" },
        { key: "use_sam_identity", label: "SAM identity (masklet)", type: "bool" },
        {
          key: "use_reid",
          label: "Live ReID (OSNet) — keep OFF for speed; Pass2 uses OSNet at end",
          type: "bool",
        },
        { key: "sam_osnet_reentry", label: "OSNet long re-entry (with SAM)", type: "bool" },
        { key: "use_offline_tracklet_link", label: "Pass2 offline tracklet link", type: "bool" },
        { key: "tracklet_link_use_reid", label: "Pass2 OSNet ReID (end of video)", type: "bool" },
        {
          key: "tracklet_link_max_gap_frames",
          label: "Pass2 max gap frames",
          type: "int",
          min: 1,
          max: 5000,
        },
        {
          key: "tracklet_link_min_sim",
          label: "Pass2 min similarity",
          type: "float",
          min: 0.1,
          max: 0.99,
          step: 0.05,
        },
        { key: "output_dir", label: "Output directory", type: "text" },
        { key: "default_prompt", label: "Class filter prompt", type: "text" },
        {
          key: "max_duration_seconds",
          label: "Max seconds (0 = всё видео)",
          type: "float",
          min: 0,
          max: 86400,
          step: 1,
        },
      ],
    },
    {
      id: "thresholds",
      title: "Thresholds",
      open: true,
      fields: [
        { key: "detect_conf", label: "Detect conf", type: "float", min: 0.01, max: 1, step: 0.05 },
        { key: "seg_conf", label: "Seg conf", type: "float", min: 0.01, max: 1, step: 0.05 },
        { key: "appearance_thresh", label: "ReID appearance", type: "float", min: 0.1, max: 0.99, step: 0.05 },
        { key: "track_buffer", label: "Track buffer", type: "int", min: 10, max: 600 },
        { key: "match_iou_min", label: "Match IoU", type: "float", min: 0.1, max: 0.95, step: 0.05 },
        {
          key: "sam_match_iou",
          label: "SAM match IoU",
          type: "float",
          min: 0.05,
          max: 0.95,
          step: 0.05,
        },
      ],
    },
    {
      id: "gpu",
      title: "GPU pipeline",
      open: true,
      fields: [
        {
          key: "inference_device",
          label: "Inference device",
          type: "select",
          options: [
            { value: "cuda", label: "GPU (CUDA)" },
            { value: "cpu", label: "CPU" },
          ],
        },
        { key: "realtime_mode", label: "Realtime ~1×", type: "bool" },
        { key: "frame_stride", label: "Frame stride (2 = половина кадров, 0=auto)", type: "int", min: 0, max: 32 },
        { key: "gpu_full_batch", label: "Full video GPU batch", type: "bool" },
        { key: "infer_batch_size", label: "Infer batch size (0=auto)", type: "int", min: 0, max: 8192 },
        { key: "gpu_queue_depth", label: "GPU queue depth", type: "int", min: 1, max: 8 },
        { key: "decode_prefetch", label: "Batch prefetch queue", type: "int", min: 1, max: 16 },
        { key: "preload_video", label: "Preload whole video", type: "bool" },
        {
          key: "frame_source_mode",
          label: "Frame source mode",
          type: "select",
          options: [
            { value: "auto", label: "auto" },
            { value: "preload", label: "preload" },
            { value: "stream", label: "stream" },
            { value: "windowed", label: "windowed (fast)" },
          ],
        },
        { key: "smart_ram_budget", label: "Smart RAM limit", type: "bool" },
        { key: "max_process_ram_gb", label: "Process RAM limit (GB)", type: "float", min: 4, max: 128, step: 1 },
      ],
    },
    {
      id: "trt",
      title: "TensorRT",
      fields: [
        { key: "use_tensorrt", label: "Use TensorRT (.engine)", type: "bool" },
        { key: "tensorrt_imgsz", label: "TRT imgsz", type: "int", min: 320, max: 1280, step: 32 },
        { key: "tensorrt_max_batch", label: "TRT max batch", type: "int", min: 1, max: 256 },
        { key: "tensorrt_fp16", label: "TRT FP16", type: "bool" },
        { key: "tensorrt_workspace_gb", label: "TRT workspace (GB)", type: "float", min: 0.5, max: 16, step: 0.5 },
        { key: "tensorrt_autocast_fast", label: "Fast AutoCast (TRT 11)", type: "bool" },
      ],
    },
    {
      id: "cross",
      title: "Cross-check (helmet)",
      open: true,
      fields: [
        { key: "cross_check_enabled", label: "Enable cross-check", type: "bool" },
        { key: "cross_check_object_prompt", label: "Object class prompt", type: "text" },
        { key: "cross_check_warning_text", label: "Warning text", type: "text" },
        { key: "cross_check_conf", label: "Cross-check conf", type: "float", min: 0.01, max: 1, step: 0.05 },
        { key: "cross_check_draw_head_box", label: "Draw head box", type: "bool" },
        { key: "cross_check_draw_boxes", label: "Draw cross-check boxes", type: "bool" },
      ],
    },
    {
      id: "overlay",
      title: "Overlay",
      fields: [
        { key: "overlay_alpha", label: "Overlay alpha", type: "float", min: 0.1, max: 0.9, step: 0.05 },
        { key: "draw_boxes", label: "Draw boxes", type: "bool" },
        { key: "draw_masks", label: "Draw masks", type: "bool" },
        { key: "draw_centers", label: "Draw centers", type: "bool" },
        { key: "draw_pose", label: "Draw pose", type: "bool" },
        { key: "pose_kpt_conf", label: "Pose keypoint conf", type: "float", min: 0.05, max: 0.99, step: 0.05 },
      ],
    },
    {
      id: "advanced",
      title: "Advanced",
      fields: [
        { key: "seg_fallback_iou_min", label: "Seg fallback IoU", type: "float", min: 0.05, max: 0.95, step: 0.05 },
        { key: "recovery_thresh", label: "ReID recovery thresh", type: "float", min: 0.1, max: 0.99, step: 0.05 },
        { key: "reid_debug_log", label: "ReID debug log", type: "bool" },
        { key: "reid_gallery_size", label: "ReID gallery size", type: "int", min: 1, max: 50 },
        { key: "w_iou", label: "ReID w_iou", type: "float", min: 0, max: 1, step: 0.05 },
        { key: "w_app", label: "ReID w_app", type: "float", min: 0, max: 1, step: 0.05 },
        { key: "infer_imgsz", label: "Infer imgsz (0=default)", type: "int", min: 0, max: 1280, step: 32 },
        { key: "seg_stride", label: "Seg stride", type: "int", min: 1, max: 8 },
        { key: "preview_every_n", label: "Preview every N", type: "int", min: 1, max: 60 },
        { key: "use_amp", label: "Use AMP", type: "bool" },
        { key: "gpu_pipeline", label: "GPU pipeline", type: "bool" },
        { key: "max_infer_batch_size", label: "Max infer batch", type: "int", min: 1, max: 512 },
        { key: "max_job_batch_size", label: "Max job batch (ReID)", type: "int", min: 0, max: 512 },
        { key: "reid_embed_chunk", label: "ReID embed chunk (0=all)", type: "int", min: 0, max: 512 },
        { key: "use_batch_detect", label: "Batch detect", type: "bool" },
        { key: "reid_batch_across_frames", label: "ReID batch across frames", type: "bool" },
        { key: "reid_gpu_overlap", label: "ReID GPU overlap", type: "bool" },
        { key: "gpu_mask_resize", label: "GPU mask resize", type: "bool" },
        { key: "parallel_post", label: "Parallel post", type: "bool" },
        { key: "post_workers", label: "Post workers", type: "int", min: 1, max: 32 },
        { key: "max_preload_ram_gb", label: "Max preload RAM (GB)", type: "float", min: 1, max: 128, step: 1 },
        { key: "max_window_ram_gb", label: "Max window RAM (GB)", type: "float", min: 1, max: 32, step: 0.5 },
        { key: "window_frames", label: "Window frames (0=auto)", type: "int", min: 0, max: 8192 },
        { key: "windows_in_ram", label: "Windows in RAM", type: "int", min: 1, max: 4 },
        { key: "parallel_models", label: "Parallel models (legacy)", type: "bool" },
        {
          key: "encode_mode",
          label: "Encode mode",
          type: "select",
          options: [
            { value: "manual", label: "manual" },
            { value: "parallel", label: "parallel" },
            { value: "deferred", label: "deferred" },
          ],
        },
        { key: "async_encode", label: "Async encode", type: "bool" },
        { key: "encode_preset", label: "Encode preset", type: "text" },
        { key: "encode_crf", label: "Encode CRF", type: "int", min: 0, max: 51 },
        { key: "encode_codec", label: "Encode codec", type: "text" },
        { key: "encode_workers", label: "Encode workers (0=post)", type: "int", min: 0, max: 32 },
        { key: "cross_check_min_intersection_px", label: "Cross min intersection px", type: "float", min: 0, max: 100, step: 0.5 },
        { key: "cross_check_min_iou", label: "Cross min IoU", type: "float", min: 0, max: 1, step: 0.05 },
        { key: "ram_budget_system_reserve_gb", label: "RAM system reserve (GB)", type: "float", min: 0, max: 16, step: 0.25 },
        { key: "ram_budget_models_gb", label: "RAM models (GB, 0=auto)", type: "float", min: 0, max: 32, step: 0.5 },
        { key: "ram_budget_spill_gb", label: "RAM spill (GB)", type: "float", min: 0, max: 8, step: 0.25 },
        { key: "ram_budget_safety_margin_gb", label: "RAM safety margin (GB)", type: "float", min: 0, max: 8, step: 0.25 },
        {
          key: "sam_identity_backend",
          label: "SAM identity backend",
          type: "select",
          options: [
            { value: "memory", label: "memory" },
            { value: "sam2", label: "sam2" },
          ],
        },
      ],
    },
  ];

  /** Mirrors app/config/ui_fast_profile.json speed/identity pins (api.main bake). */
  const FAST_PROFILE_PINS = {
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
  };

  function fieldId(key) {
    return `set-${key.replace(/_/g, "-")}`;
  }

  function renderField(field) {
    const id = fieldId(field.key);
    if (field.type === "bool") {
      return `
        <label class="check-field">
          <input type="checkbox" id="${id}" data-key="${field.key}" data-type="bool" />
          <span>${field.label}</span>
        </label>`;
    }
    if (field.type === "select") {
      const opts = (field.options || [])
        .map((o) => `<option value="${o.value}">${o.label}</option>`)
        .join("");
      return `
        <label class="field">
          <span>${field.label}</span>
          <select id="${id}" data-key="${field.key}" data-type="select">${opts}</select>
        </label>`;
    }
    if (field.type === "model") {
      return `
        <label class="field">
          <span>${field.label}</span>
          <select id="${id}" data-key="${field.key}" data-type="model" data-group="${field.group || "detect"}"></select>
        </label>`;
    }
    const inputType = field.type === "int" || field.type === "float" ? "number" : "text";
    const attrs = [
      `id="${id}"`,
      `data-key="${field.key}"`,
      `data-type="${field.type}"`,
      field.min != null ? `min="${field.min}"` : "",
      field.max != null ? `max="${field.max}"` : "",
      field.step != null ? `step="${field.step}"` : "",
    ]
      .filter(Boolean)
      .join(" ");
    return `
      <label class="field">
        <span>${field.label}</span>
        <input type="${inputType}" ${attrs} />
      </label>`;
  }

  function render(container) {
    container.innerHTML = SECTIONS.map(
      (sec) => `
      <details class="settings-section" data-section="${sec.id}" ${sec.open ? "open" : ""}>
        <summary>${sec.title}</summary>
        <div class="settings-grid">${sec.fields.map(renderField).join("")}</div>
      </details>`
    ).join("");
  }

  function fillModelSelects(models) {
    const byGroup = {
      detect: models.detect || [],
      seg: models.seg || [],
      reid: models.reid || [],
    };
    document.querySelectorAll('select[data-type="model"]').forEach((sel) => {
      const group = sel.dataset.group || "detect";
      const paths = byGroup[group] || [];
      const current = sel.value;
      sel.innerHTML = paths.map((p) => `<option value="${p}">${p.split(/[/\\]/).pop()}</option>`).join("");
      if (current && paths.includes(current)) sel.value = current;
    });
  }

  function setValues(settings, models) {
    if (!settings) return;
    SECTIONS.forEach((sec) => {
      sec.fields.forEach((field) => {
        const el = document.getElementById(fieldId(field.key));
        if (!el) return;
        const val = settings[field.key];
        if (field.type === "bool") {
          el.checked = Boolean(val);
        } else if (val != null) {
          el.value = String(val);
        }
      });
    });
    if (models) fillModelSelects(models);
    SECTIONS.forEach((sec) => {
      sec.fields.forEach((field) => {
        if (field.type !== "model") return;
        const el = document.getElementById(fieldId(field.key));
        const val = settings[field.key];
        if (el && val) {
          if (![...el.options].some((o) => o.value === val)) {
            const opt = document.createElement("option");
            opt.value = val;
            opt.textContent = String(val).split(/[/\\]/).pop() + " (custom)";
            el.appendChild(opt);
          }
          el.value = val;
        }
      });
    });
  }

  function collect() {
    const out = {};
    SECTIONS.forEach((sec) => {
      sec.fields.forEach((field) => {
        const el = document.getElementById(fieldId(field.key));
        if (!el) return;
        if (field.type === "bool") {
          out[field.key] = el.checked;
        } else if (field.type === "int") {
          out[field.key] = parseInt(el.value, 10) || 0;
        } else if (field.type === "float") {
          const v = parseFloat(el.value);
          if (field.key === "max_duration_seconds" && (Number.isNaN(v) || v <= 0)) {
            out[field.key] = null;
          } else {
            out[field.key] = Number.isNaN(v) ? 0 : v;
          }
        } else {
          out[field.key] = el.value;
        }
      });
    });
    if (out.cross_check_model === "") out.cross_check_model = null;
    return out;
  }


  // Keep in sync with api/settings_codec.py UI_PIPELINE_KEYS.
  const UI_PIPELINE_KEYS = [
    "detect_model", "seg_model", "reid_model", "output_dir",
    "detect_conf", "seg_conf", "match_iou_min", "appearance_thresh", "track_buffer",
    "use_seg", "use_sam_identity", "use_reid", "sam_osnet_reentry",
    "sam_identity_backend", "sam_match_iou", "sam_model",
    "use_offline_tracklet_link", "tracklet_link_max_gap_frames",
    "tracklet_link_min_sim", "tracklet_link_use_reid",
    "overlay_alpha", "draw_boxes", "draw_masks", "draw_centers", "draw_pose", "pose_kpt_conf",
    "cross_check_enabled", "cross_check_model", "cross_check_object_prompt",
    "cross_check_conf", "cross_check_warning_text", "cross_check_draw_head_box", "cross_check_draw_boxes",
    "gpu_full_batch", "infer_batch_size", "gpu_queue_depth", "decode_prefetch",
    "realtime_mode", "frame_stride", "frame_source_mode", "preload_video",
    "use_tensorrt", "tensorrt_max_batch", "tensorrt_fp16", "tensorrt_autocast_fast", "inference_device",
    "encode_mode", "default_prompt", "smart_ram_budget", "max_process_ram_gb", "max_duration_seconds",
  ];

  function collectUi(options = {}) {
    const all = collect();
    const out = {};
    UI_PIPELINE_KEYS.forEach((k) => {
      if (k in all) out[k] = all[k];
    });
    if (!out.encode_mode) out.encode_mode = "manual";
    // Pin baked fast profile so empty/unchecked form fields cannot regress Pass1.
    if (options.pinFastProfile) {
      Object.assign(out, FAST_PROFILE_PINS);
    }
    return out;
  }

  const OVERLAY_KEYS = [
    "overlay_alpha",
    "draw_boxes",
    "draw_masks",
    "draw_centers",
    "draw_pose",
    "pose_kpt_conf",
    "cross_check_enabled",
    "cross_check_draw_head_box",
    "cross_check_draw_boxes",
  ];

  function collectOverlay() {
    const all = collect();
    const out = {};
    OVERLAY_KEYS.forEach((k) => {
      if (k in all) out[k] = all[k];
    });
    return out;
  }

  window.SettingsForm = {
    render,
    setValues,
    collect,
    collectUi,
    collectOverlay,
    fillModelSelects,
    SECTIONS,
    OVERLAY_KEYS,
    UI_PIPELINE_KEYS,
    FAST_PROFILE_PINS,
  };
})();
