# Training this on a desktop GPU — measured, on an RTX 3060 12GB

`TRAINING.md` documents the recipe. This documents what the recipe costs on a
consumer card that is also driving a Windows desktop, because that turned out to
be the binding constraint and none of it is guessable from the recipe.

Machine: RTX 3060 12GB · i5-14600K (14C/20T, **has Intel UHD 770 iGPU**) ·
32GB RAM · Windows 11 · single 1080p monitor.

## What actually fits

Measured on the real dataset, `coco-l` (yolo12l, 26.4M params) at `imgsz=1024`.
`GPU_mem` is what ultralytics reports; anything above the card's physical 12.29GB
is Windows silently spilling to system RAM over PCIe, **not** an error.

| batch | GPU_mem | Fits on-card? | s/epoch | 300 epochs |
|---|---|---|---|---|
| 8 | — | no — hard OOM | — | — |
| 4 | **16.7 G** | no — ~8 GB spilled to RAM | 224 s | ~18.7 h |
| **2** | **8.69 G** | **yes** | **102 s** | **~8.5 h** |
| `coco-s` @ 8 | 1.9 G | yes, easily | 6.4 s | ~32 min |

**The smaller batch is 2.2x faster in wall clock.** That is not a typo and not a
tradeoff — batch=4 does not fit, so every step drags half its working set across
PCIe (~16-32 GB/s) instead of on-card GDDR6 (~360 GB/s).

### Why batch=2 costs almost nothing in quality

Because `nbs=8` stays put. Ultralytics (`engine/trainer.py:297-298`):

    accumulate   = max(round(nbs / batch), 1)
    weight_decay = weight_decay * batch * accumulate / nbs

| | batch=8 | batch=2 |
|---|---|---|
| accumulate | 1 | **4** |
| effective batch | 8 | **8** |
| optimizer steps/epoch | 8 | 7.75 |
| weight_decay | 0.0005 | **0.0005** |

Gradient accumulation restores exactly what the smaller batch gave up. The only
real difference is BatchNorm normalising over 2 samples instead of 8, since BN
cannot be accumulated. Second-order at 62 training images.

**Never fix a VRAM problem by lowering `imgsz`.** 1024 is load-bearing: box short
side is p10 23px there and 14px at 640 (`TRAINING.md`).

## The desktop overhead problem

The 3060 was driving the monitor *and* training. Windows took **~3.5-4.1 GB**
of the 12.29 GB just to composite the desktop, and it drifts upward on its own —
EdgeWebView2 respawned mid-session and ate ~900 MB back after being closed.

    12,288 (card) - 4,109 (desktop) - 8,690 (coco-l @ batch=2) = -511 MiB

That is the margin an 8-hour arm runs on. `train.py` has no resume, so an OOM at
hour six costs the whole arm.

## How to reclaim it — ranked by what it actually returns

### 1. Move the display to the integrated GPU — frees ~3.5 GB, all of it

This CPU has Intel UHD Graphics 770 sitting idle. Plug the monitor into the
**motherboard's** HDMI/DisplayPort instead of the graphics card, and Windows
composites the desktop on the iGPU while the 3060 becomes a pure compute device.

1. Reboot into BIOS (Del/F2 during POST).
2. Find **iGPU Multi-Monitor** / **Integrated Graphics** / **IGD Multi-Monitor**
   (usually under Advanced → System Agent / Chipset) and set it **Enabled**.
3. Save and shut down. Move the monitor cable from the 3060 to the motherboard's
   video port. Boot.
4. Verify: `nvidia-smi --query-gpu=memory.used --format=csv` should read a few
   hundred MiB at idle instead of ~3,500.

Keep the NVIDIA driver installed — CUDA does not need a display attached.

**What this does not do:** it will not enable `batch=4`. That needs 16.7 GB
against a 12.29 GB card, so it would still spill even with a completely free
GPU. What it buys is a ~3.6 GB safety margin instead of a deficit.

### 2. Turn off browser hardware acceleration — ~0.5-1 GB, and it persists

Chrome `chrome://settings/system` → "Use graphics acceleration when available"
→ off. Opera: Settings → System, same toggle. Edge `edge://settings/system`.

Better than closing browsers, which only helps until you reopen them. Costs some
CPU and smoothness on video and scrolling.

### 3. Disable Windows Widgets — ~200-400 MB

Settings → Personalisation → Taskbar → Widgets off. This is what keeps
respawning `msedgewebview2.exe` after you kill it.

### 4. Visual effects — ~100-300 MB

Settings → Accessibility → Visual effects → Transparency **off**, Animation
**off**.

### 5. Minor

- `rsAppUI.exe` (ReasonLabs antivirus tray UI) holds ~100-200 MB — disable its UI.
- Windows Terminal is GPU-accelerated; plain `conhost` is not.
- Any Electron app (VS Code, Discord, Slack, Teams) composites on the GPU.

## Checklist before a long unattended run

- [ ] `nvidia-smi` idle reading — under ~1 GB if the iGPU move is done, else
      expect 3.5-4 GB and size the batch accordingly
- [ ] `nvidia-smi --query-gpu=memory.used --format=csv -l 10` during epoch 1;
      **memory pegged near 12,288 with low/oscillating utilisation means it is
      spilling, not training**
- [ ] first two epoch times in `runs/<arm>/results.csv` are near the baseline for
      that arm — a large jump between epochs is the spill signature
- [ ] `powercfg /change standby-timeout-ac 0` — system sleep kills the run
- [ ] venv and `runs/` outside any OneDrive-synced folder. `.gitignore` does not
      apply to OneDrive; a ~50MB `last.pt` rewritten 300x per arm will be
      uploaded every time, and sync can hold a lock long enough to make
      `torch.save` raise `PermissionError` and kill the run.
      Junction works and OneDrive skips it:

          New-Item -ItemType Junction -Path "out\dataset\v1\runs" -Target "C:\ml\schedext-runs"

## If an arm dies anyway

`train.py` has no resume, and re-running overwrites the partial run
(`exist_ok=True`). Ultralytics stores the training args inside the checkpoint,
so recover from the repo root instead:

    python -c "from ultralytics import YOLO; YOLO(r'out\dataset\v1\runs\coco-l\weights\last.pt').train(resume=True)"

Arms that already finished are on disk and are not lost — which is why the arms
are run as separate invocations rather than `--arms all`.
