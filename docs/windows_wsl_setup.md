# Windows setup (WSL2)

The training stack (mjlab, MuJoCo Warp, the BAM actuator model) and the evaluation tools are Linux-only, so on Windows everything runs inside WSL2 Ubuntu 24.04. Both machines use the same steps; only the desktop needs the NVIDIA driver.

## 1. Desktop only: NVIDIA driver

Install the current NVIDIA Game Ready or Studio driver for Windows. That single Windows driver also serves CUDA inside WSL2. **Do not install an NVIDIA driver inside Ubuntu.** Check from PowerShell:

```powershell
nvidia-smi
```

## 2. Install WSL2 + Ubuntu 24.04

In an Administrator PowerShell, from the repo folder (or download just the script):

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup\windows\install_wsl.ps1
```

Reboot if asked, then open **Ubuntu 24.04** from the Start menu once and create your Linux user.

Without the script: `wsl --install -d Ubuntu-24.04`.

## 3. Clone and set up inside Ubuntu

Clone into the Linux filesystem (`~/`), not `/mnt/c/...`. File access across the Windows boundary is many times slower and makes builds crawl.

```bash
git clone <repo URL> ~/microduck-pretraining
cd ~/microduck-pretraining
bash setup/setup.sh
```

The first `uv sync` downloads about 5 GB (PyTorch with CUDA libraries). Expect 10-20 minutes.

## 4. Check

```bash
uv run md-check
```

- Desktop: `CUDA GPU for training` should show `NVIDIA GeForce RTX 3090, 24 GB` and `Warp sees CUDA` should pass.
- Laptop: the GPU lines are warnings, which is expected. It is the evaluation machine.
- `Display for 3D viewer`: Windows 11 WSLg provides one automatically. On Windows 10, the headless tools (`md-eval`, `md-report`) still work.

## Editing from Windows

- VS Code: install the **WSL** extension, then run `code .` inside Ubuntu.
- File Explorer: `\\wsl$\Ubuntu-24.04\home\<user>\microduck-pretraining`.
- SolidWorks/OpenSCAD files for payload mounts can live in `hardware/`; keep exports (STL/STEP) there too.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `torch.cuda.is_available()` is False on the desktop | Update the Windows NVIDIA driver; run `wsl --update`; restart WSL with `wsl --shutdown` |
| `uv sync` fails building `better-actuator-models` | `sudo apt-get install -y build-essential git`, then re-run |
| Viewer window doesn't open | Windows 11 + `wsl --update` for WSLg; check `echo $DISPLAY` |
| Training runs out of GPU memory | `uv run md-train c3 --num-envs 2048` (use the same value for every condition) |
