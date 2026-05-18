# Tailscale Phone Setup

This project now has a sidecar Streamlit phone control panel. It does not replace
the main market apps. It reads project artifacts and writes feedback/task files in
the project root.

## 1. Install Tailscale

Install Tailscale on this Windows computer and on your iPhone.

Windows:

```powershell
winget install --id Tailscale.Tailscale -e
```

If this terminal does not immediately recognize `tailscale`, use the installed
Windows path:

```powershell
& "C:\Program Files\Tailscale\tailscale.exe" status
& "C:\Program Files\Tailscale\tailscale.exe" up
```

iPhone:

Install Tailscale from the App Store and sign in with the same Tailscale account.

## 2. Start Tailscale

Open Tailscale on Windows, sign in, and make sure this computer appears as online.
Open Tailscale on the iPhone too.

## 3. Run The Phone App

From the project root:

```powershell
.\run_phone_control.ps1
```

That starts Streamlit on all network interfaces:

```text
http://localhost:8501
```

## 4. Open It From iPhone

In the Tailscale app, find this computer's device name or Tailscale IP.

Then open one of these in Safari:

```text
http://DEVICE-NAME:8501
http://TAILSCALE-IP:8501
```

## Notes

- Keep this computer awake.
- Keep the Streamlit terminal running.
- The phone app writes `pattern_feedback.db`, `codex_tasks.md`, and
  `codex_feedback_summary.md` in the project root.
- The current integration reads recent rows from `mach2_group_feedback.csv`.
- No main analyzer files are changed by this setup.
