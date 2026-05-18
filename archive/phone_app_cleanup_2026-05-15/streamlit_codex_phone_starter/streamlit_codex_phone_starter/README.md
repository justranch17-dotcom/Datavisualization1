# Streamlit Codex Phone Control

This is a sidecar phone control panel for the Datavisulaization project.

It is intentionally separate from `Mach2AImarket.py`, `Marketvisual.py`, and
`AImarketvisual.py`, so we can use it from an iPhone without risking the current
scanner workflow.

## What It Does

- Runs a Streamlit app that works well from a phone.
- Reads recent Mach 2 feedback from `mach2_group_feedback.csv`.
- Saves phone feedback into `pattern_feedback.db` at the project root.
- Saves Codex instructions into `codex_tasks.md` at the project root.
- Saves readable feedback summaries into `codex_feedback_summary.md`.

## Run From The Project Root

```powershell
.\run_phone_control.ps1
```

Then open this on the computer:

```text
http://localhost:8501
```

## Use From iPhone With Tailscale

See the project-root file:

```text
TAILSCALE_PHONE_SETUP.md
```

Once Tailscale is running on both devices, open one of these on the iPhone:

```text
http://DEVICE-NAME:8501
http://TAILSCALE-IP:8501
```

## Integration Notes

The phone app does not import the main Streamlit apps, because importing them
would execute their UI immediately. Instead, `scanner.py` reads safe project
artifacts.

Current bridge:

```text
mach2_group_feedback.csv -> phone pattern cards
```

Future bridge we can add safely:

```text
Mach2AImarket.py scan results -> saved CSV/JSON -> phone pattern cards
```
