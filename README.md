# SIGN IT ULTRA Sign Language Translator

Lightweight sign-language translator built with a simple Flask backend and MediaPipe-powered browser frontends. It supports live translation, quick training (single-sample and video), auto-speech, and a "meeting" view tuned for low-latency translation.

This repository contains several front-end HTML UIs and two trainer backends (a simple in-memory trainer used by default and an optional DB-backed trainer). The primary files you will interact with are:

- `app.py` — the default Flask backend (in-memory trainer, recommended for quick experiments)
- `translator.html` — full-featured translator UI with training controls, auto-speech, and sentence builder
- `meeting.html` — lightweight, low-latency meeting-style translator UI
- `README-BACKEND.md` — detailed backend API and run instructions

Why this project
- Fast local prototyping for real-time sign detection using MediaPipe hands in the browser
- Simple, explainable feature-based trainer (no heavy ML dependencies required)
- Quick training workflow (add samples, video-based training, train model) and persistence options

Quick start (Windows / PowerShell)

1. Start the backend (default: `app.py`):

```powershell
python .\app.py
```

This starts a Flask server on `http://127.0.0.1:5000` by default.

2. Serve or open the frontend

- For quick local testing, you can open `translator.html` or `meeting.html` directly in your browser (Chrome/Edge). Some browsers disallow camera access from file:// URLs — if you see problems, run a simple static server:

```powershell
# from the project root
python -m http.server 8000
# Visit http://localhost:8000/translator.html
```

3. Use the UI
- `translator.html`: start camera, train signs (Add Training Sample / Record Video / Train Model), and test translations.
- `meeting.html`: start camera and use the low-latency translator for meetings.

API server
- The frontends expect `http://127.0.0.1:5000` by default. If you run the backend on another host/port, update the `API_URL` constant in the HTML files.

Where to look next
- `app.py` — backend request handling and the simple trainer
- `translator.html` — training UX and debug helpers
- `meeting.html` — debouncer and low-latency logic

Issues & troubleshooting
- 400 Invalid label when sending video samples: ensure the selected sign in the frontend matches the backend `sign_classes` (labels are normalized to lowercase). The frontend has been updated to match `app.py` but legacy labels cause rejections.
- Camera not working: if your browser blocks camera access on `file://` origins, run a local static server (see Quick start).
- Backend unreachable: confirm `app.py` is running and `http://127.0.0.1:5000/api/health` returns `{"status":"healthy"}`.

License
- MIT by default. Add a `LICENSE` file to the repo if you want to publish it on GitHub.

Contributing
- PRs welcome. For significant model changes (augmentation, multi-prototype profiles, or CNN training), open an issue describing dataset requirements and desired accuracy targets.

Contact
- Add a short README or CONTRIBUTING file if you want to guide downstream contributors.