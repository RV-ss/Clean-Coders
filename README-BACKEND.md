# Backend API — ULTRA Sign Language Translator

This document describes the local Flask backend used by the front-end HTML files (`translator.html`, `meeting.html`). The active backend is `app.py` (a lightweight, in-memory trainer).

Base URL: `http://127.0.0.1:5000`

Endpoints

- `GET /api/health`
  - Returns: `{ "status": "healthy" }`
  - Use to confirm the server is running.

- `POST /api/predict`
  - Accepts: JSON `{ "landmarks": [x1, y1, z1, x2, y2, z2, ...] }` (flattened 21 landmarks × 3 = 63 numbers)
  - Returns: `{ "success": true, "prediction": { "sign": "hello", "confidence": 0.92 } }
  - Notes: The front end calls this for live translation and test prediction.

- `POST /api/add_sample`
  - Accepts: `{ "landmarks": [...], "label": "hello" }`
  - Returns: `{ "success": true, "total_samples": N }`
  - Adds a single feature sample (extracted server-side) to the in-memory training set.

- `POST /api/add_video_sample`
  - Accepts: `{ "frames": [{ "landmarks": [...], "timestamp": 123456 }, ...], "label": "hello" }`
  - Returns: `{ "success": true, "samples_added": K, "total_samples": N }`
  - Notes: Used by the UI video recorder to upload multiple frames from a short clip (4s by default).

- `POST /api/train`
  - Trains the simple feature-based model using the current training samples in memory.
  - Returns: `{ "success": true, "accuracy": 0.85, "samples_used": N, "message": "Trained on ..." }`

- `GET /api/status`
  - Returns training metadata: total samples, per-sign counts, whether the model is trained and a simple accuracy estimate.

- `POST /api/reset`
  - Resets training data and model state.

- `POST /api/debug_prediction`
  - Accepts: same as `/api/predict`
  - Returns: debug information including extracted feature counts and per-sign similarity scores. Useful to tune thresholds.

Implementation notes

- `app.py` implements a feature-based trainer and predicts using normalized distances against per-sign prototypes.
- Labels are normalized to lowercase on the server; make sure the frontend's sign selections match server labels.

Running the server

```powershell
python .\app.py
```

If you want to expose the server on another port, edit the `app.run(...)` call in `app.py` or run with environment variables.

Debugging tips
- If predict responses are slow, test `/api/health` timing and ensure CPU usage is not saturated.
- Use `/api/debug_prediction` to see per-sign similarity scores and understand why the model favors a sign.

Extending the backend

- The codebase includes TODOs for multi-prototype profiles and augmentation. Adding these increases cross-user generalization.

License
- MIT (add your own LICENSE file when publishing).

Contact
- Add your maintainer info here before publishing to GitHub.