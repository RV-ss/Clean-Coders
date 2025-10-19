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

Render deployment notes
-----------------------

If you deploy the backend to Render (or a similar service) the repository includes a simple `Procfile` which starts the app using `gunicorn`.

- Before starting the service on Render, set an environment variable to restrict CORS origins for safety:

  ALLOWED_ORIGINS=https://clean-coders-asl-ai-translator.onrender.com

  You can also include multiple origins separated by commas (for example, add `http://localhost:5000` while testing locally).

- Render start command (Procfile provided):

  web: gunicorn app:app -b 0.0.0.0:$PORT --workers 2

The front-end files (`translator.html`, `meeting.html`) are already configured to prefer the deployed Render API at:

  https://clean-coders-asl-ai-translator.onrender.com

so once your Render service is live the static pages will point to the deployed API by default. If you host the front-end separately, set an env var `__API_URL__` to override the built-in preference.

How to finish deployment on Render (manual steps)
------------------------------------------------

1. Push your repository to GitHub.

2. Go to https://dashboard.render.com and create a new Web Service.
  - Connect your GitHub repo and select the branch to deploy.
  - For the Start Command use: `gunicorn app:app -b 0.0.0.0:$PORT --workers 2` (or rely on `render.yaml`).
  - Set the build command to `pip install -r requirements.txt` (or leave blank if using `render.yaml`).

3. Set environment variables (in the Render dashboard -> Environment):
  - `ALLOWED_ORIGINS` = `https://clean-coders-asl-ai-translator.onrender.com`
  - `FLASK_DEBUG` = `0`

4. Deploy and watch the logs for successful startup. Your service should be reachable at the Render-generated URL.

Render CLI (optional)
----------------------
If you prefer the CLI, you can use the `render.yaml` file in this repository and run:

1. Install the Render CLI: https://render.com/docs/deploy-to-render
2. From your repository root run:

  render deploy

This will use the `render.yaml` configuration to create/update the service.

If you run into errors, copy the recent logs from the Render dashboard and paste them here and I will help you interpret and fix them.

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