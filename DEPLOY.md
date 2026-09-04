# Deploying SAST IQ to Render

This repo ships a `Dockerfile` and a `render.yaml` Blueprint. Render runs the
FastAPI app as a long-lived container, so SQLite and the ML model files behave
the way they do locally.

## 1. Push this repo to GitHub

```bash
git remote add origin https://github.com/<you>/sast-iq.git
git branch -M main
git push -u origin main
```

## 2. Create the service on Render

1. Sign in at <https://dashboard.render.com>.
2. **New +  ->  Blueprint**.
3. Pick this GitHub repo. Render reads `render.yaml` and proposes a web
   service named `sast-iq`.
4. **Apply**. First build takes a few minutes (installs scikit-learn/scipy).
5. When it goes live you get `https://sast-iq-XXXX.onrender.com`.
   - Dashboard: `/dashboard`
   - API docs: `/docs`

`autoDeploy` is on, so every push to `main` redeploys.

## Free plan caveats

- **Ephemeral filesystem.** Feedback labels and any models trained via the UI
  are lost on each redeploy and after the service spins down (~15 min idle).
  On every boot the container re-seeds the 22 training samples and trains a
  fresh baseline model, so the dashboard is never empty.
- **Cold starts.** After idle spin-down the first request waits ~30-60 s while
  the container starts and loads scikit-learn.

## Keeping data between deploys (paid)

Render persistent disks require a paid instance type. In `render.yaml`:

1. Change `plan: free` to `plan: starter`.
2. Uncomment the `disk:` block (mounts a 1 GB volume at `/data`).
3. Uncomment the `DATABASE_URL` and `MODEL_DIR` env vars (point the DB file and
   the model registry at `/data`).

Both paths are env-driven in the code (`backend/database.py`,
`backend/ml/model_registry.py`) — no code changes needed.

## The `/api/scan` endpoint on a hosted box

Scanning targets a git repo **on the server's filesystem**. On Render there are
no local repos to point at, so scanning is only useful if you either
(a) bake sample repos into the image, or (b) extend `git_utils.get_diff_chunks`
to `git clone` a remote URL into a temp dir first. The rest of the app
(feedback loop, retraining, dashboard, smart memory) works as-is.

## Security note

No endpoint is authenticated, CORS is open, and `/api/scan` reads arbitrary
filesystem paths. Add auth before sharing the URL beyond a trusted group.
