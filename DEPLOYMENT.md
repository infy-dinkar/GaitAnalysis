# Deployment Guide

This guide walks you through deploying MotionLens to production:

- **Backend (FastAPI + MediaPipe)** → Hugging Face Spaces (Docker)
- **Frontend (Next.js)** → Vercel

Total time: **~30 minutes** if you have GitHub, Hugging Face, and Vercel accounts ready.

---

## Table of contents

1. [Architecture overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Push to GitHub](#step-1-push-to-github)
4. [Deploy backend to Hugging Face Spaces](#step-2-deploy-backend-to-hugging-face-spaces)
5. [Deploy frontend to Vercel](#step-3-deploy-frontend-to-vercel)
6. [Connect frontend to backend](#step-4-connect-frontend-to-backend)
7. [Lock down CORS](#step-5-lock-down-cors-recommended)
8. [Verify everything works](#step-6-verify)
9. [Updating the deployment](#updating-the-deployment)
10. [Common issues](#common-issues)

---

## Architecture overview

```
┌──────────────────────┐       HTTPS       ┌─────────────────────────┐
│  Next.js frontend    │  ──────────────▶  │  FastAPI backend         │
│  Vercel              │     POST /api     │  Hugging Face Spaces     │
│  https://*.vercel.app│  ◀──────────────  │  https://*.hf.space      │
└──────────────────────┘     JSON          └─────────────────────────┘
```

**What needs the backend:**
- Gait analysis (video upload → full kinematic report)

**What runs entirely in the browser (no backend needed):**
- Live biomech (shoulder, neck, knee, hip, ankle ROM via webcam)
- Biomech video upload (browser-side MoveNet)
- Posture analysis (front + side photo screening)

So the backend is only critical for the gait module.

---

## Prerequisites

You'll need accounts on:

- **GitHub** — [github.com](https://github.com) (free)
- **Hugging Face** — [huggingface.co/join](https://huggingface.co/join) (free)
- **Vercel** — [vercel.com/signup](https://vercel.com/signup) (free Hobby plan)

Tools installed locally:

- **Git** — `git --version` should work
- That's it. Everything else (Docker, Node) only matters if you want local dev.

---

## Step 1 — Push to GitHub

If your code isn't already on GitHub:

```bash
# in the project root (D:\Gait_Analysis or similar)
git init
git add .
git commit -m "Initial deploy"

# create a new empty repo on github.com first, then:
git remote add origin https://github.com/YOUR-USERNAME/motionlens.git
git branch -M main
git push -u origin main
```

If it's already on GitHub, just make sure your latest changes are pushed:

```bash
git add .
git commit -m "Prepare for deploy"
git push
```

---

## Step 2 — Deploy backend to Hugging Face Spaces

### 2.1 Create a new Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in:
   - **Space name** — e.g. `motionlens-api` (your URL will be `https://YOUR-USERNAME-motionlens-api.hf.space`)
   - **License** — pick whatever fits your project
   - **Space SDK** — select **Docker** → **Blank**
   - **Hardware** — `CPU basic · 2 vCPU · 16 GB RAM` (free tier — sufficient)
   - **Visibility** — Public is fine
3. Click **Create Space**.

### 2.2 Push your code to the Space

The Space gives you a git URL like `https://huggingface.co/spaces/YOUR-USERNAME/motionlens-api`. There are two ways to push:

**Option A — Mirror your GitHub repo (simplest)**

In the Space's **Settings** tab, scroll to **"Linked to a repository"** and link the GitHub repo you pushed in Step 1. Hugging Face will sync automatically on every push to GitHub.

**Option B — Manual git push**

```bash
git remote add hf https://huggingface.co/spaces/YOUR-USERNAME/motionlens-api
git push hf main
```

You'll be prompted for your Hugging Face username + access token (create one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), Write scope).

### 2.3 Wait for the build

The Space will detect the `Dockerfile` and start building automatically. The first build takes **5–10 minutes** because it has to install MediaPipe, OpenCV, and download the pose model (~10 MB).

You can watch progress in the Space's **Logs** tab. When the build succeeds you'll see:

```
INFO:     Uvicorn running on http://0.0.0.0:7860
```

### 2.4 Verify the backend is alive

Open your Space's URL in a browser and append `/api/health`:

```
https://YOUR-USERNAME-motionlens-api.hf.space/api/health
```

You should see:

```json
{ "status": "healthy", "service": "MotionLens API", "version": "1.0.0" }
```

✅ Backend deployed. Copy your Space URL — you'll need it in Step 4.

---

## Step 3 — Deploy frontend to Vercel

### 3.1 Import the repo

1. Go to [vercel.com/new](https://vercel.com/new)
2. Click **Import Git Repository**, select your GitHub repo.
3. Vercel will auto-detect the project. **IMPORTANT**: change the **Root Directory** to `motionlens-web` (the Next.js app lives in a subfolder).

### 3.2 Configure environment variables

Before clicking Deploy, expand **Environment Variables** and add:

| Name | Value |
|---|---|
| `NEXT_PUBLIC_API_BASE_URL` | `https://YOUR-USERNAME-motionlens-api.hf.space` |

Use the URL from Step 2.4 — **without a trailing slash**.

### 3.3 Deploy

Click **Deploy**. Vercel will build the Next.js app (takes ~2–4 minutes the first time) and give you a URL like:

```
https://motionlens-xxx.vercel.app
```

✅ Frontend deployed.

---

## Step 4 — Connect frontend to backend

If you set the env var correctly in Step 3.2, the frontend already points at the backend. Nothing else to do here.

To verify, open your Vercel URL and try uploading a gait video — if it works, the wiring is correct.

If you forgot or need to change the URL:

1. Go to your project on Vercel
2. **Settings → Environment Variables**
3. Edit `NEXT_PUBLIC_API_BASE_URL`
4. **Deployments tab → ⋯ menu on the latest deploy → Redeploy** (env var changes need a rebuild)

---

## Step 5 — Lock down CORS (recommended)

By default the backend accepts requests from any origin (`*`). For production, lock it to your Vercel domain.

1. Open your Hugging Face Space.
2. **Settings → Variables and secrets → New variable** (use **Variable**, not Secret — it's not sensitive):
   - **Name**: `MOTIONLENS_ALLOWED_ORIGINS`
   - **Value**: `https://motionlens-xxx.vercel.app`
3. **Restart the Space** (Settings → "Factory reboot" or push any change).

If you have a custom domain, list both:

```
https://motionlens.com,https://www.motionlens.com,https://motionlens-xxx.vercel.app
```

---

## Step 6 — Verify

Open your Vercel URL on:

- **Desktop browser** (Chrome/Firefox/Safari) — every feature should work
- **Mobile browser** — camera-based features (live biomech, posture capture) work because Vercel serves over HTTPS

End-to-end checklist:

- [ ] Home page loads
- [ ] `/biomech/shoulder/live` — camera permission prompt appears, skeleton renders ~50 FPS
- [ ] `/biomech/shoulder/upload` — video upload, in-browser analysis, report renders
- [ ] `/posture` — both photos required, annotated report renders
- [ ] `/gait/upload` — video upload reaches the backend, report includes Plotly charts (this is the only path that requires the backend)

---

## Updating the deployment

Both platforms watch GitHub:

- **Push to GitHub `main` branch** → Vercel rebuilds the frontend automatically (~2 min)
- **HF Space mirrored to GitHub** → builds the backend automatically on push (~5 min on existing layers)

Manual triggers:

- Vercel: Deployments → ⋯ → Redeploy
- HF Spaces: Settings → Factory reboot

---

## Common issues

### Backend

| Symptom | Fix |
|---|---|
| Space build fails on `mediapipe` install | HF Spaces sometimes runs out of build memory. Restart the build (Settings → Factory reboot). |
| `Application startup failed` in logs | Check that `requirements_api.txt` and `requirements.txt` are at the repo root, not nested. |
| `/api/health` works but POST returns 500 | Check Space logs for the traceback — usually a missing system lib. The included Dockerfile handles all known cases. |
| Cold start is slow (~30 sec) | Free tier Spaces sleep when idle. First request wakes it. Upgrade hardware tier if you need always-warm. |

### Frontend

| Symptom | Fix |
|---|---|
| Vercel build fails on `npm install` | Make sure **Root Directory** is set to `motionlens-web`. |
| Build OK, but uploads fail with "Network error" | `NEXT_PUBLIC_API_BASE_URL` is wrong or missing. Verify in Vercel → Settings → Environment Variables, then redeploy. |
| Camera doesn't work on mobile/desktop LAN | Camera APIs require HTTPS or localhost. Vercel provides HTTPS automatically — just use the Vercel URL, not the IP address. |
| `Cannot read properties of undefined (reading 'getUserMedia')` | Browsing over HTTP from a non-localhost origin. Use the HTTPS Vercel URL. |
| Reports render but Plotly charts don't show | Plotly bundle is ~3 MB; on slow connections give it a few seconds. If it never loads, check the browser console for errors. |

### CORS

| Symptom | Fix |
|---|---|
| `CORS policy: No 'Access-Control-Allow-Origin'` in browser console | After locking down CORS in Step 5, your Vercel URL doesn't match. Add it to `MOTIONLENS_ALLOWED_ORIGINS` and restart the Space. |
| CORS error only on preview deployments | Vercel preview URLs change per branch (`*-git-feature.vercel.app`). Add the wildcard via specific URLs, or temporarily relax to `*` while testing. |

---

## Cost summary

Both platforms have generous free tiers that fit MotionLens:

| Platform | Free tier limit | Sufficient for MotionLens? |
|---|---|---|
| **Hugging Face Spaces** | 16 GB RAM, 2 vCPU, sleeps after 48h idle | ✅ Yes |
| **Vercel Hobby** | 100 GB bandwidth, unlimited static, 100 GB-hours serverless | ✅ Yes |

If you outgrow these:
- HF Spaces: $9/mo gets you "always-on" hardware (no cold starts)
- Vercel Pro: $20/mo for team features + higher limits

---

## You're done

Share your Vercel URL with users. Live biomech, posture, and biomech upload work entirely in their browsers. Gait analysis hits your HF backend.

Questions or issues? Check the [main README](README.md) for architecture details and the [troubleshooting section](#common-issues) above.
