# AI Launchpad

A comprehensive learning portal for AI/ML students — structured roadmaps, curated resources, recommended videos, programming projects, and an interactive Gemini-powered doubt solver.

🌐 **Live:** `https://<your-username>.github.io/<your-repo-name>/`

---

## Tech Stack

- **Frontend** — React 19, TypeScript, Tailwind CSS v4, Vite
- **AI** — Gemini API (`gemini-2.0-flash`) called directly from the browser
- **Hosting** — GitHub Pages (fully static, no server needed)

---

## Local Development

**Prerequisites:** Node.js 20+

```bash
npm install
npm run dev
```

App runs at `http://localhost:3000`. You'll be prompted to enter your Gemini API key inside the chat UI — it's saved to `localStorage` so you only need to enter it once.

---

## Deploy to GitHub Pages

### 1. Enable GitHub Pages

Go to your repo → **Settings → Pages → Source → GitHub Actions**

### 2. Set the base path variable

Go to **Settings → Secrets and variables → Actions → Variables tab** and add:

| Name | Value |
|---|---|
| `VITE_BASE_PATH` | `/your-repo-name/` ← must start and end with `/` |

Skip this if you're using a custom domain (leave it as `/`).

### 3. Push to `main`

The workflow fires automatically and your site goes live at:
```
https://<your-username>.github.io/<your-repo-name>/
```

---

## GitHub Actions Workflows

| File | Triggers | What it does |
|---|---|---|
| `deploy.yml` | Push to `main`, manual | Typechecks → builds → deploys to Pages |
| `ci.yml` | Pull requests | Typechecks → builds (no deploy) |

---

## Gemini API Key

Since this is a static site, the Gemini API key is entered by the user in the chat UI and stored in their browser's `localStorage`. No key is ever hardcoded or committed to the repo.

---

## Available Scripts

| Command | Description |
|---|---|
| `npm run dev` | Start development server |
| `npm run build` | Typecheck + build for production |
| `npm run lint` | TypeScript typecheck only |
| `npm run clean` | Remove `dist/` folder |
