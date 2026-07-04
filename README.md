# AI Launchpad

A free, structured, textbook-style website for learning **Machine Learning and Deep Learning from scratch** - in the spirit of W3Schools / GeeksforGeeks reference sites, not a landing page.

## Features

- 13-module curriculum: math foundations → Python → EDA → classical ML → deep learning → CNNs → sequence models → Transformers
- Every topic page follows the same 8-step template: Intuition → Theory → Math → Diagram → Code → Pros/Cons → Interview Q&A → Quiz
- Left sidebar navigation with search, collapsible modules, and per-module progress counters
- Read-tracking, module progress bars, and a "Continue where you left off" button (localStorage, no account)
- KaTeX for math, Prism for syntax-highlighted Python with copy buttons, inline SVG diagrams
- Light/dark mode, fully responsive, on-page table of contents

## Tech

Pure static HTML/CSS/JS - no build step, no dependencies to install. KaTeX and Prism are loaded from CDNs. Routing is hash-based (`#/module/topic`), so it works on GitHub Pages without any 404 tricks.

```
index.html          app shell
css/style.css       all styling (CSS variables for theming)
js/curriculum.js    the full module/topic tree (single source of truth)
js/app.js           router, sidebar, search, progress, quizzes
js/content/*.js     one file per module; registers pages into CONTENT["module/topic"]
```

### Adding a new topic page

1. Make sure the topic exists in `js/curriculum.js` (all planned topics are already listed).
2. In the matching `js/content/<module>.js` file, add:
   ```js
   CONTENT["module-id/topic-slug"] = { html: String.raw`...page HTML...` };
   ```
3. Follow the 8-step template used by the existing pages (see `js/content/ml.js`).

Topics without registered content automatically render as "coming soon" stubs.

## Run locally

Just open `index.html` in a browser, or serve the folder:

```
python -m http.server 8000
```

## Deploy to GitHub Pages

Already wired up: `.github/workflows/deploy.yml` deploys on every push to `main`.

One-time setup in the GitHub repo: **Settings → Pages → Source → GitHub Actions**.
