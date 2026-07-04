/* ============================================================
   AI Launchpad - app shell: routing, sidebar, search, theme,
   progress tracking, quizzes, code copy buttons, page TOC.
   ============================================================ */

(function () {
  "use strict";

  /* ---------- persistent state (localStorage) ---------- */

  var LS_PROGRESS = "ail-progress";
  var LS_THEME = "ail-theme";
  var LS_OPEN = "ail-open-modules";

  function loadJSON(key, fallback) {
    try {
      var raw = localStorage.getItem(key);
      return raw ? JSON.parse(raw) : fallback;
    } catch (e) { return fallback; }
  }

  function saveJSON(key, val) {
    try { localStorage.setItem(key, JSON.stringify(val)); } catch (e) { /* private mode */ }
  }

  var progress = loadJSON(LS_PROGRESS, {});           // { "ml/linear-regression": true }
  var openModules = loadJSON(LS_OPEN, { "getting-started": true });

  /* ---------- helpers ---------- */

  function $(sel, root) { return (root || document).querySelector(sel); }
  function $all(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }

  function topicKey(moduleId, slug) { return moduleId + "/" + slug; }
  function topicHref(moduleId, slug) { return "#/" + moduleId + "/" + slug; }

  function findModule(id) {
    for (var i = 0; i < CURRICULUM.length; i++) if (CURRICULUM[i].id === id) return CURRICULUM[i];
    return null;
  }

  /* Flat ordered list of all topics for prev/next navigation. */
  var FLAT = [];
  CURRICULUM.forEach(function (m) {
    m.topics.forEach(function (t) {
      FLAT.push({ module: m, topic: t, key: topicKey(m.id, t.slug) });
    });
  });

  function flatIndex(key) {
    for (var i = 0; i < FLAT.length; i++) if (FLAT[i].key === key) return i;
    return -1;
  }

  function moduleReadCount(m) {
    var n = 0;
    m.topics.forEach(function (t) { if (progress[topicKey(m.id, t.slug)]) n++; });
    return n;
  }

  function totalReadCount() {
    var n = 0;
    FLAT.forEach(function (e) { if (progress[e.key]) n++; });
    return n;
  }

  /* ---------- theme ---------- */

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    $("#icon-moon").style.display = theme === "light" ? "" : "none";
    $("#icon-sun").style.display = theme === "dark" ? "" : "none";
  }

  var savedTheme = null;
  try { savedTheme = localStorage.getItem(LS_THEME); } catch (e) { }
  applyTheme(savedTheme === "dark" ? "dark" : "light");

  $("#theme-toggle").addEventListener("click", function () {
    var next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    applyTheme(next);
    try { localStorage.setItem(LS_THEME, next); } catch (e) { }
  });

  /* ---------- sidebar ---------- */

  var sidebarNav = $("#sidebar-nav");
  var searchTerm = "";

  function renderSidebar() {
    var currentKey = currentRouteKey();
    var html = "";
    var anyShown = false;

    CURRICULUM.forEach(function (m) {
      var topics = m.topics.filter(function (t) {
        if (!searchTerm) return true;
        return t.title.toLowerCase().indexOf(searchTerm) !== -1 ||
               m.title.toLowerCase().indexOf(searchTerm) !== -1;
      });
      if (searchTerm && topics.length === 0) return;
      anyShown = true;

      var isOpen = searchTerm ? true : !!openModules[m.id];
      var read = moduleReadCount(m);

      html += '<div class="nav-module' + (isOpen ? " open" : "") + '" data-module="' + m.id + '">' +
        '<button class="nav-module-header" data-toggle="' + m.id + '">' +
          '<span class="nav-module-num">' + m.num + '</span>' +
          '<span class="nav-module-title">' + m.title + '</span>' +
          '<span class="nav-module-count">' + read + "/" + m.topics.length + '</span>' +
          '<svg class="nav-module-chevron" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>' +
        '</button><ul class="nav-topics">';

      topics.forEach(function (t) {
        var key = topicKey(m.id, t.slug);
        var isRead = !!progress[key];
        var isActive = key === currentKey;
        html += '<li class="nav-topic' + (isRead ? " is-read" : "") + '">' +
          '<a href="' + topicHref(m.id, t.slug) + '"' + (isActive ? ' class="active"' : "") + '>' +
          '<span class="read-mark">✓</span><span>' + t.title + '</span></a></li>';
      });

      html += "</ul></div>";
    });

    if (!anyShown) html = '<p class="nav-empty">No topics match "' + escapeHTML(searchTerm) + '".</p>';
    sidebarNav.innerHTML = html;
  }

  function escapeHTML(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  sidebarNav.addEventListener("click", function (e) {
    var btn = e.target.closest("[data-toggle]");
    if (btn) {
      var id = btn.getAttribute("data-toggle");
      if (searchTerm) return; // while searching, everything stays expanded
      openModules[id] = !openModules[id];
      saveJSON(LS_OPEN, openModules);
      renderSidebar();
      return;
    }
    if (e.target.closest("a")) closeMobileSidebar();
  });

  /* search */
  $("#search-input").addEventListener("input", function (e) {
    searchTerm = e.target.value.trim().toLowerCase();
    renderSidebar();
    if (searchTerm && window.innerWidth <= 900) document.body.classList.add("sidebar-open");
  });

  /* mobile sidebar */
  $("#menu-toggle").addEventListener("click", function () {
    document.body.classList.toggle("sidebar-open");
  });
  $("#sidebar-overlay").addEventListener("click", closeMobileSidebar);
  function closeMobileSidebar() { document.body.classList.remove("sidebar-open"); }

  /* ---------- routing ---------- */

  function parseHash() {
    var h = location.hash.replace(/^#\/?/, "");
    if (!h) return { page: "home" };
    var parts = h.split("/");
    var m = findModule(parts[0]);
    if (m && parts[1]) {
      for (var i = 0; i < m.topics.length; i++) {
        if (m.topics[i].slug === parts[1]) return { page: "topic", module: m, topic: m.topics[i] };
      }
    }
    return { page: "home" };
  }

  function currentRouteKey() {
    var r = parseHash();
    return r.page === "topic" ? topicKey(r.module.id, r.topic.slug) : null;
  }

  window.addEventListener("hashchange", route);

  function route() {
    var r = parseHash();
    if (r.page === "home") renderHome();
    else renderTopic(r.module, r.topic);
    window.scrollTo(0, 0);
    closeMobileSidebar();
  }

  /* ---------- home page ---------- */

  function renderHome() {
    var total = FLAT.length;
    var read = totalReadCount();
    var next = null;
    for (var i = 0; i < FLAT.length; i++) { if (!progress[FLAT[i].key]) { next = FLAT[i]; break; } }
    if (!next) next = FLAT[0];

    var html =
      '<h1>AI Launchpad</h1>' +
      '<p class="lead">A free, structured course for learning Machine Learning and Deep Learning from scratch - ' +
      'written like a textbook, organized like a syllabus. Every topic follows the same path: ' +
      '<em>intuition → theory → math → diagram → code → quiz</em>.</p>' +

      '<p><a class="btn" href="' + topicHref(next.module.id, next.topic.slug) + '">' +
      (read === 0 ? "Start learning" : "Continue: " + next.topic.title) + ' →</a> ' +
      '<a class="btn secondary" href="#/getting-started/roadmap">View the roadmap</a></p>' +

      '<div class="module-progress"><div class="mp-label"><span>Course progress</span><span>' +
      read + " of " + total + ' topics read</span></div>' +
      '<div class="mp-track"><div class="mp-fill" style="width:' + Math.round(100 * read / total) + '%"></div></div></div>' +

      '<h2>Modules</h2><div class="home-modules">';

    CURRICULUM.forEach(function (m) {
      var mr = moduleReadCount(m);
      var pct = Math.round(100 * mr / m.topics.length);
      html += '<div class="home-module">' +
        '<span class="hm-num">' + m.num + '</span>' +
        '<div class="hm-body">' +
          '<a class="hm-title" href="' + topicHref(m.id, m.topics[0].slug) + '">' + m.title + '</a>' +
          '<div class="hm-desc">' + m.desc + '</div>' +
        '</div>' +
        '<div class="hm-progress">' + mr + "/" + m.topics.length +
          '<div class="mp-track"><div class="mp-fill" style="width:' + pct + '%"></div></div>' +
        '</div></div>';
    });

    html += "</div>";
    setContent(html, []);
    document.title = "AI Launchpad - Learn ML & Deep Learning Step by Step";
  }

  /* ---------- topic pages ---------- */

  function renderTopic(m, t) {
    var key = topicKey(m.id, t.slug);
    var entry = CONTENT[key];
    var idx = m.topics.indexOf(t);

    /* mark as read up front (real pages only) so the tracker below counts it */
    if (entry && !progress[key]) {
      progress[key] = true;
      saveJSON(LS_PROGRESS, progress);
    }

    var crumbs = '<nav class="breadcrumbs"><a href="#/">Home</a><span class="sep">›</span>' +
      '<a href="' + topicHref(m.id, m.topics[0].slug) + '">' + m.num + ". " + m.title + '</a>' +
      '<span class="sep">›</span><span>' + t.title + "</span></nav>";

    var read = moduleReadCount(m);
    var tracker = '<div class="module-progress"><div class="mp-label">' +
      '<span>Topic ' + (idx + 1) + " of " + m.topics.length + " in <strong>" + m.title + "</strong></span>" +
      '<span>' + read + " read</span></div>" +
      '<div class="mp-track"><div class="mp-fill" style="width:' + Math.round(100 * read / m.topics.length) + '%"></div></div></div>';

    var body;
    if (entry) {
      body = entry.html;
    } else {
      body = stubPage(t);
    }

    var html = crumbs + tracker + body + resourcesHTML(key) + pagerHTML(key);
    setContent(html, entry ? null : []);
    document.title = t.title + " - AI Launchpad";
    renderSidebar();
  }

  var RESOURCE_META = {
    paper: { label: "Paper", cls: "res-paper" },
    video: { label: "Video", cls: "res-video" },
    read: { label: "Read", cls: "res-read" },
    interactive: { label: "Interactive", cls: "res-interactive" }
  };

  function resourcesHTML(key) {
    var list = window.RESOURCES && RESOURCES[key];
    if (!list || !list.length) return "";
    var html = '<div class="go-deeper"><h2>Go deeper</h2><ul class="resource-list">';
    list.forEach(function (r) {
      var type = r[0], title = r[1], source = r[2], url = r[3];
      var meta = RESOURCE_META[type] || RESOURCE_META.read;
      if (!/^https?:\/\//i.test(url)) return; // safety: only allow http(s) links
      html += '<li class="resource-item">' +
        '<span class="res-tag ' + meta.cls + '">' + meta.label + '</span>' +
        '<a href="' + escapeHTML(url) + '" target="_blank" rel="noopener noreferrer">' + escapeHTML(title) + '</a>' +
        '<span class="res-source">' + escapeHTML(source) + '</span></li>';
    });
    return html + "</ul></div>";
  }

  function stubPage(t) {
    return '<h1>' + t.title + '</h1>' +
      '<div class="stub-note"><strong>This chapter is being written.</strong> ' +
      'Use the Next button to continue - the surrounding chapters may already be live.</div>' +
      '<div class="stub-outline">' +
      '<p>When published, this page will teach the concept the same way every other ' +
      'chapter does: intuition first, then the formal idea and the math, with diagrams ' +
      'and runnable code wherever they genuinely help - plus interview questions and a ' +
      'quick self-check quiz at the end.</p></div>';
  }

  function pagerHTML(key) {
    var i = flatIndex(key);
    var prev = i > 0 ? FLAT[i - 1] : null;
    var next = i >= 0 && i < FLAT.length - 1 ? FLAT[i + 1] : null;
    var html = '<div class="pager">';
    if (prev) {
      html += '<a href="' + topicHref(prev.module.id, prev.topic.slug) + '">' +
        '<span class="pager-dir">← Previous</span><span class="pager-title">' + prev.topic.title + "</span></a>";
    } else html += '<span class="pager-spacer"></span>';
    if (next) {
      html += '<a class="pager-next" href="' + topicHref(next.module.id, next.topic.slug) + '">' +
        '<span class="pager-dir">Next →</span><span class="pager-title">' + next.topic.title + "</span></a>";
    } else html += '<span class="pager-spacer"></span>';
    return html + "</div>";
  }

  /* ---------- content post-processing ---------- */

  var contentEl = $("#content");
  var tocEl = $("#page-toc");

  /**
   * Render HTML into the content area, then run KaTeX, Prism,
   * copy buttons, quiz wiring and the on-page TOC.
   * @param tocOverride pass [] to hide the TOC (home/stub pages).
   */
  function setContent(html, tocOverride) {
    contentEl.innerHTML = html;

    /* KaTeX */
    if (window.renderMathInElement) {
      renderMathInElement(contentEl, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "\\(", right: "\\)", display: false }
        ],
        throwOnError: false
      });
    }

    /* Prism + copy buttons */
    $all("pre code", contentEl).forEach(function (code) {
      var pre = code.parentElement;
      if (!pre.parentElement.classList.contains("code-block")) {
        var wrap = document.createElement("div");
        wrap.className = "code-block";
        pre.parentNode.insertBefore(wrap, pre);
        wrap.appendChild(pre);
        var btn = document.createElement("button");
        btn.className = "copy-btn";
        btn.textContent = "Copy";
        btn.addEventListener("click", function () {
          var text = code.textContent;
          function done() { btn.textContent = "Copied!"; setTimeout(function () { btn.textContent = "Copy"; }, 1500); }
          if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(done);
          } else {
            var ta = document.createElement("textarea");
            ta.value = text; document.body.appendChild(ta); ta.select();
            document.execCommand("copy"); document.body.removeChild(ta); done();
          }
        });
        wrap.appendChild(btn);
      }
    });
    if (window.Prism) Prism.highlightAllUnder(contentEl);

    /* on-page TOC from h2 headings */
    var headings = tocOverride !== null && tocOverride !== undefined ? tocOverride : $all("h2", contentEl);
    if (headings.length >= 2) {
      var toc = '<div class="toc-title">On this page</div><ul>';
      headings.forEach(function (h, i) {
        if (!h.id) h.id = "sec-" + i + "-" + h.textContent.toLowerCase().replace(/[^a-z0-9]+/g, "-").slice(0, 40);
        toc += '<li><a href="#' + h.id + '" data-scroll>' + escapeHTML(h.textContent) + "</a></li>";
      });
      tocEl.innerHTML = toc + "</ul>";
    } else {
      tocEl.innerHTML = "";
    }
  }

  /* TOC links: smooth-scroll without breaking the hash router */
  tocEl.addEventListener("click", function (e) {
    var a = e.target.closest("a[data-scroll]");
    if (!a) return;
    e.preventDefault();
    var target = document.getElementById(a.getAttribute("href").slice(1));
    if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
  });

  /* quiz interaction (event delegation, works for all rendered quizzes) */
  contentEl.addEventListener("click", function (e) {
    var opt = e.target.closest(".quiz-opt");
    if (!opt || opt.disabled) return;
    var q = opt.closest(".quiz-q");
    var answer = parseInt(q.getAttribute("data-answer"), 10);
    var opts = $all(".quiz-opt", q);
    opts.forEach(function (o, i) {
      o.disabled = true;
      if (i === answer) o.classList.add("correct");
    });
    if (opts.indexOf(opt) !== answer) opt.classList.add("wrong");
    var explain = $(".quiz-explain", q);
    if (explain) explain.classList.remove("hidden");
  });

  /* ---------- boot ---------- */

  /* auto-open the module of the current route */
  (function () {
    var r = parseHash();
    if (r.page === "topic") { openModules[r.module.id] = true; saveJSON(LS_OPEN, openModules); }
  })();

  renderSidebar();
  route();
})();
