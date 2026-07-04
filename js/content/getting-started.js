/* Module 0 - Getting Started */

CONTENT["getting-started/how-to-use"] = {
  html: String.raw`
<h1>How to use this site</h1>
<p class="lead">AI Launchpad is a structured course, not a collection of blog posts.
Read it in order, top to bottom, and you will go from "what is a vector?" to
"I can explain a Transformer" without gaps.</p>

<h2>The idea: one topic per page</h2>
<p>Each page covers exactly one concept and takes 5–10 minutes to read.
Big topics (like Backpropagation) are split across several pages instead of
one giant wall of text. Long derivations and full code listings are folded
into collapsible blocks - open them when you're ready, skip them on a first pass.</p>

<h2>How every page teaches</h2>
<p>Every page builds the concept the same way a good teacher would:
<strong>plain-English intuition first, then the formal idea, then the math</strong> -
never the other way around. Along the way you'll find:</p>
<ul>
  <li><strong>Boxed formulas</strong> - the equations that matter, highlighted so they're easy to find again when revising.</li>
  <li><strong>Diagrams</strong> - wherever a picture explains something better than a paragraph.</li>
  <li><strong>Runnable code</strong> - wherever the concept is best understood by implementing it. Every block has a Copy button, and the code is commented line-by-line to map back to the formulas.</li>
  <li><strong>Collapsible derivations</strong> - full proofs and long code listings are folded away so the page stays readable; open them when you're ready.</li>
  <li><strong>Interview questions and a quick quiz</strong> - at the end of concept pages, so you can check yourself before moving on.</li>
</ul>

<h2>Tracking your progress</h2>
<p>The site remembers what you've read (stored locally in your browser - no account needed):</p>
<ul>
  <li>A <strong>✓ check mark</strong> appears next to finished topics in the sidebar.</li>
  <li>Each page shows a <strong>module tracker</strong> ("Topic 4 of 12 in Deep Learning Foundations").</li>
  <li>The <a href="#/">home page</a> shows overall course progress and a
      "Continue" button that jumps to your next unread topic.</li>
</ul>
<p>Use the <strong>← Previous / Next →</strong> buttons at the bottom of every page
to move linearly through the course.</p>

<h2>How to study (honest advice)</h2>
<ul>
  <li><strong>Type the code yourself.</strong> Don't copy-paste on the first read.
      Typing forces you to notice every line.</li>
  <li><strong>Do the quizzes.</strong> They're short on purpose. If you get one wrong,
      re-read that section before moving on.</li>
  <li><strong>Don't skip the math modules.</strong> Everything after Module 1 quietly
      assumes you know what a gradient and a dot product are. An hour there saves
      ten hours of confusion in Deep Learning.</li>
  <li><strong>Come back.</strong> The Practice Zone has cheat sheets and a glossary
      designed for revision after your first pass.</li>
</ul>

<h2>Notation used everywhere on this site</h2>
<div class="callout">
  <span class="co-title">One notation, all pages</span>
  We fix the symbols once so every formula on this site reads the same way:
  \(x\) = input features, \(y\) = true target, \(\hat{y}\) = model prediction,
  \(w\) = weights, \(b\) = bias, \(m\) = number of training examples,
  \(n\) = number of features, \(\alpha\) = learning rate, \(J\) = cost function.
  Bold symbols (\(\mathbf{x}, \mathbf{w}\)) are vectors; capital bold (\(\mathbf{X}, \mathbf{W}\)) are matrices.
</div>
<p>Next, check the <a href="#/getting-started/roadmap">prerequisites and full roadmap</a>
to see where you're headed.</p>
`};

CONTENT["getting-started/roadmap"] = {
  html: String.raw`
<h1>Prerequisites &amp; Roadmap</h1>
<p class="lead">Here is the whole journey on one page - what you need before starting,
and how the 13 modules connect.</p>

<h2>Prerequisites</h2>
<p>You need surprisingly little:</p>
<ul>
  <li><strong>Basic Python</strong> - variables, loops, functions, lists and dictionaries.
      If you can write a function that sums a list, you're ready.
      (You do <em>not</em> need to be a software engineer.)</li>
  <li><strong>High-school math</strong> - comfort with algebra. Everything beyond that
      (linear algebra, calculus, statistics) is taught in Module 1.</li>
  <li><strong>A way to run Python</strong> - either locally
      (install <a href="https://www.anaconda.com/download" target="_blank" rel="noopener">Anaconda</a>)
      or free in the browser with
      <a href="https://colab.research.google.com" target="_blank" rel="noopener">Google Colab</a>.</li>
</ul>

<h2>The roadmap</h2>
<p>The course is five phases. Each phase builds strictly on the previous one.
Click any box to jump to that module.</p>

<div class="diagram">
<svg viewBox="0 0 680 640" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Course roadmap: five phases from foundations to deep learning and beyond">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" style="fill: var(--text-muted)"/>
    </marker>
  </defs>

  <!-- Phase 1 -->
  <rect x="10" y="10" width="660" height="90" rx="10" class="d-box-muted"/>
  <text x="24" y="32" class="d-text-accent">PHASE 1 · Foundations</text>
  <a href="#/getting-started/how-to-use"><rect x="24" y="46" width="196" height="38" rx="7" class="d-box-soft"/>
    <text x="122" y="70" text-anchor="middle" class="d-text">0 · Getting Started</text></a>
  <a href="#/math/linear-algebra"><rect x="238" y="46" width="196" height="38" rx="7" class="d-box-soft"/>
    <text x="336" y="70" text-anchor="middle" class="d-text">1 · Math &amp; Stats</text></a>
  <a href="#/python/numpy"><rect x="452" y="46" width="196" height="38" rx="7" class="d-box-soft"/>
    <text x="550" y="70" text-anchor="middle" class="d-text">2 · Python for DS</text></a>

  <line x1="340" y1="100" x2="340" y2="128" class="d-line" marker-end="url(#arr)"/>

  <!-- Phase 2 -->
  <rect x="10" y="132" width="660" height="90" rx="10" class="d-box-muted"/>
  <text x="24" y="154" class="d-text-accent">PHASE 2 · Data Skills</text>
  <a href="#/dataviz/why-visualization"><rect x="24" y="168" width="196" height="38" rx="7" class="d-box-soft"/>
    <text x="122" y="192" text-anchor="middle" class="d-text">3 · Visualization</text></a>
  <a href="#/eda/workflow"><rect x="238" y="168" width="196" height="38" rx="7" class="d-box-soft"/>
    <text x="336" y="192" text-anchor="middle" class="d-text">4 · EDA</text></a>
  <a href="#/features/encoding"><rect x="452" y="168" width="196" height="38" rx="7" class="d-box-soft"/>
    <text x="550" y="192" text-anchor="middle" class="d-text">5 · Feature Engineering</text></a>

  <line x1="340" y1="222" x2="340" y2="250" class="d-line" marker-end="url(#arr)"/>

  <!-- Phase 3 -->
  <rect x="10" y="254" width="660" height="90" rx="10" class="d-box-muted"/>
  <text x="24" y="276" class="d-text-accent">PHASE 3 · Classical Machine Learning</text>
  <a href="#/ml/learning-paradigms"><rect x="190" y="290" width="300" height="38" rx="7" class="d-box-soft"/>
    <text x="340" y="314" text-anchor="middle" class="d-text">6 · ML Algorithms (17 topics)</text></a>

  <line x1="340" y1="344" x2="340" y2="372" class="d-line" marker-end="url(#arr)"/>

  <!-- Phase 4 -->
  <rect x="10" y="376" width="660" height="90" rx="10" class="d-box-muted"/>
  <text x="24" y="398" class="d-text-accent">PHASE 4 · Deep Learning</text>
  <a href="#/dl/perceptron"><rect x="20" y="412" width="152" height="38" rx="7" class="d-box-soft"/>
    <text x="96" y="436" text-anchor="middle" class="d-text" style="font-size:12px">7 · DL Foundations</text></a>
  <a href="#/cnn/why-cnns"><rect x="184" y="412" width="152" height="38" rx="7" class="d-box-soft"/>
    <text x="260" y="436" text-anchor="middle" class="d-text" style="font-size:12px">8 · CNNs</text></a>
  <a href="#/sequence/why-sequence-models"><rect x="348" y="412" width="152" height="38" rx="7" class="d-box-soft"/>
    <text x="424" y="436" text-anchor="middle" class="d-text" style="font-size:12px">9 · Sequence Models</text></a>
  <a href="#/transformers/self-attention"><rect x="512" y="412" width="152" height="38" rx="7" class="d-box-soft"/>
    <text x="588" y="436" text-anchor="middle" class="d-text" style="font-size:12px">10 · Transformers</text></a>

  <line x1="340" y1="466" x2="340" y2="494" class="d-line" marker-end="url(#arr)"/>

  <!-- Phase 5 -->
  <rect x="10" y="498" width="660" height="90" rx="10" class="d-box-muted"/>
  <text x="24" y="520" class="d-text-accent">PHASE 5 · Beyond the Basics</text>
  <a href="#/advanced/autoencoders"><rect x="90" y="534" width="230" height="38" rx="7" class="d-box-soft"/>
    <text x="205" y="558" text-anchor="middle" class="d-text">11 · Advanced Topics</text></a>
  <a href="#/practice/cheat-sheets"><rect x="360" y="534" width="230" height="38" rx="7" class="d-box-soft"/>
    <text x="475" y="558" text-anchor="middle" class="d-text">12 · Practice Zone</text></a>

  <text x="340" y="620" text-anchor="middle" class="d-text-sm">Modules 1 and 2 can be studied in parallel · Practice Zone is useful throughout</text>
</svg>
<div class="caption">The five phases of the course. Each phase assumes only the phases above it.</div>
</div>

<h2>Suggested pacing</h2>
<p>At a steady pace of ~1 hour per day:</p>
<table>
  <tr><th>Phase</th><th>Modules</th><th>Rough time</th></tr>
  <tr><td>1 · Foundations</td><td>0–2</td><td>2–3 weeks</td></tr>
  <tr><td>2 · Data Skills</td><td>3–5</td><td>2–3 weeks</td></tr>
  <tr><td>3 · Classical ML</td><td>6</td><td>3–4 weeks</td></tr>
  <tr><td>4 · Deep Learning</td><td>7–10</td><td>4–6 weeks</td></tr>
  <tr><td>5 · Beyond</td><td>11–12</td><td>1–2 weeks</td></tr>
</table>
<p>These are estimates, not deadlines. Slower with full understanding beats
faster with gaps - especially in Phases 1 and 4.</p>

<h2>Where am I in the big picture?</h2>
<p>Whenever you feel lost, come back to this page (it's always the second link in the
sidebar). The module tracker at the top of every topic page also tells you exactly
where you are within the current module.</p>
<p>Ready? Start with <a href="#/math/linear-algebra">Linear Algebra for ML</a> →</p>
`};
