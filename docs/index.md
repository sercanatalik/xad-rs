<!-- Landing page of the documentation site. Written as HTML so the layout can
     be richer than a chapter; the classes are styled in theme/xad.css. Figures
     here are drawn from README.md and the examples, not measured separately:
     when a number changes there, change it here too. -->

<div class="xad-landing-page">

<header class="xl-hero">
  <div class="xl-hero-text">
    <div class="xl-eyebrow">Automatic differentiation · Rust</div>
    <h1 class="xl-title">Write the function once. Choose the derivatives later.</h1>
    <p class="xl-lede">A body written once as <code>fn f&lt;R: Real&gt;</code> runs under forward mode, reverse mode, first order or second. The mode decides which derivatives come back with the value, never what the value is. Every one of them is exact to machine precision: no bump size to tune, no symbolic blow-up.</p>
    <div class="xl-actions">
      <a class="xl-btn xl-btn-primary" href="theory/01-automatic-differentiation.html">Read the theory</a>
      <a class="xl-btn xl-btn-secondary" href="https://docs.rs/xad-rs">API on docs.rs</a>
    </div>
    <dl class="xl-facts">
      <div><dt>Cargo.toml</dt><dd><code>xad-rs = "8.0"</code></dd></div>
      <div><dt>MSRV</dt><dd>Rust 1.85, edition 2024</dd></div>
      <div><dt>Licence</dt><dd>MIT</dd></div>
    </dl>
  </div>
  <figure class="xl-figure">
    <svg viewBox="0 0 560 300" role="img" aria-label="Schematic of gradient cost against number of inputs for forward, K-lane forward and reverse mode">
      <line x1="56" y1="20" x2="56" y2="250" stroke="#101820" stroke-width="1"></line>
      <line x1="56" y1="250" x2="540" y2="250" stroke="#101820" stroke-width="1"></line>
      <g stroke="#E4E8ED" stroke-width="1">
        <line x1="56" y1="200" x2="540" y2="200"></line><line x1="56" y1="150" x2="540" y2="150"></line><line x1="56" y1="100" x2="540" y2="100"></line><line x1="56" y1="50" x2="540" y2="50"></line>
        <line x1="177" y1="20" x2="177" y2="250"></line><line x1="298" y1="20" x2="298" y2="250"></line><line x1="419" y1="20" x2="419" y2="250"></line>
      </g>
      <text x="56" y="270" fill="#4A5563" text-anchor="middle">0</text>
      <text x="177" y="270" fill="#4A5563" text-anchor="middle">8</text>
      <text x="298" y="270" fill="#4A5563" text-anchor="middle">16</text>
      <text x="419" y="270" fill="#4A5563" text-anchor="middle">24</text>
      <text x="540" y="270" fill="#4A5563" text-anchor="middle">32</text>
      <text x="540" y="292" fill="#4A5563" text-anchor="end">inputs n</text>
      <text x="20" y="24" fill="#4A5563" transform="rotate(-90 20 24)" text-anchor="end">cost per gradient</text>
      <polyline points="56,250 116,190 176,130 236,70 296,10" fill="none" stroke="#101820" stroke-width="2"></polyline>
      <polyline points="56,250 177,232 177,214 298,196 298,178 419,160 419,142 540,124" fill="none" stroke="#4A5563" stroke-width="2" stroke-dasharray="6 4"></polyline>
      <line x1="56" y1="214" x2="540" y2="214" stroke="#1F4FD8" stroke-width="2.5"></line>
      <circle cx="116" cy="214" r="4" fill="#FFFFFF" stroke="#1F4FD8" stroke-width="2"></circle>
      <text x="124" y="206" fill="#1F4FD8">n ≈ 4</text>
      <text x="300" y="24" fill="#101820">Jet1 × n</text>
      <text x="452" y="116" fill="#4A5563">JetK&lt;8&gt;, ⌈n/8⌉ passes</text>
      <text x="452" y="238" fill="#1F4FD8">one reverse sweep</text>
    </svg>
    <figcaption><b>Fig. 1</b> Cost of a full gradient against the number of inputs, schematic. Forward mode pays one pass per input, K lanes per pass move the crossover out, reverse mode pays one sweep. Measured values are in <code>examples/jetk_gradient.rs</code>.</figcaption>
  </figure>
</header>

<section class="xl-section" id="modes">
  <div class="xl-head">
    <span class="xl-num">§ 1</span>
    <h2 class="xl-h">Pick a mode</h2>
    <p>Three numbers decide it: inputs n, outputs m, and the order. Every mode returns a bit-identical value; the mode only decides which derivatives come back with it.</p>
  </div>
  <div class="xl-table">
    <div class="xl-row xl-row-head"><span>Your problem</span><span>Mode</span><span>Chapter</span></div>
    <div class="xl-row"><span>Just the value, no derivatives</span><span><code>f64</code></span><a href="theory/01-automatic-differentiation.html">§ 1</a></div>
    <div class="xl-row"><span>One input direction, any number of outputs</span><span><code>Jet1&lt;T&gt;</code> or <code>compute_derivative_fwd</code></span><a href="theory/02-forward-mode-and-dual-numbers.html">§ 2</a></div>
    <div class="xl-row"><span>Full gradient, n ≲ 16 inputs, no tape</span><span><code>compute_gradient_fwd_k::&lt;K, _&gt;</code>, ⌈n/K⌉ passes of <code>JetK</code></span><a href="theory/02-forward-mode-and-dual-numbers.html">§ 2</a></div>
    <div class="xl-row"><span>Full gradient, any number of inputs, scalar output</span><span><code>compute_gradient_rev</code> or <code>Tape</code> + <code>AReal&lt;T&gt;</code></span><a href="theory/03-reverse-mode-and-taped-adjoints.html">§ 3</a></div>
    <div class="xl-row"><span>Gamma or diagonal Hessian along one direction</span><span><code>Jet2&lt;T&gt;</code></span><a href="theory/04-second-order-and-k-jets.html">§ 4</a></div>
    <div class="xl-row"><span>Full n × n Hessian</span><span><code>compute_hessian_k::&lt;K, _&gt;</code>, ⌈n/K⌉ passes of <code>Tape&lt;JetK&gt;</code></span><a href="theory/04-second-order-and-k-jets.html">§ 4</a></div>
    <div class="xl-row"><span>Value, gradient and Hessian from one evaluation, no tape</span><span><code>Jet2Vec</code> via <code>compute_full_hessian</code></span><a href="theory/04-second-order-and-k-jets.html">§ 4</a></div>
  </div>
  <div class="xl-caption"><b>Table 1</b> Mode selection. Against one forward pass per input, reverse mode breaks even around n ≈ 4. Pick K ≈ n rounded up to the next of 4, 8 or 16; idle lanes cost register pressure.</div>
</section>

<section class="xl-section" id="theory">
  <div class="xl-head">
    <span class="xl-num">§ 2</span>
    <h2 class="xl-h">Theory</h2>
    <p>Six chapters, each a 15 to 25 minute read. Rustdoc says how to call the API; these say why the math works and how the implementation behaves.</p>
  </div>
  <div class="xl-grid">
    <a class="xl-card" href="theory/01-automatic-differentiation.html"><span class="xl-num">§ 1</span><span class="xl-card-title">Automatic differentiation as a discipline</span><span class="xl-card-body">Why AD is exact, how it differs from bumping and from symbolic algebra, the Wengert list, and the role of the <code>Real</code> trait.</span></a>
    <a class="xl-card" href="theory/02-forward-mode-and-dual-numbers.html"><span class="xl-num">§ 2</span><span class="xl-card-title">Forward mode and dual numbers</span><span class="xl-card-body">Dual numbers as ℝ[ε]/(ε²), the chain rule as a one-line proof, Jacobian-vector products, and the K-wide extension that gives <code>JetK</code>.</span></a>
    <a class="xl-card" href="theory/03-reverse-mode-and-taped-adjoints.html"><span class="xl-num">§ 3</span><span class="xl-card-title">Reverse mode and taped adjoints</span><span class="xl-card-body">The adjoint recurrence, Baur–Strassen and the cheap-gradient principle, the packed three-buffer tape, and tape reuse.</span></a>
    <a class="xl-card" href="theory/04-second-order-and-k-jets.html"><span class="xl-num">§ 4</span><span class="xl-card-title">Second order and k-jets</span><span class="xl-card-body">Truncated Taylor series, Faà di Bruno, <code>Jet2</code> and <code>Jet2Vec</code>, forward-over-adjoint, and K Hessian columns per sweep.</span></a>
    <a class="xl-card" href="theory/05-implementation-tradeoffs.html"><span class="xl-num">§ 5</span><span class="xl-card-title">Implementation tradeoffs</span><span class="xl-card-body">Operator overloading versus source transformation, the tape layout, thread-local active tapes, and where Rust's type system helps.</span></a>
    <a class="xl-card" href="theory/06-numerical-analysis-of-ad.html"><span class="xl-num">§ 6</span><span class="xl-card-title">Numerical analysis of AD</span><span class="xl-card-body">Round-off in AD against truncation and cancellation in finite differences, the √u accuracy floor, and the complex-step trick.</span></a>
  </div>
</section>

<section class="xl-section" id="examples">
  <div class="xl-head">
    <span class="xl-num">§ 3</span>
    <h2 class="xl-h">Examples</h2>
    <code class="xl-cmd">cargo run --release --example swap_pricer</code>
  </div>
  <div class="xl-examples">
    <a href="https://github.com/sercanatalik/xad-rs/blob/main/examples/real_generic_pricer.rs"><code>real_generic_pricer.rs</code><span>One <code>call_price&lt;R: Real&gt;</code> body priced under f64, AReal, Jet1 and Jet2, the three first-order drivers, and <code>weighted_sum</code> recording one tape statement where a <code>+</code> chain records 16.</span></a>
    <a href="https://github.com/sercanatalik/xad-rs/blob/main/examples/swap_pricer.rs"><code>swap_pricer.rs</code><span>30-input IRS: DV01 via reverse, diagonal gamma via Jet2, and the full 30×30 Hessian via <code>compute_hessian_k</code>.</span></a>
    <a href="https://github.com/sercanatalik/xad-rs/blob/main/examples/fx_option.rs"><code>fx_option.rs</code><span>Garman–Kohlhagen FX option greeks via reverse mode and Jet2 spot gamma, cross-checked against analytic.</span></a>
    <a href="https://github.com/sercanatalik/xad-rs/blob/main/examples/fixed_rate_bond.rs"><code>fixed_rate_bond.rs</code><span>Yield to maturity, duration and convexity.</span></a>
    <a href="https://github.com/sercanatalik/xad-rs/blob/main/examples/jetk_gradient.rs"><code>jetk_gradient.rs</code><span>The K-lane forward gradient against reverse mode and Jet1 × n, on a 6-input and a 30-input body, with the crossover figures.</span></a>
    <a href="https://github.com/sercanatalik/xad-rs/blob/main/examples/jacobian.rs"><code>jacobian.rs</code><span>A 4×4 Jacobian via reverse mode.</span></a>
    <a href="https://github.com/sercanatalik/xad-rs/blob/main/examples/hessian.rs"><code>hessian.rs</code><span>A 4×4 Hessian via <code>compute_full_hessian</code> with an analytic cross-check.</span></a>
    <a href="https://github.com/sercanatalik/xad-rs/blob/main/examples/adjoint_first_order.rs"><code>adjoint_first_order.rs</code><span>A full 4-input gradient in one reverse sweep.</span></a>
    <a href="https://github.com/sercanatalik/xad-rs/blob/main/examples/fwd_adj_second_order.rs"><code>fwd_adj_second_order.rs</code><span>Forward-over-adjoint: one seeded sweep over <code>Tape&lt;Jet1&lt;f64&gt;&gt;</code> gives the gradient and a Hessian row, cross-checked against <code>compute_hessian</code>.</span></a>
  </div>
</section>

<section class="xl-section" id="performance">
  <div class="xl-head">
    <span class="xl-num">§ 4</span>
    <h2 class="xl-h">Performance</h2>
    <p>Built for many small valuations: a curve bootstrap is a Newton loop, risk is positions times scenarios. Measured on Apple M-series with fat LTO; the examples that measure ship with the crate.</p>
  </div>
  <div class="xl-table xl-perf">
    <div class="xl-row xl-row-head"><span>Lever</span><span>What it does</span><span class="xl-right">Measured</span></div>
    <div class="xl-row"><span><strong>Reuse the tape</strong></span><span class="xl-muted"><code>Tape::record</code> instead of a fresh tape per valuation. A fresh tape reserves room for a small valuation, so this matters most once a recording outgrows the reserve.</span><span class="xl-right">~1.3× at n = 6</span></div>
    <div class="xl-row"><span><strong>Widen the Hessian pass</strong></span><span class="xl-muted"><code>compute_hessian_k::&lt;K, _&gt;</code> seeds K tangent lanes per recording, so an n × n Hessian costs ⌈n/K⌉ passes instead of n. On a 48-input, 2000-op kernel at K = 8.</span><span class="xl-right">8.2×, 14.7× parallel</span></div>
    <div class="xl-row"><span><strong>Vector reverse mode</strong></span><span class="xl-muted"><code>compute_jacobian_rev</code> recovers a full m × n Jacobian in one sweep.</span><span class="xl-right">~1.75×</span></div>
    <div class="xl-row"><span><strong>Parallelise</strong></span><span class="xl-muted">Independent valuations with <code>rayon</code> plus one <code>Tape::record</code> per worker. The tape is thread-local, so workers need no coordination.</span><span class="xl-right">—</span></div>
    <div class="xl-row"><span><strong>Primal is free</strong></span><span class="xl-muted">A <code>fn f&lt;R: Real&gt;</code> body at R = f64 against hand-written f64. Monomorphization erases the trait.</span><span class="xl-right">within ~1%</span></div>
  </div>
  <div class="xl-caption"><b>Table 2</b> Measured levers. Rejected after measurement: a struct-of-arrays tape layout (regresses small cache-resident tapes 12–16%) and expression-template fusion (no isolated bottleneck).</div>
</section>

</div>
