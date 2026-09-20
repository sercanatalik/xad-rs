// Site behaviour on top of mdBook's own scripts:
//   1. renders ```math fences (GitHub's syntax, which the chapters use) with KaTeX;
//   2. builds the "On this page" rail from the chapter's h2/h3 headings;
//   3. turns the sidebar's "1." section labels into "§1";
//   4. gives the previous/next links their chapter titles' section numbers.
(function () {
  'use strict';

  var KATEX = 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/';

  function loadKatex(done) {
    var blocks = document.querySelectorAll('pre > code.language-math');
    if (blocks.length === 0) { return; }
    var css = document.createElement('link');
    css.rel = 'stylesheet';
    css.href = KATEX + 'katex.min.css';
    document.head.appendChild(css);
    var js = document.createElement('script');
    js.src = KATEX + 'katex.min.js';
    js.onload = function () { done(blocks); };
    document.head.appendChild(js);
  }

  function renderMath(blocks) {
    Array.prototype.forEach.call(blocks, function (code) {
      var pre = code.parentNode;
      var box = document.createElement('div');
      box.className = 'xad-math';
      try {
        window.katex.render(code.textContent, box, { displayMode: true, throwOnError: false });
      } catch (e) {
        box.textContent = code.textContent;
      }
      pre.parentNode.replaceChild(box, pre);
    });
  }

  function buildRail() {
    var rail = document.querySelector('.xad-rail');
    var main = document.querySelector('#mdbook-content main');
    if (!rail || !main || document.documentElement.classList.contains('xad-landing')) { return; }
    var heads = main.querySelectorAll('h2, h3');
    if (heads.length < 2) { rail.remove(); return; }
    var label = document.createElement('div');
    label.className = 'xad-rail-label';
    label.textContent = 'On this page';
    rail.appendChild(label);
    var list = document.createElement('ol');
    Array.prototype.forEach.call(heads, function (h) {
      var anchor = h.querySelector('a.header') || h;
      var id = h.id || (anchor.getAttribute('href') || '').replace(/^#/, '');
      if (!id) { return; }
      var li = document.createElement('li');
      li.className = 'xad-rail-' + h.tagName.toLowerCase();
      var a = document.createElement('a');
      a.href = '#' + id;
      a.textContent = h.textContent.trim();
      li.appendChild(a);
      list.appendChild(li);
    });
    rail.appendChild(list);

    // Highlight the heading currently in view.
    var links = list.querySelectorAll('a');
    var byId = {};
    Array.prototype.forEach.call(links, function (a) { byId[a.getAttribute('href').slice(1)] = a; });
    if ('IntersectionObserver' in window) {
      var current = null;
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) {
          if (en.isIntersecting) {
            if (current) { current.classList.remove('active'); }
            current = byId[en.target.id];
            if (current) { current.classList.add('active'); }
          }
        });
      }, { rootMargin: '-72px 0px -70% 0px', threshold: 0 });
      Array.prototype.forEach.call(heads, function (h) { if (h.id) { io.observe(h); } });
    }
  }

  function sectionLabels() {
    var strongs = document.querySelectorAll('#mdbook-sidebar .chapter-item strong');
    Array.prototype.forEach.call(strongs, function (s) {
      var n = s.textContent.trim().replace(/\.$/, '');
      if (/^\d+$/.test(n)) { s.textContent = '§' + n; }
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    loadKatex(renderMath);
    buildRail();
    // The sidebar is populated by a custom element; give it a tick.
    sectionLabels();
    setTimeout(sectionLabels, 0);
  });
})();
