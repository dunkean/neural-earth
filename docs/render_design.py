"""Render the complete design offline. Requires Markdown==3.8.2.

Usage: python docs/render_design.py [--markdown-lib /path/to/extra/packages]
The optional path keeps this documentation dependency out of the app environment.
"""
import argparse
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--markdown-lib", type=Path)
args = parser.parse_args()
if args.markdown_lib:
    sys.path.insert(0, str(args.markdown_lib))

import markdown

directory = Path(__file__).resolve().parent
source = directory / "TERRAIN_REALTIME_DESIGN.md"
renderer = markdown.Markdown(extensions=["tables", "fenced_code", "toc"],
                             extension_configs={"toc": {"toc_depth": "2-3"}})
body = renderer.convert(source.read_text(encoding="utf-8"))
body = body.replace("<table>", '<div class="table-scroll"><table>')
body = body.replace("</table>", "</table></div>")
template = r'''<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Agent WebGPU prioritaire, agent natif open source optionnel et adaptation aux GPU 3090, 4090, 5090 : design Terrain Diffusion à 30 m/pixel.">
<title>Terrain Diffusion — design du moteur temps réel</title>
<style>
:root{color-scheme:light;--paper:#fff;--bg:#f3f5f2;--ink:#172924;--muted:#56665e;--line:#dce5de;--accent:#146947;--code:#edf2ef}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:26px}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.68 system-ui,-apple-system,"Segoe UI",sans-serif}a{color:var(--accent);text-decoration-thickness:1px;text-underline-offset:3px}a:hover{color:#0c462e}button,input{font:inherit}button{cursor:pointer}header{background:#173c2d;color:#fff;padding:42px max(24px,calc((100vw - 1390px)/2));border-bottom:5px solid #72bd92}.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:.13em;color:#b5d6bf}.hero-title{font-size:clamp(28px,4vw,46px);line-height:1.12;font-weight:700;margin:11px 0 18px;max-width:950px}.hero-subtitle{color:#d5e6d9;max-width:850px;font-size:17px;margin:0}.hero-links{display:flex;gap:18px;flex-wrap:wrap;margin-top:22px}.hero-links a{color:#fff;font-size:14px}.layout{max-width:1440px;margin:auto;display:grid;grid-template-columns:285px minmax(0,1fr);gap:26px;padding:26px 24px 60px}aside{position:sticky;top:20px;align-self:start;max-height:calc(100vh - 40px);overflow-y:auto;padding:0 10px 0 0;font-size:13px}aside h2{font-size:13px;text-transform:uppercase;letter-spacing:.09em;margin:0 0 12px}.toc ul{list-style:none;padding:0;margin:0}.toc li{margin:0}.toc a{display:block;padding:7px 9px;color:#40564a;text-decoration:none;border-left:2px solid transparent;line-height:1.4;border-radius:0 5px 5px 0}.toc a:hover,.toc a.active{background:#e5ede7;border-left-color:var(--accent);color:var(--accent)}.toc ul ul{padding-left:11px;font-size:12px}.sidebar-note{color:var(--muted);border-top:1px solid var(--line);padding-top:12px;font-size:12px}main{background:var(--paper);padding:32px clamp(20px,4vw,55px) 48px;border:1px solid var(--line);border-radius:12px;min-width:0;box-shadow:0 4px 24px #193c2a06}article h1{font-size:27px;line-height:1.25;margin-top:0}article h2{font-size:24px;line-height:1.3;margin:52px 0 19px;border-top:1px solid var(--line);padding-top:27px}article h3{font-size:19px;line-height:1.35;margin:31px 0 14px}p{margin:0 0 17px}li{margin-bottom:10px}ul,ol{padding-left:25px}strong{font-weight:650}.table-scroll{overflow:auto;margin:22px 0 26px;border:1px solid var(--line);border-radius:7px}table{width:100%;border-collapse:collapse;font-size:14px;line-height:1.5}th,td{text-align:left;padding:12px 13px;vertical-align:top;border-bottom:1px solid var(--line);min-width:140px}th{font-weight:650;background:#eaf1ec;color:#194b34}tr:last-child td{border-bottom:0}tr:nth-child(even) td{background:#fafcfb}code{font:13px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;background:var(--code);padding:2px 4px;border-radius:3px;overflow-wrap:anywhere}pre{background:#172c24;color:#e6f2e9;padding:20px;border-radius:8px;overflow:auto;line-height:1.55}pre code{background:none;color:inherit;padding:0;overflow-wrap:normal}blockquote{border-left:3px solid var(--accent);padding-left:17px;color:var(--muted)}.calculator{background:#edf5ef;border:1px solid #caddcc;border-radius:8px;padding:22px;margin-bottom:32px}.calculator h2{font-size:18px;margin:0 0 6px}.calculator p{font-size:14px;color:var(--muted);margin:0 0 14px}.inputs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.inputs label{font-size:12px;color:var(--muted);display:block}.inputs input{width:100%;padding:7px;border:1px solid #b6cbbb;border-radius:5px;color:var(--ink);background:#fff}.calc-output{font-size:14px;margin-top:14px;border-top:1px solid #caddcc;padding-top:12px}.calc-output strong{font-variant-numeric:tabular-nums}.toolbar{display:none}.footer{margin-top:40px;padding-top:17px;border-top:1px solid var(--line);font-size:12px;color:var(--muted)}.reading-progress{position:fixed;top:0;left:0;height:3px;width:0;background:#76c79b;z-index:20}
@media(max-width:950px){.layout{grid-template-columns:220px minmax(0,1fr);gap:15px;padding:20px 14px}main{padding:24px}aside{font-size:12px}.toc ul ul{display:none}.inputs{grid-template-columns:1fr 1fr}}
@media(max-width:700px){header{padding:30px 21px}.layout{display:block;padding:14px 10px}aside{position:static;max-height:none;padding:0 12px 16px;display:none}aside.open{display:block}main{padding:24px 18px;border-radius:8px}.toolbar{display:flex;gap:10px;padding:0 12px 14px}.toolbar button{background:#fff;color:var(--accent);border:1px solid var(--line);border-radius:5px;padding:7px 11px;font-size:13px}article h2{font-size:22px}article h1{font-size:24px}th,td{padding:10px;min-width:150px}.inputs{grid-template-columns:1fr 1fr}.calculator{padding:17px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
@media print{body{background:#fff;color:#000;font-size:10pt}header,aside,.toolbar,.reading-progress,.calculator{display:none}.layout{display:block;padding:0}main{border:0;box-shadow:none;padding:0}article h1{font-size:19pt}article h2{font-size:15pt;break-after:avoid;margin-top:24pt;padding-top:12pt}article h3{break-after:avoid;font-size:12pt}pre{white-space:pre-wrap;background:#eee;color:#000;break-inside:avoid}.table-scroll{overflow:visible;border-radius:0}table{font-size:8pt}th,td{padding:6pt;min-width:0}tr{break-inside:avoid}a{color:#000}.footer{display:none}}
</style>
</head>
<body>
<div class="reading-progress" aria-hidden="true"></div>
<header>
  <div class="eyebrow">Analyse & design · 7 octobre 2026 · v1.1</div>
  <div class="hero-title">Un moteur pour explorer<br>un monde généré à la demande.</div>
  <p class="hero-subtitle">Terrain Diffusion à 30 m/pixel : agent WebGPU prioritaire, agent natif open source optionnel et usage adaptatif des GPU, de la RTX 3090 à la machine RTX 5090 + 4090.</p>
  <div class="hero-links"><a href="TERRAIN_REALTIME_DESIGN.md" download>Markdown source</a><a href="audit-session.json">Mesures de session</a><a href="#11-site-public-agent-webgpu-prioritaire-wasm-et-gpu-des-visiteurs">Agent WebGPU et adaptation GPU</a></div>
</header>
<div class="layout">
  <div class="toolbar"><button id="toc-toggle" aria-expanded="false" aria-controls="navigation">Sommaire</button><button id="print">Imprimer</button></div>
  <aside id="navigation" aria-label="Sommaire"><h2>Dans ce document</h2>__TOC__<p class="sidebar-note">Architecture proposée. Les budgets sont des objectifs à mesurer ; aucun gain futur n’est présenté comme acquis.</p></aside>
  <main>
    <section class="calculator" aria-labelledby="calc-title">
      <h2 id="calc-title">Dimensionner la demande de navigation</h2>
      <p>Exemple à 30 m/pixel, déplacement horizontal. Le facteur requis apparaît seulement si tu fournis un débit de référence mesuré.</p>
      <div class="inputs">
        <label>Largeur, pixels natifs<input id="width" type="number" min="1" max="16000" value="1920"></label>
        <label>Hauteur, pixels natifs<input id="height" type="number" min="1" max="16000" value="1080"></label>
        <label>Déplacement, pixels/s<input id="speed" type="number" min="0" max="100000" value="600"></label>
        <label>Débit mesuré, pixels finaux/s<input id="baseline" type="number" min="0" placeholder="À mesurer"></label>
      </div>
      <div id="calc-output" class="calc-output" aria-live="polite"></div>
    </section>
    <article>__BODY__</article>
    <div class="footer">Document complet généré depuis le Markdown du projet. Lecture hors ligne, sans bibliothèque ni ressource externe chargée par cette page. Les sources techniques restent accessibles par leurs liens.</div>
  </main>
</div>
<script>
const $=id=>document.getElementById(id);
const fmt=n=>new Intl.NumberFormat('fr-FR',{maximumFractionDigits:2}).format(n);
function calculate(){
  const w=Math.max(1,Number($('width').value)||1),h=Math.max(1,Number($('height').value)||1);
  const v=Math.max(0,Number($('speed').value)||0),rate=h*Math.min(v,w),baseline=Number($('baseline').value);
  const required=baseline>0?` · facteur requis avec 20 % de marge : <strong>×${fmt(1.2*rate/baseline)}</strong>`:'';
  $('calc-output').innerHTML=`Demande sur un intervalle d’une seconde : <strong>${fmt(rate)} pixels/s</strong> · <strong>${fmt(rate/65536)} tuiles 256²/s</strong> · ${fmt(v*30/1000)} km/s${required}`;
}
document.querySelectorAll('.inputs input').forEach(i=>i.addEventListener('input',calculate));calculate();
$('toc-toggle').onclick=()=>{const open=$('navigation').classList.toggle('open');$('toc-toggle').setAttribute('aria-expanded',String(open));};
$('print').onclick=()=>window.print();
const links=[...document.querySelectorAll('.toc a')],headings=[...document.querySelectorAll('article h2,article h3')];
function progress(){
 const max=document.documentElement.scrollHeight-innerHeight;
 document.querySelector('.reading-progress').style.width=`${max>0?100*scrollY/max:0}%`;
 let current=headings[0];for(const h of headings){if(h.getBoundingClientRect().top<130)current=h;else break;}
 for(const link of links)link.classList.toggle('active',!!current&&link.hash==='#'+current.id);
}
addEventListener('scroll',progress,{passive:true});addEventListener('resize',progress);progress();
</script>
</body>
</html>'''
result = template.replace("__TOC__", renderer.toc).replace("__BODY__", body)
destination = directory / "TERRAIN_REALTIME_DESIGN.html"
destination.write_text(result, encoding="utf-8")
print(f"Rendered {destination.name}: {len(body)} characters, complete Markdown body")
