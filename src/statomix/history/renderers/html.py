"""Self-contained interactive HTML project-history renderer."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

from statomix.history.model import ProjectHistory
from statomix.reporting.html_theme import STATOMIX_HTML_CSS

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ — Statomix history</title>
<style>__CSS__</style>
</head>
<body>
<header><div class="eyebrow">Statomix · read-only project history</div><h1>__TITLE__</h1><p>Artifact lineage, curation decisions, transformations, analyses, and reports · history ID <span id="history-id"></span></p></header>
<div class="controls">
<label>Dataset <select id="dataset-filter"></select></label>
<label>Pipeline <select id="pipeline-filter"></select></label>
<label>Role <select id="role-filter"></select></label>
<label>Status <select id="status-filter"></select></label>
<label>Search <input id="search" type="search" placeholder="ID, name, reason"></label>
<label><input id="reports" type="checkbox"> Show report nodes</label>
</div>
<div class="layout"><main class="canvas"><svg id="graph" aria-label="Artifact lineage graph"></svg></main><aside class="side"><div id="details"><h2>Node details</h2><p class="empty">Select a node.</p></div><hr><div id="warnings"></div></aside></div>
<script id="history-data" type="application/json">__GRAPH_JSON__</script>
<script>
const data=JSON.parse(document.getElementById('history-data').textContent);
const stage={source:0,cleaner:1,reference:1,transformer:2,analyzer:3,report:4};
const colors={source:'#22221e',cleaner:'#3f6f85',reference:'#7c5bb5',transformer:'#3c8066',analyzer:'#a96720',report:'#77766d'};
const svg=document.getElementById('graph'),NS='http://www.w3.org/2000/svg';
document.getElementById('history-id').textContent=data.history_id.slice(0,16);
function displayLabel(dataset){const node=data.nodes.find(n=>(n.dataset||'Project')===dataset);return node?.display_label||node?.dataset_label||dataset}
function options(id,values,labels={}){const el=document.getElementById(id);el.replaceChildren(new Option('All',''));[...values].sort().forEach(x=>el.appendChild(new Option(labels[x]||x,x)))}
function escapeHtml(x){return String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]))}
const datasetNames=new Set(data.nodes.map(n=>n.dataset||'Project'));options('dataset-filter',datasetNames,Object.fromEntries([...datasetNames].map(d=>[d,displayLabel(d)])));options('pipeline-filter',new Set(data.nodes.map(n=>n.pipeline)));options('role-filter',new Set(data.nodes.map(n=>n.dataset_role).filter(Boolean)));options('status-filter',new Set(data.nodes.map(n=>n.status)));
function add(tag,attrs,parent=svg){const el=document.createElementNS(NS,tag);Object.entries(attrs||{}).forEach(([k,v])=>el.setAttribute(k,v));parent.appendChild(el);return el}
function visibleNodes(){const dataset=document.getElementById('dataset-filter').value,pipeline=document.getElementById('pipeline-filter').value,role=document.getElementById('role-filter').value,status=document.getElementById('status-filter').value,q=document.getElementById('search').value.toLowerCase(),reports=document.getElementById('reports').checked;return data.nodes.filter(n=>(!dataset||(n.dataset||'Project')===dataset)&&(!pipeline||n.pipeline===pipeline)&&(!role||n.dataset_role===role)&&(!status||n.status===status)&&(reports||n.node_type!=='report')&&(!q||JSON.stringify(n).toLowerCase().includes(q)))}
function addEdgeLabel(x,y,relationship){const width=Math.max(52,String(relationship).length*6.2+16);add('rect',{x:x-width/2,y:y-12,width,height:20,rx:9,class:'edge-label-bg'});const t=add('text',{x,y:y+2,'text-anchor':'middle',class:'edge-label'});t.textContent=relationship}
function render(){
  svg.replaceChildren();
  const defs=add('defs',{}),marker=add('marker',{id:'arrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:6,markerHeight:6,orient:'auto-start-reverse'},defs);
  add('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:'#8c8a81'},marker);
  const nodes=visibleNodes(),ids=new Set(nodes.map(n=>n.node_id)),datasets=[...new Set(nodes.map(n=>n.dataset||'Project'))].sort(),positions={},lanes={};let y=70;
  for(const d of datasets){const dn=nodes.filter(n=>(n.dataset||'Project')===d),byStage={};dn.forEach(n=>{const currentStage=stage[n.node_type]??2;if(!byStage[currentStage])byStage[currentStage]=[];byStage[currentStage].push(n)});const maximum=Math.max(1,...Object.values(byStage).map(items=>items.length));const h=Math.max(170,64+maximum*92);lanes[d]=[y,h];for(let s=0;s<5;s++){(byStage[s]||[]).sort((a,b)=>a.node_id.localeCompare(b.node_id)).forEach((n,i)=>positions[n.node_id]=[190+s*270,y+70+i*92])}y+=h+18}
  svg.setAttribute('viewBox',`0 0 1450 ${Math.max(y,320)}`);svg.setAttribute('height',Math.max(y,320));['Source','Curation / reference','Transformation','Analysis','Reports'].forEach((x,i)=>{const t=add('text',{x:190+i*270,y:35,'text-anchor':'middle',class:'stage-title'});t.textContent=x});
  for(const d of datasets){const [top,h]=lanes[d];add('rect',{x:12,y:top,width:1426,height:h,rx:16,class:'lane'});const t=add('text',{x:30,y:top+29,class:'lane-title'});t.textContent=displayLabel(d)}
  const visibleEdges=data.edges.filter(e=>ids.has(e.source)&&ids.has(e.target));
  for(const e of visibleEdges){const [sx,sy]=positions[e.source],[tx,ty]=positions[e.target],x1=sx+100,x2=tx-100,m=(x1+x2)/2;add('path',{d:`M ${x1} ${sy} C ${m} ${sy}, ${m} ${ty}, ${x2} ${ty}`,class:'edge','marker-end':'url(#arrow)'})}
  for(const n of nodes){const [x,ny]=positions[n.node_id],g=add('g',{class:'node',tabindex:0});const bad=['invalid','incomplete'].includes(n.status);add('rect',{x:x-100,y:ny-29,width:200,height:58,rx:14,fill:bad?'#b33e32':(colors[n.node_type]||'#4d4b44')},g);n.label.split('\n').slice(0,2).forEach((line,i)=>{const t=add('text',{x,y:ny-3+i*16,'text-anchor':'middle'},g);t.textContent=line.slice(0,30)});g.addEventListener('click',()=>show(n,g));g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' ')show(n,g)})}
  for(const e of visibleEdges){const [sx,sy]=positions[e.source],[tx,ty]=positions[e.target];addEdgeLabel((sx+100+tx-100)/2,(sy+ty)/2,e.relationship)}
}
function show(n,g){document.querySelectorAll('.node.selected').forEach(x=>x.classList.remove('selected'));g.classList.add('selected');document.getElementById('details').innerHTML=`<h2>${escapeHtml(n.label.replace('\n',' · '))}</h2><div><span class="badge">${escapeHtml(n.node_type)}</span><span class="badge">${escapeHtml(n.status)}</span>${n.dataset_role?`<span class="badge">${escapeHtml(n.dataset_role)}</span>`:''}</div><p><b>Dataset:</b> ${escapeHtml(n.display_label||n.dataset||'Project')}</p><p><b>Version:</b> ${escapeHtml(n.version??'—')} / ${escapeHtml(n.config_version??'—')}</p><p><b>Shape:</b> ${escapeHtml(n.rows??'—')} × ${escapeHtml(n.columns??'—')}</p><p><b>Reason:</b> ${escapeHtml(n.reason||'Not recorded')}</p><p><b>Path:</b> ${escapeHtml(n.path||'Embedded manifest')}</p><details><summary>Complete metadata</summary><pre>${escapeHtml(JSON.stringify(n,null,2))}</pre></details>`}
['dataset-filter','pipeline-filter','role-filter','status-filter','search','reports'].forEach(id=>document.getElementById(id).addEventListener(id==='search'?'input':'change',render));render();
const warningBox=document.getElementById('warnings');warningBox.innerHTML=`<h2>Validation (${data.warnings.length})</h2>`+(data.warnings.length?data.warnings.map(w=>`<div class="warning ${w.severity}"><b>${escapeHtml(w.code)}</b><br>${escapeHtml(w.message)}</div>`).join(''):'<p class="empty">No warnings detected.</p>');
</script>
</body></html>
"""


def render_history_html(*, history: ProjectHistory, destination: Path) -> None:
    """Write an interactive report with no network or widget dependency."""

    payload = history.to_dict()
    payload["history_id"] = history.history_id
    graph_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).replace("</", "<\\/")
    document = (
        HTML_TEMPLATE.replace(
            "__TITLE__",
            escape(history.project_name),
        )
        .replace("__CSS__", STATOMIX_HTML_CSS)
        .replace("__GRAPH_JSON__", graph_json)
    )
    destination.write_text(document, encoding="utf-8")
