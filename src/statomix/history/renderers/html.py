"""Self-contained interactive HTML project-history renderer."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

from statomix.history.model import ProjectHistory

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ — Statomix history</title>
<style>
:root{--bg:#f8fafc;--panel:#fff;--text:#0f172a;--muted:#64748b;--line:#cbd5e1}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px Inter,system-ui,sans-serif}
header{padding:18px 22px;background:#0f172a;color:white}header h1{margin:0;font-size:20px}header p{margin:5px 0 0;color:#cbd5e1}
.controls{display:flex;gap:10px;flex-wrap:wrap;padding:12px 18px;background:white;border-bottom:1px solid #e2e8f0}
select,input{border:1px solid #cbd5e1;border-radius:6px;padding:7px 9px;background:white}.controls label{display:flex;align-items:center;gap:6px}
.layout{display:grid;grid-template-columns:minmax(600px,1fr) 370px;height:calc(100vh - 126px)}
.canvas{overflow:auto;padding:12px}.side{overflow:auto;background:white;border-left:1px solid #e2e8f0;padding:16px}
#graph{min-width:1300px;background:white;border:1px solid #e2e8f0;border-radius:9px}.lane{fill:#f8fafc;stroke:#e2e8f0}.lane-title{font-weight:700;fill:#0f172a}.stage-title{font-size:12px;font-weight:700;fill:#475569}
.edge{stroke:#94a3b8;stroke-width:1.6;fill:none}.edge-label{font-size:10px;fill:#475569}.node{cursor:pointer}.node rect{stroke:white;stroke-width:2}.node text{fill:white;font-size:11px;font-weight:650;pointer-events:none}.node:hover rect,.node.selected rect{stroke:#111827;stroke-width:3}
.badge{display:inline-block;padding:3px 7px;border-radius:999px;background:#e2e8f0;margin:2px;font-size:12px}.warning{padding:8px;border-left:4px solid #f59e0b;background:#fffbeb;margin:8px 0}.warning.error{border-color:#dc2626;background:#fef2f2}
pre{white-space:pre-wrap;word-break:break-word;background:#f1f5f9;border-radius:6px;padding:10px;font-size:12px}.empty{color:var(--muted)}
@media(max-width:1000px){.layout{grid-template-columns:1fr;height:auto}.side{border-left:0;border-top:1px solid #e2e8f0;min-height:350px}}
</style>
</head>
<body>
<header><h1>__TITLE__</h1><p>Read-only Statomix artifact lineage · history ID <span id="history-id"></span></p></header>
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
const colors={source:'#334155',cleaner:'#2563eb',reference:'#7c3aed',transformer:'#059669',analyzer:'#d97706',report:'#64748b'};
const svg=document.getElementById('graph'),NS='http://www.w3.org/2000/svg';
document.getElementById('history-id').textContent=data.history_id.slice(0,16);
function options(id,values){const el=document.getElementById(id);el.innerHTML='<option value="">All</option>'+[...values].sort().map(x=>`<option>${escapeHtml(x)}</option>`).join('')}
function escapeHtml(x){return String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]))}
options('dataset-filter',new Set(data.nodes.map(n=>n.dataset||'Project')));options('pipeline-filter',new Set(data.nodes.map(n=>n.pipeline)));options('role-filter',new Set(data.nodes.map(n=>n.dataset_role).filter(Boolean)));options('status-filter',new Set(data.nodes.map(n=>n.status)));
function add(tag,attrs,parent=svg){const el=document.createElementNS(NS,tag);Object.entries(attrs||{}).forEach(([k,v])=>el.setAttribute(k,v));parent.appendChild(el);return el}
function visibleNodes(){const dataset=document.getElementById('dataset-filter').value,pipeline=document.getElementById('pipeline-filter').value,role=document.getElementById('role-filter').value,status=document.getElementById('status-filter').value,q=document.getElementById('search').value.toLowerCase(),reports=document.getElementById('reports').checked;return data.nodes.filter(n=>(!dataset||(n.dataset||'Project')===dataset)&&(!pipeline||n.pipeline===pipeline)&&(!role||n.dataset_role===role)&&(!status||n.status===status)&&(reports||n.node_type!=='report')&&(!q||JSON.stringify(n).toLowerCase().includes(q)))}
function render(){svg.innerHTML='';const defs=add('defs',{}),marker=add('marker',{id:'arrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:6,markerHeight:6,orient:'auto-start-reverse'},defs);add('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:'#94a3b8'},marker);const nodes=visibleNodes(),ids=new Set(nodes.map(n=>n.node_id)),datasets=[...new Set(nodes.map(n=>n.dataset||'Project'))].sort(),positions={},lanes={};let y=55;
for(const d of datasets){const dn=nodes.filter(n=>(n.dataset||'Project')===d),counts={};dn.forEach(n=>counts[stage[n.node_type]??2]=(counts[stage[n.node_type]??2]||0)+1);const h=Math.max(150,55+Math.max(1,...Object.values(counts))*88);lanes[d]=[y,h];for(let s=0;s<5;s++){dn.filter(n=>(stage[n.node_type]??2)===s).sort((a,b)=>a.node_id.localeCompare(b.node_id)).forEach((n,i)=>positions[n.node_id]=[170+s*270,y+48+i*88])}y+=h+18}
svg.setAttribute('viewBox',`0 0 1400 ${Math.max(y,300)}`);svg.setAttribute('height',Math.max(y,300));['Source','Curation / reference','Transformation','Analysis','Reports'].forEach((x,i)=>{const t=add('text',{x:170+i*270,y:28,'text-anchor':'middle',class:'stage-title'});t.textContent=x});
for(const d of datasets){const [top,h]=lanes[d];add('rect',{x:10,y:top,width:1380,height:h,rx:9,class:'lane'});const t=add('text',{x:24,y:top+26,class:'lane-title'});t.textContent=d}
for(const e of data.edges.filter(e=>ids.has(e.source)&&ids.has(e.target))){const [sx,sy]=positions[e.source],[tx,ty]=positions[e.target],x1=sx+100,x2=tx-100,m=(x1+x2)/2;add('path',{d:`M ${x1} ${sy} C ${m} ${sy}, ${m} ${ty}, ${x2} ${ty}`,class:'edge','marker-end':'url(#arrow)'});const t=add('text',{x:m,y:(sy+ty)/2-4,'text-anchor':'middle',class:'edge-label'});t.textContent=e.relationship}
for(const n of nodes){const [x,ny]=positions[n.node_id],g=add('g',{class:'node',tabindex:0});const bad=['invalid','incomplete'].includes(n.status);add('rect',{x:x-100,y:ny-29,width:200,height:58,rx:8,fill:bad?'#dc2626':(colors[n.node_type]||'#475569')},g);const lines=n.label.split('\n').slice(0,2);lines.forEach((line,i)=>{const t=add('text',{x,y:ny-3+i*16,'text-anchor':'middle'},g);t.textContent=line.slice(0,30)});g.addEventListener('click',()=>show(n,g));g.addEventListener('keydown',e=>{if(e.key==='Enter')show(n,g)})}}
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
    document = HTML_TEMPLATE.replace(
        "__TITLE__",
        escape(history.project_name),
    ).replace("__GRAPH_JSON__", graph_json)
    destination.write_text(document, encoding="utf-8")
