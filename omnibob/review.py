"""Build an offline, owner-visible review from a fully verified action trace."""
import argparse
import json
from pathlib import Path

from .evaluate import replay


def build_review(directory, output):
    directory, output = Path(directory), Path(output)
    frames = []

    def visible(event, view):
        frames.append({"event": event, "view": view})

    verification = replay(directory, on_visible_decision=visible)
    if not frames:
        raise ValueError("no candidate-visible decisions in trace")
    manifest = json.loads((directory/"manifest.json").read_text())
    results = [json.loads(line) for line in (directory/"games.jsonl").read_text().splitlines()]
    payload = {"frames": frames, "results": results, "verification": verification,
               "source": str(directory.resolve()), "environment": manifest["environment"],
               "policies": manifest["policy_versions"], "inference": manifest.get("inference"),
               "split": manifest["split"], "opponents": manifest["opponents"]}
    # Data is never executable markup. Card text and model reasons render as textContent.
    encoded = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    html = TEMPLATE.replace("__PAYLOAD__", encoded)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(html)
    return {"output": str(output.resolve()), "visible_decisions": len(frames), **verification,
            "scope": "local simulator replay; owner-visible information; no network or live-game proof"}


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>OmniBob · 本地对局复盘</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#10151e;color:#e8edf6}
body{max-width:1400px;margin:auto;padding:24px}h1{font-size:24px;margin:0 0 8px}h2{font-size:17px;margin:20px 0 10px}
p{line-height:1.55}.muted{color:#a8b7cb;font-size:13px}.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;position:sticky;top:0;background:#10151ef5;padding:14px 0;z-index:2}
button,select{background:#26364e;color:inherit;border:1px solid #435773;border-radius:6px;padding:9px;cursor:pointer}button:disabled{opacity:.35;cursor:default}input[type=range]{flex:1;min-width:200px}
.panel{background:#192334;border:1px solid #32425a;border-radius:10px;padding:16px;margin:12px 0}.stats{display:flex;gap:18px;flex-wrap:wrap}.cards{display:flex;gap:9px;flex-wrap:wrap}.card{background:#212f44;border:1px solid #3c506c;border-radius:7px;padding:10px;width:156px;min-height:76px;white-space:pre-wrap;font-size:13px;line-height:1.5}.card strong{display:block;color:#f1d99d;font-size:14px}.card small{color:#b5c5db}.chosen{outline:2px solid #65dcc0}.action{color:#87ead2;font-size:19px;margin:5px 0}.reason{font-family:ui-monospace,monospace;font-size:12px;overflow-wrap:anywhere}pre{overflow:auto;white-space:pre-wrap;font-size:12px}summary{cursor:pointer}.empty{color:#8392a9;padding:12px}a{color:#91c7ff}
</style>
<h1>OmniBob · 本地对局复盘</h1>
<p class="muted">逐动作重放已核对可见观测摘要和最终名次。页面只显示当时己方可见信息；这是本地模拟器对局，不是联网比赛或真实客户端验证。左右方向键逐步查看。</p>
<div class="toolbar"><select id="game" aria-label="选择对局"></select><button id="prev">← 上一步</button><button id="next">下一步 →</button><input id="slider" aria-label="决策进度" type="range" min="0" value="0"><span id="position"></span></div>
<div class="panel"><div class="stats" id="stats"></div><p class="action" id="action"></p><div class="reason" id="reason"></div><p class="muted" id="result"></p></div>
<h2>己方战场</h2><div id="board" class="cards"></div>
<h2>手牌</h2><div id="hand" class="cards"></div>
<h2>酒馆商店</h2><div id="shop" class="cards"></div>
<h2>当前选项 / 目标</h2><div id="choices" class="cards"></div>
<details class="panel"><summary>本步完整玩家可见视图</summary><pre id="visible"></pre></details>
<details class="panel"><summary>策略、环境与复现信息</summary><pre id="metadata"></pre></details>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('payload').textContent), el=id=>document.getElementById(id);
const key=e=>`${e.policy}/${e.seed}/${e.candidate_seat}`, groups=new Map();
for(const frame of data.frames){const k=key(frame.event);if(!groups.has(k))groups.set(k,[]);groups.get(k).push(frame)}
for(const [k,rows] of groups){const o=document.createElement('option');o.value=k;o.textContent=`${rows[0].event.policy} · 种子 ${rows[0].event.seed} · 座位 ${rows[0].event.candidate_seat+1}`;el('game').append(o)}
let index=0;
function cards(id,items,selected){const parent=el(id);parent.replaceChildren();if(!items?.length){const n=document.createElement('div');n.className='empty';n.textContent='无';parent.append(n);return}
items.forEach((item,i)=>{const card=item.card||item;const node=document.createElement('div');node.className='card'+(selected.includes(i)?' chosen':'');const title=document.createElement('strong');title.textContent=`${i+1}. ${item.name_zh||card.name_zh||item.name||card.name||item.id||card.id||item.kind||'选项'}`;node.append(title);const details=document.createElement('div');const stats=item.atk!==undefined?`${item.atk} / ${item.hp}  `:'';details.textContent=stats+(item.price!==undefined?`费用 ${item.price}  `:'')+(item.golden?'金色':'');node.append(details);const text=document.createElement('small');text.textContent=item.text_zh||card.text_zh||item.text||card.text||(item.location?JSON.stringify(item.location):'');node.append(text);parent.append(node)})}
function render(){const rows=groups.get(el('game').value),{event:e,view:v}=rows[index],me=v.me;el('slider').max=rows.length-1;el('slider').value=index;el('position').textContent=`${index+1} / ${rows.length}`;el('prev').disabled=index===0;el('next').disabled=index===rows.length-1;
el('stats').replaceChildren();for(const s of [`回合 ${e.turn}`,`英雄 ${me.hero?.name_zh||me.hero?.name||'待选'}`,`生命 ${me.health} + 护甲 ${me.armor}`,`金币 ${me.gold}`,`酒馆等级 ${me.tier}`,`存活 ${v.alive}`]){const n=document.createElement('span');n.textContent=s;el('stats').append(n)}
el('action').textContent=`${e.action_name}  [${e.action}]`;el('reason').textContent=e.reason;
const r=data.results.find(r=>r.policy===e.policy&&r.seed===e.seed&&r.seat===e.candidate_seat);el('result').textContent=`最终名次 ${r?.placement??'未知'} · 状态 ${r?.status??'未知'} · 单步决策 ${e.decision_ms.toFixed(2)} ms · 高亮表示动作涉及的位置（从 1 开始显示）`;
let b=[],h=[],s=[],ch=[];const a=e.action;if(a>=8&&a<16)s=[a-8];else if(a>=16&&a<23)b=[a-16];else if(a>=23&&a<30)b=[a-23];else if(a>=30&&a<46)ch=[a-30];else if(a>=46&&a<126){h=[Math.floor((a-46)/8)];b=[(a-46)%8]}else if(a>=126&&a<196){h=[Math.floor((a-126)/7)];b=[(a-126)%7]}else if(a>=196){b=[Math.floor((a-196)/7),(a-196)%7]}
cards('board',me.board,b);cards('hand',me.hand,h);cards('shop',me.shop,s);const p=me.pending;cards('choices',p?.options||p?.targets||p?.choices||[],ch);el('visible').textContent=JSON.stringify(v,null,2)}
el('game').onchange=()=>{index=0;render()};el('slider').oninput=()=>{index=Number(el('slider').value);render()};function step(n){index=Math.max(0,Math.min(groups.get(el('game').value).length-1,index+n));render()}el('prev').onclick=()=>step(-1);el('next').onclick=()=>step(1);document.addEventListener('keydown',e=>{if(['INPUT','SELECT'].includes(e.target.tagName))return;if(e.key==='ArrowLeft')step(-1);if(e.key==='ArrowRight')step(1)});
el('metadata').textContent=JSON.stringify(Object.fromEntries(Object.entries(data).filter(([k])=>!['frames','results'].includes(k))),null,2);render();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="completed play or evaluate --trace directory")
    parser.add_argument("--output", type=Path, required=True, help="new self-contained HTML file")
    args = parser.parse_args()
    print(json.dumps(build_review(args.directory, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
