/* TODO(trace-interface): accept validated, recorded events via a separate adapter.
 * Static scenario steps must never be labeled as measured or live execution.
 * TODO(trace-interface): trace scopes must provide exact layer instance + source IDs;
 * do not infer individual operation order, tensor values, or timings from this UI.
 */
(() => {
  'use strict';
  const model=globalThis.VIEWER_MODEL,data=globalThis.PROJECT_DATA;
  if(!model||!data){document.body.textContent='데이터가 없습니다. tools/execution-viewer/build.mjs를 실행한 뒤 다시 여세요.';return;}
  const $=id=>document.getElementById(id);
  let route='p1',scenario=model.scenarios[0],index=0,selected='x-init',timer=null;
  const allNodes=new Map(model.graphs.flatMap(g=>g.nodes.map(n=>[n.id,n])));
  const rendered=new Map(),edgeElements=[],fileButtons=new Map();
  const shown=n=>route==='legacy'&&n.legacy?{...n,...n.legacy}:n;
  const sourcesFor=n=>shown(n).sources;
  const make=(tag,cls,text)=>{const el=document.createElement(tag);if(cls)el.className=cls;if(text!==undefined)el.textContent=text;return el;};
  const svg=(tag,attrs)=>{const el=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))el.setAttribute(k,v);return el;};
  function buildGraph(g){
    const section=make('section','diagram');section.setAttribute('aria-label',g.title);
    const head=make('div','diagram-head'),heading=make('div');heading.append(make('h2','',g.title),make('small','',g.subtitle));head.append(heading);
    const zoom=make('div','zoom');head.append(zoom);section.append(head);
    const scroll=make('div','graph-scroll'),size=make('div','graph-size'),canvas=make('div','graph');
    const width=450,height=Math.ceil(g.nodes.length/2)*110+15;
    canvas.style.width=width+'px';canvas.style.height=height+'px';
    let scale=1;
    function resize(){size.style.width=width*scale+'px';size.style.height=height*scale+'px';canvas.style.transform=`scale(${scale})`;}
    for(const[label,delta]of [['−',-.15],['+',.15],['1:1',0]]){const b=make('button','',label);b.setAttribute('aria-label',g.title+' '+(delta===0?'원래 크기':delta>0?'확대':'축소'));b.onclick=()=>{scale=delta?Math.min(1.75,Math.max(.55,scale+delta)):1;resize();};zoom.append(b);}
    resize();const edges=svg('svg',{class:'edges',width,height,'aria-hidden':'true'});canvas.append(edges);
    const positions=new Map(g.nodes.map((n,i)=>[n.id,{x:14+(i%2)*224,y:10+Math.floor(i/2)*110}]));
    const defs=svg('defs',{});for(const type of ['idle','related','active']){const marker=svg('marker',{id:g.id+'-'+type,viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:5,markerHeight:5,orient:'auto-start-reverse'});marker.append(svg('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:`var(--${type==='idle'?'idle':type==='related'?'secondary':'active'})`}));defs.append(marker);}edges.append(defs);
    for(const edge of g.edges){const a=positions.get(edge.from),b=positions.get(edge.to);const down=b.y>a.y;
      const x1=a.x+93,y1=a.y+(down?65:32),x2=b.x+93,y2=b.y+(down?0:32);
      let d;if(a.y===b.y)d=`M ${a.x+186} ${a.y+32} L ${b.x} ${b.y+32}`;
      else if(down)d=`M ${x1} ${y1} C ${x1} ${y1+28}, ${x2} ${y2-28}, ${x2} ${y2}`;
      else d=`M ${a.x+186} ${y1} C 440 ${y1}, 440 ${y2}, ${b.x+186} ${y2}`;
      const el=svg('path',{d,class:'edge','marker-end':`url(#${g.id}-idle)`});const title=svg('title',{});title.textContent=edge.label;el.append(title);edges.append(el);edgeElements.push({el,edge,graph:g.id});
    }
    for(const n of g.nodes){const pos=positions.get(n.id),button=make('button','node');button.style.left=pos.x+'px';button.style.top=pos.y+'px';button.dataset.nodeId=n.id;
      button.append(make('span','node-label',n.label),make('span','node-note',n.note),make('span','status-dot'));
      button.onclick=()=>{selected=n.id;update();};canvas.append(button);rendered.set(n.id,button);
    }
    size.append(canvas);scroll.append(size);section.append(scroll);$('diagrams').append(section);
  }
  model.graphs.forEach(buildGraph);
  for(const s of model.scenarios){const option=make('option','',s.label);option.value=s.id;$('scenario').append(option);}
  function addSource(ref){
    const block=make('div','source'),s=data.sources[ref.path+'#'+(ref.anchor||'')];
    block.append(make('div','',ref.path+(s?':'+s.line:'')));
    if(s){const pre=make('pre'),code=make('code','',s.code.split('\n').map((line,i)=>String(s.start+i).padStart(4)+'  '+line).join('\n'));pre.append(code);block.append(pre);if(!s.verified)block.append(make('p','error','소스 anchor 확인 필요'));}
    else block.append(make('p','muted','목록에 존재하는 파일입니다. 상세 실행 계측은 연결되지 않았습니다.'));
    $('detail-sources').append(block);
  }
  function inspect(active,related){
    const n=allNodes.get(selected);if(!n)return;const value=shown(n);
    $('detail-title').textContent=value.label;$('detail-description').textContent=value.description||value.note;
    $('detail-status').replaceChildren(make('span','tag',active.has(n.id)?'현재 설명 단계':related.has(n.id)?'정적 연관 · 실행 여부 미측정':'현재 단계 비활성'));
    $('detail-sources').replaceChildren();sourcesFor(n).forEach(addSource);
    $('detail-relations').replaceChildren();for(const g of model.graphs)for(const edge of g.edges)if(edge.from===n.id||edge.to===n.id){$('detail-relations').append(make('div','relation',`${shown(allNodes.get(edge.from)).label} → ${shown(allNodes.get(edge.to)).label} · ${edge.label}${edge.kind==='dependency'?' [정적 의존]':''}`));}
    $('detail-todo').textContent='TODO(trace-interface): 이 노드의 의미 ID를 추가 추적 인터페이스의 scope / operation 이벤트와 연결합니다. 현재는 소스 분석이며 debugging feature 및 실행 계측을 사용하지 않습니다.';
  }
  function update(){
    const s=scenario.steps[index],active=new Set(s.active),related=new Set(s.related);
    $('step-count').textContent=String(index+1).padStart(2,'0')+' / '+scenario.steps.length;$('step-title').textContent=s.title;$('step-description').textContent=s.description;$('phase').textContent=s.phase;
    $('timeline').max=scenario.steps.length-1;$('timeline').value=index;
    $('previous').disabled=index===0;$('next').disabled=index===scenario.steps.length-1;
    $('p1').setAttribute('aria-pressed',route==='p1');$('legacy').setAttribute('aria-pressed',route==='legacy');
    $('route-description').textContent=route==='p1'?'P1 · tests/diffusion.rs의 기본 실행 경로. 표시된 단계는 정적 분석이며 실제 이벤트·소요 시간·텐서 값은 수집하지 않았습니다.':'Legacy · 동일 공개 모델/API의 대응 경로를 분석한 페이지. 원본 tests/diffusion.rs는 Legacy로 실행하지 않으며, 기존 diffusion_routes.rs가 양 경로를 별도 검증합니다.';
    for(const[id,el]of rendered){const n=shown(allNodes.get(id));el.querySelector('.node-label').textContent=n.label;el.querySelector('.node-note').textContent=n.note;el.classList.toggle('active',active.has(id));el.classList.toggle('related',!active.has(id)&&related.has(id));el.classList.toggle('selected',id===selected);el.querySelector('.status-dot').textContent=active.has(id)?'●':'';el.setAttribute('aria-label',n.label+' · '+n.note+(active.has(id)?' · 현재 설명 단계':''));el.setAttribute('aria-pressed',id===selected);}
    for(const{el,edge,graph}of edgeElements){
      const direct=(active.has(edge.from)&&active.has(edge.to)&&edge.kind!=='dependency')||(s.flow||[]).includes(edge.id);
      const relation=!direct&&(active.has(edge.from)||related.has(edge.from))&&(active.has(edge.to)||related.has(edge.to));
      el.classList.toggle('active',direct);el.classList.toggle('related',relation);el.classList.toggle('flow',direct&&timer!==null);el.setAttribute('marker-end',`url(#${graph}-${direct?'active':relation?'related':'idle'})`);
    }
    const paths=new Set([...active,...related].flatMap(id=>sourcesFor(allNodes.get(id)).map(s=>s.path)));
    for(const[path,button]of fileButtons)button.classList.toggle('relevant',paths.has(path));
    inspect(active,related);
  }
  function pause(){if(timer!==null)clearInterval(timer);timer=null;$('play').textContent='재생';$('play').classList.remove('playing');}
  function play(){if(timer!==null){pause();update();return;}if(index===scenario.steps.length-1)index=0;timer=setInterval(()=>{if(index>=scenario.steps.length-1){pause();update();return;}index++;update();},Number($('speed').value));$('play').textContent='일시정지';$('play').classList.add('playing');update();}
  $('play').onclick=play;$('previous').onclick=()=>{pause();index=Math.max(0,index-1);update();};$('next').onclick=()=>{pause();index=Math.min(scenario.steps.length-1,index+1);update();};$('timeline').oninput=event=>{pause();index=Number(event.target.value);update();};
  $('scenario').onchange=event=>{pause();scenario=model.scenarios.find(s=>s.id===event.target.value);index=0;selected=scenario.steps[0].active[0];update();};
  $('speed').onchange=()=>{if(timer!==null){pause();play();}};
  $('p1').onclick=()=>{route='p1';update();};$('legacy').onclick=()=>{route='legacy';update();};
  // Route switches update labels and emphasis only; no node or edge is recreated.
  const groups=new Map();for(const file of data.inventory){const dir=file.path.includes('/')?file.path.slice(0,file.path.lastIndexOf('/')):'(root)';if(!groups.has(dir))groups.set(dir,[]);groups.get(dir).push(file);}
  for(const[dir,files]of groups){const group=make('details');group.open=true;group.append(make('summary','',dir+' / '+files.length));for(const file of files){const button=make('button','file',file.path.slice(file.path.lastIndexOf('/')+1));button.onclick=()=>{
      selected=null;for(const el of rendered.values()){el.classList.remove('selected');el.setAttribute('aria-pressed','false');}
      $('detail-title').textContent=file.path;$('detail-description').textContent=`${file.bytes.toLocaleString()} bytes · repository inventory`;$('detail-status').replaceChildren(make('span','tag','정적 파일 정보'));
      $('detail-sources').replaceChildren();const refs=Object.values(data.sources).filter(s=>s.path===file.path);const pre=make('pre'),code=make('code','',refs[0]?.code||file.preview||'바이너리 / 문서 파일: 소스 미리보기 대상이 아닙니다.');pre.append(code);const block=make('div','source');block.append(pre);$('detail-sources').append(block);
      $('detail-relations').replaceChildren();for(const declaration of data.modules.filter(m=>m.path===file.path))$('detail-relations').append(make('div','relation',`${declaration.line}: ${declaration.context.join(' ')} ${declaration.declaration}`));$('detail-todo').textContent='TODO(trace-interface): 실제 호출이 확인된 파일·심볼에만 runtime event를 연결합니다.';
    };group.append(button);fileButtons.set(file.path,button);} $('file-tree').append(group);}
  $('inventory-count').textContent=data.inventory.length+' files / '+data.modules.length+' module declarations';
  $('provenance').textContent='분석 생성 '+new Date(data.generatedAt).toLocaleString('ko-KR')+' · SHA256 '+data.fingerprint.slice(0,12)+' · Rust 코드 변경 없음';
  window.addEventListener('pagehide',pause);update();
})();
