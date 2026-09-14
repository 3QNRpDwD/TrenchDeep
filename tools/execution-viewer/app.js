/* TODO(trace-interface): accept validated, recorded events via a separate adapter.
 * Static scenario steps must never be labeled as measured or live execution.
 * TODO(trace-interface): trace scopes must provide exact layer instance + source IDs;
 * do not infer individual operation order, tensor values, or timings from this UI.
 */
(() => {
  'use strict';
  const model=globalThis.VIEWER_MODEL,data=globalThis.PROJECT_DATA;
  const UX=globalThis.VIEWER_EXPERIENCE;
  if(!model||!data){document.body.textContent='데이터가 없습니다. tools/execution-viewer/build.mjs를 실행한 뒤 다시 여세요.';return;}
  const $=id=>document.getElementById(id);
  let route='p1',scenario=model.scenarios[0],index=0,selected='x-init',timer=null;
  const allNodes=new Map(model.graphs.flatMap(g=>g.nodes.map(n=>[n.id,n])));
  const rendered=new Map(),edgeElements=[],fileButtons=new Map(),viewports=new Map();
  let lastStepKey=null;
  const panelOrder=[...UX.panelIds];
  const playbackFrames=()=>UX.frames(scenario.steps,$('playback-depth').value);
  const currentStep=()=>{const frames=playbackFrames();return frames[UX.frameIndex(frames,index)].step;};
  const motion=()=>matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth';
  function openDetail(){pause();update(false);if(!$('detail-dialog').open)$('detail-dialog').showModal();}
  const shown=n=>route==='legacy'&&n.legacy?{...n,...n.legacy}:n;
  const sourcesFor=n=>shown(n).sources;
  const make=(tag,cls,text)=>{const el=document.createElement(tag);if(cls)el.className=cls;if(text!==undefined)el.textContent=text;return el;};
  const svg=(tag,attrs)=>{const el=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))el.setAttribute(k,v);return el;};
  function buildGraph(g){
    const section=make('section','diagram');section.dataset.graphId=g.id;section.setAttribute('aria-label',g.title);
    const head=make('div','diagram-head'),heading=make('div');heading.append(make('h2','',g.title),make('small','',g.subtitle));head.append(heading);
    const zoom=make('div','zoom');head.append(zoom);section.append(head);
    const scroll=make('div','graph-scroll'),size=make('div','graph-size'),canvas=make('div','graph');
    let {width,height,positions}=globalThis.VIEWER_CAMERA.layout(g.nodes.map(n=>n.id),'horizontal');
    canvas.style.width=width+'px';canvas.style.height=height+'px';
    let scale=1,mini=null,miniViewport=null;
    const miniNodes=new Map();
    function resize(){size.style.width=width*scale+'px';size.style.height=height*scale+'px';canvas.style.transform=`scale(${scale})`;paintMini();}
    for(const[label,delta]of [['−',-.15],['+',.15],['1:1',0]]){const b=make('button','',label);b.setAttribute('aria-label',g.title+' '+(delta===0?'원래 크기':delta>0?'확대':'축소'));b.onclick=()=>{scale=delta?Math.min(1.75,Math.max(.55,scale+delta)):1;resize();if($('auto-follow').checked)followGraph(g.id,true);};zoom.append(b);}
    resize();const edges=svg('svg',{class:'edges',width,height,'aria-hidden':'true'});canvas.append(edges);
    const defs=svg('defs',{});for(const type of ['idle','related','active']){const marker=svg('marker',{id:g.id+'-'+type,viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:5,markerHeight:5,orient:'auto-start-reverse'});marker.append(svg('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:`var(--${type==='idle'?'idle':type==='related'?'secondary':'active'})`}));defs.append(marker);}edges.append(defs);
    for(const [edgeIndex,edge] of g.edges.entries()){const a=positions.get(edge.from),b=positions.get(edge.to);
      const d=globalThis.VIEWER_CAMERA.edgePath(a,b,edgeIndex,width,height,'horizontal');
      const el=svg('path',{d,class:'edge','marker-end':`url(#${g.id}-idle)`});const title=svg('title',{});title.textContent=edge.label;el.append(title);edges.append(el);edgeElements.push({el,edge,graph:g.id});
    }
    for(const n of g.nodes){const pos=positions.get(n.id),button=make('button','node');button.style.left=pos.x+'px';button.style.top=pos.y+'px';button.dataset.nodeId=n.id;
      button.append(make('span','node-label',n.label),make('span','node-note',n.note),make('span','status-dot'));
      button.onclick=()=>{selected=n.id;openDetail();};canvas.append(button);rendered.set(n.id,button);
    }
    size.append(canvas);scroll.append(size);section.append(scroll);$('diagrams').append(section);
    const controls=make('div','panel-controls');
    for(const[text,delta]of [['앞으로',-1],['뒤로',1]]){const button=make('button','',text);button.setAttribute('aria-label',g.title+' 패널 '+text);button.onclick=()=>{const from=panelOrder.indexOf(g.id),to=Math.max(0,Math.min(panelOrder.length-1,from+delta));[panelOrder[from],panelOrder[to]]=[panelOrder[to],panelOrder[from]];orderPanels();};controls.append(button);}
    const heightSelect=make('select');heightSelect.setAttribute('aria-label',g.title+' 패널 높이');for(const[value,text]of [['small','낮게'],['normal','보통 높이'],['large','높게']]){const o=make('option','',text);o.value=value;heightSelect.append(o);}heightSelect.value='normal';
    const wide=make('button','','넓게');wide.setAttribute('aria-pressed','false');wide.setAttribute('aria-label',g.title+' 패널 전체 열 사용');
    heightSelect.onchange=()=>{scroll.style.height=UX.sizes[heightSelect.value]+'px';paintMini();};
    wide.onclick=()=>{const enabled=!section.classList.contains('wide-panel');section.classList.toggle('wide-panel',enabled);wide.setAttribute('aria-pressed',enabled);paintMini();};controls.append(heightSelect,wide);head.after(controls);
    const miniBar=make('div','minimap-bar');miniBar.append(make('span','muted','미니맵 · 클릭/방향키로 이동'));
    mini=svg('svg',{class:'minimap',viewBox:`0 0 ${width} ${height}`,tabindex:'0',role:'group','aria-label':g.title+' 미니맵: 클릭 또는 방향키로 화면 이동, Home으로 처음'});
    for(const n of g.nodes){const rect=svg('rect',{class:'mini-node',rx:5});mini.append(rect);miniNodes.set(n.id,rect);}
    miniViewport=svg('rect',{class:'mini-viewport'});mini.append(miniViewport);miniBar.append(mini);section.append(miniBar);
    function paintMini(){
      if(!mini)return;mini.setAttribute('viewBox',`0 0 ${width} ${height}`);
      const active=new Set(currentStep().active);
      for(const[id,rect]of miniNodes){const p=positions.get(id);for(const[k,value]of Object.entries({x:p.x,y:p.y,width:186,height:65}))rect.setAttribute(k,value);rect.classList.toggle('active',active.has(id));rect.classList.toggle('selected',selected===id);}
      const x=Math.max(0,(scroll.scrollLeft-12)/scale),y=Math.max(0,(scroll.scrollTop-12)/scale);
      for(const[k,value]of Object.entries({x:Math.min(width,x),y:Math.min(height,y),width:Math.max(0,Math.min(width-x,(scroll.clientWidth-24)/scale)),height:Math.max(0,Math.min(height-y,(scroll.clientHeight-24)/scale))}))miniViewport.setAttribute(k,value);
    }
    mini.onclick=event=>{const matrix=mini.getScreenCTM();if(!matrix)return;pause();update(false);const point=new DOMPoint(event.clientX,event.clientY).matrixTransform(matrix.inverse());scroll.scrollTo({left:Math.max(0,point.x*scale-scroll.clientWidth/2+12),top:Math.max(0,point.y*scale-scroll.clientHeight/2+12),behavior:motion()});};
    mini.onkeydown=event=>{const directions={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]};if(!directions[event.key]&&event.key!=='Home')return;event.preventDefault();pause();update(false);const delta=directions[event.key]||[0,0];scroll.scrollTo({left:event.key==='Home'?0:scroll.scrollLeft+delta[0]*scroll.clientWidth*.5,top:event.key==='Home'?0:scroll.scrollTop+delta[1]*scroll.clientHeight*.5,behavior:motion()});};
    scroll.addEventListener('scroll',paintMini,{passive:true});
    const observer=new ResizeObserver(paintMini);observer.observe(scroll);
    const nav=make('button','',g.title);nav.onclick=()=>{section.scrollIntoView({block:'start',inline:'nearest',behavior:motion()});followGraph(g.id,true);};$('diagram-nav').append(nav);
    function relayout(direction){
      ({width,height,positions}=globalThis.VIEWER_CAMERA.layout(g.nodes.map(n=>n.id),direction));
      canvas.style.width=width+'px';canvas.style.height=height+'px';edges.setAttribute('width',width);edges.setAttribute('height',height);
      for(const n of g.nodes){const p=positions.get(n.id),el=rendered.get(n.id);el.style.left=p.x+'px';el.style.top=p.y+'px';}
      edgeElements.filter(e=>e.graph===g.id).forEach(({el,edge},i)=>el.setAttribute('d',globalThis.VIEWER_CAMERA.edgePath(positions.get(edge.from),positions.get(edge.to),i,width,height,direction)));
      viewports.get(g.id).positions=positions;resize();
    }
    function fit(){scale=Math.min(1,(scroll.clientWidth-24)/width,(scroll.clientHeight-24)/height);resize();scroll.scrollTo({left:0,top:0,behavior:'instant'});}
    function saveState(){return {size:heightSelect.value,wide:section.classList.contains('wide-panel'),scale,left:scroll.scrollLeft,top:scroll.scrollTop};}
    function restoreState(p){heightSelect.value=p.size;scroll.style.height=UX.sizes[p.size]+'px';section.classList.toggle('wide-panel',p.wide);wide.setAttribute('aria-pressed',p.wide);scale=p.scale;resize();}
    viewports.set(g.id,{scroll,positions,scale:()=>scale,previous:new Set(),focusId:null,nav,relayout,fit,section,paintMini,saveState,restoreState});
  }
  // Camera movement only: positions, graph topology and DOM identities stay fixed.
  // Do not call element.focus(): playback must not steal keyboard focus.
  function followGraph(id,force=false){
    const v=viewports.get(id),s=currentStep();
    const candidates=s.active.filter(n=>v.positions.has(n));
    const available=candidates.length?candidates:s.related.filter(n=>v.positions.has(n));
    const target=globalThis.VIEWER_CAMERA.chooseTarget(available,v.previous,v.focusId);
    v.previous=new Set(available);
    if(!target)return;
    if(!force&&v.focusId===target)return;
    v.focusId=target;const p=v.positions.get(target),scale=v.scale();
    v.scroll.scrollTo({...globalThis.VIEWER_CAMERA.offset(p,scale,v.scroll.clientWidth,v.scroll.clientHeight),behavior:force?'instant':motion()});
  }
  function followAll(force=false){for(const id of viewports.keys())followGraph(id,force);}
  model.graphs.forEach(buildGraph);
  function orderPanels(){for(const id of panelOrder){const v=viewports.get(id);$('diagrams').append(v.section);$('diagram-nav').append(v.nav);}for(const v of viewports.values())v.paintMini();}
  function applyLayout(follow=true){
    const panel=$('panel-layout').value,direction=$('node-layout').value;
    $('diagrams').dataset.layout=panel;
    $('diagrams').setAttribute('aria-label',`패널 ${$('panel-layout').selectedOptions[0].textContent} · 노드 ${$('node-layout').selectedOptions[0].textContent}`);
    for(const v of viewports.values())v.relayout(direction);
    if(follow&&$('auto-follow').checked)followAll(true);
  }
  $('panel-layout').onchange=applyLayout;$('node-layout').onchange=applyLayout;
  $('fit-all').onclick=()=>{for(const v of viewports.values())v.fit();};
  function settings(){return {version:1,panel:$('panel-layout').value,direction:$('node-layout').value,depth:$('playback-depth').value,follow:$('auto-follow').checked,order:panelOrder,panels:Object.fromEntries([...viewports].map(([id,v])=>[id,v.saveState()]))};}
  function restoreSettings(value){const p=UX.validateSettings(value);pause();$('panel-layout').value=p.panel;$('node-layout').value=p.direction;$('playback-depth').value=p.depth;$('auto-follow').checked=p.follow;panelOrder.splice(0,panelOrder.length,...p.order);orderPanels();applyLayout(false);for(const[id,v]of viewports)v.restoreState(p.panels[id]);for(const[id,v]of viewports){v.scroll.scrollTo({left:p.panels[id].left,top:p.panels[id].top,behavior:'instant'});v.paintMini();}update(false);}
  const storageKey='trenchdeep.execution-viewer.layout.v1';
  function loadLayout(automatic=false){try{const raw=localStorage.getItem(storageKey);if(!raw){if(!automatic)$('layout-status').textContent='저장한 배치가 없습니다.';return;}restoreSettings(JSON.parse(raw));$('layout-status').textContent='저장한 배치를 불러왔습니다.';}catch(error){$('layout-status').textContent='배치를 불러올 수 없습니다: '+error.message;}}
  $('save-layout').onclick=()=>{try{localStorage.setItem(storageKey,JSON.stringify(settings()));$('layout-status').textContent='이 브라우저에 저장했습니다. 다음에 열면 자동 복원합니다.';}catch(error){$('layout-status').textContent='브라우저 저장을 사용할 수 없습니다. 현재 화면에서는 계속 조절할 수 있습니다.';}};
  $('load-layout').onclick=()=>loadLayout();
  $('apply-preset').onclick=()=>{restoreSettings(UX.preset($('layout-preset').value));if($('layout-preset').value==='overview')for(const v of viewports.values())v.fit();$('layout-status').textContent='프리셋 적용 · 보관하려면 내 배치 저장을 누르세요.';};
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
  function update(allowFollow=true){
    const frames=playbackFrames(),position=UX.frameIndex(frames,index),s=frames[position].step,active=new Set(s.active),related=new Set(s.related);
    $('step-count').textContent=String(position+1).padStart(2,'0')+' / '+frames.length;$('step-title').textContent=s.title;$('step-description').textContent=s.description;$('phase').textContent=s.phase;
    $('timeline').max=frames.length-1;$('timeline').value=position;
    $('previous').disabled=position===0;$('next').disabled=position===frames.length-1;
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
    for(const [id,v]of viewports){const count=s.active.filter(n=>v.positions.has(n)).length;v.nav.textContent=model.graphs.find(g=>g.id===id).title+(count?' · '+count:'');v.nav.classList.toggle('has-active',count>0);v.paintMini();}
    const key=scenario.id+':'+index+':'+route+':'+$('playback-depth').value;
    if(allowFollow&&$('auto-follow').checked&&key!==lastStepKey)followAll();
    lastStepKey=key;
  }
  function pause(){if(timer!==null)clearInterval(timer);timer=null;$('play').textContent='재생';$('play').classList.remove('playing');}
  function play(){if(timer!==null){pause();update();return;}let frames=playbackFrames();if(UX.frameIndex(frames,index)===frames.length-1)index=0;timer=setInterval(()=>{frames=playbackFrames();if(UX.frameIndex(frames,index)===frames.length-1){pause();update();return;}index=UX.move(frames,index,1);update();},Number($('speed').value));$('play').textContent='일시정지';$('play').classList.add('playing');update();}
  $('play').onclick=play;$('previous').onclick=()=>{pause();index=UX.move(playbackFrames(),index,-1);update();};$('next').onclick=()=>{pause();index=UX.move(playbackFrames(),index,1);update();};$('timeline').oninput=event=>{pause();index=playbackFrames()[Number(event.target.value)].start;update();};
  $('playback-depth').onchange=()=>{pause();update();};
  $('scenario').onchange=event=>{pause();scenario=model.scenarios.find(s=>s.id===event.target.value);index=0;selected=scenario.steps[0].active[0];update();};
  $('speed').onchange=()=>{if(timer!==null){pause();play();}};
  $('p1').onclick=()=>{route='p1';update();refreshSearch();};$('legacy').onclick=()=>{route='legacy';update();refreshSearch();};
  $('auto-follow').onchange=()=>{if($('auto-follow').checked)followAll(true);};
  $('focus-active').onclick=()=>followAll(true);
  $('close-detail').onclick=()=>$('detail-dialog').close();
  $('detail-dialog').addEventListener('click',event=>{if(event.target===$('detail-dialog')){const r=$('detail-dialog').getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)$('detail-dialog').close();}});
  function revealNode(id){
    pause();selected=id;update(false);const v=[...viewports.values()].find(v=>v.positions.has(id));
    v.section.scrollIntoView({block:'center',inline:'nearest',behavior:motion()});v.scroll.scrollTo({...globalThis.VIEWER_CAMERA.offset(v.positions.get(id),v.scale(),v.scroll.clientWidth,v.scroll.clientHeight),behavior:'instant'});
    rendered.get(id).focus({preventScroll:true});v.paintMini();
  }
  function refreshSearch(){
    const query=$('structure-search').value;
    const hits=UX.search([...allNodes.values()].map(shown),data.inventory,query);
    const nodeHits=new Set(hits.filter(h=>h.kind==='node').map(h=>h.id)),fileHits=new Set(hits.filter(h=>h.kind==='file').map(h=>h.id));
    for(const[id,el]of rendered)el.classList.toggle('search-hit',nodeHits.has(id));for(const[id,el]of fileButtons)el.classList.toggle('search-hit',fileHits.has(id));
    $('search-status').textContent=query.trim()?(hits.length?`${hits.length}개 일치 · ${Math.min(hits.length,60)}개 결과 표시 · 전체 구조 유지`:'검색 결과가 없습니다.'):'검색 결과를 선택하면 해당 위치로 이동합니다.';
    $('search-results').replaceChildren();
    for(const hit of hits.slice(0,60)){const button=make('button','search-result');button.append(make('strong','',hit.label),make('span','muted',hit.detail));button.onclick=()=>{if(hit.kind==='node')revealNode(hit.id);else fileButtons.get(hit.id)?.click();};$('search-results').append(button);}
  }
  $('structure-search').oninput=refreshSearch;
  $('structure-search').onkeydown=event=>{if(event.key==='Escape'){$('structure-search').value='';refreshSearch();}if(event.key==='Enter')$('search-results').querySelector('button')?.click();};
  $('clear-search').onclick=()=>{$('structure-search').value='';refreshSearch();$('structure-search').focus();};
  // Route switches update labels and emphasis only; no node or edge is recreated.
  const groups=new Map();for(const file of data.inventory){const dir=file.path.includes('/')?file.path.slice(0,file.path.lastIndexOf('/')):'(root)';if(!groups.has(dir))groups.set(dir,[]);groups.get(dir).push(file);}
  for(const[dir,files]of groups){const group=make('details');group.open=true;group.append(make('summary','',dir+' / '+files.length));for(const file of files){const button=make('button','file',file.path.slice(file.path.lastIndexOf('/')+1));button.onclick=()=>{
      selected=null;pause();update(false);for(const el of rendered.values()){el.classList.remove('selected');el.setAttribute('aria-pressed','false');}
      $('detail-title').textContent=file.path;$('detail-description').textContent=`${file.bytes.toLocaleString()} bytes · repository inventory`;$('detail-status').replaceChildren(make('span','tag','정적 파일 정보'));
      $('detail-sources').replaceChildren();const refs=Object.values(data.sources).filter(s=>s.path===file.path);const pre=make('pre'),code=make('code','',refs[0]?.code||file.preview||'바이너리 / 문서 파일: 소스 미리보기 대상이 아닙니다.');pre.append(code);const block=make('div','source');block.append(pre);$('detail-sources').append(block);
      $('detail-relations').replaceChildren();for(const declaration of data.modules.filter(m=>m.path===file.path))$('detail-relations').append(make('div','relation',`${declaration.line}: ${declaration.context.join(' ')} ${declaration.declaration}`));$('detail-todo').textContent='TODO(trace-interface): 실제 호출이 확인된 파일·심볼에만 runtime event를 연결합니다.';
      if(!$('detail-dialog').open)$('detail-dialog').showModal();
    };group.append(button);fileButtons.set(file.path,button);} $('file-tree').append(group);}
  $('inventory-count').textContent=data.inventory.length+' files / '+data.modules.length+' module declarations';
  $('provenance').textContent='분석 생성 '+new Date(data.generatedAt).toLocaleString('ko-KR')+' · SHA256 '+data.fingerprint.slice(0,12)+' · Rust 코드 변경 없음';
  window.addEventListener('pagehide',pause);update();refreshSearch();loadLayout(true);
})();
