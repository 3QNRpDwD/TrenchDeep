// Local presentation state only. No Rust execution or telemetry.
(function(root){
  const panelIds=['execution','files','api','runtime','model','data'];
  const sizes={small:240,normal:326,large:480};
  const unique=items=>[...new Set(items)];
  function frames(steps,depth){
    if(depth!=='summary')return steps.map((step,start)=>({start,end:start,step}));
    const result=[];
    for(let i=0;i<steps.length;i++){
      const prefix=/^(manual|train)-(?!back$)/.exec(steps[i].id)?.[1];
      if(!prefix){result.push({start:i,end:i,step:steps[i]});continue;}
      const start=i;while(i+1<steps.length&&steps[i+1].id.startsWith(prefix+'-')&&steps[i+1].id!==prefix+'-back')i++;
      const group=steps.slice(start,i+1);
      result.push({start,end:i,step:{...steps[start],title:`${steps[start].phase} · U-Net 순전파와 손실`,description:`${group.length}개 설명 단계를 묶어 표시합니다. q_sample → 시간 임베딩 → down/middle/up → prediction → MSE. 블록별 진행은 ‘블록 상세’로 전환해 확인할 수 있습니다.`,active:unique(group.flatMap(s=>s.active)),related:unique(group.flatMap(s=>s.related)),flow:unique(group.flatMap(s=>s.flow||[]))}});
    }
    return result;
  }
  function frameIndex(list,index){return Math.max(0,list.findIndex(f=>index>=f.start&&index<=f.end));}
  function move(list,index,delta){return list[Math.max(0,Math.min(list.length-1,frameIndex(list,index)+delta))].start;}
  function search(nodes,files,query){
    const tokens=query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);if(!tokens.length)return [];
    const matches=text=>tokens.every(t=>text.toLocaleLowerCase().includes(t));
    const out=[];
    for(const n of nodes){const text=[n.label,n.note,n.description,...n.sources.flatMap(s=>[s.path,s.anchor||''])].join(' ');if(matches(text))out.push({kind:'node',id:n.id,label:n.label,detail:n.sources[0]?.path||n.note});}
    for(const f of files)if(matches(f.path))out.push({kind:'file',id:f.path,label:f.path,detail:'프로젝트 파일'});
    return out;
  }
  function validateSettings(value){
    if(!value||value.version!==1)throw Error('지원하지 않는 배치 저장 형식입니다.');
    const oneOf=(v,values,fallback)=>values.includes(v)?v:fallback;
    const finite=(v,min,max,fallback)=>typeof v==='number'&&Number.isFinite(v)?Math.min(max,Math.max(min,v)):fallback;
    const order=unique((Array.isArray(value.order)?value.order:[]).filter(id=>panelIds.includes(id)));
    const panels={};for(const id of panelIds){const p=value.panels?.[id]||{};panels[id]={size:oneOf(p.size,Object.keys(sizes),'normal'),wide:p.wide===true,scale:finite(p.scale,.1,1.75,1),left:finite(p.left,0,100000,0),top:finite(p.top,0,100000,0)};}
    return {version:1,panel:oneOf(value.panel,['stack','row','grid2','grid3'],'stack'),direction:oneOf(value.direction,['horizontal','vertical'],'horizontal'),depth:oneOf(value.depth,['summary','detail'],'detail'),follow:value.follow!==false,order:[...order,...panelIds.filter(id=>!order.includes(id))],panels};
  }
  function preset(name){
    const p=validateSettings({version:1});
    if(name==='overview'){p.panel='grid3';p.direction='vertical';p.depth='summary';for(const v of Object.values(p.panels))v.size='small';}
    if(name==='model'){p.panel='grid2';p.order=['model','data','execution','api','runtime','files'];p.panels.model={...p.panels.model,size:'large',wide:true};}
    if(name==='path'){p.panel='grid2';p.order=['api','runtime','execution','files','model','data'];p.direction='vertical';}
    return p;
  }
  const api={panelIds,sizes,frames,frameIndex,move,search,validateSettings,preset};root.VIEWER_EXPERIENCE=api;
  if(typeof module!=='undefined')module.exports=api;
})(globalThis);
