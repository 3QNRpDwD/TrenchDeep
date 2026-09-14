const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const model=require('./model.js');
const camera=require('./camera.js');
const ux=require('./experience.js');
const context={};vm.runInNewContext(fs.readFileSync(path.join(__dirname,'project-data.js'),'utf8'),context);
const data=context.PROJECT_DATA;
test('every scenario and edge resolves to stable graph IDs',()=>{
  const nodes=model.graphs.flatMap(g=>g.nodes),ids=new Set(nodes.map(n=>n.id));
  assert.equal(ids.size,nodes.length);assert.equal(model.graphs.length,6);
  for(const g of model.graphs){const local=new Set(g.nodes.map(n=>n.id));for(const e of g.edges){assert(local.has(e.from),e.id);assert(local.has(e.to),e.id);}}
  const edges=new Set(model.graphs.flatMap(g=>g.edges.map(e=>e.id)));
  for(const scenario of model.scenarios){assert.equal(new Set(scenario.steps.map(s=>s.id)).size,scenario.steps.length);for(const s of scenario.steps){for(const id of [...s.active,...s.related])assert(ids.has(id),id);for(const id of s.flow||[])assert(edges.has(id),id);}}
});
test('all source anchors exist in the current repository snapshot',()=>{
  assert.equal(data.warnings.length,0);
  for(const g of model.graphs)for(const n of g.nodes)for(const ref of [...n.sources,...(n.legacy?.sources||[])]){const s=data.sources[ref.path+'#'+(ref.anchor||'')];assert(s?.verified,ref.path);assert(s.line>=1);assert.equal(s.hash.length,64);}
});
test('main scenario respects test order and reverse timestep order',()=>{
  const ids=model.scenarios[0].steps.map(s=>s.id);
  const before=(a,b)=>assert(ids.indexOf(a)<ids.indexOf(b),`${a} before ${b}`);
  before('manual-back','register');before('register','dataset');before('dataset','batch');before('train-back','update');before('update','cleanup');before('cleanup','sample-noise');before('sample-1-reverse','sample-0-predict');before('sample-0-reverse','check');
  const dataset=model.scenarios[0].steps.find(s=>s.id==='dataset');assert(dataset.active.includes('d-dataset'));assert(!dataset.active.includes('d-host'));
  for(const s of model.scenarios[0].steps.filter(s=>s.id.startsWith('sample-')))assert(!s.active.includes('r-record'));
});
test('two routes share structure; checkpoint does not activate incompatible Unet map',()=>{
  for(const g of model.graphs)for(const n of g.nodes)if(n.legacy){assert(!('id' in n.legacy));assert(!('edges' in n.legacy));}
  for(const s of model.scenarios.find(s=>s.id==='checkpoint').steps)assert(!s.active.some(id=>id.startsWith('m-')));
  assert.equal(model.mode,'static-analysis');
});
test('auto-follow prioritizes newly activated nodes and otherwise preserves the camera target',()=>{
  assert.equal(camera.chooseTarget(['context','down1'],new Set(['context','down0']),'context'),'down1');
  assert.equal(camera.chooseTarget(['context','down1'],new Set(['context','down1']),'down1'),'down1');
  assert.equal(camera.chooseTarget(['context'],new Set(['context','down1']),'down1'),'context');
  assert.equal(camera.chooseTarget([],new Set(['context']),'context'),null);
});
test('camera offsets center a node at each zoom level without changing node coordinates',()=>{
  const p=Object.freeze({x:686,y:185});
  assert.deepEqual(camera.offset(p,1,720,326),{left:431,top:66});
  assert.deepEqual(camera.offset(p,1.5,720,326),{left:820.5,top:174.5});
  assert.deepEqual(camera.offset({x:14,y:45},.55,720,326),{left:0,top:0});
  assert.deepEqual(p,{x:686,y:185});
});
test('layout switches preserve all IDs and keep nodes inside both canvas directions',()=>{
  for(const graph of model.graphs){
    const ids=graph.nodes.map(n=>n.id);
    for(const direction of ['horizontal','vertical']){
      const layout=camera.layout(ids,direction);
      assert.deepEqual([...layout.positions.keys()],ids);
      for(const p of layout.positions.values()){assert(p.x>=0&&p.y>=0);assert(p.x+186<=layout.width);assert(p.y+65<=layout.height);}
      graph.edges.forEach((e,i)=>{const d=camera.edgePath(layout.positions.get(e.from),layout.positions.get(e.to),i,layout.width,layout.height,direction);assert(!/NaN|undefined/.test(d));});
    }
    assert.deepEqual(camera.layout(ids,'horizontal'),camera.layout(ids,'horizontal'));
  }
});
test('summary playback covers every detailed step and keeps training/sampling boundaries',()=>{
  for(const scenario of model.scenarios){
    const frames=ux.frames(scenario.steps,'summary');
    assert.deepEqual(frames.flatMap(f=>Array.from({length:f.end-f.start+1},(_,i)=>f.start+i)),scenario.steps.map((_,i)=>i));
    for(let i=0;i<scenario.steps.length;i++){const f=frames[ux.frameIndex(frames,i)];assert(f.start<=i&&f.end>=i);}
    const backwards=scenario.steps.map((s,i)=>s.id.endsWith('-back')?i:-1).filter(i=>i>=0);
    for(const i of backwards)assert(frames.some(f=>f.start===i&&f.end===i));
  }
  const steps=model.scenarios[0].steps,frames=ux.frames(steps,'summary');
  assert(frames.length<steps.length);assert.equal(ux.move(frames,0,-1),0);
  assert.equal(ux.move(frames,steps.length-1,1),steps.length-1);
  const middle=steps.findIndex(s=>s.id==='train-mid');
  assert.equal(steps[ux.move(frames,middle,1)].id,'train-back');
});
test('search finds API anchors and files without modifying the structural model',()=>{
  const before=JSON.stringify(model),nodes=model.graphs.flatMap(g=>g.nodes);
  assert(ux.search(nodes,data.inventory,'backward').some(h=>h.kind==='node'&&h.id==='a-back'));
  assert(ux.search(nodes,data.inventory,'diffusion.rs').some(h=>h.kind==='file'&&h.id==='tests/diffusion.rs'));
  assert(ux.search(nodes,data.inventory,'q_sample').some(h=>h.id==='a-q'));
  assert.equal(ux.search(nodes,data.inventory,'not_an_existing_symbol_xyz').length,0);
  assert.equal(ux.search(nodes,data.inventory,'   ').length,0);assert.equal(JSON.stringify(model),before);
});
test('saved layout validates ranges, restores missing panels and rejects incompatible versions',()=>{
  assert.throws(()=>ux.validateSettings({version:2}));
  const p=ux.validateSettings({version:1,order:['model','model','unknown'],panel:'bad',panels:{model:{scale:Infinity,size:'bad'},data:{scale:8,left:-2}}});
  assert.equal(p.panel,'stack');assert.equal(p.order[0],'model');assert.equal(new Set(p.order).size,6);
  assert.equal(p.panels.model.scale,1);assert.equal(p.panels.model.size,'normal');assert.equal(p.panels.data.scale,1.75);assert.equal(p.panels.data.left,0);
  for(const key of ['overview','model','path','default']){const preset=ux.preset(key);assert.deepEqual(ux.validateSettings(JSON.parse(JSON.stringify(preset))),preset);}
});
