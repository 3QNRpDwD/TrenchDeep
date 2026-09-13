const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const model=require('./model.js');
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
