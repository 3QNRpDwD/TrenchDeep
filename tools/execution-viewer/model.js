/* Static, reviewed source map. This is NOT an execution trace.
 * TODO(trace-interface): replace scenario cursor input with recorded TraceEvent IDs
 * only after an opt-in trace interface exists. Never enable `debugging` here.
 */
(function (root) {
  const graphs = [];
  const source = (path, anchor) => ({ path, anchor });
  const n = (id, label, note, path, anchor, description, legacy) => ({id,label,note,sources:path?[source(path,anchor)]:[],description:description||note,legacy});
  const e = (from,to,label,kind='call') => ({id:from+'>'+to,from,to,label,kind});
  const diff='src/nn/diffusion.rs', rt='src/runtime/mod.rs', test='tests/diffusion.rs';
  graphs.push({id:'execution',title:'01 / 실행 과정',subtitle:'테스트 → 학습 → 샘플링',nodes:[
    n('x-init','Context · 모델 초기화','U-Net / scheduler',test,'let ctx = ExecutionContext::new()'),
    n('x-input','합성 이미지 · 잡음','[1,1,4,4]',test,'let image = ctx.input'),
    n('x-forward','수동 forward + loss','고정 noise · t=1',test,'let (prediction, loss)'),
    n('x-backward','수동 backward','parameter gradient 검증',test,'loss.backward()'),
    n('x-register','Optimizer 등록','SGD · lr=0.001',test,'let mut optimizer'),
    n('x-dataset','데이터셋 구성','이미 생성한 image 참조',test,'let dataset ='),
    n('x-fit','Trainer.fit','1 epoch / 1 batch',test,'UnsupervisedTrainer::silent'),
    n('x-sample','역확산 샘플링','t=1 → t=0',test,'let sample = model.sample'),
    n('x-check','결과 검증','shape · finite · graph=0',test,'assert!(sample.to_vec()'),
    n('x-scheduler','Scheduler 독립 테스트','q_sample · t0 noise 무시',test,'fn scheduler_matches'),
    n('x-checkpoint','Checkpoint 독립 테스트','저장 → 복원 → 고정 noise',test,'fn diffusion_checkpoint')
  ],edges:[e('x-init','x-input','생성'),e('x-input','x-forward','입력'),e('x-forward','x-backward','loss'),e('x-backward','x-register','검증 후'),e('x-register','x-dataset','등록 후'),e('x-dataset','x-fit','prebatched'),e('x-fit','x-sample','학습 후'),e('x-sample','x-check','출력 검증')]});
  graphs.push({id:'files',title:'02 / 파일 · 모듈',subtitle:'디스크 경로 ≠ 모듈 이름',nodes:[
    n('f-test','tests/diffusion.rs','통합 테스트',test,'fn full_unet'),
    n('f-lib','src/lib.rs','공개 모듈 · re-export','src/lib.rs','pub mod backend'),
    n('f-nn','nn/diffusion.rs','Diffusion · Unet · scheduler',diff,'pub struct Unet'),
    n('f-layers','nn/layers.rs','공개 Layer 구현','src/nn/layers.rs','pub trait Layer'),
    n('f-trainer','trainer/runners.rs','공개 UnsupervisedTrainer','src/trainer/runners.rs','pub struct UnsupervisedTrainer'),
    n('f-service','trainer/service.rs','공통 학습 서비스','src/trainer/service.rs','pub fn finish_step'),
    n('f-data','data/prebatched.rs','기존 Variable 참조','src/trainer/data/prebatched.rs','pub struct UnsupervisedDataset'),
    n('f-runtime','runtime/dispatch.rs','Operation 분배','src/runtime/dispatch.rs','pub fn execute'),
    n('f-engine','runtime/backward.rs','P1 역전파','src/runtime/backward.rs','pub fn backward',null,{label:'runtime/legacy_session.rs',note:'Legacy bridge',sources:[source('src/runtime/legacy_session.rs','pub(super) fn execute_legacy')]}),
    n('f-cpu','cpu/operations/forward.rs','P1 CPU provider','src/backend/cpu/operations/forward.rs','impl OperationProvider',null,{label:'tensor/operators/*',note:'구형 연산 구현',sources:[source('src/tensor/operators/conv2d.rs','fn backward')]}),
    n('f-optim','optimizer/algorithms.rs','공통 SGD / Adam','src/optimizer/algorithms.rs','fn step(&mut self)'),
    n('f-other','다른 모델 · prepared','이 시나리오에서 호출하지 않음','src/runtime/prepared/mod.rs','//! Explicit P1'),
    n('f-checkpoint','nn/checkpoint.rs','직렬화 · 파라미터 복원','src/nn/checkpoint.rs','pub struct ModelState'),
    n('f-oldtrainer','trainer/unsupervised.rs','구형 Trainer · 이 경로 미사용','src/trainer/legacy_mod.rs','#[path = "unsupervised.rs"]')
  ],edges:[e('f-test','f-lib','use','dependency'),e('f-lib','f-nn','re-export','dependency'),e('f-nn','f-layers','Layer','dependency'),e('f-trainer','f-service','위임','dependency'),e('f-test','f-trainer','fit','dependency'),e('f-test','f-data','dataset','dependency'),e('f-nn','f-runtime','Tensor API','dependency'),e('f-runtime','f-engine','route','dependency'),e('f-runtime','f-cpu','execute','dependency'),e('f-service','f-optim','step','dependency'),e('f-nn','f-checkpoint','ModelState','dependency')]});
  graphs.push({id:'api',title:'03 / API · 호출',subtitle:'공개 인터페이스와 구현',nodes:[
    n('a-context','ExecutionContext','builder / input / tensor',rt,'pub fn new() -> Self'),
    n('a-model','Diffusion','UnsupervisedModel',diff,'impl UnsupervisedModel for Diffusion'),
    n('a-noise','noise · timestep','StdRng / Box–Muller',diff,'fn noise(&mut self'),
    n('a-q','Scheduler.q_sample','x₀ + ε → xₜ',diff,'pub fn q_sample'),
    n('a-unet','Unet.forward / predict','Layer.forward',diff,'pub fn forward(&self, input: &Variable'),
    n('a-loss','Variable.mse_loss','Reduction::Mean',diff,'let loss = prediction.mse_loss'),
    n('a-back','Variable.backward','context.backward',rt,'pub fn backward(&self)'),
    n('a-dataset','UnsupervisedDataset','IntoBatchLoader → Prebatched','src/trainer/data/prebatched.rs','impl IntoBatchLoader for &UnsupervisedDataset'),
    n('a-trainer','UnsupervisedTrainer.fit','TrainingService.fit','src/trainer/runners.rs','impl UnsupervisedTrainer'),
    n('a-opt','Optimizer.step / zero_grad','Parameter → update','src/optimizer/algorithms.rs','pub trait Optimizer'),
    n('a-sample','sample_with_noise','no_grad / reversed timestep',diff,'pub fn sample_with_noise'),
    n('a-reverse','Scheduler.reverse_step','t=0은 noise 미사용',diff,'pub fn reverse_step'),
    n('a-save','CheckpointableModel','save / load',diff,'impl crate::trainer::CheckpointableModel')
  ],edges:[e('a-model','a-noise','noise 생성'),e('a-model','a-q','오염'),e('a-q','a-unet','noisy image'),e('a-unet','a-loss','prediction'),e('a-loss','a-back','scalar loss'),e('a-dataset','a-trainer','batch loader'),e('a-trainer','a-model','forward_loss'),e('a-trainer','a-back','finish_step'),e('a-trainer','a-opt','갱신'),e('a-sample','a-unet','predict'),e('a-unet','a-reverse','ε prediction'),e('a-model','a-save','trait 구현','dependency')]});
  graphs.push({id:'runtime',title:'04 / 실행 의존성',subtitle:'경로별 계산과 수명 관리',nodes:[
    n('r-context','ExecutionContext','TensorId / ContextId',rt,'pub struct ExecutionContext'),
    n('r-store','CpuTensorStore','TensorBuffer 저장','src/backend/cpu/storage.rs','pub struct CpuTensorStore',null,{label:'Bridge Store',note:'TensorId ↔ NativeHandle',sources:[source('src/runtime/legacy_session.rs','impl TensorStore for Store')]}),
    n('r-dispatch','execute / tracking','입력 검증 · no_grad','src/runtime/dispatch.rs','pub fn execute'),
    n('r-provider','OperationProvider','CpuBackend.execute','src/backend/cpu/operations/forward.rs','impl OperationProvider',null,{label:'NativeSession',note:'execute_many → 구형 연산',sources:[source('src/runtime/legacy_session.rs','fn execute_many')]}),
    n('r-kernel','CPU forward 구현','개별 커널 내부는 미계측','src/backend/cpu/operations/kernels.rs',null,null,{label:'Legacy operators',note:'Function / AutogradFunction',sources:[source('src/tensor/operators/conv2d.rs','fn forward')]}),
    n('r-record','GradientRecord','tracked일 때만 기록','src/runtime/dispatch.rs','fn commit',null,{label:'ComputationGraph',note:'구형 계산 그래프',sources:[source('src/tensor/graph.rs','pub(crate) fn add_input')]}),
    n('r-back','Reverse traversal','BackwardOp · gradient 누적','src/runtime/backward.rs','for record in records.iter().rev()',null,{label:'Native backward',note:'구형 역순 순회 · gradient 복사',sources:[source('src/tensor/graph.rs','pub(crate) fn backward'),source('src/runtime/legacy_session.rs','pub(super) fn backward_legacy')]}),
    n('r-update','Parameter 갱신','sub_assign / clear_grad','src/optimizer/algorithms.rs','self.context.sub_assign'),
    n('r-scope','TrainingScope cleanup','graph · gradients · collect','src/runtime/training.rs','fn cleanup'),
    n('r-nograd','no_grad scope','샘플링 graph 생성 억제',diff,'self.context.no_grad(||'),
    n('r-rand','rand / StdRng','host 잡음 생성',diff,'fn noise(&mut self'),
    n('r-serde','serde / ModelState','checkpoint I/O','src/nn/checkpoint.rs',null)
  ],edges:[e('r-context','r-store','소유','dependency'),e('r-context','r-dispatch','execute'),e('r-store','r-provider','input views / handles','data'),e('r-dispatch','r-provider','route'),e('r-provider','r-kernel','forward'),e('r-provider','r-record','tracked graph'),e('r-kernel','r-store','output','data'),e('r-record','r-back','역순'),e('r-back','r-update','gradient','data'),e('r-update','r-store','weights','data'),e('r-scope','r-record','clear'),e('r-scope','r-store','collect'),e('r-nograd','r-dispatch','tracking off','dependency'),e('r-rand','r-store','noise','data')]});
  graphs.push({id:'model',title:'05 / U-Net 구조',subtitle:'두 해상도 · skip connection',nodes:[
    n('m-time','시간 임베딩','sin/cos → Linear → SiLU → Linear',diff,'let frequencies ='),
    n('m-in','Initial Conv2D','1×1×4×4 → 1×2×4×4',diff,'let initial = self.initial.forward'),
    n('m-down0','Down stage 0','Residual ×2 + Attention',diff,'for stage in &self.down'),
    n('m-resize0','Downsample 0','1×2×4×4 → 1×2×2×2',diff,'h = stage.resize.forward(&h)?'),
    n('m-down1','Down stage 1','Residual ×2 + Attention',diff,'fn body(&self'),
    n('m-resize1','Downsample 1','1×4×2×2 → 1×4×1×1',diff,'h = stage.resize.forward(&h)?'),
    n('m-mid','Middle','Residual → Attention → Residual',diff,'h = self.middle2.forward'),
    n('m-up1','Up stage 1','upsample · concat skip · body',diff,'for (stage, skip) in self.up'),
    n('m-up0','Up stage 0','upsample · concat skip · body',diff,'for (stage, skip) in self.up'),
    n('m-final','Final residual + conv','concat initial → 1×1×4×4',diff,'self.final_conv'),
    n('m-residual','Residual 상세','GN → SiLU → Conv + time → GN → SiLU → Conv + skip',diff,'fn forward(&self, input: &Variable, time:'),
    n('m-attn','Attention 상세','GN → Q/K/V → softmax → projection + input',diff,'let scores = q.matmul')
  ],edges:[e('m-in','m-down0','features','data'),e('m-down0','m-resize0','down','data'),e('m-resize0','m-down1','features','data'),e('m-down1','m-resize1','down','data'),e('m-resize1','m-mid','bottleneck','data'),e('m-mid','m-up1','upsample','data'),e('m-up1','m-up0','upsample','data'),e('m-up0','m-final','output','data'),e('m-down1','m-up1','skip','data'),e('m-down0','m-up0','skip','data'),e('m-in','m-final','initial skip','data'),e('m-time','m-residual','conditioning','dependency'),e('m-mid','m-attn','attention','dependency')]});
  graphs.push({id:'data',title:'06 / 데이터 이동',subtitle:'값 · 참조 · gradient · 갱신',nodes:[
    n('d-host','Host Vec<f32>','i/16 · 고정 noise=0.1',test,'let image = ctx.input'),
    n('d-image','image : Variable','[1,1,4,4] · input leaf',test,'let image = ctx.input'),
    n('d-noise','ε : Tensor','고정 또는 StdRng 생성',diff,'fn noise(&mut self'),
    n('d-dataset','Dataset → Batch','Variable handle clone · 픽셀 재생성 없음','src/trainer/data/prebatched.rs','impl IntoBatchLoader for &UnsupervisedDataset'),
    n('d-noisy','xₜ : Variable','q_sample → as_variable',diff,'let noisy = self'),
    n('d-features','중간 feature / skips','NCHW · concat 채널 축',diff,'let mut skips ='),
    n('d-pred','ε prediction','[1,1,4,4]',diff,'let prediction = self.unet.forward'),
    n('d-loss','loss : Variable','MSE Mean · scalar',diff,'let loss = prediction.mse_loss'),
    n('d-grad','Parameter gradients','역전파 후 leaf gradient','src/runtime/backward.rs','let keep = state'),
    n('d-weight','Parameters','SGD: W ← W − lr·g','src/optimizer/algorithms.rs','Algorithm::Sgd =>'),
    n('d-sample','sampling image','xₜ → reverse_step → output',diff,'pub fn sample_with_noise'),
    n('d-clean','Graph / gradient 정리','핸들·참조 소멸 / collect','src/runtime/training.rs','fn cleanup'),
    n('d-disk','Checkpoint .tdw','ModelState · parameter values',test,'original.save_checkpoint')
  ],edges:[e('d-host','d-image','input allocation','data'),e('d-image','d-dataset','참조 / clone','data'),e('d-image','d-noisy','q_sample','data'),e('d-noise','d-noisy','q_sample','data'),e('d-dataset','d-noisy','training input','data'),e('d-noisy','d-features','U-Net','data'),e('d-weight','d-features','layer parameters','data'),e('d-features','d-pred','final conv','data'),e('d-pred','d-loss','prediction','data'),e('d-noise','d-loss','target constant','data'),e('d-loss','d-grad','backward','data'),e('d-grad','d-weight','update','data'),e('d-grad','d-clean','clear','data'),e('d-noise','d-sample','initial / step noise','data'),e('d-pred','d-sample','reverse_step','data'),e('d-weight','d-disk','save / load','data')]});

  const runtime=['r-context','r-store','r-dispatch','r-provider','r-kernel'];
  const files=['f-lib','f-nn','f-layers','f-runtime','f-engine','f-cpu'];
  const step=(id,title,description,active,related=[],phase='테스트')=>({id,title,description,active,related,phase});
  const compute=(id,title,description,active,phase)=>({...step(id,title,description,[...active,...runtime],files,phase),flow:graphs.find(g=>g.id==='model').edges.filter(e=>e.kind==='data'&&active.includes(e.to)).map(e=>e.id)});
  function forward(prefix,execution,phase){
    const common=[execution,'a-unet','d-features','r-record'];
    return [
      compute(prefix+'-q','잡음을 섞어 xₜ 생성','q_sample: sqrt(ᾱ)·image + sqrt(1−ᾱ)·noise. 학습 전 수동 검증은 t=1, Trainer는 난수로 t를 선택합니다.',[execution,'a-q','d-image','d-noise','d-noisy','r-record'],phase),
      compute(prefix+'-time','시간 임베딩','정규화 timestep → sin/cos embedding → time1 → SiLU → time2.',[...common,'m-time'],phase),
      ...[['in','Initial convolution','m-in'],['down0','Down stage 0','m-down0'],['resize0','Downsample 0','m-resize0'],['down1','Down stage 1','m-down1'],['resize1','Downsample 1','m-resize1'],['mid','Middle block','m-mid'],['up1','Up stage 1 + skip','m-up1'],['up0','Up stage 0 + skip','m-up0'],['final','최종 prediction','m-final']].map(([id,label,node])=>compute(prefix+'-'+id,label,'구조의 해당 블록을 통과합니다. 개별 연산의 실제 순서·호출 횟수·값은 추가 추적 인터페이스 구현 후 연결합니다.',[...common,node,...(['down0','down1','mid','up1','up0'].includes(id)?['m-residual','m-attn']:[])],phase)),
      compute(prefix+'-loss','MSE 손실 계산','prediction과 noise target을 비교하고 Reduction::Mean으로 scalar loss를 만듭니다.',[execution,'a-loss','d-pred','d-noise','d-loss','r-record'],phase)
    ];
  }
  const backward=(id,x,phase)=>step(id,'역전파 · gradient 검증','실제 역순 연산 이벤트는 아직 없습니다. P1은 BackwardOp, Legacy는 구형 graph backward를 사용합니다.',[x,'a-back','r-record','r-back','d-loss','d-grad'],['f-engine','f-runtime','f-cpu','d-weight'],phase);
  const main=[
    step('init','Context와 모델 초기화','ExecutionContext::new는 P1 기본 경로입니다. Unet(1,2,[1,2],1,[0,1]), linear scheduler(2,0.001,0.02), diffusion seed=7. Legacy 페이지는 동일 설정의 분석상 대응 경로입니다.',['x-init','a-context','a-model','r-context','r-store','d-weight'],['f-test','f-lib','f-nn','f-layers']),
    step('input','합성 이미지·고정 잡음 생성','16개 i/16 값을 input Variable로 생성하고, 값 0.1의 noise Tensor를 생성합니다. 외부 데이터 다운로드는 없습니다.',['x-input','a-context','d-host','d-image','d-noise','r-context','r-store'],['f-test']),
    ...forward('manual','x-forward','수동 검증'),backward('manual-back','x-backward','수동 검증'),
    step('drop','prediction · loss 핸들 해제','모든 파라미터의 gradient 존재를 검증한 뒤 prediction과 loss를 drop합니다.',['x-backward','d-grad','d-clean'],['f-test','f-engine']),
    step('register','SGD 생성 · 파라미터 등록','학습률 0.001로 SGD를 생성하고 register_all(model.parameters())를 호출합니다. 아직 파라미터 갱신 단계가 아닙니다.',['x-register','a-opt','d-weight'],['f-optim','f-test']),
    step('dataset','UnsupervisedDataset 구성','images=[&image]를 검증하고 참조를 보관합니다. 이미 존재하는 텐서를 감쌀 뿐, 이미지 픽셀을 새로 생성하지 않습니다.',['x-dataset','a-dataset','d-image','d-dataset'],['f-data','f-test','r-context']),
    step('batch','Trainer · prebatched 로더','IntoBatchLoader가 Variable 핸들을 clone합니다. with_training_scope에서 기존 gradient를 정리한 뒤 한 배치를 읽습니다.',['x-fit','a-trainer','a-dataset','d-dataset','r-scope'],['f-trainer','f-service','f-data'],'학습 1/1'),
    step('noise','학습용 noise · timestep 생성','Diffusion.forward_loss는 StdRng로 Gaussian noise와 timestep을 생성합니다. 이번 실행의 실제 t 값은 계측 전에는 알 수 없습니다.',['x-fit','a-trainer','a-model','a-noise','r-rand','r-store','d-noise'],['f-nn','f-service'],'학습 1/1'),
    ...forward('train','x-fit','학습 1/1'),backward('train-back','x-fit','학습 1/1'),
    step('update','SGD 파라미터 갱신','TrainingService.finish_step → optimizer.step → context.sub_assign. 이후 optimizer.zero_grad를 호출합니다.',['x-fit','a-trainer','a-opt','r-update','r-store','d-grad','d-weight'],['f-optim','f-service'],'학습 1/1'),
    step('cleanup','학습 scope 정리','graph와 gradients를 정리하고 collect합니다. 테스트는 graph_nodes=0을 확인합니다.',['x-fit','r-scope','r-record','r-store','d-clean'],['f-service','f-runtime'],'학습 1/1'),
    step('sample-noise','샘플링 noise 준비','초기 noise와 timestep별 noise 2개를 먼저 생성한 후 sample_with_noise를 호출합니다.',['x-sample','a-noise','a-sample','r-rand','r-store','d-noise','d-sample'],['f-nn'],'샘플링'),
    ...[1,0].flatMap(t=>[
      compute('sample-'+t+'-predict',`t=${t} · U-Net.predict`,`no_grad 안에서 U-Net 전체를 통과합니다. normalized time=${t}/2. graph record는 생성하지 않습니다.`,['x-sample','a-sample','a-unet','r-nograd','d-sample','d-features','d-pred','m-time','m-in','m-down0','m-resize0','m-down1','m-resize1','m-mid','m-up1','m-up0','m-final'],`샘플링 t=${t}`),
      compute('sample-'+t+'-reverse',`t=${t} · reverse_step`,t?'prediction으로 mean을 구하고 해당 step noise를 더합니다.':'t=0은 mean을 반환하며 noise를 더하지 않습니다.',['x-sample','a-reverse','d-pred','d-sample',...(t?['d-noise']:[]),'r-nograd'],`샘플링 t=${t}`)
    ]),
    step('check','출력 검증 · 종료','sample shape=[1,1,4,4], 모든 값 finite, graph_nodes=0을 검증합니다. 화면은 결과값을 측정하거나 실행하지 않습니다.',['x-check','d-sample','r-context'],['f-test'])
  ];
  const scenarios=[{id:'main',label:'Diffusion · 학습과 샘플링',steps:main},{id:'scheduler',label:'Scheduler · 독립 검증',steps:[
    step('scheduler-init','Scheduler fixture 생성','betas=[0.1,0.2], scalar x=2.0, noise=0.5.',['x-scheduler','a-context','r-context','r-store','d-image','d-noise'],['f-test']),
    compute('scheduler-q','q_sample 닫힌식 검증','t=1의 ᾱ=0.72를 사용한 닫힌식과 결과를 비교합니다.',['x-scheduler','a-q','d-image','d-noise','d-noisy'],'독립 검증'),
    compute('scheduler-zero','t=0 noise 무시 검증','noise=1과 noise=999를 전달한 reverse_step 결과가 같은지 검증합니다.',['x-scheduler','a-reverse','d-sample'],'독립 검증')
  ]},{id:'checkpoint',label:'Checkpoint · 독립 검증',steps:[
    step('checkpoint-init','서로 다른 모델 인스턴스 생성','이 테스트는 multipliers=[1], attention_at=[], cosine scheduler(2,0.008)을 사용합니다. 주 테스트의 두-stage 모델 그림은 비활성으로 유지합니다.',['x-checkpoint','a-model','r-context','d-weight'],['f-test','f-nn']),
    step('save','파라미터 · 설정 저장','save_checkpoint → ModelState.save. 임시 .tdw 파일을 사용합니다.',['x-checkpoint','a-save','d-weight','d-disk','r-serde'],['f-checkpoint']),
    step('load','Checkpoint 검증 · 복원','architecture, scheduler, parameter identity와 shape를 검증하고 replace_parameter로 복원합니다.',['x-checkpoint','a-save','d-disk','d-weight','r-store','r-serde'],['f-checkpoint']),
    step('checkpoint-sample','고정 noise로 샘플링 비교','임시 파일을 제거한 뒤 동일 initial=0.1과 step noises=0으로 두 모델의 sample_with_noise 결과를 비교합니다. RNG 상태 복원 검증이 아닙니다.',['x-checkpoint','a-sample','a-unet','a-reverse','r-nograd','d-noise','d-sample'],['f-nn','f-test'])
  ]}];
  const model={schemaVersion:1,mode:'static-analysis',graphs,scenarios};
  root.VIEWER_MODEL=model;
  if(typeof module!=='undefined')module.exports=model;
})(globalThis);
