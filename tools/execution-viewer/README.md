# TrenchDeep Execution Atlas

기존 Rust 소스·테스트·Cargo 설정을 변경하지 않는 로컬 구조 뷰어다. 외부 패키지·CDN·서버가 필요하지 않는다.

## 열기

`tools/execution-viewer/index.html`을 브라우저에서 연다. 생성된 `project-data.js`가 함께 있어야 한다. 프로젝트 루트에서 소스 정보를 갱신하려면:

```powershell
node tools/execution-viewer/build.mjs
node --test tools/execution-viewer/viewer.test.cjs
```

생성기는 프로젝트 파일을 읽고 **이 폴더의 project-data.js만** 쓴다. Cargo 명령이나 Rust 코드를 실행하지 않는다. 직접 HTML을 열어도 동작하도록 일반 script를 사용하며 network 요청은 없다.

## 현재 구현

- P1/Legacy 페이지, 재생·일시정지·이전·다음·속도·타임라인 탐색.
- 실행 과정, 파일/모듈, API, runtime 의존성, U-Net, 데이터 이동의 6개 고정 다이어그램.
- 비활성 노드·텍스트는 유지하고 엣지만 흐리게 표시. 재생 중 활성 엣지에 흐름 표시.
- 공통 노드 좌표·개수와 선택 단계는 경로 전환에서도 유지.
- 각 다이어그램 확대/축소 및 스크롤. 파일 목록은 수동으로만 접을 수 있으며 단계 변경으로 숨기지 않는다.
- 검토된 코드 anchor, 발췌, 파일별 hash, 전체 파일 목록 및 Rust mod/pub use 선언.
- 주 diffusion 테스트, scheduler 검증, checkpoint 검증의 독립 시나리오.

## 해석 범위

**코드 분석 기반 재생이며 실행 기록이 아니다.** 그래프의 흐름 속도는 설명 속도다. 실행 시간, 개별 연산 발생 순서, 관측한 tensor 값이나 gradient 값을 제공하지 않는다. `debugging` feature를 사용하지 않는다.

원본 `tests/diffusion.rs`는 P1을 사용한다. Legacy 페이지는 같은 공개 모델·Trainer의 내부 대응 경로를 설명하며, 해당 테스트를 Legacy로 실행한 기록이 아니다. `tests/diffusion_routes.rs`는 두 경로를 실제로 검증하지만, 주 테스트와 모델 크기·optimizer·loader가 다르다. 해당 비교 테스트의 별도 애니메이션은 아직 추가하지 않았다.

U-Net 지도는 주 테스트의 두-stage 설정이다. checkpoint 테스트는 한-stage 모델이므로 주 테스트의 구조 노드를 비활성으로 유지한다. Sampling 단계는 U-Net 전체 호출을 강조하고, 아직 각 timestep 내부의 레이어 단위 이벤트를 수집하지 않는다.

프로젝트 파일 목록은 `.git`, 빌드 산출물, 에디터 설정, 도구 자신의 폴더를 제외한다. 파일 시스템 링크는 따라가지 않는다. 모듈 선언 추출은 문법 패턴에 따른 인덱스이며 Rust 이름 해석기나 전체 호출 그래프 분석기가 아니다. 파일/의존성 지도는 검토한 diffusion 경로를 중심으로 구성했다. CPU 내부 함수 실행 여부는 측정하지 않는다.

`src/nn/mod.rs`는 `layers.rs`와 `diffusion.rs`를 공개한다. 현재 공개 Trainer는 `runners.rs → service.rs`, optimizer는 `algorithms.rs`다. 구형 `trainer/unsupervised.rs`, `nn/conv2d.rs`, `optimizer/sgd.rs`를 이름만 보고 현재 실행 경로로 연결하면 안 된다. `runtime/prepared`는 이 시나리오에서 실행하지 않는다.

## 다음 작업

추가 추적 인터페이스의 선행 조건과 삽입 위치는 [TRACE_INTERFACE_TODO.md](TRACE_INTERFACE_TODO.md)에 기록했다. 실제 Rust 코드에는 주석도 추가하지 않았다. 뷰어의 TODO 주석이 미래 event adapter 연결 위치를 가리킨다.

기록 로딩, 실제 연산 이벤트, 실시간 연결, 전체 텐서 덤프는 미구현이다. 현재 제한을 해결하기 위해 전체 debugging을 켜는 fallback은 제공하지 않는다.
