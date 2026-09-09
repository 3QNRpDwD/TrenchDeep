# P1 구현·검증 현황

갱신: 2026-09-09.

## 실행 구조

기본 ExecutionContext는 P1이다. legacyBenchmark feature에서 Legacy route를 선택하면
별도 패키지 없이 같은 crate의 src/legacy.rs가 root src의 native tensor/graph/연산을 호출한다.
src/legacy.rs는 내부 네임스페이스 경계이며 격리된 빌드나 소스 복사본이 아니다.
P1 모델·Trainer·Adam은 유지한다. 내부 타입 계약이 다른 파일은 legacy_ 접두사로 구분한다.

구형 Context 구현 26개를 제거하고, 내용이 동일한 feature·시각화 파일 5개를 공유했다.
parameter 이름 접근자는 source에 직접 포함했다. comparison도 같은 원본 모델 모듈을 재수출한다.
원본 보정 6개와 통합 경로/삭제 이력은 src/native_provenance/에 보존한다.
기존 보정은 Conv2d, Sigmoid, Tanh, Softmax, Div, Sum이다.
TensorHandle 소멸자의 tracing 호출을 제거해 TLS 종료 후 접근 오류를 방지했다.
원본 DOT 테스트는 scalar 입력 숨김 정책을 정확히 검사하도록 수정했다.

## 유지한 검증

| 테스트 | 책임 |
|---|---|
| diffusion_routes | 제품 Diffusion·Adam·공통 Trainer 직접 사용, ctx route만 변경; 3 epoch loss·전체 parameter 갱신·정리 |
| legacy_reference_diffusion | 원본 timestep/noise 추출, 예측/loss/gradient/Adam 재생 및 fixture 유효성 |
| diffusion | U-Net 학습·sampling·scheduler·checkpoint |
| execution_route, legacy_operations | 생성·capability·실제 route 연산·gradient·scope 수명 |
| legacy_conv_gradient, legacy_pilots | 보정 회귀 및 다른 모델의 원본 수치 비교 |
| providers, cleanup, metrics, visualization, rl_checkpoint, parameter_mapping | 독립 구현·실패 정리·관찰·RL·parameter 계약 |

benchmark 사례 3개는 benches/support/로 이동했다. run_case의 수치 검사는 유지하고
integration test 진입점은 제거했다. tests/support/diffusion_trainer.rs의 중복 재생
Trainer pass도 제거했다. 원본 수치 재생과 제품 Trainer route 검증은 각각 유지된다.
P1-only 기준 구성 Trainer 테스트와 단일-convolution migration pilot은 실제 U-Net 검증으로 대체했다.

## 이번 검증

- 통합 all-features lib/integration: 375 passed, 기존 4 ignored.
  target/p1/integrated-all.log 및 최종 공용 파일 통합 후 integrated-final-all.log.
- legacy/ 제거 후 직접 Diffusion route·원본 재생·route 경계 검사:
  target/p1/integrated-after-removal.log.
- no-default provider/cleanup/metrics: target/p1/integrated-no-default.log.
- Legacy-only 빌드: target/p1/integrated-native-only.log.
- benchmark 빌드: target/p1/integrated-benches.log.
- 소스 매핑/해시: scripts/verify_legacy.py.

원본의 고유 unit test가 같은 crate 검사에 포함되므로 과거 147/148개와
현재 테스트 수는 직접 비교할 수 없다. 삭제한 테스트 수만큼 감소한 수치가 아니다.
정적 그래프·완전 resume·optimizer snapshot·최신 성능 측정 완료를 의미하지 않는다.

## 재현

```powershell
python scripts/verify_legacy.py
cargo test --all-features --lib --tests
cargo test --features legacyBenchmark --test diffusion_routes --test legacy_reference_diffusion
cargo test --no-default-features --test providers --test metrics --test cleanup
cargo check --no-default-features --features legacyBenchmark
cargo check --benches --features legacyBenchmark
```

fixture의 보정 JSON 위치는 src/native_provenance/CORRECTIONS.json이다.
과거 benchmark 결과를 통합된 코드의 최신 성능으로 인용하지 않는다.
잔여 연산과 정적 그래프 방향은 P1_REVISED_PLAN.md 및 P1_LEGACY_OPERATIONS.md를 따른다.
