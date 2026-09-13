# 추가 계측 인터페이스 연결 지점

현재 구현은 **정적 소스 분석 재생**이다. 실행을 수집하지 않고 `debugging`이나 다른 Cargo feature를 켜지 않는다. 아래 TODO는 기존 프로젝트 코드에 넣지 않고 이 독립 도구에 보관한다. 추후 명시적으로 계측 변경을 진행할 때 적용한다.

## 선행 인터페이스

```rust,ignore
// TODO(trace-interface): 아래는 제안이며 현재 crate에 존재하지 않는 인터페이스다.
// opt-in feature 및 context별 세션. 기본 빌드에서는 이벤트 객체도 만들지 않는다.
// TraceSession / TraceScope / TraceEvent / SourceId / SemanticNodeId
// event: sequence, parent_scope, phase, route, semantic_id, source_id,
//        operation, input_ids, output_ids, shapes, outcome
// 이벤트에는 Tensor/Variable 자체를 보관하지 않는다(수명·graph 정리 불변).
// append는 runtime borrow/legacy graph lock 안에서 재진입 콜백을 호출하지 않는다.
```

`app.js`의 정적 시나리오 입력 앞에 별도 recorded-event adapter를 구현한다. `model.js`의 의미 ID를 재사용하되, 관측하지 않은 연산을 만들어 채우지 않는다. 정적 재생과 관측 기록은 독립 모드로 표시하고 후자의 기록에만 타임스탬프·값을 표시한다.

## TODO 연결 목록

| 위치 | 필요한 이벤트 | 현재 처리 |
|---|---|---|
| tests/diffusion.rs의 테스트 호출 경계 | scenario 시작/끝, assertion, image/noise 생성, dataset 구성 | 실행 순서를 수동 검토한 단계 |
| nn/diffusion.rs의 noise/forward_loss | 실제 timestep, noise 생성, q_sample, loss | 실제 난수 t·noise 값 미표시 |
| Unet.forward / Stage.body / Residual.forward / Attention.forward | 블록 인스턴스 ID, enter/exit, 개별 Layer 호출 | U-Net 블록 지도 및 코드 발췌 |
| runtime/dispatch.rs의 execute/commit | 연산 시작/성공/실패, 입출력 ID와 shape, record 여부 | P1 정적 경로 |
| runtime/backward.rs의 역순 순회 | 원래 순전파 연산 ID를 참조하는 VJP 시작/끝 | 역전파 전체 단계만 표시 |
| runtime/legacy_session.rs | TensorId ↔ NativeHandle, execute_many, no_grad | Legacy 대응 경로만 표시 |
| tensor/graph.rs의 구형 역전파 순회 | native node ID와 순전파 source/scope 대응 | 구형 역전파 코드 근거 |
| trainer/service.rs / runtime/training.rs | epoch/batch, forward, backward, cleanup/error | 1 epoch/1 batch의 코드 경로 |
| optimizer/algorithms.rs | register, step, zero_grad, parameter update | SGD 단계·정적 수식 |
| sample_with_noise의 반복문 | 실제 t별 predict/reverse_step 및 no_grad 종료 | t=1→0의 설명 단계 |
| checkpoint 호출 경계 | save/load/remove 결과, architecture 검증 | 별도 독립 시나리오 |

## 도입 시 불변 조건

- debugging 전체 추적을 우회 수단으로 사용하지 않는다.
- RNG를 추가 소비하지 않고 연산을 재실행하지 않는다. 입력·gradient·parameter 값도 바꾸지 않는다.
- scope/그래프 잠금 내부에서 외부 observer 호출 금지. 메타데이터만 버퍼링한다.
- Legacy graph_snapshot은 현재 미지원이다. 기존 API를 호출하거나 구형 VisualizationCapture를 bridge 세션에 겹쳐 실행하지 않는다.
- 기록 누락·중단·버전 불일치는 명시적으로 표시한다. 성공한 실제 실행처럼 재생하지 않는다.
- CPU 내부 반복문, 전체 Tensor 값, prepared 실행은 이번 인터페이스 기본 범위 밖이다.
- 기록 모드에서도 전체 노드·엣지는 먼저 구성하고 위치를 고정한다. 이벤트는 강조만 바꾼다.
- 계측 전후 동일 seed의 loss·gradient·parameter·sampling 결과와 graph 수명 검증을 통과해야 한다.
