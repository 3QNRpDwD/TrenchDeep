# Snapshot 참조 전환 검사 — 2026-09-07

## 결론과 범위

일반 forward/backward 입력의 데이터 복사는 빌린 TensorView로 전환 가능하다.
정적 그래프 전환이 선행 조건은 아니다. 이번 작업은 코드 감사이며 실행 구현,
공개 API, 수치 의미를 변경하지 않았고 성능 측정이나 참조 실행 실험도 하지 않았다.
사용자는 논리적 안전 조건이 갖춰진 구조에서 unsafe 사용을 허용했다.
아래 설계는 후보이며 정적 그래프 전환은 기존 결정대로 P1 완료 후 진행한다.

## 실제 복사 위치

| 경로 | 현재 동작 | 전환 방향 |
|---|---|---|
| runtime/dispatch.rs `with_inputs` | 모든 입력 snapshot을 owned buffer로 복사 | 연산 호출 동안 여러 입력의 읽기 view 유지 |
| backend/cpu/operations/forward.rs `execute` | 받은 view를 다시 `to_owned` | CPU helper를 TensorView/슬라이스 입력으로 변경 |
| runtime/backward.rs | 입력/saved snapshot, 출력 gradient clone | 읽기 구간에서 VJP 계산·shape 검증 후 참조 해제, gradient 누적 |
| backend/cpu/operations/mod.rs | 여러 VJP에서 입력/gradient/saved를 다시 복사 | VJP helper의 읽기 입력을 view로 변경; loss 경로 등 이미 view를 쓰는 부분 재사용 |
| forward.rs `saved.push(output.clone())` | 일부 activation 출력을 backward용으로 복제 | 저장 값이 기존 출력임을 나타내는 참조 계약과 수명 보장 필요 |
| runtime/mod.rs `update` | parameter를 복사해 갱신한 뒤 replace | 별도의 제한된 쓰기 접근 계약 필요; 읽기 참조 치환으로 해결되지 않음 |

출력·새 gradient 할당, broadcast 작업 배열, 시각화/파일 저장용 owned snapshot은
입력 복사와 구분한다. Context만 고치면 CPU provider의 두 번째 복사가 남는다.
backend 병렬 helper에도 복사가 있지만 사용 경로별 비용은 이번 감사에서 측정하지 않았다.

## 현재 계약의 제약

`TensorStore::with_view`는 callback 안에서만 유효한 참조를 제공한다.
CPU 저장소는 `HashMap<TensorId, Rc<RefCell<TensorBuffer>>>`이며 callback 동안
RefCell 읽기 guard가 살아 있다. alias는 같은 Rc를 공유한다. replace는 TensorBuffer
전체를 대입해 이전 Vec를 해제할 수 있으므로 Rc로 객체를 유지하는 것만으로 내부
데이터 포인터가 유효해지지는 않는다. graph pin도 버퍼 교체를 막는 읽기 guard가 아니다.
외부 저장소에는 callback 밖 주소 안정성이나 다중 view 제공 의무가 없다.

forward는 현재 snapshot을 만든 다음 state borrow를 해제하고 provider를 호출한다.
provider/custom op가 context를 보유하면 재진입이 가능하므로, 읽기 borrow를 호출 내내
유지하는 변경은 같은 context를 수정하는 재진입을 BorrowConflict로 바꿀 수 있다.
이 의미 변화는 명시적인 실행 계약 또는 호환 경로로 처리해야 한다.
backward는 이미 State를 가변 borrow한 상태로 VJP를 호출한다. 읽기 계산과 쓰기 누적을
분리하면 참조를 사용하기 용이하지만 실제 borrow 검사와 외부 provider 검증은 필요하다.

## 권장하는 최소 설계 후보

1. 저장소에 여러 ID를 동시에 읽는 callback/guard 경계를 제공한다. CPU 구현은
   해당 버퍼의 읽기 guard를 유지하고 기존 TensorView를 provider로 전달한다.
   같은 입력의 반복 및 alias는 읽기 공유를 허용한다. 외부 저장소 호환 정책도 정의한다.
2. 연산 실행 중 입력 교체·삭제·쓰기 및 충돌하는 재진입을 거부하고, 읽기 종료 후
   output 등록 또는 gradient 누적을 수행한다. 오류와 panic에서도 guard가 해제돼야 한다.
3. CPU helper의 읽기 전용 `&TensorBuffer`를 view/슬라이스로 옮겨 이중 복사를 제거한다.
   backend 계산 수식, 모델, Trainer는 재작성하지 않는다.
4. saved output 참조와 parameter in-place 갱신은 각각 수명·alias/오류 원자성 계약을
   검토한 다음 적용한다. 사용자에게 반환하는 owned snapshot 의미는 유지한다.

우선 safe Rust의 callback/guard로 표현 가능한지 검증하는 것이 자연스럽다.
unsafe가 필요하면 주소가 안정된 버퍼의 소유권과 읽기 guard를 함께 가진 내부 객체로
국한하고, guard 수명을 벗어난 raw pointer 참조를 만들지 않는다. `transmute`로 기존
callback 참조 수명만 늘리는 방식은 현재 계약으로 정당화할 수 없다.
모델 흐름의 단순함보다 alias, replace, cleanup, callback 재진입을 실제로 차단하는
실행 규칙이 안전성의 근거여야 한다. 세대 번호 검증만으로 살아 있는 참조의 무효화를
방지할 수 없으며, 쓰기와 해제를 참조 사용 기간 내내 막아야 한다.

## 구현 시 확인할 항목

- 입력과 provider 사이 데이터 포인터/복사량 계측으로 실제 무복사 경로 확인.
- `x * x`, detach alias, 공유 parameter, 분기 합류의 forward/backward 동등성.
- provider 재진입, 입력 replace/clear 시도, 오류/panic 후 guard·scope 정리.
- 외부 저장소/연산 구현 및 기본 구현 제외 빌드의 계약 호환성.
- 기존 DDPM 모든 gradient/Adam 3 step parity와 반복 배치 메모리 검증.
- unsafe를 도입하면 Miri 등으로 해당 경계 검증. 속도 개선은 계측 전 주장하지 않음.
