# P1 실행 경계 감사 요약

정리일: 2026-09-09. 초기 감사의 미완료 표현을 현재 구현 사실로 통합했다.

| 대상 | 확인된 경계와 반영 내용 |
|---|---|
| Trainer | 공개 with_training_scope로 기존 guard/cleanup 재사용; 별도 Legacy Trainer 없음 |
| RL·optimizer | receiver 연산·buffer accessor 사용; 기존 알고리즘 유지 |
| 모델·레이어 | Context Diffusion/U-Net과 공개 parameter 계약 재사용 |
| Parameter 매핑 | 이름·shape·공유 그룹 검증; 원본 접근자를 root src에 직접 포함, build.rs/OUT_DIR 사본 제거 |
| AutogradEngine | graph record/get/remove/nodes/order 계약; 이것만 교체해도 native backward가 실행되는 것은 아님 |
| Legacy 실행 | runtime/legacy_session.rs가 native storage/forward/backward/gradient/replace와 scope 정리에 연결 |
| DDPM 재현 | 실제 원본 timestep/noise 및 version-2 fixture 재생; 공통 Trainer의 Legacy DDPM E2E는 남음 |

공개 scope는 기존 graph·중첩 학습을 거부하고 성공/오류 후 graph·gradient를 정리한다.
주 오류와 cleanup 오류를 함께 보존하며 parameter rollback은 보장하지 않는다.
공개 반환 handle 수명과 내부 실험용 owned-buffer training_step 계약을 구분한다.

Legacy 세션은 thread당 하나다. 같은 thread의 raw legacy API 혼용은 지원하지 않는다.
상위 계층은 native graph/registry나 구체 handle을 다루지 않는다.
원본 forward/VJP를 P1 provider로 연결하는 것과 원본 graph/backward 실행을 구분한다.

[Snapshot 감사](P1_SNAPSHOT_REFERENCE_AUDIT.md)는 복사·참조 가능성의 분석이며
무복사 구현이나 비용 측정 완료를 뜻하지 않는다.
현재 제한은 [연산 지원표](P1_LEGACY_OPERATIONS.md),
후속 순서는 [계획](P1_REVISED_PLAN.md)에 유지한다.
