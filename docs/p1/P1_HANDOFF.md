# P1 인계

갱신: 2026-09-09. Legacy 소스 통합과 테스트 정리 완료 상태.

- 기본 P1 / 명시적 Legacy route와 동일 Context Diffusion·공통 Trainer 구조를 유지한다.
- 별도 trench-deep-legacy dependency, legacy/build.rs, 루트 legacy/ 폴더를 제거했다.
- Legacy route는 같은 crate의 src/legacy.rs를 통해 src/tensor, src/nn 등의 native 구현에 연결된다. P1과 계약이 다른 내부 구현은 legacy_mod.rs 등으로 구분한다.
- 원본 parameter 접근자는 실제 source에 포함했다. OUT_DIR 소스 복사는 없다.
- 구형 tensor/context, nn/context, optimizer/context, trainer/context와 native pilot 사본 26개를 제거했다. 동일 내용의 공용 feature/시각화 파일 5개는 공유한다.
- 보정·이동·삭제 매핑은 src/native_provenance/에 보존한다. scripts/verify_legacy.py는 현재 src 경로를 검사한다.
- tests/의 benchmark 사례 3개는 benches/support/로 옮기고 중복 테스트 진입점을 제거했다. 재생 Trainer 래퍼, 중복 P1-only 기준 Trainer 테스트, 단일-convolution pilot도 제거했다.
- tests/diffusion_routes.rs는 제품 Diffusion·Adam을 공통 Trainer에 직접 전달해 양쪽 route를 비교한다. 원본 예측·gradient 재생은 legacy_reference_diffusion.rs에 남는다.
- 통합 검사: 375 passed, 기존 4 ignored. Native-only 빌드, benchmark 빌드, no-default provider/cleanup/metrics 검사 통과. 상세 로그는 P1_STATUS.md.
- 기본 ctx는 여전히 동적 그래프다. 정적 그래프는 P1 완료 후 학습 전 수명·참조·버퍼·필요 복사를 분석하는 별도 단계다.

다음: 동일-noise sampling 비교, 미지원 연산/계약 정리, 수치 검증 후 benchmark.
기존 사용자 워크트리와 .gitignore 변경을 보존한다.
