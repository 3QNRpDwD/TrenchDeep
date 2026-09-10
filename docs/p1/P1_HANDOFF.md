# P1 인계

갱신: 2026-09-10. Legacy 소스 통합·테스트 정리 및 제품 sampling route 비교 완료 상태.

- 기본 P1 / 명시적 Legacy route와 동일 Context Diffusion·공통 Trainer 구조를 유지한다.
- 별도 trench-deep-legacy dependency, legacy/build.rs, 루트 legacy/ 폴더를 제거했다.
- Legacy route는 같은 crate의 src/legacy.rs를 통해 src/tensor, src/nn 등의 native 구현에 연결된다. P1과 계약이 다른 내부 구현은 legacy_mod.rs 등으로 구분한다.
- 원본 parameter 접근자는 실제 source에 포함했다. OUT_DIR 소스 복사는 없다.
- 구형 tensor/context, nn/context, optimizer/context, trainer/context와 native pilot 사본 26개를 제거했다. 동일 내용의 공용 feature/시각화 파일 5개는 공유한다.
- 보정·이동·삭제 매핑은 src/native_provenance/에 보존한다. scripts/verify_legacy.py는 현재 src 경로를 검사한다.
- tests/의 benchmark 사례 3개는 benches/support/로 옮기고 중복 테스트 진입점을 제거했다. 재생 Trainer 래퍼, 중복 P1-only 기준 Trainer 테스트, 단일-convolution pilot도 제거했다.
- tests/diffusion_routes.rs는 제품 Diffusion·Adam을 공통 Trainer에 직접 전달해 양쪽 route를 비교한다. 원본 예측·gradient 재생은 legacy_reference_diffusion.rs에 남는다.
- 연산 추가 시 전체 검사: 377 passed, 기존 4 ignored (unary-all-features.log). 아래 모델 재배치 검사로 갱신됐다. 이전 검사 기록은 P1_STATUS.md.
- 기본 ctx는 여전히 동적 그래프다. 정적 그래프는 P1 완료 후 학습 전 수명·참조·버퍼·필요 복사를 분석하는 별도 단계다.

- 제품 sampling: 동일 초기 가중치·이미지·단계별 noise로 P1/Legacy의 10단계 prediction·image 및 최종 결과 비교 통과(abs 또는 rel 1e-3). 반복 실행·중간 shape 오류 후 tensor/graph 정리, 입력 보존, 추적 복원도 확인했다. tests/diffusion_routes.rs의 학습·sampling 2 tests 통과(target/p1/sampling-routes.log).

- Abs·Log·Sqrt native backward와 tracked route 연결을 완료했다. Scalar 해석적 gradient, 다차원 가중 gradient/no-grad, 정의역 경계를 검사한다. 음수 Log는 기존 P1 -Inf/native NaN 차이를 유지한다. 변경 이력·해시는 INTEGRATED.json에 기록했다.

- 모델 소유권 정리: native 모델 18파일을 src/nn/native_models로 이동했다. legacy::nn::models가 정식 경로이며 tests::common::model은 호환 재수출만 남는다. Context pilot 테스트는 tests/models/, nn·Trainer 내부 테스트는 src/tests/nn 및 src/tests/trainer로 분리했다. Convergence 구현·테스트는 공용화했다.
- 재배치 후 전체 371 passed, 기존 4 ignored(model-layout-final.log). 6개 감소는 중복 Convergence 테스트 제거분이다. benchmark/example·native-only 빌드와 해시 검사도 통과했다.

- Sub: nonempty trailing-axis broadcasting과 입력 shape별 gradient 합산, backward shape 검증 및 native 대입 경로 공유를 완료했다. 공용 shape 함수의 빈 축 문제 때문에 shape가 다른 빈 입력은 오류로 차단한다.
- 최신 전체 372 passed, 기존 4 ignored(sub-all-features.log). Native-only 빌드·해시 검사 통과.

다음: 단일 입력 Concat부터 Shape 연산 지원 확장, 빈 축 broadcasting 계약, 수치 검증 후 benchmark. Native Trainer는 원본 회귀 검증용으로 여전히 존재하며 제거 완료로 간주하지 않는다.
기존 사용자 워크트리와 .gitignore 변경을 보존한다.
