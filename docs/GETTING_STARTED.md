# TrenchDeep 시작하기

TrenchDeep은 Rust로 텐서 연산, 자동 미분, 신경망 학습을 구현하는 딥러닝 프레임워크다. 기본 사용 흐름은 **실행 환경 생성 → 데이터 준비 → 모델 생성 → optimizer 설정 → 학습 → 예측**이다.

이 문서에서는 간단한 mlp 학습을 실행하는 데 필요한 내용만 다룬다. 기능별 설정과 확장 방법은 별도 문서로 연결할 예정이다.

## 프로젝트에 추가하기

TrenchDeep 저장소를 로컬에 내려받은 뒤, 사용할 Rust 프로젝트의 `Cargo.toml`에 의존성을 추가한다. `path`는 해당 프로젝트에서 TrenchDeep 저장소까지의 경로로 바꾼다.

```toml
[dependencies]
trench-deep = { path = "../TrenchDeep", features = ["enableBackward"] }
```

기본 feature에는 내장 텐서 저장소와 연산 구현이 포함된다. 학습과 자동 미분에는 `enableBackward`가 필요하다. Rust 코드에서는 패키지 이름의 하이픈 대신 `trench_deep`을 사용한다.

## 알아둘 구성 요소

| 구성 요소 | 역할 |
| --- | --- |
| `ExecutionContext` | 텐서·연산·자동 미분을 관리하는 실행 환경 |
| `Tensor` / `Variable` | 수치 데이터와 자동 미분 연산에 사용하는 값 |
| `Model` | 입력으로부터 예측을 계산 |
| `Loss` | 모델이 예측한 값과 타겟 데이터 사이의 loss 를 계산 |
| `Optimizer` | 등록된 모델 파라미터를 gradient로 갱신 |
| `Trainer` | 데이터 반복, 역전파, 파라미터 갱신과 학습 결과 집계를 수행 |

함께 사용할 데이터·모델·optimizer·Trainer는 같은 `ExecutionContext`로 만든다. 텐서 생성 시 값과 shape를 함께 전달하며, 값의 개수는 shape의 각 차원을 곱한 값과 일치해야 한다.

## 작은 분류 모델 학습하기

아래 코드를 사용하는 프로젝트의 `src/main.rs`에 넣고 `cargo run`으로 실행한다. 입력 특성 2개를 받아 클래스 2개를 구분하는 작은 MLP 예제다. 실제 성능 평가를 위한 데이터셋이 아니라 API 사용 흐름을 보여주기 위한 예제다.

```rust
use trench_deep::{ExecutionContext, MlResult};
use trench_deep::nn::Mlp;
use trench_deep::loss::SoftmaxCrossEntropyLoss;
use trench_deep::optimizer::{Adam, Optimizer};
use trench_deep::trainer::{
    EpochSchedule, SupervisedDataset, TrainableModel, Trainer,
};

fn main() -> MlResult<()> {
    // 1. 실행 환경과 모델: 입력 2개 → 은닉 노드 4개 → 클래스 2개
    let ctx = ExecutionContext::new();
    let mut model = Mlp::new(&ctx, 2, 4, 2)?;

    // 2. 학습 데이터: 각 항목의 shape는 [배치 크기, 특성/클래스 수]
    let inputs = [
        ctx.input(vec![0.0, 0.0], &[1, 2])?,
        ctx.input(vec![1.0, 1.0], &[1, 2])?,
    ];
    let targets = [
        ctx.tensor(vec![1.0, 0.0], &[1, 2])?, // 클래스 0
        ctx.tensor(vec![0.0, 1.0], &[1, 2])?, // 클래스 1
    ];
    let input_refs: Vec<_> = inputs.iter().collect();
    let target_refs: Vec<_> = targets.iter().collect();
    let dataset = SupervisedDataset::new(&ctx, &input_refs, &target_refs)?;

    // 3. Optimizer 생성과 모델 파라미터 등록
    let mut optimizer = Adam::new(&ctx, 0.05, 0.9, 0.999, 1e-8)?;
    optimizer.register_all(&model.parameters())?;

    // 4. 학습: 데이터 전체를 20번 반복
    let loss = SoftmaxCrossEntropyLoss::new();
    let result = Trainer::supervised(&ctx).minimal().fit(
        &mut model,
        &loss,
        &mut optimizer,
        &dataset,
        EpochSchedule::new(20)?,
    )?;
    println!("마지막 epoch loss: {}", result.final_loss);

    // 5. 새 입력에 대한 클래스별 확률 확인
    let input = ctx.tensor(vec![1.0, 1.0], &[1, 2])?;
    let probabilities = model.predict(&input)?.to_vec()?;
    println!("클래스별 확률: {:?}", probabilities);

    Ok(())
}
```

`ctx.input(...)`은 모델에 넣을 `Variable`을, `ctx.tensor(...)`는 정답 등의 `Tensor`를 만든다. 예제의 정답은 해당 클래스 위치만 1인 one-hot 형식이다. 모델의 `forward`는 logits를 반환하고, 외부 `SoftmaxCrossEntropyLoss`가 loss를 계산한다. `predict`는 클래스별 확률을 반환한다. Loss를 교체할 때 모델을 수정할 필요가 없다.

`SupervisedDataset`은 이미 배치로 구성된 입력·정답을 빌려 사용한다. 위 예제에서는 항목 하나가 sample 하나인 배치다. sample들을 모아 배치를 만들거나 섞어서 학습하려면 `DataLoader`를 사용한다.

## 다음 단계

- **학습 설정:** `Trainer::supervised(&ctx).minimal()`처럼 학습 방식과 프리셋을 선택한다. `.silent()`, `.default()`, `.verbose()`도 제공하며, 이후 `.with_seed(42)`나 `.metrics(...)` 등으로 설정을 덮어쓸 수 있다.
- **모델 직접 작성:** `TrainableModel`로 context와 파라미터를 제공하고, `ForwardModel`로 예측 연산을 제공한다. 일반 지도학습의 입력 준비와 loss 계산은 Trainer가 담당한다.
- **Prepared 실행:** `.prepared()`로 forward와 loss 그래프를 준비해 재사용한다. 일반 지도학습은 `ForwardModel`을 그대로 사용하며, 연산·provider가 prepared 실행을 지원해야 한다. 사용자 정의 학습 방식은 `PreparedTrainingStrategy`가 필요하다.
- **학습 결과:** `TrainResult`에서 마지막 loss, 완료한 epoch 수, 종료 사유와 활성화된 메트릭을 확인한다.

### 기능별 문서

아래 항목은 세부 문서가 작성되면 링크를 연결할 자리다.

| 주제 | 문서에서 다룰 내용 | 상태 |
| --- | --- | --- |
| 텐서와 자동 미분 | shape, 연산, gradient, context 수명 | 작성 예정 |
| 데이터 입력 | Dataset, DataLoader, batching, shuffle, 파일 읽기 | 작성 예정 |
| 모델과 레이어 | 내장 레이어, 모델 구성, 사용자 정의 모델 | 작성 예정 |
| 학습과 optimizer | 학습 방식, loss, 프리셋, seed, clipping | [API 안내](tp/TRAINER_API_MIGRATION.md) |
| Prepared 실행 | 입력 준비, 그래프 재사용, 지원 조건 | 작성 예정 |
| 메트릭과 시각화 | hook, observer, 학습 상태와 그래프 확인 | 작성 예정 |
| 저장과 체크포인트 | 모델 저장, interrupt 처리, 재개 지원 범위 | 작성 예정 |

기존 Trainer API를 사용한 코드를 전환하려면 [Trainer API 통합 및 전환 안내](tp/TRAINER_API_MIGRATION.md)를 참고한다.
