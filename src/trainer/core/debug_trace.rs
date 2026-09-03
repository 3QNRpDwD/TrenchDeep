use super::*;
use crate::trainer::TrainableModel;

impl TrainerCore {
    /// `debugging` feature 전용 모델 구조 요약.
    ///
    /// 일반 빌드에서는 빈 함수로 컴파일되어 파라미터 순회 비용이 없다.
    #[cfg(feature = "debugging")]
    pub(crate) fn trace_model<M: TrainableModel + ?Sized>(
        &self,
        paradigm: &str,
        model: &M,
        work_units: usize,
        batches_per_unit: Option<usize>,
    ) {
        let params = model.params();
        let total_elements: usize = params.iter().map(|p| p.tensor().data().len()).sum();

        tracing::debug!(
            target: "trench_deep::trainer::debug",
            paradigm,
            model = std::any::type_name::<M>(),
            parameters = params.len(),
            total_elements,
            work_units,
            batches_per_unit = ?batches_per_unit,
            "training model structure"
        );

        for (index, param) in params.iter().enumerate() {
            #[cfg(feature = "enableVisualization")]
            let label = param.label();
            #[cfg(not(feature = "enableVisualization"))]
            let label = "unlabeled";

            tracing::debug!(
                target: "trench_deep::trainer::debug",
                index,
                label,
                node_id = ?param.node_id(),
                shape = ?param.tensor().shape(),
                elements = param.tensor().data().len(),
                retain_grad = param.is_retain_grad(),
                "model parameter"
            );
            crate::tensor::operators::debug::stats(
                &format!("model.param[{index}]/{label}"),
                param.tensor(),
            );
        }
    }

    #[cfg(not(feature = "debugging"))]
    #[inline(always)]
    pub(crate) fn trace_model<M: TrainableModel + ?Sized>(
        &self,
        _paradigm: &str,
        _model: &M,
        _work_units: usize,
        _batches_per_unit: Option<usize>,
    ) {
    }
}
