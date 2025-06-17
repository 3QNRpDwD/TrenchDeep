use super::*;

#[derive(Debug, Serialize, Deserialize)]
pub struct TestConfig {
    pub n_train: u32,
    pub n_val: u32,
    pub n_features: usize,
    pub n_classes: usize,
    pub n_hidden_1: usize,
    pub n_hidden_2: usize,
    pub learning_rate: f32,
    pub epochs: usize,
    pub tolerance: f32,
    pub required_accuracy: f32,
    pub model_save_path: String,
    pub visualization_path: String,
}

impl Default for TestConfig {
    fn default() -> Self {
        TestConfig {
            n_train: 5000,
            n_val: 500,
            n_features: 784, // 28*28 MNIST 이미지 크기
            n_hidden_1: 128,
            n_hidden_2: 30,
            n_classes: 10,   // 0-9 숫자 클래스
            learning_rate: 0.01,    
            epochs: 15,
            tolerance: 1e-5,
            required_accuracy: 80.0, // 테스트 통과를 위한 최소 정확도
            model_save_path: "model_parameters.json".to_string(),
            visualization_path: "graph/twolayer_refactored.svg".to_string(),
        }
    }
}
