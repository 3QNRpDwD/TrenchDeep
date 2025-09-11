# TrenchDeep 🚀

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![Version](https://img.shields.io/badge/version-0.0.2-blue.svg)](https://github.com/3QNRpDwD/TrenchDeep)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

A **from-scratch deep learning framework** built in Rust for educational purposes and high-performance machine learning research.

## 📖 Table of Contents

- [About](#about)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Configuration Features](#configuration-features)
- [Testing](#testing)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## 🎯 About

**TrenchDeep** is an educational deep learning framework designed to provide a deep understanding of neural networks, automatic differentiation, and optimization algorithms. Built entirely in Rust, it offers both performance and memory safety while maintaining educational clarity.

### 🤔 Why TrenchDeep?

- **Educational Focus**: Learn deep learning concepts by building from first principles
- **Rust Performance**: Memory-safe, zero-cost abstractions with C++-level performance
- **Modular Design**: Feature-gated architecture for selective compilation
- **Research-Ready**: Extensible for custom architectures and algorithms
- **Visualization**: Built-in computation graph visualization for debugging

## ✨ Features

### Core Features
- 🔢 **Tensor Operations**: N-dimensional arrays with automatic broadcasting
- 🎯 **Automatic Differentiation**: Reverse-mode AD (backpropagation) engine
- 🧠 **Neural Network Layers**: Linear, Convolutional, Pooling, Activation layers
- 📊 **Loss Functions**: Cross-entropy, MSE, and custom loss implementations
- 🔄 **Optimizers**: SGD, Adam, and extensible optimizer framework
- 📈 **Visualization**: Computation graph export to DOT format

### Advanced Features
- 🔗 **Higher-Order Differentiation**: Second-order gradients for advanced optimization
- 💾 **Memory Pool**: Efficient tensor memory management
- 🎨 **Model Serialization**: Save and load trained models
- 📊 **Progress Tracking**: Built-in training progress bars and logging
- 🧪 **Benchmarking**: Performance testing utilities

## 🏗️ Project Structure

```
TrenchDeep/
├── src/
│   ├── lib.rs              # Main library entry point
│   ├── backend/            # Hardware abstraction layer
│   │   ├── cpu/           # CPU-specific implementations
│   │   └── config.rs      # Backend configuration
│   ├── tensor/            # Core tensor operations
│   │   ├── operators/     # Mathematical operations
│   │   ├── allocator.rs   # Memory management
│   │   ├── graph.rs       # Computation graph
│   │   └── visualization.rs # Graph visualization
│   ├── nn/                # Neural network components
│   │   ├── activation/    # Activation functions
│   │   ├── linear.rs      # Fully connected layers
│   │   ├── conv.rs        # Convolutional layers
│   │   └── parameter.rs   # Learnable parameters
│   ├── loss/              # Loss functions
│   ├── optimizer/         # Optimization algorithms
│   └── tests/             # Integration tests
│       └── common/        # Test utilities and models
├── Cargo.toml             # Project configuration
└── README.md              # This file
```

## 🚀 Installation

### Prerequisites

- **Rust** 1.70 or higher
- **Cargo** (comes with Rust)

### Install from Source

```bash
git clone https://github.com/3QNRpDwD/TrenchDeep.git
cd TrenchDeep
cargo build --release
```

### Add as Dependency

Add to your `Cargo.toml`:

```toml
[dependencies]
trench-deep = { git = "https://github.com/3QNRpDwD/TrenchDeep" }
```

## ⚡ Quick Start

### Basic Tensor Operations

```rust
use trench_deep::{tensor::Tensor, MlResult};

fn main() -> MlResult<()> {
    // Create tensors
    let a = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let b = Tensor::new(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
    
    // Matrix multiplication
    let c = &a * &b;
    println!("Result: {:?}", c.data());
    
    Ok(())
}
```

### Automatic Differentiation

```rust
use trench_deep::{tensor::Tensor, var_input, MlResult};

fn main() -> MlResult<()> {
    // Create variables that require gradients
    let x = var_input!(Tensor::scalar(2.0));
    let y = var_input!(Tensor::scalar(3.0));
    
    // Define function: f(x,y) = x² + y²
    let z = sphere_function(&x, &y)?;
    
    // Compute gradients
    z.backward()?;
    
    println!("∂f/∂x = {:?}", x.grad().data()); // [4.0]
    println!("∂f/∂y = {:?}", y.grad().data()); // [6.0]
    
    Ok(())
}

fn sphere_function(x: &Variable, y: &Variable) -> MlResult<Variable> {
    let square = Square::new();
    let add = Add::new();
    
    add.apply(&[
        &square.apply(&[x])?,
        &square.apply(&[y])?
    ])
}
```

## 📚 Usage Examples

### Building a Neural Network

```rust
use trench_deep::{
    nn::{Sequential, Linear, activation::ReLULayer},
    loss::CrossEntropyLoss,
    MlResult
};

fn create_mlp() -> MlResult<Sequential> {
    let model = Sequential::new()
        .add_layer(Linear::new(784, 128, "hidden1")?)
        .add_layer(ReLULayer::new("relu1"))
        .add_layer(Linear::new(128, 64, "hidden2")?)
        .add_layer(ReLULayer::new("relu2"))
        .add_layer(Linear::new(64, 10, "output")?);
    
    Ok(model)
}
```

### Training Loop

```rust
#[cfg(feature = "enableBackpropagation")]
fn train_model(model: &mut impl Model, 
               x_train: &[&Variable], 
               y_train: &[&Variable]) -> MlResult<()> {
    let learning_rate = 0.001;
    let epochs = 100;
    
    for epoch in 0..epochs {
        for (x, y) in x_train.iter().zip(y_train.iter()) {
            // Forward pass
            let prediction = model.apply(x)?;
            let loss = CrossEntropyLoss::new().apply(&[&prediction, y])?;
            
            // Backward pass
            loss.backward()?;
            
            // Update parameters
            model.update(&Tensor::scalar(learning_rate))?;
            model.zero_grad()?;
            
            ComputationGraph::reset_graph();
        }
    }
    
    Ok(())
}
```

### MNIST Classification Example

```rust
use trench_deep::{
    tests::common::{
        data::MnistDataset,
        model::{SoftmaxRegression, Model}
    },
    MlResult
};

#[test]
fn mnist_classification() -> MlResult<()> {
    // Load MNIST dataset
    let dataset = MnistDataset::load_and_prepare_data(1000, 200, 784, 10)?;
    
    // Create model
    let mut model = SoftmaxRegression::build_model(784, 10)?;
    
    // Train model
    #[cfg(feature = "enableBackpropagation")]
    model.train(
        &dataset.x_train(),
        &dataset.t_train(),
        50,        // epochs
        0.01,      // learning rate
        1e-6       // tolerance
    )?;
    
    // Evaluate
    let accuracy = evaluate_model(&mut model, 
                                 &dataset.x_test(), 
                                 &dataset.t_test())?;
    
    println!("Test accuracy: {:.2}%", accuracy);
    Ok(())
}
```

## 🛠️ Configuration Features

TrenchDeep uses Cargo features for modular compilation:

```toml
[features]
default = []
enableBackpropagation = []
requiresGrad = ["enableBackpropagation"]
enableHigherOrderDifferentiation = ["requiresGrad"]
enableVisualization = ["enableBackpropagation"]
```

### Feature Descriptions

- **`enableBackpropagation`**: Enables automatic differentiation and training
- **`requiresGrad`**: Enables gradient storage for intermediate computations
- **`enableHigherOrderDifferentiation`**: Supports second-order gradients
- **`enableVisualization`**: Computation graph visualization to DOT/SVG

### Usage with Features

```bash
# functionality
cargo test --features enableVisualization

# Minimal build (inference only)
cargo build --release

# Enable all features
cargo build --all-features
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests with backpropagation
cargo test --features "enableBackpropagation"

# Run specific benchmark tests
cargo test benchmark --features "enableBackpropagation" -- --nocapture

# Run MNIST integration test
cargo test mnist --features "enableBackpropagation,enableVisualization"
```

### Available Test Functions

- **Optimization Functions**: Sphere, Rosenbrock, Goldstein-Price, Matyas
- **Neural Networks**: MLP, Softmax Regression
- **Integration Tests**: MNIST classification
- **Benchmarks**: Performance profiling utilities

### Example Test Output

```
test benchmark::sphere ... ok
test benchmark::rosenbrock ... ok
test mnist_test::softmax_regression_mnist_classification_integration_test ... ok

Epoch   1/50 [████████████████████] 1000/1000 Batches | AL: 0.245167 | AC: 91.20%
```

## 🚧 Roadmap

### Version 0.1.0 (Planned)
- [ ] GPU backend support (CUDA/OpenCL)
- [ ] Additional optimizers (Adam, RMSprop)
- [ ] Batch normalization layers
- [ ] Dropout regularization
- [ ] Model checkpointing

### Version 0.2.0 (Future)
- [ ] Convolutional neural networks
- [ ] Recurrent neural networks (LSTM/GRU)
- [ ] Distributed training
- [ ] ONNX model export/import

## 🐛 Bug Reports & Debugging

### Common Issues

1. **Compilation Errors**: Ensure Rust 1.70+ and enable required features
2. **NaN Gradients**: Check learning rate and input data normalization
3. **Memory Issues**: Use pooled tensors for large computations

### Debug Features

Enable debug logging:
```rust
use trench_deep::tests::common::utils::setup_logging;

fn main() {
    setup_logging("debug"); // trace, debug, info, warn, error
    // Your code here
}
```

### Reporting Bugs

Please open an issue on GitHub with:
- Rust version (`rustc --version`)
- TrenchDeep version
- Minimal reproducible example
- Error messages and stack traces

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
git clone https://github.com/your-username/TrenchDeep.git
cd TrenchDeep
cargo check --all-features
cargo test --all-features
```

### Code Style

- Follow Rust naming conventions
- Add documentation for public APIs
- Include tests for new functionality
- Run `cargo fmt` and `cargo clippy`

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

```
Copyright 2024 TrenchDeep Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## 📞 Contact

**Author**: TrenchDeep Development Team
- **Email**: 2QRNpDwD@gmail.com
- **GitHub**: [@3QNRpDwD](https://github.com/3QNRpDwD)
- **Repository**: https://github.com/3QNRpDwD/TrenchDeep

For questions, suggestions, or collaboration opportunities, feel free to reach out!

## 🙏 Acknowledgments

- **Rust Community** for excellent documentation and crates
- **PyTorch** and **TensorFlow** for deep learning inspiration
- **Educational Resources**:
  - ["밑바닥 부터 시작하는 딥러닝 3" by 사이토 고키]([https://www.oreilly.com/library/view/deep-learning-from/9781492041405/](https://www.google.co.kr/books/edition/%EB%B0%91%EB%B0%94%EB%8B%A5%EB%B6%80%ED%84%B0_%EC%8B%9C%EC%9E%91%ED%95%98%EB%8A%94_%EB%94%A5%EB%9F%AC%EB%8B%9D_3/2uQKEAAAQBAJ?hl=ko&gbpv=0))
- **Open Source Libraries**:
  - `mnist` crate for dataset loading
  - `indicatif` for progress bars
  - `serde` for serialization

---

<div align="center">

**⭐ Star this repository if you find it helpful! ⭐**

Made with ❤️ in Rust

</div>
