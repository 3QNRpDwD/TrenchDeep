# TrenchDeep 🚀

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![Version](https://img.shields.io/badge/version-0.0.2-blue.svg)](https://github.com/3QNRpDwD/TrenchDeep)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

A **from-scratch deep learning framework** built in Rust for educational purposes and high-performance machine learning research.

## Known Issues

⚠️ A bug was introduced in [commit `677ebf5`](https://github.com/3QNRpDwD/TrenchDeep/commit/677ebf5abe7427fb548f766bfa21753b22be30c0).  
See [Issue #3](https://github.com/OWNER/REPO/issues/3) for details.

Until this bug is resolved, please use the stable version at  
[commit `d5dc143`](https://github.com/3QNRpDwD/TrenchDeep/tree/d5dc143a25c6cfb6a8c126aaa553a53eb9b93ce9).

## 📖 Table of Contents

- [About](#about)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Computation Graph Visualization](#computation-graph-visualization)
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
├── graph/                 # Generated visualization files
│   ├── *.dot             # DOT format graphs
│   └── *.svg             # SVG rendered graphs
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

## 🎨 Computation Graph Visualization

TrenchDeep provides powerful visualization capabilities to help understand and debug neural networks and optimization functions. The framework can export computation graphs to DOT format, which can then be converted to various image formats.

### 📊 Available Visualizations

The repository includes several example computation graphs:

#### 1. **Optimization Functions**
- **`graph/rosenbrock.dot`**: Rosenbrock function (classic optimization benchmark)
- **`graph/goldstein.dot`**: Goldstein-Price function (multimodal test function)

#### 2. **Neural Networks** 
- **`graph/twolayer.dot`**: Two-layer neural network with sigmoid activations
- **`graph/twolayer_refactored.svg`**: Refactored two-layer network (pre-rendered SVG)

### 🎭 Node Types and Color Scheme

The visualization uses a modern, color-coded design system:

| Node Type | Shape | Color | Description |
|-----------|-------|-------|-------------|
| **Input** | House | 🟢 Green (`#10B981`) | Input data/variables |
| **Function** | Hexagon | 🔵 Blue (`#3B82F6`) | Mathematical operations |
| **Variable** | Ellipse | 🟡 Amber (`#F59E0B`) | Intermediate values |
| **Output** | Inverted House | 🔴 Red (`#EF4444`) | Final outputs |
| **Weight** | Diamond | 🟣 Purple (`#8B5CF6`) | Learnable parameters |
| **Bias** | Circle | 🟠 Orange (`#F97316`) | Bias terms |
| **Loss** | Octagon | 🔴 Pink (`#EC4899`) | Loss functions |
| **Activation** | Double Circle | 🔵 Cyan (`#06B6D4`) | Activation functions |

### 🔧 Generating Visualization Files

#### Enable Visualization Features

```bash
# Build with visualization support
cargo build --features enableVisualization

# Run tests with visualization
cargo test --features enableVisualization
```

#### Programmatic Graph Export

```rust
use trench_deep::{
    tensor::{graph::ComputationGraph, visualization::GraphViz},
    MlResult
};

#[cfg(feature = "enableVisualization")]
fn export_computation_graph() -> MlResult<()> {
    // ... build your computation graph ...
    
    // Export to DOT format
    let graph_viz = GraphViz::new();
    let dot_content = graph_viz.export_to_dot(&ComputationGraph::current())?;
    
    // Save to file
    std::fs::write("my_graph.dot", dot_content)?;
    
    Ok(())
}
```

### 🖼️ Converting DOT Files to Images

#### Prerequisites for Image Conversion

Install Graphviz on your system:

```bash
# Ubuntu/Debian
sudo apt-get install graphviz

# macOS (with Homebrew)
brew install graphviz

# Windows (with Chocolatey)
choco install graphviz

# Fedora/CentOS
sudo dnf install graphviz
```

#### Conversion Commands

```bash
# Convert to PNG (high quality, good for documentation)
dot -Tpng graph/rosenbrock.dot -o rosenbrock.png

# Convert to SVG (vector format, scalable)
dot -Tsvg graph/twolayer.dot -o twolayer.svg

# Convert to PDF (print-ready)
dot -Tpdf graph/goldstein.dot -o goldstein.pdf

# Convert to JPG (smaller file size)
dot -Tjpg graph/twolayer.dot -o twolayer.jpg

# High-resolution PNG with DPI setting
dot -Tpng -Gdpi=300 graph/goldstein.dot -o goldstein_hires.png
```

#### Batch Conversion Script

Create a script to convert all DOT files at once:

```bash
#!/bin/bash
# convert_graphs.sh

# Create output directory
mkdir -p images

# Convert all .dot files to PNG
for dotfile in graph/*.dot; do
    filename=$(basename "$dotfile" .dot)
    echo "Converting $filename..."
    dot -Tpng -Gdpi=200 "$dotfile" -o "images/${filename}.png"
done

echo "All graphs converted to images/ directory"
```

Make it executable and run:

```bash
chmod +x convert_graphs.sh
./convert_graphs.sh
```

### 🎯 Visualization Best Practices

#### For Educational Purposes
- Use **PNG** format for crisp documentation images
- Set DPI to 200-300 for high-quality prints
- **SVG** format for interactive web documentation

#### For Publications
- **PDF** format for academic papers
- **EPS** format for LaTeX documents
- High DPI PNG (300+) for conference presentations

#### Example Output Formats

```bash
# Academic paper quality
dot -Tpdf -Gsize="8,6" -Gdpi=300 graph/rosenbrock.dot -o paper_figure.pdf

# Web documentation
dot -Tsvg -Gsize="10,8" graph/twolayer.dot -o web_diagram.svg

# Presentation slide
dot -Tpng -Gdpi=150 -Gsize="12,9" graph/goldstein.dot -o slide.png
```

### 📈 Understanding the Generated Graphs

#### Rosenbrock Function Visualization
The Rosenbrock function graph shows:
- Two input variables (x, y)
- Squared terms and their combinations  
- The classic "banana-shaped" valley optimization landscape

#### Two-Layer Neural Network
The neural network graph displays:
- Input layer with data flow
- Weight matrices (diamonds) and biases (circles)
- Matrix multiplication operations
- Sigmoid activation functions (double circles)
- Output layer with loss computation

#### Goldstein-Price Function  
Complex multimodal function showing:
- Multiple local minima structure
- Polynomial term combinations
- Nested mathematical operations

### 🔍 Debugging with Visualizations

#### Common Debugging Patterns

```rust
#[cfg(feature = "enableVisualization")]
fn debug_gradient_flow() -> MlResult<()> {
    // 1. Build your model
    let model = create_model()?;
    
    // 2. Forward pass
    let output = model.forward(&input)?;
    let loss = loss_function.apply(&[&output, &target])?;
    
    // 3. Export BEFORE backward pass
    export_graph("forward_pass.dot")?;
    
    // 4. Backward pass
    loss.backward()?;
    
    // 5. Export AFTER backward pass (shows gradients)
    export_graph("with_gradients.dot")?;
    
    Ok(())
}
```

This workflow helps identify:
- **Gradient vanishing/exploding**: Long chains of small/large values
- **Dead neurons**: Missing connections or zero gradients  
- **Architecture issues**: Unexpected graph topology

## 🛠️ Configuration Features

TrenchDeep uses Cargo features for modular compilation:

```toml
[features]
default = []
enableBackpropagation = []
requiresGrad = ["enableBackpropagation"]
enableHigherOrderDifferentiation = ["requiresGrad"]
enableVisualization = ["enableBackpropagation"]
#debugging = []
```

### Feature Descriptions

- **`enableBackpropagation`**: Enables automatic differentiation and training
- **`requiresGrad`**: Enables gradient storage for intermediate computations
- **`enableHigherOrderDifferentiation`**: Supports second-order gradients
- **`enableVisualization`**: Computation graph visualization to DOT/SVG

### Usage with Features

```bash
# Full functionality with visualization
cargo test --features "enableVisualization"

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
cargo test --lib benchmark --features enableBackpropagation

# Run MNIST integration test
cargo test --package trench-deep --lib tests::mnist_test::softmax_regression_mnist_classification_integration_test --features enableBackpropagation
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

```bash
cargo test --package trench-deep --lib <test_path> --features debugging
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
  - ["밑바닥 부터 시작하는 딥러닝 3" by 사이토 고키](https://www.google.co.kr/books/edition/%EB%B0%91%EB%B0%94%EB%8B%A5%EB%B6%80%ED%84%B0_%EC%8B%9C%EC%9E%91%ED%95%98%EB%8A%94_%EB%94%A5%EB%9F%AC%EB%8B%9D_3/2uQKEAAAQBAJ?hl=ko&gbpv=0)
- **Open Source Libraries**:
  - `mnist` crate for dataset loading
  - `indicatif` for progress bars
  - `serde` for serialization
- **Graphviz** for powerful graph visualization capabilities

---

<div align="center">

**⭐ Star this repository if you find it helpful! ⭐**

Made with ❤️ in Rust

</div>
