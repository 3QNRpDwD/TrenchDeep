## 📁 src/

### 🏗️ Core Modules

#### `backend/mod.rs`
- **Purpose**: Hardware backend abstraction layer
- **Key Features**:
  - CPU computation support (`CpuBackend`)
  - Device type management (`Device`, `DeviceType`)
  - Basic mathematical operation interfaces (add, multiply, matmul, etc.)
  - FLOPS calculation and performance measurement
- **Design**: Extensible architecture through `Backend` trait

#### `tensor/mod.rs`
- **Purpose**: Tensor data structures and operation management
- **Key Features**:
  - Tensor creation and management (`Tensor`, `GlobalTensor`, `PooledTensor`)
  - Memory allocation optimization (`TensorAllocator`)
  - Computation graph management (`ComputationGraph`)
  - Automatic differentiation system
  - Visualization support (`VisualizationGraph`)
- **Macros**: `tensor_ops!`, `scalar_ops!`, `scalar!`

#### `tensor/operators/mod.rs`
- **Purpose**: Tensor operator definitions and implementations
- **Supported Operations**:
  - Basic operations: Add, Sub, Mul, Div, Neg
  - Mathematical functions: Exp, Log, Sqrt, Sin, Cos, Pow
  - Matrix operations: Matmul, Transpose
  - Statistics: Sum, Mean, Topk, Matmax
  - Neural network: Conv2d, MaxPool, AvgPool
- **Design**: `Function` trait-based with forward/backward propagation support

### 🧠 Neural Network Modules

#### `nn/mod.rs`
- **Purpose**: Neural network layers and model architecture
- **Core Components**:
  - `Variable`: Automatic differentiation-enabled variables
  - `Parameter`: Trainable parameters
  - `Layer` trait: Layer interface
  - `Sequential`: Sequential model composition
- **Layer Types**:
  - `Linear`: Fully connected layer
  - `Conv`: Convolutional layer
  - `MaxPooling`, `AvgPooling`: Pooling layers
- **Macros**: `variable!`, `sequential!`

#### `nn/activation/mod.rs`
- **Purpose**: Activation function implementations
- **Supported Functions**:
  - Sigmoid, Tanh, ReLU, Softmax
- **Design**: Unified interface through `Activation` trait

### 🎯 Training & Optimization

#### `optimizer/mod.rs`
- **Purpose**: Optimization algorithm implementations
- **Supported Optimizers**:
  - Basic: BGD, SGD, MiniBGD
  - Momentum-based: Momentum, NAG
  - Adaptive: AdaGrad, AdaDelta, RMSProp
  - Advanced: Adam, AdamW
- **Interface**: `Optimizer<T>` trait

#### `loss/mod.rs`
- **Purpose**: Loss function definitions
- **Supported Loss Functions**:
  - Regression: MeanSquaredError, MeanAbsoluteError, HuberLoss
  - Classification: BinaryCrossEntropyLoss, CrossEntropyLoss, SoftmaxCrossEntropyLoss
- **Design**: Unified interface through `Loss` trait

### 🧪 Testing & Examples

#### `tests/mod.rs`
- **Purpose**: Integration tests and examples
- **Main Tests**:
  - MNIST classification (Softmax Regression, MLP)
  - Model performance evaluation
  - Visualization generation
- **Benchmark Functions**:
  - Sphere, Matyas, Goldstein-Price, Rosenbrock function optimization

#### `lib.rs`
- **Purpose**: Library main entry point
- **Key Features**:
  - Module organization and re-exports
  - Error type definitions (`MlError`, `TensorError`)
  - Result type aliases (`MlResult<T>`)
  - Benchmark tests included

## 🔧 Key Features

### Feature Flags
- `enableBackpropagation`: Enable automatic differentiation
- `enableVisualization`: Enable computation graph visualization
- `requiresGrad`: Gradient computation requirements

### Macro System
- `define_op!`: Automate operator definitions
- `tensor_ops!`: Provide tensor operation convenience
- `variable!`: Simplify variable creation
- `sequential!`: Simplify model composition

### Memory Management
- Efficient memory management using thread-local storage
- Tensor pooling system for memory reuse
- Computation graph optimization

### Extensibility
- Backend abstraction enables diverse hardware support
- Trait-based design facilitates easy addition of new operators/layers
- Modular structure allows independent feature development
