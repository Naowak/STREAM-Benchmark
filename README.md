# Stream Benchmark

A comprehensive benchmark suite for evaluating sequence modeling capabilities of neural networks, particularly focusing on memory, long-term dependencies, and temporal reasoning tasks.

## 🚀 Features

- **12 Diverse Tasks**: From simple memory tests to complex pattern recognition
- **Multiple Difficulty Levels**: Small, medium and large configurations. Advice : if you're building an architecture, you should begin with small.
- **Unified Interface**: Consistent API across all tasks with standardized evaluation metrics
- **Ready-to-Use**: Pre-configured datasets with train/validation/test splits
- **Flexible**: Support for both classification and regression tasks

## 📦 Installation

```bash
pip install stream-benchmark
```

Or install from source:

```bash
git clone https://github.com/Naowak/stream-benchmark.git
cd stream-benchmark
pip install -e .
```

## 🎯 Quick Start

```python
import stream_benchmark as sb

# Build a task
task_data = sb.build_task('simple_copy', difficulty='small', seed=0)

# Access the data
X_train = task_data['X_train']  # Training inputs
Y_train = task_data['Y_train']  # Training targets
T_train = task_data['T_train']  # Prediction timesteps

# Train your model (example with dummy predictions)
Y_pred = your_model.predict(X_train)

# Evaluate performance
score = sb.compute_score(
    Y=Y_train, 
    Y_hat=Y_pred, 
    prediction_timesteps=T_train,
    classification=task_data['classification']
)
print(f"Score: {score}")
```

## 📚 Available Tasks

### Memory Tests
### Postcasting Tasks
- **`discrete_postcasting`**: Copy discrete sequences after a delay  

    ![discrete_postcasting](https://raw.githubusercontent.com/Naowak/stream-benchmark/main/images/discrete_postcasting.png)

- **`continuous_postcasting`**: Copy continuous sequences after a delay  

    ![continuous_postcasting](https://raw.githubusercontent.com/Naowak/stream-benchmark/main/images/continuous_postcasting.png)

### Signal Processing
- **`sinus_forecasting`**: Predict frequency-modulated sinusoidal signals 

    ![sinus_forecasting](https://raw.githubusercontent.com/Naowak/stream-benchmark/main/images/sinus_forecasting.png)

- **`chaotic_forecasting`**: Forecast Lorenz system dynamics 

    ![chaotic_forecasting](https://raw.githubusercontent.com/Naowak/stream-benchmark/main/images/chaotic_forecasting.png)


### Long-term Dependencies
- **`discrete_pattern_completion`**: Complete masked repetitive patterns (discrete)  

    ![discrete_pattern_completion](https://raw.githubusercontent.com/Naowak/stream-benchmark/main/images/discrete_pattern_completion.png)

- **`continuous_pattern_completion`**: Complete masked repetitive patterns (continuous) 

    ![continuous_pattern_completion](https://raw.githubusercontent.com/Naowak/stream-benchmark/main/images/continuous_pattern_completion.png)

- **`simple_copy`**: Memorize and reproduce sequences after delay + trigger  

    ![simple_copy](https://raw.githubusercontent.com/Naowak/stream-benchmark/main/images/simple_copy.png)

- **`selective_copy`**: Memorize only marked elements and reproduce them  

    ![selective_copy](https://raw.githubusercontent.com/Naowak/stream-benchmark/main/images/selective_copy.png)


### Information Manipulation
- **`adding_problem`**: Add numbers at marked positions  

    ![adding_problem](https://raw.githubusercontent.com/Naowak/stream-benchmark/main/images/adding_problem.png)

- **`sorting_problem`**: Sort sequences according to given positions 

    ![sorting_problem](https://raw.githubusercontent.com/Naowak/stream-benchmark/main/images/sorting_problem.png)

- **`bracket_matching`**: Validate parentheses sequences

    ![bracket_matching](https://raw.githubusercontent.com/Naowak/stream-benchmark/main/images/bracket_matching.png)

- **`sequential_mnist`**: Classify MNIST digits read column by column 

    ![sequential_mnist](https://raw.githubusercontent.com/Naowak/stream-benchmark/main/images/sequential_mnist.png)


## 🔧 Task Configuration

Each task supports three difficulty levels (the following numbers may vary in function of the task):

### Small 
- Reduced sequence lengths and sample counts
- Suitable for quick experiments and debugging
- Example: 100 training samples, sequences of ~50-100 timesteps

### Medium 
- Realistic problem sizes
- Suitable for thorough model evaluation
- Example: 1,000 training samples, sequences of ~100-200 timesteps

### Large 
- Big Data configurations
- Suitable for high-performance models
- Example: 10,000 training samples, sequences of ~200-500 timesteps

For the tasks that allow it, the difficulty is also adjusted by the dimensions of input & output.

```python
# Small configuration (fast)
task_small = sb.build_task('bracket_matching', difficulty='small', seed=0)

# Medium configuration (thorough)
task_medium = sb.build_task('bracket_matching', difficulty='medium', seed=0)

# Large configuration (comprehensive)
task_large = sb.build_task('bracket_matching', difficulty='large', seed=0)
```

## 📊 Data Format

All tasks return a standardized dictionary:

```python
{
    'X_train': np.ndarray,      # Training inputs [batch, time, features]
    'Y_train': np.ndarray,      # Training targets [batch, time, outputs]
    'T_train': np.ndarray,      # Training prediction timesteps [batch, n_predictions]
    'X_valid': np.ndarray,      # Validation inputs
    'Y_valid': np.ndarray,      # Validation targets  
    'T_valid': np.ndarray,      # Validation prediction timesteps
    'X_test': np.ndarray,       # Test inputs
    'Y_test': np.ndarray,       # Test targets
    'T_test': np.ndarray,       # Test prediction timesteps
    'classification': bool      # True for classification, False for regression : used to select the correspongind loss
}
```

## 🎨 Example: Complete Evaluation Pipeline

```python
import stream_benchmark as sb
import numpy as np
from MyModel import MyModel

def evaluate_model_on_all_tasks(model, difficulty='small'):
    """Evaluate a model on all available tasks."""
    
    results = {}
    task_names = [
        'simple_copy', 'selective_copy', 'adding_problem',
        'discrete_postcasting', 'continuous_postcasting',
        'discrete_pattern_completion', 'continuous_pattern_completion',
        'bracket_matching', 'sorting_problem', 'sequential_mnist',
        'sinus_forecasting', 'chaotic_forecasting'
    ]
    
    for task_name in task_names:
        print(f"Evaluating on {task_name}...")
        
        # Load task
        task_data = sb.build_task(task_name, difficulty=difficulty)
        
        # Train model (simplified)
        model = MyModel(...)
        model.train(task_data['X_train'], task_data['Y_train'], task_data['T_train'])
        # If you want your model to learn all timesteps, including the ones that are not evaluated :
        # Comment the previous line and uncomment the following one
        # model.train(task_data['X_train'], task_data['Y_train'])

        # Predict on test set
        Y_pred = model.predict(task_data['X_test'])
        
        # Compute score
        score = sb.compute_score(
            Y=task_data['Y_test'],
            Y_hat=Y_pred,
            prediction_timesteps=task_data['T_test'],
            classification=task_data['classification']
        )
        
        results[task_name] = score
        print(f"  Score: {score:.4f}")
    
    return results

# Usage
# results = evaluate_model_on_all_tasks(your_model, difficulty='medium')
```

## 🧪 Task Details

### Memory and Copy Tasks
Test the model's ability to store and retrieve information over time.

### Pattern Recognition Tasks  
Evaluate pattern detection and completion capabilities with both discrete and continuous sequences.

### Algorithmic Reasoning Tasks
Assess the model's ability to perform computations on stored information (adding, sorting).

### Real-world Sequence Tasks
Include practical applications like MNIST digit classification and signal forecasting.

## 📈 Evaluation Metrics

- **Classification tasks**: Error rate (1 - accuracy)
- **Regression tasks**: Mean Squared Error (MSE)

Lower scores indicate better performance for both metrics.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use Stream Benchmark in your research, please cite:

(paper in progress...)

```bibtex
@software{stream_benchmark,
  title={Stream Benchmark: Sequential Task Review to Evaluate Artificial Memory},
  author={Yannis Bendi-Ouis, Xavier Hinaut},
  year={2025},
  url={https://github.com/Naowak/stream-benchmark}
}
```

## 🙏 Acknowledgments

- Inspired by classic sequence modeling benchmarks
- Built with NumPy and Hugging Face Datasets
- MNIST task uses the official MNIST dataset

## 📞 Support

- 🐛 Issues: [GitHub Issues](https://github.com/Naowak/stream-benchmark/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/Naowak/stream-benchmark/discussions)

---

*Stream Benchmark - Advancing sequence modeling evaluation, one task at a time.*
