from stream_benchmark.tasks import *

# ----- Define the set configs -----
stream_small = {
    'sinus_forecasting': {
        'fct': generate_sinus_forecasting,
        'params': {"sequence_length": 200, "forecast_length": 5, "training_ratio": 0.45, "validation_ratio": 0.1, "testing_ratio": 0.45},
        'classification': False,
    },
    'chaotic_forecasting': {
        'fct': generate_chaotic_forecasting,
        'params': {"sequence_length": 200, "forecast_length": 5, "training_ratio": 0.45, "validation_ratio": 0.1, "testing_ratio": 0.45},
        'classification': False,
    },
    'discrete_postcasting': {
        'fct': generate_discrete_postcasting,
        'params': {"n_train": 100, "n_valid": 20, "n_test": 100, "sequence_length": 50, "delay": 5, "n_symbols": 3},
        'classification': True,
    },
    'continuous_postcasting': {
        'fct': generate_continuous_postcasting,
        'params': {"n_train": 100, "n_valid": 20, "n_test": 100, "sequence_length": 50, "delay": 5},
        'classification': False,
    },
    'discrete_pattern_completion': {
        'fct': generate_discrete_pattern_completion,
        'classification': True,
        'params': {"n_train": 100, "n_valid": 20, "n_test": 100, "sequence_length": 60, "n_symbols": 3, "base_length": 4, "mask_ratio": 0.2}, 
    },
    'continuous_pattern_completion': {
        'fct': generate_continuous_pattern_completion,
        'classification': False,
        'params': {"n_train": 100, "n_valid": 20, "n_test": 100, "sequence_length": 60, "base_length": 4, "mask_ratio": 0.2}, 
    },
    'bracket_matching': {
        'fct': generate_bracket_matching,
        'params': {"n_train": 100, "n_valid": 20, "n_test": 100, "sequence_length": 50, "max_depth": 5},
        'classification': True,
    },
    'simple_copy': {
        'fct': generate_simple_copy,
        'classification': True,
        'params': {"n_train": 100, "n_valid": 20, "n_test": 100, "sequence_length": 22, "delay": 5, "n_symbols": 3}, 
    },
    'selective_copy': {
        'fct': generate_selective_copy,
        'classification': True,
        'params': {"n_train": 100, "n_valid": 20, "n_test": 100, "sequence_length": 40, "delay": 5, "n_markers": 5, "n_symbols": 3},
    },
    'adding_problem': {
        'fct': generate_adding_problem,
        'classification': True,
        'params': {"n_train": 100, "n_valid": 20, "n_test": 100, "sequence_length": 10, "max_number": 3},
    },
    'sorting_problem': {
        'fct': generate_sorting_problem,
        'classification': True,
        'params': {"n_train": 100, "n_valid": 20, "n_test": 100, "sequence_length": 10, "n_symbols": 3}, 
    },
    'sequential_mnist': {
        'fct': generate_sequential_mnist,
        'classification': True,
        'params': {"n_train": 100, "n_valid": 20, "n_test": 100, "path": "./data/mnist/", "cache_dir": "./data/"},
    },
}

stream_medium = {
    'sinus_forecasting': {
        'fct': generate_sinus_forecasting,
        'params': {"sequence_length": 20000, "forecast_length": 5, "training_ratio": 0.45, "validation_ratio": 0.1, "testing_ratio": 0.45},
        'classification': False,
    },
    'chaotic_forecasting': {
        'fct': generate_chaotic_forecasting,
        'params': {"sequence_length": 20000, "forecast_length": 5, "training_ratio": 0.45, "validation_ratio": 0.1, "testing_ratio": 0.45},
        'classification': False,
    },
    'discrete_postcasting': {
        'fct': generate_discrete_postcasting,
        'params': {"n_train": 10000, "n_valid": 200, "n_test": 1000, "sequence_length": 50, "delay": 5, "n_symbols": 3},
        'classification': True,
    },
    'continuous_postcasting': {
        'fct': generate_continuous_postcasting,
        'params': {"n_train": 10000, "n_valid": 200, "n_test": 1000, "sequence_length": 50, "delay": 5},
        'classification': False,
    },
    'discrete_pattern_completion': {
        'fct': generate_discrete_pattern_completion,
        'classification': True,
        'params': {"n_train": 10000, "n_valid": 200, "n_test": 1000, "sequence_length": 60, "n_symbols": 3, "base_length": 4, "mask_ratio": 0.2}, 
    },
    'continuous_pattern_completion': {
        'fct': generate_continuous_pattern_completion,
        'classification': False,
        'params': {"n_train": 10000, "n_valid": 200, "n_test": 1000, "sequence_length": 60, "base_length": 4, "mask_ratio": 0.2}, 
    },
    'bracket_matching': {
        'fct': generate_bracket_matching,
        'params': {"n_train": 10000, "n_valid": 200, "n_test": 1000, "sequence_length": 50, "max_depth": 5},
        'classification': True,
    },
    'simple_copy': {
        'fct': generate_simple_copy,
        'classification': True,
        'params': {"n_train": 10000, "n_valid": 200, "n_test": 1000, "sequence_length": 22, "delay": 5, "n_symbols": 3}, 
    },
    'selective_copy': {
        'fct': generate_selective_copy,
        'classification': True,
        'params': {"n_train": 10000, "n_valid": 200, "n_test": 1000, "sequence_length": 40, "delay": 5, "n_markers": 5, "n_symbols": 3},
    },
    'adding_problem': {
        'fct': generate_adding_problem,
        'classification': True,
        'params': {"n_train": 10000, "n_valid": 200, "n_test": 1000, "sequence_length": 10, "max_number": 3},
    },
    'sorting_problem': {
        'fct': generate_sorting_problem,
        'classification': True,
        'params': {"n_train": 10000, "n_valid": 200, "n_test": 1000, "sequence_length": 10, "n_symbols": 3}, 
    },
    'sequential_mnist': {
        'fct': generate_sequential_mnist,
        'classification': True,
        'params': {"n_train": 10000, "n_valid": 200, "n_test": 1000, "path": "./data/mnist/", "cache_dir": "./data/"},
    },
}