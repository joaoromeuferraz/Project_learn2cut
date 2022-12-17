from config import easy_config, hard_config

params_1 = {
    "run_name": "easy_config1",
    "config": easy_config,
    "r_thresh": [0.2, 0.4, 0.8],
    "params_thresh": [(0.001, 10, 0.2), (0.001, 10, 0.2), (0.001, 10, 0.2), (0.001, 10, 0.2)],
    "num_eval": 3,
    "num_cuts": 10,
    "gamma": 1.,
    "strategies": ["es", "pg"],
    "units": [64, 64, 64],
    "activations": ['relu', 'relu', 'linear'],
    "num_episodes": 50,
    "num_test": 10
}

params_2 = params_1.copy()
params_2["run_name"] = "easy_config2"
params_2["params_thresh"] = [(0.1, 10, 0.2), (0.01, 10, 0.2), (0.001, 10, 0.2), (0.0001, 10, 0.2)]

params_3 = params_1.copy()
params_3["run_name"] = "easy_config3"
params_3["params_thresh"] = [(0.1, 10, 0.3), (0.01, 10, 0.2), (0.001, 10, 0.1), (0.0001, 10, 0.05)]

params_4 = params_1.copy()
params_4["run_name"] = "hard_config1"
params_4["config"] = hard_config

params_5 = params_1.copy()
params_5["run_name"] = "hard_config2"
params_5["config"] = hard_config
params_5["params_thresh"] = [(0.1, 10, 0.2), (0.01, 10, 0.2), (0.001, 10, 0.2), (0.0001, 10, 0.2)]

params_6 = params_1.copy()
params_6["run_name"] = "hard_config3"
params_6["config"] = hard_config
params_6["params_thresh"] = [(0.1, 10, 0.3), (0.01, 10, 0.2), (0.001, 10, 0.1), (0.0001, 10, 0.05)]

params_7 = {
    "run_name": "easy_config4",
    "config": easy_config,
    "r_thresh": [0.2, 0.4, 0.8],
    "params_thresh": [(0.1, 10, 0.3), (0.01, 10, 0.2), (0.001, 10, 0.1), (0.0001, 10, 0.05)],
    "num_eval": 3,
    "num_cuts": 10,
    "gamma": 1.,
    "strategies": ["es", "pg"],
    "units": [100, 100, 64],
    "activations": ['tanh', 'tanh', 'linear'],
    "num_episodes": 50,
    "num_test": 10
}

params_8 = {
    "run_name": "hard_config4",
    "config": hard_config,
    "r_thresh": [0.2, 0.4, 0.8],
    "params_thresh": [(0.1, 10, 0.3), (0.01, 10, 0.2), (0.001, 10, 0.1), (0.0001, 10, 0.05)],
    "num_eval": 3,
    "num_cuts": 10,
    "gamma": 1.,
    "strategies": ["es", "pg"],
    "units": [100, 100, 64],
    "activations": ['tanh', 'tanh', 'linear'],
    "num_episodes": 50,
    "num_test": 10
}
