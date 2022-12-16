import gymenv_v2
from gymenv_v2 import make_multiple_env
import numpy as np
from config import custom_config, easy_config, hard_config
from layers import Embedding
import tensorflow as tf
from policy import Policy, RandomPolicy
from rollout import rollout, rollout_multiple
import tensorflow_probability as tfp
from utils import discounted_rewards, AdamOptimizer
import os
import time
import wandb

def get_params(prev_reward, r_thresh, params_thresh):
    """
    r_thresh = [0.2, 0.4, 0.8]
    params_thresh = [(0.1, 10, 0.30), (0.01, 10, 0.2), (0.001, 10, 0.15), (0.0001, 10, 0.10)]
    """
    for i, r in enumerate(r_thresh):
        if prev_reward <= r:
            return params_thresh[i]
    return params_thresh[-1]

def es(env, units, activations, r_thresh, params_thresh, num_eval, num_cuts, gamma, num_episodes, num_test):
    print("")
    print("--------------RUNNING EVOLUTION STRATEGY--------------")
    print("")
    lr, num_trajectories, delta_std = get_params(0., r_thresh, params_thresh)
    policy = Policy(units, activations, lr)
    s = env.reset()
    _ = policy.compute_prob(s)
    
    optimizers = [AdamOptimizer(lr=lr) for _ in range(len(policy.get_weights()))]
    train_rrecord = []
    prev_reward = 0.
    movingAverage = 0.
    fixedWindow = 10
    all_rewards = []
    
    for e in range(num_episodes):
        print(f"Episode {e+1}")
        
        start_t = time.time()
        lr, num_trajectories, delta_std = get_params(prev_reward, r_thresh, params_thresh)
        for i in range(len(optimizers)):
            optimizers[i].lr = lr
        w_orig = policy.get_weights()
        epsilons = []
        rewards_table = np.zeros(num_trajectories)
        
        print(f"Simulating {num_trajectories} trajectories...")
        for t in range(num_trajectories):
            eps = [np.random.randn(*x.shape)*delta_std for x in w_orig]
            w_new = [w_orig[i] + eps[i] for i in range(len(w_orig))]
            policy.set_weights(w_new)
            rewards, states, actions = rollout(env, policy, num_cuts, gamma)
            epsilons.append(eps)
            rewards_table[t] = np.sum(rewards)
        
        rewards_table_norm = (rewards_table - np.mean(rewards_table))/(np.std(rewards_table) + 1e-8)
        
        grads = []
        print("Estimating gradient...")
        for j in range(len(w_orig)):
            arr = np.zeros(epsilons[0][j].shape)
            for i in range(len(epsilons)):
                arr += epsilons[i][j] * rewards_table[i]
            arr /= (len(epsilons) * delta_std)
            grads.append(arr)
        
        new_w = [optimizers[i].update(w_orig[i], grads[i]) for i in range(len(w_orig))]
        
        policy.set_weights(new_w)
        print("Evaluating rewards...")

        eval_r, _, _ = rollout_multiple(env, policy, num_eval, 50, gamma=1.)
        eval_r = np.array(eval_r).sum(axis=1)
        train_rrecord.append(np.mean(eval_r))
        
        for aux_r in eval_r:
            wandb.log({"training rewards": aux_r})
            all_rewards.append(aux_r)
        
        if len(all_rewards) >= fixedWindow:
            movingAverage = np.mean(all_rewards[-fixedWindow:])
            wandb.log({f"moving average training rewards": movingAverage})
        
        print("Evaluated rewards: %.4f" % np.mean(eval_r))
        print('mean',np.mean(eval_r),'max',np.max(eval_r),'min',np.min(eval_r),'std',np.std(eval_r))
        
        end_t = time.time()
        print("Time elapsed: %.4f minutes" % ((end_t - start_t)/60))
        print("-------------------------------------------------")
        prev_reward = np.mean(eval_r)
    
    print("Evaluating policy")
    start_t = time.time()
    test_rrecord = eval_policy(policy, env, num_test)
    end_t = time.time()
    print(f"Average test reward: {np.mean(test_rrecord)}")
    print("Time elapsed: %.4f minutes" % ((end_t - start_t)/60))
    
    return train_rrecord, test_rrecord, all_rewards, policy.get_weights()
    
def pg(env, units, activations, r_thresh, params_thresh, num_eval, num_cuts, gamma, num_episodes, num_test):
    print("")
    print("--------------RUNNING POLICY GRADIENT ALGORITHM--------------")
    print("")
    
    lr, num_trajectories, delta_std = get_params(0., r_thresh, params_thresh)
        
    policy = Policy(units, activations, lr)
    s = env.reset()
    _ = policy.compute_prob(s)
    
    train_rrecord = []
    prev_reward = 0.
    movingAverage = 0.
    fixedWindow = 10
    all_rewards = []
    
    for e in range(num_episodes):
        print(f"Episode {e+1}")
        lr, num_trajectories, delta_std = get_params(prev_reward, r_thresh, params_thresh)
        policy.optimizer.learning_rate = lr
        start_t = time.time()
        
        rewards_train = []
        states_train = []
        actions_train = []
        
        print(f"Simulating {num_trajectories} trajectories...")
        for t in range(num_trajectories):
            rewards, states, actions = rollout(env, policy, num_cuts, gamma)
            rewards_train.append(np.flip(np.flip(rewards).cumsum()))
            states_train.append([[aux2.astype(float) for aux2 in aux1] for aux1 in states])
            actions_train.append(actions)
        
        print("Estimating gradient...")
        for s_, r_, a_ in zip(states_train, rewards_train, actions_train):
            _, _ = policy.train(s_, r_, a_)
        
        print("Evaluating rewards...")
        eval_r, _, _ = rollout_multiple(env, policy, num_eval, 50, gamma=1.)
        eval_r = np.array(eval_r).sum(axis=1)
        train_rrecord.append(np.mean(eval_r))
        
        for aux_r in eval_r:
            wandb.log({"training rewards": aux_r})
            all_rewards.append(aux_r)
        
        if len(all_rewards) >= fixedWindow:
            movingAverage = np.mean(all_rewards[-fixedWindow:])
            wandb.log({f"moving average training reward": movingAverage})
            
        print("Evaluated rewards: %.4f" % np.mean(eval_r))
        print('mean',np.mean(eval_r),'max',np.max(eval_r),'min',np.min(eval_r),'std',np.std(eval_r))
        print("")
        
        end_t = time.time()
        print("Time elapsed: %.4f minutes" % ((end_t - start_t)/60))
        prev_reward = np.mean(eval_r)
        print("-------------------------------------------------")
    
    print("Evaluating policy")
    start_t = time.time()
    test_rrecord = eval_policy(policy, env, num_test)
    end_t = time.time()
    print(f"Average test reward: {np.mean(test_rrecord)}")
    print("Time elapsed: %.4f minutes" % ((end_t - start_t)/60))
    
    return train_rrecord, test_rrecord, all_rewards, policy.get_weights()


def eval_policy(policy, env, num_evals):
    test_rewards = []
    for i in range(num_evals):
        rewards, states, actions = rollout(env, policy, 50, 1.)
        test_rewards.append(np.sum(rewards))
    return test_rewards
    
    
def main(run_name, config, r_thresh, params_thresh, num_eval=10, num_cuts=10, gamma=1., strategies=["es", "pg"], 
         units=[64, 64, 64], activations=['relu', 'relu', 'linear'], num_episodes=50, num_test=50):
    env = make_multiple_env(**config) 
    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists(f"results/{run_name}"):
        os.mkdir(f"results/{run_name}")
    
    all_params = {
        "units": units, "activations": activations, "r_thresh": r_thresh, "params_thresh": params_thresh, 
        "num_eval": num_eval, "num_cuts": num_cuts, "gamma": gamma, "num_episodes": num_episodes, "num_test": num_test
    }
    
    np.save(f"results/{run_name}/params", all_params)
    for strategy in strategies:
        tag = "training-easy" if len(config["idx_list"]) ==10 else "training-hard" 
        run=wandb.init(project="finalproject", entity="orcs4529", tags=[tag])
        if not os.path.exists(f"results/{run_name}/{strategy}"):
                os.mkdir(f"results/{run_name}/{strategy}")
                
        if strategy == "es":
            train_rrecord, test_rrecord, all_rewards, weights = es(env, **all_params)
        else:
            train_rrecord, test_rrecord, all_rewards, weights = pg(env, **all_params)
        
        np.save(f"results/{run_name}/{strategy}/train_rrecord", train_rrecord)
        np.save(f"results/{run_name}/{strategy}/test_rrecord", test_rrecord)
        np.save(f"results/{run_name}/{strategy}/all_rewards", all_rewards)
        np.save(f"results/{run_name}/{strategy}/weights", weights)
        run_info = {
            "id": run.id,
            "url": run.get_url()
        }
        np.save(f"results/{run_name}/{strategy}/run_info", run_info)
        run.finish()
