from utils import discounted_rewards
# def rollout(env, policy, num_rollouts, rollout_length, gamma, return_cum_rewards=True):
#     rewards = []
#     times = []
#     states = []
#     for i in range(num_rollouts):
#         s = env.reset()
#         cum_rewards = []
#         states_ = []
#         factor = 1.
#         done = False
#         t = 0
#         rsum = 0
#         while not done and t <= rollout_length:
#             action = policy.act(s)
#             s, reward, done, info = env.step(action)
            
#             rsum += reward * factor
#             cum_rewards.append(rsum)
#             factor *= gamma
#             t += 1
#         if return_cum_rewards:
#             rewards.append(cum_rewards)
#         else:
#             rewards.append(rsum)
#         times.append(t)
#     return rewards, times


# def rollout_envs(envs, policy, num_rollouts, rollout_length, gamma, return_cum_rewards=True):
#     rewards = []
#     times = []
#     for env in envs:
#         r, t = rollout(env, policy, num_rollouts, rollout_length, gamma, return_cum_rewards=return_cum_rewards)
#         rewards += r
#         times += t
#     return rewards, times


# def rollout_evaluate(envs, policy, num_rollouts, rollout_length, gamma):
#     rewards = []
#     times = []
#     for env in envs:
#         for i in range(num_rollouts):
#             r, t = rollout(env, policy, 1, rollout_length, gamma,return_cum_rewards=False)
#             rewards += r
#             times += t
#     return rewards, times


def rollout(env, policy, num_cuts, gamma=1.):
    rewards = []
    states = []
    actions = []
    s = env.reset()
    done = False
    t = 0
    while not done and t <= num_cuts:
        states.append(s)
        a = policy.act(s)
        actions.append(a)
        s, reward, done, info = env.step(a)
        rewards.append(reward)
        t += 1
    if gamma != 1.:
        rewards = discounted_rewards(rewards, gamma)
    return rewards, states, actions

def rollout_multiple(env, policy, num_trajectories, num_cuts, gamma=1.):
    rewards = []
    states = []
    actions = []
    for _ in range(num_trajectories):
        r, s, a = rollout(env, policy, num_cuts, gamma)
        rewards.append(r)
        states += s
        actions += a
    return rewards, states, actions
    
        
        
    