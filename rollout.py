
def rollout(env, policy, num_rollouts, rollout_length, gamma):
    rewards = []
    times = []
    for i in range(num_rollouts):
        s = env.reset()
        factor = 1.
        done = False
        t = 0
        rsum = 0
        while not done and t <= rollout_length:
            action = policy.act(s)
            s, reward, done, info = env.step(action)
            rsum += reward * factor
            factor *= gamma
            t += 1
        rewards.append(rsum)
        times.append(t)
    return rewards, times


def rollout_envs(envs, policy, num_rollouts, rollout_length, gamma):
    rewards = []
    times = []
    for env in envs:
        r, t = rollout(env, policy, num_rollouts, rollout_length, gamma)
        rewards += r
        times += t
    return rewards, times


def rollout_evaluate(envs, policy, num_rollouts, rollout_length, gamma):
    rewards = []
    times = []
    for env in envs:
        for i in range(num_rollouts):
            r, t = rollout(env, policy, 1, rollout_length, gamma)
            rewards += r
            times += t
    return rewards, times

