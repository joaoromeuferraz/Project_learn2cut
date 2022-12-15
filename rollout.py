from utils import discounted_rewards

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
    
        
        
    