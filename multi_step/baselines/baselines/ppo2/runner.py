import numpy as np
from baselines.common.runners import AbstractEnvRunner

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, mb_h, mb_std, mb_neglogstd = [],[],[],[],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs, h, std = self.model.step(self.obs, S=self.states, M=self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_neglogstd.append(-np.log(std))
            mb_dones.append(self.dones)
            mb_h.append(h)
            mb_std.append(std)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_neglogstd = np.asarray(mb_neglogstd, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values, last_hs = self.model.value(self.obs, S=self.states, M=self.dones)
        # last_hs = self.model.hvalues(self.obs, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        mb_h = np.asarray(mb_h, dtype=np.float32)
        mb_std = np.asarray(mb_std, dtype=np.float32)
        mb_std = np.mean(mb_std, axis=2)
        mb_std = 0.5 * np.log(2 * np.pi * np.e * mb_std * mb_std)
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        mb_hadvs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        lasthlam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
                ### H reward
                nexthvalues = last_hs
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
                nexthvalues = mb_h[t + 1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            delta_h = mb_std[t] + self.gamma * nexthvalues * nextnonterminal - mb_h[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
            mb_hadvs[t] = lasthlam = delta_h + self.gamma * self.lam * nextnonterminal * lasthlam
        mb_returns = mb_advs + mb_values
        mb_hreturns = mb_hadvs + mb_h
        return (*map(sf01, (mb_obs, mb_returns, mb_hreturns, mb_dones, mb_actions, mb_values, mb_h, mb_neglogpacs, mb_neglogstd)),
            mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


