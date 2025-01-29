from flexibuff import FlexibleBuffer
from flexibuddiesrl import DDPG, TD3, Agent, PG, DQN
import gymnasium as gym
import numpy as np


import os
import pygame

# Set the desired window position (x, y)
x = 0
y = 0
# Set the environment variable
os.environ["SDL_VIDEO_WINDOW_POS"] = f"{x},{y}"

# from fasttttsandbox import TTTNvN

gym_disc_env = "CartPole-v1"  # "LunarLander-v2"  # "CartPole-v1"  #
gym_cont_env = "LunarLander-v2"  # "LunarLander-v2"  # "Pendulum-v1"  # "HalfCheetah-v4"


def __close():
    return 0


def test_single_env(
    env: gym.Env,
    agent: DDPG,
    buffer: FlexibleBuffer,
    n_episodes=100,
    n_steps=1000000,
    joint_obs_dim=7,
    discrete=False,
    debug=False,
    online=False,
):
    agent: DDPG
    rewards = []
    smooth_rewards = []
    aloss_return = []
    closs_return = []
    step = 0
    episode = 0
    m_aloss, m_closs = 0, 0
    n_updates = 0
    while step < n_steps and episode < n_episodes:

        ep_reward = 0
        obs, info = env.reset()

        obs = np.pad(obs, (0, joint_obs_dim - len(obs)), "constant")

        terminated, truncated = False, False
        while not (terminated or truncated):

            discrete_actions, continuous_actions, disc_lp, cont_lp, value = (
                agent.train_actions(obs, step=True, debug=debug)
            )

            # dac, cac, _, __, ___ = agent.train_actions(obs, step=True, debug=debug)
            # print(discrete_actions, continuous_actions)
            if discrete:
                actions = discrete_actions[0]  # [int(discrete_actions[0]), int(dac[0])]
            else:
                actions = continuous_actions

            # print(actions)
            if cont_lp is None:
                cont_lp = 0
            if disc_lp is None:
                disc_lp = 0
            # print(actions)
            # print(
            #     f"c_a: {continuous_actions}, d_a: {discrete_actions}, actions: {actions}"
            # )

            obs_, reward, terminated, truncated, _ = env.step(actions)
            obs_ = np.pad(obs_, (0, joint_obs_dim - len(obs_)), "constant")

            # print(
            #    f"obs: {obs}, obs_:{obs_}, continuous: {continuous_actions}, discrete: {discrete_actions}, reward: {reward}, dlp: {disc_lp},clp: {cont_lp}"
            # )
            # print(disc_lp, cont_lp)
            buffer.save_transition(
                terminated=terminated or truncated,
                registered_vals={
                    "obs": obs,
                    "obs_": obs_,
                    "continuous_actions": continuous_actions,
                    "discrete_actions": discrete_actions,
                    "global_rewards": reward,  # + abs(obs[1]) * 100,
                    "discrete_log_probs": disc_lp,
                    "continuous_log_probs": cont_lp,
                },
            )

            # if env.render_mode == "human":
            # env.display_board(env.board)
            # if terminated or truncated:
            # print(terminated or truncated)
            # print(abs(obs[1]) * 50)
            ep_reward += reward  # + abs(obs[1]) * 100
            obs = obs_

            # print(buffer.steps_recorded)
            if (
                buffer.steps_recorded > 255
                and buffer.episode_inds is not None
                and not online
            ) or (
                # and len(buffer.episode_inds) > 5
                online
                and buffer.steps_recorded > 511
            ):
                # print(online)
                if online:
                    batch = buffer.sample_transitions(
                        idx=np.arange(0, buffer.steps_recorded), as_torch=True
                    )
                    # agent.mini_batch_size = buffer.steps_recorded - 1
                    # print(buffer.steps_recorded - 1)
                    # print(batch.discrete_actions[0, :, 0])
                    # print(torch.from_numpy(np.array(term_array)).to("cuda:0"))
                    # print(batch.discrete_log_probs[0, :, 0])
                    # input()
                    # print(agent.actor_logstd)
                    # print(batch.global_rewards)
                else:
                    batch = buffer.sample_transitions(batch_size=256, as_torch=True)

                # for ep in episodes:
                aloss, closs = agent.reinforcement_learn(
                    batch, agent_num=0, debug=debug
                )
                # print(aloss, closs)
                m_aloss += aloss
                m_closs += closs
                n_updates += 1
                if online:
                    buffer.reset()
            step += 1
        # print(aloss, closs)
        aloss_return.append(m_aloss / max(n_updates, 1))
        closs_return.append(m_closs / max(n_updates, 1))
        rewards.append(ep_reward)
        rlen = min(len(rewards), 20)
        smooth_rewards.append(sum(rewards[-rlen:]) / rlen)
        # print(smooth_rewards[-1])
        er = 1

        if episode % er == 0 and episode > 1:

            print(
                f"n_ep: {episode} r: {sum(rewards[-10:])/10}, step: {step}, best: {max(rewards[-10:])} m_aloss: {m_aloss}, m_closs: {m_closs}"
            )
            m_aloss = 0
            m_closs = 0
            n_updates = 0
            # print(f"lr: {agent.optimizer.param_groups[0]["lr"]}, G: {agent.g_mean}")
            # plt.plot(rewards)
            # plt.show()
            obs, info = env.reset()

            obs = np.pad(obs, (0, joint_obs_dim - len(obs)), "constant")
            # print(agent.train_actions(obs, step=False))
            # print(agent.Q1(obs))
            # print(agent.eps)

            # env = TTTNvN(2, 2, "human", True, True)
            # env.__dict__["close"] = __close

        episode += 1
        # env = gym.make("CartPole-v1")

    # env.close()
    return smooth_rewards, aloss_return, closs_return


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    if gym_disc_env == "LunarLander-v2":
        discrete_env = gym.make(
            gym_disc_env, continuous=False, render_mode="human"
        )  # """MountainCar-v0")  # )   # , render_mode="human")

    else:
        discrete_env = gym.make(gym_disc_env, render_mode="human")

    # discrete_env = TTTNvN(2, 2, "", True, True)
    # discrete_env.__dict__["observation_space"] = np.zeros(18)
    # discrete_env.__dict__["action_space"] = aspace(9)
    # discrete_env.__dict__["close"] = __close

    if gym_cont_env == "LunarLander-v2":
        continuous_env = gym.make(gym_cont_env, continuous=True, render_mode="human")
    else:
        continuous_env = gym.make(gym_cont_env, render_mode="human")

    joint_obs_dim = (
        discrete_env.observation_space.shape[0]
        + continuous_env.observation_space.shape[0]
    )

    def make_models():
        print("Making Model")
        names = [
            "DQN",
            "TD3",
            "PG",
            "DDPG",
        ]
        print(
            continuous_env.action_space.low,
            continuous_env.action_space.high,
            continuous_env.action_space.shape,
            continuous_env.action_space.shape[0],
        )
        models = [
            DQN(
                obs_dim=joint_obs_dim,
                continuous_action_dims=continuous_env.action_space.shape[0],
                max_actions=continuous_env.action_space.high,
                min_actions=continuous_env.action_space.low,
                discrete_action_dims=[discrete_env.action_space.n],
                hidden_dims=[64, 64],
                device="cuda:0",
                lr=3e-4,
                activation="relu",
                dueling=True,
                n_c_action_bins=5,
                entropy=0.03,
                # munchausen=0.9,
            ),
            TD3(
                obs_dim=joint_obs_dim,
                discrete_action_dims=[discrete_env.action_space.n],
                continuous_action_dim=continuous_env.action_space.shape[0],
                min_actions=continuous_env.action_space.low,
                max_actions=continuous_env.action_space.high,
                hidden_dims=np.array([64, 64]),
                gamma=0.99,
                policy_frequency=2,
                name="TD3_cd_test",
                device="cuda",
                eval_mode=False,
                rand_steps=5000,
            ),
            PG(
                obs_dim=joint_obs_dim,
                discrete_action_dims=[discrete_env.action_space.n],
                continuous_action_dim=continuous_env.action_space.shape[0],
                hidden_dims=np.array([64, 64]),
                min_actions=continuous_env.action_space.low,
                max_actions=continuous_env.action_space.high,
                gamma=0.999,
                device="cuda",
                entropy_loss=0.01,
                mini_batch_size=64,
                n_epochs=4,
                lr=3e-4,
                advantage_type="gae",
                norm_advantages=False,
                anneal_lr=2000000,
                value_loss_coef=0.5,  # 5,
                ppo_clip=0.2,
                value_clip=0.5,
                orthogonal=True,
                activation="tanh",
                starting_actorlogstd=0,
                gae_lambda=0.98,
            ),
            DDPG(
                obs_dim=joint_obs_dim,
                discrete_action_dims=[discrete_env.action_space.n],
                continuous_action_dim=continuous_env.action_space.shape[0],
                min_actions=continuous_env.action_space.low,
                max_actions=continuous_env.action_space.high,
                hidden_dims=np.array([64, 64]),
                gamma=0.99,
                policy_frequency=4,
                name="TD3_cd_test",
                device="cuda",
                eval_mode=False,
                rand_steps=5000,
            ),
        ]
        return models, names

    print("Making Discrete Flexible Buffers")
    print(continuous_env.action_space.shape)
    mem_buffer = FlexibleBuffer(
        num_steps=50000,
        track_action_mask=False,
        discrete_action_cardinalities=[discrete_env.action_space.n],
        path="./test_mem_buffer/",
        name="joint_buffer",
        n_agents=1,
        global_registered_vars={
            "global_rewards": (None, np.float32),
        },
        individual_registered_vars={
            "discrete_log_probs": ([1], np.float32),
            "continuous_log_probs": (
                [continuous_env.action_space.shape[0]],
                np.float32,
            ),
            "discrete_actions": ([1], np.int64),
            "continuous_actions": ([continuous_env.action_space.shape[0]], np.float32),
            "obs": ([joint_obs_dim], np.float32),
            "obs_": ([joint_obs_dim], np.float32),
        },
    )

    models, names = make_models()

    results = {}
    for n in names:
        results[n] = []
    while True:
        for n in range(len(names)):
            print("Testing Model ", names[n])
            mem_buffer.reset()
            print("Testing Discrete Environment")
            rewards, aloss, closs = test_single_env(
                env=discrete_env,
                agent=models[n],
                buffer=mem_buffer,
                n_episodes=30000 if names[n] == "PG" else 30000,
                n_steps=30000 if names[n] == "PG" else 15000,
                discrete=True,
                joint_obs_dim=joint_obs_dim,
                online=names[n] in ["PPO", "PG"],
            )
            plt.plot(rewards)
            plt.title("Rewards for " + names[n] + f" on {gym_disc_env}")
            plt.show()
            am = np.abs(aloss).max()
            cm = np.abs(closs).max()
            plt.plot(aloss / am)
            plt.plot(closs / cm)
            plt.legend([f"actor {am}", f"critic {cm}"])
            plt.show()
