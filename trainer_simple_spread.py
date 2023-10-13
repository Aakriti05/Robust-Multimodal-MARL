import argparse
import time
import os
import signal
import math
import pickle
import glob
import gym
import torch
import numpy as np
from tensorboardX import SummaryWriter
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from collections import defaultdict

from agents import MADDPGAgent
from models import Actor, Critic, NatureActor


def parse_args():
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents, choose from ddpg, maddpg, m3ddpg, or rmaddpg (robust-maddpg)")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries, choose from ddpg, maddpg, m3ddpg, or rmaddpg (robust-maddpg)")

    parser.add_argument('--scenario', type=str, default='simple_spread', help='name of the scenario script')
    parser.add_argument('--max-train-steps', type=int, default=500_000, help='maximum episode length')
    parser.add_argument('--max-episodes', type=int, default=100_000, help='number of episodes')
    parser.add_argument('--max-episode-len', type=int, default=100, help='maximum episode length')
    # parser.add_argument('--num-adversaries', type=int, default=0, help='number of adversaries') # set it in the chosen scenario rather than here
    parser.add_argument('--render', default=False, action='store_true', help='display agent policies')
    parser.add_argument('--eval', dest='evaluate', default=False, action='store_true', help='run agent policy without noise and training')
    parser.add_argument('--benchmark', default=False, action='store_true')
    parser.add_argument("--benchmark-iters", type=int, default=100_000, help="number of iterations run for benchmarking")
    parser.add_argument('--train-every', type=int, default=100, help='simulation steps in between network updates')
    parser.add_argument('--noise-rew', type=float, default=0.0, help='{0.0, 1.0, 2.0, 3.0, ...}, uncertainty level')
    parser.add_argument('--noise-state', type=float, default=0.0, help='{0.2, 0.4, 2.0, 3.0, ...}, uncertainty level')
    parser.add_argument('--noise-action', type=float, default=0.0, help='{0.2, 0.4, 2.0, 3.0, ...}, uncertainty level')

    # Core training parameters
    parser.add_argument('--lr-actor', type=float, default=1e-4)
    parser.add_argument('--lr-critic', type=float, default=1e-3)
    parser.add_argument('--lr-nature', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor for training of critic')
    parser.add_argument('--batch-size', type=int, default=1024, help='size of minibatch that is sampled from replay buffer')
    parser.add_argument('--hidden', type=int, default=64, help='number of hidden units in actor and critic')
    parser.add_argument('--memory-size', type=int, default=50_000, help='size of replay memory')
    parser.add_argument('--tau', type=float, default=0.01, help='update rate for exponential update of target network params')

    # perturbing actions in m3ddpg
    parser.add_argument("--adv-eps", type=float, default=1e-3, help="adversarial training rate")
    parser.add_argument("--adv-eps-s", type=float, default=1e-5, help="small adversarial training rate")

    # Checkpointing
    parser.add_argument('--exp-name', type=str, default='test', help='name of experiment')
    parser.add_argument('--save-dir', type=str, default='results')
    parser.add_argument('--eval-every', type=int, default=1000)
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")

    parser.add_argument('--exp-run-num', type=str, default='', help='run number of experiment gets appended to log dir')
    parser.add_argument('--train-steps', type=int, default=1, help='how many train steps of the networks are done')
    parser.add_argument('--save-every', type=int, default=1000)

    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--rew-flag', default=False, action='store_true')
    parser.add_argument('--state-flag', default=False, action='store_true')
    parser.add_argument('--action-flag', default=False, action='store_true')
    parser.add_argument('--staterew-flag', default=False, action='store_true')
    parser.add_argument('--resume', help='dirname of saved state')
    parser.add_argument('--load-memories', default=False, action='store_true')
    parser.add_argument('--sparse-reward', default=True, action='store_false', dest='shaped')

    ## Agent modeling (approximate other agent's policy using MLP)
    parser.add_argument('--use-agent-models', default=False, action='store_true')
    parser.add_argument('--max-past', type=int, default=5000)
    parser.add_argument('--modeling-train-steps', type=int, default=20)
    parser.add_argument('--modeling-batch-size', type=int, default=64)
    parser.add_argument('--modeling-lr', type=float, default=1e-4)
    parser.add_argument('--sigma-noise', type=float)
    parser.add_argument('--temp-noise', type=float)

    return parser.parse_args()


def make_env(scenario_name, shaped=True, noise_state=0, noise_rew=0, noise_action=0, benchmark=False):
    scenario = scenarios.load(scenario_name + '.py').Scenario()
    world = scenario.make_world()
    if scenario.has_shaped_reward:
        reward_callback = scenario.shaped_reward if shaped else scenario.sparse_reward
    else:
        reward_callback = scenario.reward
    benchmark_data = scenario.benchmark_data if benchmark and hasattr(scenario, 'benchmark_data') else None
    env = MultiAgentEnv(world,
                        reset_callback=scenario.reset_world,
                        reward_callback=reward_callback,
                        observation_callback=scenario.observation,
                        done_callback=scenario.done,
                        info_callback=benchmark_data,
                        noise_state=noise_state, noise_rew=noise_rew, noise_action=noise_action)
    return env


def create_agents(env, params):
    agents = []
    n_agents = env.n
    try:
        n_adversaries = len(env.world.adversaries)
    except:
        n_adversaries = 0
    print("adversaries", n_adversaries)
    
    action_splits = []
    for action_space in env.action_space:
        if isinstance(action_space, gym.spaces.Discrete):
            action_splits.append([action_space.n])
        elif isinstance(action_space, gym.spaces.MultiDiscrete):
            action_splits.append(action_space.nvec)

    n_obs_each = [o.shape[0] for o in env.observation_space]
    n_actions_each = [sum(a) for a in action_splits]

    for i in range(n_agents):

        policy_name = params.adv_policy if i < n_adversaries else params.good_policy

        # Actor
        actor = Actor(n_obs_each[i], action_splits[i], params.hidden)

        # Critic
        n_critic_inputs = n_obs_each[i] + n_actions_each[i] if policy_name == 'ddpg' else sum(n_obs_each) + sum(n_actions_each)
        critic = Critic(n_critic_inputs, params.hidden)

        # Nature Actor
        if policy_name == 'rmaddpg':
            n_nature_actor_inputs = sum(n_obs_each) + sum(n_actions_each)
            nature_actor = NatureActor(n_nature_actor_inputs, params.hidden)
        else:  # {ddpg, maddpg, m3ddpg}
            nature_actor = None

        # MADDPG agent
        agent = MADDPGAgent(i, 'agent_%d' % i, env, actor, critic, params, policy_name, nature_actor)

        agents.append(agent)

    return agents


def save_agent_states(dirname, agents, checkpoint, save_models=False):
    states, models = zip(*[agent.get_state() for agent in agents])
    states_filename = os.path.join(dirname, 'states_{}.pth.tar'.format(checkpoint))
    torch.save(states, states_filename)
    if save_models:
        models_filename = os.path.join(dirname, 'models_{}.pth.tar'.format(checkpoint))
        torch.save(models, models_filename)


def load_agent_states(dirname, agents, load_models=False, checkpoint=None):
    if checkpoint is None:
        tars = glob.glob(os.path.join(dirname, 'states_*.pth.tar'))
        checkpoint = max([int(s.split('/')[-1].split('.pth')[0].split('_')[1]) for s in tars])
    states_filename = os.path.join(dirname, 'states_{}.pth.tar'.format(checkpoint))

    states = torch.load(states_filename, map_location='cpu')
    if load_models:
        models_filename = os.path.join(dirname, 'models_{}.pth.tar'.format(checkpoint))
        models = torch.load(models_filename, map_location='cpu')
    for i, agent in enumerate(agents):
        state = {}
        state['state_dicts'] = states[i]
        if load_models:
            state['models'] = models[i]
        agent.load_state(state)


def evaluate(env, agents, num_runs, args, display=False, max_benchmark_iters=None):

    if args.benchmark:
        assert max_benchmark_iters is not None, 'Benchmarking, shoud define max iters!'

    dones_sum = 0.0
    rewards_all = []
    true_rewards_all = []
    agent_info = []
    train_step = 0
    done_list = []
    for _ in range(num_runs):
        obs = env.reset()
        done = False
        terminal = False
        episode_step = 0
        cum_rewards = np.zeros(len(agents), dtype=np.float)
        cum_true_rewards = np.zeros_like(cum_rewards)
        agent_info.append([])

        while not (done or terminal):
            train_step += 1
            episode_step += 1
            # act with all agents in environment and receive observation and rewards
            actions = [agent.act(o, explore=False) for o, agent in zip(obs, agents)]
            new_obs, rewards, dones, infos = env.step(actions)
            cum_rewards += rewards
            cum_true_rewards += infos['true_rewards']
            #print("Dones: ", dones)
            done = all(dones)
            terminal = episode_step >= args.max_episode_len
            obs = new_obs

            if display:
                time.sleep(0.1)
                env.render()

            if args.benchmark:
                # for i, info in enumerate(infos):
                agent_info[-1].append(infos['n'])

        rewards_all.append(cum_rewards)
        true_rewards_all.append(cum_true_rewards)

        if done:
            done_list.append(1)
            dones_sum += 1
        else:
            done_list.append(0)

    if args.benchmark:
        fname = os.path.join(args.resume, 'benchmark.pkl')
        with open(fname, 'wb') as fp:
            pickle.dump(agent_info, fp)
        return fname
    print(np.mean(done_list), np.std(done_list))
    print("std: ", np.std(np.sum(rewards_all, axis=1)), np.shape(np.sum(rewards_all, axis=1)))

    return dones_sum / num_runs, np.mean(rewards_all, axis=0), np.mean(true_rewards_all, axis=0)


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def eval_msg(start_time, train_step, episode_count, agents, sr_mean, rewards):
    msg = 'time: {}, step {}, episode {}: success rate: {:.3f}, return: {:.2f}, '
    msg += ', '.join([a.name + ': {:.2f}' for a in agents])
    cum_reward = rewards.sum()
    print(msg.format(time_since(start_time), train_step, episode_count, sr_mean, cum_reward, *rewards))


def add_entropies(writer, obs, agents, train_step):
    entropies = {}
    for o, agent in zip(obs, agents):
        device = next(agent.actor.parameters()).device
        o = torch.tensor(o, dtype=torch.float, requires_grad=False, device=device)
        ps = agent.actor.prob_dists(o)
        agent_entropies = []
        for p in ps:
            agent_entropies.append(p.base_dist._categorical.entropy().detach().cpu().numpy())
        entropies[agent.name] = np.mean(agent_entropies)
    writer.add_scalars('policy_entropy', entropies, train_step)


def log_writer(args, agents):
    if args.state_flag: experiment_name = "state"
    elif args.rew_flag: experiment_name = "rew"
    elif args.action_flag: experiment_name = "action"
    elif args.staterew_flag: experiment_name = "staterew"
    else: experiment_name = "baseline"

    if hasattr(agents[0].env.agents[0], 'adversary'):
        run_name = 'good_%s_%s_adv_%s_noise_%.2f_%.2f_%.2f_' % (experiment_name, args.good_policy, args.adv_policy, args.noise_rew, args.noise_state, args.noise_action)
    else:
        run_name = 'good_%s_%s_noise_%.2f_%.2f_%.2f_' % (experiment_name, args.good_policy, args.noise_rew, args.noise_state, args.noise_action)
    run_name += str(np.datetime64('now')) if args.exp_run_num == '' else 'num_' + args.exp_run_num + '_' + str(np.datetime64('now'))

    writer = SummaryWriter(log_dir=os.path.join('tensorboard-logs', args.exp_name, run_name))
    dirname = os.path.join(args.save_dir, args.exp_name, run_name)
    print(dirname)
    os.makedirs(dirname, exist_ok=True)

    train_rewards_file = os.path.join(dirname, 'train_rewards.csv')
    eval_rewards_file = os.path.join(dirname, 'eval_rewards.csv')
    success_rate_file = os.path.join(dirname, 'success_rate.csv')
    kl_divergence_file = os.path.join(dirname, 'kl_divergence.csv')
    times_file = os.path.join(dirname, 'times.csv')

    for rewards_file in [train_rewards_file, eval_rewards_file]:
        if not os.path.isfile(rewards_file):
            with open(rewards_file, 'w') as f:
                line = ','.join(['episode', 'cum_reward'] + [a.name for a in agents]) + '\n'
                f.write(line)
    if not os.path.isfile(success_rate_file):
        with open(success_rate_file, 'w') as f:
            f.write('episode,success_rate\n')
    if args.use_agent_models and not os.path.isfile(kl_divergence_file):
        with open(kl_divergence_file, 'w') as f:
            headers = ['step']
            for agent in agents:
                for model_idx, model in agent.agent_models.items():
                    for i in range(len(model.action_split)):
                        headers.append('%d_%d_%d' % (agent.index, model_idx, i))
            f.write(','.join(headers) + '\n')

    files = {'train_rewards_file': train_rewards_file,
             'eval_rewards_file': eval_rewards_file,
             'success_rate_file': success_rate_file,
             'kl_divergence_file': kl_divergence_file,
             'times_file': times_file}

    return dirname, writer, files


def train(args):

    def signal_handling(signum, frame):
        nonlocal terminated
        terminated = True
    signal.signal(signal.SIGINT, signal_handling)

    env = make_env(args.scenario, args.shaped, args.noise_state, args.noise_rew, args.noise_action, args.benchmark)
    agents = create_agents(env, args)
    if args.use_agent_models:
        for agent in agents:
            agent.init_agent_models(agents)
    print("Noise level: ", env.noise_state, env.noise_rew, env.noise_action)
    # load state of agents if state file is given
    args.exp_name = '%s_%s' % (args.scenario, str(args.use_agent_models))
    if args.resume:
        load_agent_states(args.resume, agents, load_models=args.use_agent_models)
        args.exp_name, args.exp_run_num = os.path.normpath(args.resume).split('/')[-2:]

    if args.evaluate:
        # render True: record video (use quicktime manually)
        # evaluate True: obtain average rewards for each agent, for plotting 0-1 normalized agent score (??)
        num_runs = 1000 if not args.render else 10
        sr = []
        mean = []
        #for i in range(5):
        sr_mean, rewards, true_rewards = evaluate(env, agents, num_runs, args, display=args.render)
        msg = 'success rate: {:.3f}, cumsum return: {:.2f}, ' + ', '.join([a.name + ': {:.2f}' for a in agents])
        print(msg.format(sr_mean, rewards.sum(), *rewards))
        print('                             true rewards: ' + ', '.join([a.name + ': {:.2f}' for a in agents]).format(*true_rewards))
        print('Finshed evaluation ...')
        #sr.append(sr_mean)
        #mean.append(rewards.sum())
        #print("sr mean: ", np.mean(sr), np.std(sr), " mean mean: ", np.mean(mean), np.std(mean))
        return
    if args.benchmark:
        # benchmark True: collect benchmark data for producing tables and figures
        if os.path.exists(args.resume + '/benchmark.pkl'):
            return
        fname = evaluate(env, agents, 1000, args, display=False, max_benchmark_iters=args.benchmark_iters)
        print('Finished benchmarking at {}'.format(fname))
        return

    # logging
    dirname, writer, files = log_writer(args, agents)

    episode_count = 0
    train_step = 0
    terminated = False

    cum_reward = []
    agents_cum_reward = [[] for _ in range(env.n)]

    start_time = time.time()
    while (episode_count <= args.max_episodes) and not terminated:
        obs = env.reset()
        done = False
        terminal = False
        episode_step = 0

        cum_reward.append(0)
        for a in agents_cum_reward:
            a.append(0)

        while not (done or terminal):
            # global step count
            train_step += 1
            # episode step count
            episode_step += 1

            # act with all agents in environment and receive observation and rewards
            actions = [agent.act(o, explore=not args.evaluate) for o, agent in zip(obs, agents)]
            if args.debug:
                add_entropies(writer, obs, agents, train_step)
            new_obs, rewards, dones, _ = env.step(actions)   # for uncertain envs, the returned rewards are uncertain
            done = all(dones)
            terminal = episode_step >= args.max_episode_len
            #print("Rewards ", rewards)

            # rendering environment
            if args.render and (episode_count % 500 == 0):
                time.sleep(0.1)
                env.render()

            # store tuples (observation, action, reward, next observation, is done) for each agent
            for i, agent in enumerate(agents):
                agent.experience(episode_count, obs[i], actions[i], rewards[i], new_obs[i], dones[i])

            # store current observation for next step
            obs = new_obs

            # store rewards
            for i, reward in enumerate(rewards):
                cum_reward[-1] += reward
                agents_cum_reward[i][-1] += reward

            # train agents
            if train_step % args.train_every == 0:
                for _ in range(args.train_steps):
                    # losses = defaultdict(dict)
                    # all_kls = []
                    for agent in agents:
                        actor_loss, critic_loss, model_loss, model_kls = agent.update(agents)
                        # print(actor_loss, critic_loss, model_loss)

                    #     # log
                    #     losses['actor_loss'][agent.name] = actor_loss[0] if agent.robust else actor_loss
                    #     losses['critic_loss'][agent.name] = critic_loss
                    #     if args.use_agent_models:  # TODO: make sure all correct
                    #         losses['model_loss'][agent.name] = model_loss
                    #         kls_dict = {}
                    #         for idx, kls in model_kls:
                    #             for i, kl in enumerate(kls):
                    #                 kls_dict['%s_%i' % (agents[idx].name, i)] = kl
                    #                 all_kls.append(kl.item())
                    #         if args.debug:
                    #             writer.add_scalars('kl_%s' % agent.name, kls_dict, train_step)
                    # if args.use_agent_models:
                    #     with open(files['kl_divergence_file'], 'a') as f:
                    #         line = ','.join(map(str, [train_step] + all_kls)) + '\n'
                    #         f.write(line)
                    # if args.debug:
                    #     for name, loss_dict in losses.items():
                    #         writer.add_scalars(name, loss_dict, train_step)

            if train_step % args.eval_every == 0:
                sr_mean, rewards, _ = evaluate(env, agents, 50, args, display=False)
                if args.debug:
                    writer.add_scalar('success_rate', sr_mean, train_step)
                eval_msg(start_time, train_step, episode_count, agents, sr_mean, rewards)

        episode_count += 1

        if episode_count % args.save_every == 0:
            if episode_count % (5 * args.save_every) == 0:
                save_agent_states(dirname, agents, episode_count, save_models=args.use_agent_models)

            # Keep track of final episode reward
            final_ep_reward = np.mean(cum_reward[-args.save_every:])
            final_ep_ag_rewards = [np.mean(rew[-args.save_every:]) for rew in agents_cum_reward]

            # save cumulative reward and agent rewards
            with open(files['train_rewards_file'], 'a') as f:
                line = ','.join(map(str, [episode_count, final_ep_reward] + final_ep_ag_rewards)) + '\n'
                f.write(line)

            # save useful info from evaluation.
            sr_mean, rewards, _ = evaluate(env, agents, 1000, args, display=False)

            with open(files['success_rate_file'], 'a') as f:
                line = '{},{}\n'.format(episode_count, sr_mean)
                f.write(line)
            with open(files['eval_rewards_file'], 'a') as f:
                line = ','.join(map(str, [episode_count, rewards.sum()] + list(rewards))) + '\n'
                f.write(line)
            with open(files['times_file'], 'a') as f:
                f.write(str(time.time()) + '\n')

            # log rewards for tensorboard
            if args.debug:
                agent_rewards_dict = {a.name: r for a, r in zip(agents, final_ep_ag_rewards)}
                writer.add_scalar('reward', final_ep_reward, episode_count)
                writer.add_scalars('agent_rewards', agent_rewards_dict, episode_count)

    print('Finished training with %d episodes, saved to %s' % (episode_count, dirname))


def main():
    args = parse_args()

    assert not (args.evaluate and args.benchmark), 'eval and benchmark cannot both be True'
    conf = 'conf:\n\tenv: %s\n\tgood vs adv: %s vs %s\n\tapprox actors: %r\n\tnoise_rew: %.2f \n\tnoise_state: %.2f \n\tnoise_action: %.2f' % (
        args.scenario, args.good_policy, args.adv_policy, args.use_agent_models, args.noise_rew, args.noise_state, args.noise_action)
    print(conf)

    train(args)
    print(conf)


if __name__ == '__main__':
    main()
