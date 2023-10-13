# Robust-Multimodal-MARL
Code for the paper [Robustness to Multi-Modal Environment Uncertainty in MARL using Curriculum Learning]. Submitted to ICLR for review.

# Instructions to run code

Training:

1. Baseline training:
`python trainer_simple_spread.py --scenario simple_spread --good-policy maddpg --adv-policy maddpg --noise-state 0.5 --noise-action 0.0 --noise-rew 0.0`
- scenario: {simple_push, simple_adversary, simple_spread}
- noise-state: uncertainty level of state
- noise-action: uncertainty level of action
- noise-rew: uncertainty level of reward

2. Single Curriculum Learning training (for CL on state uncertainty, appropriately modify for action and reward):
`python trainer_CL_simple_spread.py --scenario simple_spread --good-policy maddpg --adv-policy maddpg --noise-state 0.5 --state-flag`
- scenario: {simple_push, simple_adversary, simple_spread}
- noise-state: uncertainty level of state (initial value)
- state-flag: Flag that specifies which paramter will be subjected to CL

3. Multimodal Curriculum Learning training for action and state uncertainty (modify appropriately for other combinations):
`python trainer_multimodal_simple_spread.py --scenario simple_spread --good-policy maddpg --adv-policy maddpg --noise-state 0.5 --action-state 1.0 --state-action-flag`
- scenario: {simple_push, simple_adversary, simple_spread}
- noise-state: uncertainty level of state (initial value)
- noise-action: uncertainty level of action (initial value)
- state-action-flag: Flag that specifies which paramters will be subjected to CL, in this case we train for uncertainty in state and action.

4. Evaluation:
`python trainer_simple_spread.py --scenario simple_spread --good-policy maddpg --adv-policy maddpg --noise-state 0.5 --noise-action 0.0 --noise-rew 0.0 --eval --resume PATH_TO_CHECKPOINT_FOLDER`

Please note:
1. There is a separate multiagent/environment.py for each of the environments tested in this paper. The default environment.py corresponds to simple_spread. environment.py files for other environments can be found in the multiagent folder. Use the correct envionment.py while running experiments.
2. All the weights and training runs for all our experiments are present in the results folder. Evaluation can be run on any of them using the Evaluation command shown above.

# Weights for results shown in the paper

The weights for the training runs used to generate results in the paper are stored here: https://drive.google.com/file/d/1deZJmk9CiW7q6TyFiF80j3fY-qaKk1Sw/view?usp=sharing

# Citation

```
@article{agrawal2023RobustMultimodalMARL,
  title={Robustness to Multi-Modal Environment Uncertainty in MARL using Curriculum Learning},
  author={Agrawal, Aakriti and Aralikatti, Rohith and Sun, Yanchao and Huang, Furong},
  publisher={Arxiv},
  year={2023}
}

```
