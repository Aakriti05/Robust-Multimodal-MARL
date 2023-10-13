#!/bin/bash
#
source /cmlscratch/agrawal5/bashrc_copy.sh
echo `nvidia-smi`
echo `which python`
conda activate pyt 
echo `which python`
#python trainer_naiveCL_staterew.py --scenario simple_spread --exp-run-num 0 --good-policy rmaddpg --adv-policy rmaddpg --rew-state-flag --noise-rew 0.0 --noise-state 0.05
#python trainer_naiveCL_staterew.py --scenario simple_spread --exp-run-num 0 --good-policy rmaddpg --adv-policy rmaddpg --rew-action-flag --noise-rew 0.0 --noise-action 0.1
#python trainer_naiveCL_staterew.py --scenario simple_spread --exp-run-num 0 --good-policy rmaddpg --adv-policy rmaddpg --state-action-flag --noise-state 0.0 --noise-action 0.1
python trainer_naiveCL_staterew.py --scenario simple_spread --good-policy rmaddpg --adv-policy rmaddpg --rew-state-flag --noise-rew 1 --noise-state 0.65 --checkpoint 15000 --resume ./results/simple_spread_False/good_rew_stateCL_100_rmaddpg_noise_1.00_0.10_0.00_num_good_rew_stateCL_100_rmaddpg_noise_0.00_0.05_0.00_num_0_2023-09-25T15:26:27_2023-09-28T06:42:41/ 
#python trainer_naiveCL_staterew.py --scenario simple_spread --good-policy rmaddpg --adv-policy rmaddpg --rew-action-flag --noise-rew 4  --noise-action 2.2 --checkpoint 60000 --resume ./results/simple_spread_False/good_rew_actionCL_100_rmaddpg_noise_0.00_0.00_0.10_num_0_2023-09-25T15:27:06/
#python trainer_CL.py --scenario simple_spread --action-flag --exp-run-num 0 --good-policy rmaddpg --adv-policy rmaddpg --resume ./results/simple_spread_False/good_actionCL_rmaddpg_noise_0.00_0.00_0.10_num_0_2023-09-20T19:06:23/ --noise-action 2.2 
#python trainer_CL.py --scenario simple_spread --state-flag --exp-run-num 0 --good-policy rmaddpg --adv-policy rmaddpg --resume ./results/simple_spread_False/good_stateCL_rmaddpg_noise_0.00_0.00_num_0_2023-09-11T20:53:37/ --checkpoint 218000 --noise-state 0.55
#python trainer.py --scenario simple_spread --action-flag --exp-run-num 0 --good-policy rmaddpg --adv-policy rmaddpg --noise-action 2.4 
#python trainer.py --scenario simple_spread --staterew-flag --exp-run-num 1 --good-policy rmaddpg --adv-policy rmaddpg --noise-state 0.3 --noise-rew 21.0 &
#python trainer.py --scenario simple_adversary --exp-run-num 0 --good-policy rmaddpg --adv-policy rmaddpg &
#python trainer.py --scenario simple_spread --rew-flag --exp-run-num 0 --good-policy rmaddpg --adv-policy rmaddpg --noise-rew 11
#python trainer.py --scenario simple_spread --exp-run-num 0 --good-policy rmaddpg --adv-policy rmaddpg --action-flag --noise-action 1.6
#python trainer_CL.py --scenario simple_spread --exp-run-num 0 --action-flag --noise-action 0.1 --exp-run-num 0 --good-policy rmaddpg --adv-policy rmaddpg
#wait
