#counter=1
#while [ $counter -le 3 ]
#do
#python trainer.py --scenario simple_spread --exp-run-num $counter --max-episodes 100000 --good-policy rmaddpg --adv-policy rmaddpg --noise-std 1 --save-dir ./results/bootstrap/1/ &
#((counter++))
#done
#wait

#counter=1
#while [ $counter -le 3 ]
#do
#python trainer_bootstrapCL.py --scenario simple_spread --exp-run-num $counter --max-episodes 150000 --noise-std 1 --good-policy rmaddpg --adv-policy rmaddpg --save-dir ./results/bootstrap/1/ &
#((counter++))
#done
#wait

counter=1
while [ $counter -le 3 ]
do
python trainer_bootstrapCL.py --scenario simple_spread --exp-run-num $counter --max-episodes 150000 --resume ./results/bootstrap/2/ --noise-std 3 --good-policy rmaddpg --adv-policy rmaddpg --save-dir ./results/bootstrap/3/ &
((counter++))
done
wait

