# Example script to launch formal experiment.
num=1
echo Start Training ${num} Environments
CUDA_VISIBLE_DEVICES="" nohup python train.py --env-id MetaDrive-Tut-${num}Env-v0 --num-steps 2000 --num-envs 10 --asynchronous --algo PPO --log-dir metadrive_${num}_env > metadrive_${num}_env.log 2>&1 &
