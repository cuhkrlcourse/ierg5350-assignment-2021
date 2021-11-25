# Example script to grid search some hyper parameters.
for lr in 1e-4 5e-5 1e-5 5e-6; do
  CUDA_VISIBLE_DEVICES="" nohup python train.py --env-id MetaDrive-Tut-10Env-v0 --num-epoch 5 --num-steps 2000 --num-envs 10 --algo PPO --log-dir search_lr_${lr} --lr ${lr} >search_lr_${lr}.log 2>&1 &
done
