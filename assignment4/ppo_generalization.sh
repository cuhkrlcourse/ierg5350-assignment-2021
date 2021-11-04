for env_num in 1 5 10 20 50 100; do
    python examples/sac.py --env $1 --seed $seed --num_layer 2
done
