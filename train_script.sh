python train_RAVEN.py --dataset MNIST \
        --data-path ~/Datasets \
        --results-dir results \
        --image-size 32 --num-classes 10 \
        --epochs 100 --global-batch-size 512 --global-seed 42 \
        --num-workers 8 --log-every 100 --ckpt-every 50000 \
        --save-samples-every 1000


python train_RAVEN.py --dataset RAVEN10_abstract \
        --data-path ~/Datasets \
        --results-dir /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
        --image-size 9 --num-classes 35 --model DiT_S_1\
        --epochs 100 --global-batch-size 512 --global-seed 42 \
        --num-workers 8 --log-every 100 --ckpt-every 20000 \
        --save-samples-every 1000 --num_eval_sample 1024



# 000-RAVEN10_abstract-DiT_S_1

python train_RAVEN.py --dataset RAVEN10_abstract \
        --data-path ~/Datasets \
        --results-dir /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
        --image-size 9 --num-classes 35 --model DiT_S_1\
        --epochs 1500 --global-batch-size 512 --global-seed 42 \
        --num-workers 8 --log-every 100 --ckpt-every 20000 \
        --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 --class_dropout_prob 1.0


# 001-RAVEN10_abstract-DiT_B_1
# Note this version fogot to do shuffle in loader, bad.
python train_RAVEN.py --dataset RAVEN10_abstract \
        --data-path ~/Datasets \
        --results-dir /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
        --image-size 9 --num-classes 35 --model DiT_B_1\
        --epochs 1000 --global-batch-size 256 --global-seed 42 \
        --num-workers 8 --log-every 100 --ckpt-every 20000 \
        --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 --class_dropout_prob 1.0



# 002-RAVEN10_abstract-DiT_S_1
# Note this version fogot to do shuffle in loader, bad.
python train_RAVEN.py --dataset RAVEN10_abstract \
        --data-path ~/Datasets \
        --results-dir /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
        --image-size 9 --num-classes 35 --model DiT_S_1\
        --epochs 1500 --global-batch-size 256 --global-seed 42 \
        --num-workers 8 --log-every 100 --ckpt-every 20000 \
        --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 --class_dropout_prob 1.0


# 002-RAVEN10_abstract-DiT_S_1
# Note this version fogot to do shuffle in loader, bad.
python train_RAVEN.py --dataset RAVEN10_abstract_onehot \
        --data-path ~/Datasets \
        --results-dir /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
        --image-size 9 --num-classes 35 --model DiT_S_1\
        --epochs 1500 --global-batch-size 256 --global-seed 42 \
        --num-workers 8 --log-every 100 --ckpt-every 20000 \
        --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 --class_dropout_prob 1.0

# 004-RAVEN10_abstract-DiT_S_1 xxxx
# 005-RAVEN10_abstract_onehot-DiT_S_1 xxx 


# The sampling during training is using the random label instead of uncond. 
# so the result may not be perfect. 
# 008-RAVEN10_abstract-DiT_S_3
python train_RAVEN.py --dataset RAVEN10_abstract \
        --data-path ~/Datasets \
        --results-dir /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
        --image-size 9 --num-classes 35 --model DiT_S_3\
        --epochs 1500 --global-batch-size 256 --global-seed 42 \
        --num-workers 8 --log-every 100 --ckpt-every 20000 \
        --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 --class_dropout_prob 1.0

# 009-RAVEN10_abstract-DiT_S_1
# Conditional model 
python train_RAVEN.py --dataset RAVEN10_abstract \
        --data-path ~/Datasets \
        --results-dir /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
        --image-size 9 --num-classes 35 --model DiT_S_1\
        --epochs 1500 --global-batch-size 256 --global-seed 42 \
        --num-workers 8 --log-every 100 --ckpt-every 20000 \
        --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 --class_dropout_prob 0.1

# 010-RAVEN10_abstract-DiT_L_1 Failed. ... 40GB
python train_RAVEN.py --dataset RAVEN10_abstract \
        --data-path ~/Datasets \
        --results-dir /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
        --image-size 9 --num-classes 35 --model DiT_L_1\
        --epochs 1500 --global-batch-size 256 --global-seed 42 \
        --num-workers 8 --log-every 100 --ckpt-every 20000 \
        --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 --class_dropout_prob 1.0


# 011-RAVEN10_abstract-DiT_B_1 20GB
python train_RAVEN.py --dataset RAVEN10_abstract \
        --data-path ~/Datasets \
        --results-dir /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
        --image-size 9 --num-classes 35 --model DiT_B_1\
        --epochs 1500 --global-batch-size 256 --global-seed 42 \
        --num-workers 8 --log-every 100 --ckpt-every 20000 \
        --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 --class_dropout_prob 1.0

##########
# upate the sampling script to make it allow uncond sampling
# 012-RAVEN10_abstract-DiT_B_3
python train_RAVEN.py --dataset RAVEN10_abstract \
        --data-path ~/Datasets \
        --results-dir /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
        --image-size 9 --num-classes 0 --model DiT_B_3\
        --epochs 1500 --global-batch-size 256 --global-seed 42 \
        --num-workers 8 --log-every 100 --ckpt-every 20000 \
        --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 --class_dropout_prob 1.0

# 013-RAVEN10_abstract-DiT_S_3
python train_RAVEN.py --dataset RAVEN10_abstract \
        --data-path ~/Datasets \
        --results-dir /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
        --image-size 9 --num-classes 0 --model DiT_S_3\
        --epochs 2000 --global-batch-size 256 --global-seed 42 \
        --num-workers 8 --log-every 100 --ckpt-every 20000 \
        --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 --class_dropout_prob 1.0





python edm_rule_check_CLI.py --ep_step 1000 --ep_start 1000 --fmt %07d.pt \
        --exproot /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
        --expname 002-RAVEN10_abstract-DiT_S_1 

python edm_rule_check_CLI.py --ep_step 1000 --ep_start 1000 --fmt %07d.pt \
        --exproot /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
        --expname 001-RAVEN10_abstract-DiT_B_1 

python edm_rule_check_CLI.py --ep_step 1000 --ep_start 1000 --fmt %07d.pt \
        --exproot /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
        --expname 003-RAVEN10_abstract_onehot-DiT_S_1 --encoding onehot 

