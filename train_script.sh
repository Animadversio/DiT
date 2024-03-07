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
python train_RAVEN.py --dataset RAVEN10_abstract \
        --data-path ~/Datasets \
        --results-dir /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
        --image-size 9 --num-classes 35 --model DiT_B_1\
        --epochs 1000 --global-batch-size 256 --global-seed 42 \
        --num-workers 8 --log-every 100 --ckpt-every 20000 \
        --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 --class_dropout_prob 1.0



# 002-RAVEN10_abstract-DiT_S_1
python train_RAVEN.py --dataset RAVEN10_abstract \
        --data-path ~/Datasets \
        --results-dir /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results \
        --image-size 9 --num-classes 35 --model DiT_S_1\
        --epochs 1500 --global-batch-size 256 --global-seed 42 \
        --num-workers 8 --log-every 100 --ckpt-every 20000 \
        --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 --class_dropout_prob 1.0





