#!/bin/bash
#SBATCH -t 40:00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner          # Partition to submit to
#SBATCH -c 16               # Number of cores (-c)
#SBATCH --mem=40G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH --array 1-10       
#SBATCH -o DiT_split_RAVEN_%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e DiT_split_RAVEN_%A_%a.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=binxu_wang@hms.harvard.edu

echo "$SLURM_ARRAY_TASK_ID"
param_list=\
'--train_attr_fn train_inputs_new_split0.pt  --model DiT_S_1 --dataset RAVEN10_abstract         --num-classes 0  --class_dropout_prob 1.0          --epochs 2000 --global-batch-size 256 --global-seed 42 --num-workers 8 --log-every 100 --ckpt-every 20000 --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 
--train_attr_fn train_inputs_new_split1.pt  --model DiT_S_1 --dataset RAVEN10_abstract         --num-classes 0  --class_dropout_prob 1.0          --epochs 2000 --global-batch-size 256 --global-seed 42 --num-workers 8 --log-every 100 --ckpt-every 20000 --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 
--train_attr_fn train_inputs_new_split2.pt  --model DiT_S_1 --dataset RAVEN10_abstract         --num-classes 0  --class_dropout_prob 1.0          --epochs 2000 --global-batch-size 256 --global-seed 42 --num-workers 8 --log-every 100 --ckpt-every 20000 --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 
--train_attr_fn train_inputs_new_split3.pt  --model DiT_S_1 --dataset RAVEN10_abstract         --num-classes 0  --class_dropout_prob 1.0          --epochs 2000 --global-batch-size 256 --global-seed 42 --num-workers 8 --log-every 100 --ckpt-every 20000 --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 
--train_attr_fn train_inputs_new_split4.pt  --model DiT_S_1 --dataset RAVEN10_abstract         --num-classes 0  --class_dropout_prob 1.0          --epochs 2000 --global-batch-size 256 --global-seed 42 --num-workers 8 --log-every 100 --ckpt-every 20000 --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 
--train_attr_fn train_inputs_new_split5.pt  --model DiT_S_1 --dataset RAVEN10_abstract         --num-classes 0  --class_dropout_prob 1.0          --epochs 2000 --global-batch-size 256 --global-seed 42 --num-workers 8 --log-every 100 --ckpt-every 20000 --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 
--train_attr_fn train_inputs_new_split6.pt  --model DiT_S_1 --dataset RAVEN10_abstract         --num-classes 0  --class_dropout_prob 1.0          --epochs 2000 --global-batch-size 256 --global-seed 42 --num-workers 8 --log-every 100 --ckpt-every 20000 --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 
--train_attr_fn train_inputs_new_split7.pt  --model DiT_S_1 --dataset RAVEN10_abstract         --num-classes 0  --class_dropout_prob 1.0          --epochs 2000 --global-batch-size 256 --global-seed 42 --num-workers 8 --log-every 100 --ckpt-every 20000 --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 
--train_attr_fn train_inputs_new_split8.pt  --model DiT_S_1 --dataset RAVEN10_abstract         --num-classes 0  --class_dropout_prob 1.0          --epochs 2000 --global-batch-size 256 --global-seed 42 --num-workers 8 --log-every 100 --ckpt-every 20000 --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 
--train_attr_fn train_inputs_new_split9.pt  --model DiT_S_1 --dataset RAVEN10_abstract         --num-classes 0  --class_dropout_prob 1.0          --epochs 2000 --global-batch-size 256 --global-seed 42 --num-workers 8 --log-every 100 --ckpt-every 20000 --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 
'
# --model DiT_S_1 --dataset RAVEN10_abstract         --num-classes 0  --class_dropout_prob 1.0          --epochs 2000 --global-batch-size 256 --global-seed 42 --num-workers 8 --log-every 100 --ckpt-every 20000 --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 
# --model DiT_S_1 --dataset RAVEN10_abstract         --num-classes 35 --class_dropout_prob 0.1  --cond  --epochs 2000 --global-batch-size 256 --global-seed 42 --num-workers 8 --log-every 100 --ckpt-every 20000 --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 
# --model DiT_B_1 --dataset RAVEN10_abstract         --num-classes 0  --class_dropout_prob 1.0          --epochs 2000 --global-batch-size 256 --global-seed 42 --num-workers 8 --log-every 100 --ckpt-every 20000 --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 
# --model DiT_B_1 --dataset RAVEN10_abstract         --num-classes 35 --class_dropout_prob 0.1  --cond  --epochs 2000 --global-batch-size 256 --global-seed 42 --num-workers 8 --log-every 100 --ckpt-every 20000 --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 
# --model DiT_S_1 --dataset RAVEN10_abstract_onehot  --num-classes 0  --class_dropout_prob 1.0          --epochs 2000 --global-batch-size 256 --global-seed 42 --num-workers 8 --log-every 100 --ckpt-every 20000 --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 
# --model DiT_S_1 --dataset RAVEN10_abstract_onehot  --num-classes 35 --class_dropout_prob 0.1  --cond   --epochs 2000 --global-batch-size 256 --global-seed 42 --num-workers 8 --log-every 100 --ckpt-every 20000 --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 
# --model DiT_B_1 --dataset RAVEN10_abstract_onehot  --num-classes 0  --class_dropout_prob 1.0          --epochs 2000 --global-batch-size 256 --global-seed 42 --num-workers 8 --log-every 100 --ckpt-every 20000 --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 
# --model DiT_B_1 --dataset RAVEN10_abstract_onehot  --num-classes 35 --class_dropout_prob 0.1  --cond   --epochs 2000 --global-batch-size 256 --global-seed 42 --num-workers 8 --log-every 100 --ckpt-every 20000 --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 
# --model DiT_S_3 --dataset RAVEN10_abstract         --num-classes 0  --class_dropout_prob 1.0          --epochs 2000 --global-batch-size 256 --global-seed 42 --num-workers 8 --log-every 100 --ckpt-every 20000 --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 
# --model DiT_S_3 --dataset RAVEN10_abstract         --num-classes 35 --class_dropout_prob 0.1  --cond  --epochs 2000 --global-batch-size 256 --global-seed 42 --num-workers 8 --log-every 100 --ckpt-every 20000 --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 
# --model DiT_B_3 --dataset RAVEN10_abstract         --num-classes 0  --class_dropout_prob 1.0          --epochs 2000 --global-batch-size 256 --global-seed 42 --num-workers 8 --log-every 100 --ckpt-every 20000 --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 
# --model DiT_B_3 --dataset RAVEN10_abstract         --num-classes 35 --class_dropout_prob 0.1  --cond  --epochs 2000 --global-batch-size 256 --global-seed 42 --num-workers 8 --log-every 100 --ckpt-every 20000 --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 
# --model DiT_S_3 --dataset RAVEN10_abstract_onehot  --num-classes 0  --class_dropout_prob 1.0          --epochs 2000 --global-batch-size 256 --global-seed 42 --num-workers 8 --log-every 100 --ckpt-every 20000 --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 
# --model DiT_S_3 --dataset RAVEN10_abstract_onehot  --num-classes 35 --class_dropout_prob 0.1  --cond  --epochs 2000 --global-batch-size 256 --global-seed 42 --num-workers 8 --log-every 100 --ckpt-every 20000 --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 
# --model DiT_B_3 --dataset RAVEN10_abstract_onehot  --num-classes 0  --class_dropout_prob 1.0          --epochs 2000 --global-batch-size 256 --global-seed 42 --num-workers 8 --log-every 100 --ckpt-every 20000 --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 
# --model DiT_B_3 --dataset RAVEN10_abstract_onehot  --num-classes 35 --class_dropout_prob 0.1  --cond  --epochs 2000 --global-batch-size 256 --global-seed 42 --num-workers 8 --log-every 100 --ckpt-every 20000 --save-samples-every 1000 --num_eval_sample 1024 --eval_sampler ddim100 


export param_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$param_name"

# load modules
module load python
conda deactivate
conda activate torch
which python
which python3
# run code
cd /n/home12/binxuwang/Github/DiT

python train_RAVEN.py --data-path ~/Datasets --image-size 9 \
    --cmb_per_class 4000 \
    $param_name        


