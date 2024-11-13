#!/bin/bash
#SBATCH -t 20:00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner_h100          # Partition to submit to
#SBATCH -c 16               # Number of cores (-c)
#SBATCH --mem=50G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH --array 7-12
#SBATCH -o /n/home12/binxuwang/Github/DiT/cluster_logs/DiT_heldout_incr_RAVEN_%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/home12/binxuwang/Github/DiT/cluster_logs/DiT_heldout_incr_RAVEN_%A_%a.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=binxu_wang@hms.harvard.edu

echo "$SLURM_ARRAY_TASK_ID"
param_list=\
'--expname stream0_16M_heldout0     --model DiT_S_1 --dataset RAVEN10_abstract  --cmb_per_class   4000  --num-classes 0  --class_dropout_prob 1.0   --heldout_ids 1  16  20  34  37 
--expname stream0_16M_heldout01     --model DiT_S_1 --dataset RAVEN10_abstract  --cmb_per_class   4000  --num-classes 0  --class_dropout_prob 1.0   --heldout_ids 1  16  20  34  37   8  12  24  36  39
--expname stream0_16M_heldout012    --model DiT_S_1 --dataset RAVEN10_abstract  --cmb_per_class   4000  --num-classes 0  --class_dropout_prob 1.0   --heldout_ids 1  16  20  34  37   8  12  24  36  39   5  17  21  33  38
--expname stream0_16M_heldout0123   --model DiT_S_1 --dataset RAVEN10_abstract  --cmb_per_class   4000  --num-classes 0  --class_dropout_prob 1.0   --heldout_ids 1  16  20  34  37   8  12  24  36  39   5  17  21  33  38   3  10  29  31  37
--expname stream0_16M_heldout01234  --model DiT_S_1 --dataset RAVEN10_abstract  --cmb_per_class   4000  --num-classes 0  --class_dropout_prob 1.0   --heldout_ids 1  16  20  34  37   8  12  24  36  39   5  17  21  33  38   3  10  29  31  37   0  14  27  35  38
--expname stream0_16M_heldout012345 --model DiT_S_1 --dataset RAVEN10_abstract  --cmb_per_class   4000  --num-classes 0  --class_dropout_prob 1.0   --heldout_ids 1  16  20  34  37   8  12  24  36  39   5  17  21  33  38   3  10  29  31  37   0  14  27  35  38   4  19  26  30  39
--expname stream0_16M_heldout_attr0     --model DiT_S_1 --dataset RAVEN10_abstract  --cmb_per_class   4000  --num-classes 0  --class_dropout_prob 1.0   --heldout_ids 0 1 2 3 4 5 6 7 8 9
--expname stream0_16M_heldout_attr3     --model DiT_S_1 --dataset RAVEN10_abstract  --cmb_per_class   4000  --num-classes 0  --class_dropout_prob 1.0   --heldout_ids 30 31 32 33 34 35 36 37 38 39
--expname stream0_16M_heldout_rel0     --model DiT_S_1 --dataset RAVEN10_abstract  --cmb_per_class   4000  --num-classes 0  --class_dropout_prob 1.0    --heldout_ids  0 10 20 30 
--expname stream0_16M_heldout_rel3     --model DiT_S_1 --dataset RAVEN10_abstract  --cmb_per_class   4000  --num-classes 0  --class_dropout_prob 1.0    --heldout_ids  3 13 23 33 
--expname stream0_16M_heldout_rel5     --model DiT_S_1 --dataset RAVEN10_abstract  --cmb_per_class   4000  --num-classes 0  --class_dropout_prob 1.0    --heldout_ids  5 15 25 35 
--expname stream0_16M_heldout_rel8     --model DiT_S_1 --dataset RAVEN10_abstract  --cmb_per_class   4000  --num-classes 0  --class_dropout_prob 1.0    --heldout_ids  8 18 28 38 
'
# --model DiT_S_1 --dataset RAVEN10_abstract  --cmb_per_class 400000  --num-classes 0  --class_dropout_prob 1.0    --epochs 2000 --global-batch-size 256  
# --model DiT_S_1 --dataset RAVEN10_abstract  --cmb_per_class 400000  --num-classes 0  --class_dropout_prob 1.0    --epochs 2000 --global-batch-size 256  
# --model DiT_S_1 --dataset RAVEN10_abstract  --cmb_per_class 400000  --num-classes 0  --class_dropout_prob 1.0    --epochs 2000 --global-batch-size 256  
# --model DiT_S_1 --dataset RAVEN10_abstract  --cmb_per_class 400000  --num-classes 0  --class_dropout_prob 1.0    --epochs 2000 --global-batch-size 256  
# --model DiT_S_1 --dataset RAVEN10_abstract  --cmb_per_class 400000  --num-classes 0  --class_dropout_prob 1.0    --epochs 2000 --global-batch-size 256  
# --model DiT_S_1 --dataset RAVEN10_abstract  --cmb_per_class 400000  --num-classes 0  --class_dropout_prob 1.0    --epochs 2000 --global-batch-size 256  
# --model DiT_S_1 --dataset RAVEN10_abstract  --cmb_per_class 400000  --num-classes 0  --class_dropout_prob 1.0    --epochs 2000 --global-batch-size 256  

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
mamba activate torch2 
which python
which python3
# run code
cd /n/home12/binxuwang/Github/DiT

python train_RAVEN_stream.py --data-path ~/Datasets --image-size 9 \
    --train_attr_fn attr_all_1000k.pt \
    --dataset_root $STORE_DIR/Datasets/RPM_dataset/RPM1000k \
    --global-seed 42 --num-workers 8 --log-every 100 \
    --total_steps 1000000 --global-batch-size 256 \
    --ckpt-every 20000 --save-samples-every 2500 \
    --num_eval_sample 2048 --eval_sampler ddim100 \
    $param_name 

