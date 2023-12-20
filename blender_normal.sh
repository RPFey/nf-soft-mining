#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --array=0-7
#SBATCH --partition=batch
#SBATCH --qos=normal
##SBATCH -w=kd-a40-0.grasp.maas
#SBATCH --time=4:00:00
#SBATCH --exclude=kd-2080ti-1.grasp.maas,mp-2080ti-0.grasp.maas,dj-2080ti-0.grasp.maas,kd-2080ti-2.grasp.maas,kd-2080ti-3.grasp.maas,kd-2080ti-4.grasp.maas
##SBATCH --exclude=ee-3090-0.grasp.maas,ee-a6000-0.grasp.maas
##SBATCH --exclude=node-a6000-0,node-v100-0
#SBATCH --signal=SIGUSR1@180
#SBATCH --output=./output/cluster/%x-%j.out

hostname
echo $SLURM_ARRAY_TASK_ID '/' $SLURM_ARRAY_TASK_COUNT

DATASET_PATH=/mnt/kostas-graid/datasets/blender/nerf_synthetic/
EXP_BASE_NAME=$1
OBJS=("lego" "ship" "chair"  "drums"  "ficus"  "hotdog"  "materials"  "mic")
OBJ=${OBJS[$SLURM_ARRAY_TASK_ID]}
EXP_NAME=${OBJ}-${EXP_BASE_NAME}

source /mnt/kostas-graid/sw/envs/boshu/miniconda3/bin/activate nerfacc
cd ~/nf-soft-mining
pwd

# echo python active_train.py -s $DATASET_PATH/${OBJ} -m ./output/${EXP_NAME} --eval ${@:2}
# srun python active_train.py -s $DATASET_PATH/${OBJ} -m ./output/${EXP_NAME} --eval ${@:2}

srun python examples/train_ngp_nerf_prop.py --sampling_type ${EXP_BASE_NAME} \
        --data_root ${DATASET_PATH} \
        --scene ${OBJ} --logdir logs/${EXP_NAME}

# mkdir -p ${EXP_BASE_NAME}/${OBJ}
# echo python render_inflate.py -m ./output/${OBJ}-render-all --out_path ${EXP_BASE_NAME}/${OBJ} --white_background ${@:2} 
# srun python render_inflate.py -m ./output/${OBJ}-render-all --out_path ${EXP_BASE_NAME}/${OBJ} --white_background ${@:2} 