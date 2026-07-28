#!/bin/bash -l
#$ -N picai_mv
#$ -l h_rt=24:00:00
#$ -l mem=16G
#$ -l gpu=1
#$ -pe smp 4
#$ -wd /home/rmapshu/picai_segmentation
#$ -o outputs/train_multiview_$JOB_ID.log
#$ -e outputs/train_multiview_$JOB_ID.err

export PATH="/home/rmapshu/Scratch/miniconda3/envs/orena/bin:$PATH"

python train_multiview.py
