#!/bin/bash -l
#$ -N picai_eval
#$ -l h_rt=1:00:00
#$ -l mem=16G
#$ -l gpu=1
#$ -pe smp 4
#$ -wd /home/rmapshu/picai_segmentation
#$ -o outputs/eval_$JOB_ID.log
#$ -e outputs/eval_$JOB_ID.err

export PATH="/home/rmapshu/Scratch/miniconda3/envs/orena/bin:$PATH"

python evaluate.py
