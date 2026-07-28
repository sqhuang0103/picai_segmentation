#!/bin/bash -l
#$ -N picai_prep
#$ -l h_rt=4:00:00
#$ -l mem=8G
#$ -pe smp 8
#$ -wd /home/rmapshu/picai_segmentation
#$ -o outputs/preprocess_$JOB_ID.log
#$ -e outputs/preprocess_$JOB_ID.err

export PATH="/home/rmapshu/Scratch/miniconda3/envs/orena/bin:$PATH"

python data/preprocess.py
