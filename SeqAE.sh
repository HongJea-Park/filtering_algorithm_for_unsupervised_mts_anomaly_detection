#!/bin/bash

# ------------------------------ SeqAE ------------------------------
# -------------------------  anomaly type 1 -------------------------
# ----------------------------- ratio 0 ------------------------------
python main.py \
    --anomaly_type type1 \
    --model_name SeqAE \
    --filtering False \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0
    
python main.py \
    --anomaly_type type1 \
    --model_name SeqAE \
    --filtering True \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0 \
    --iqr_multiplier 1.5

python main.py \
    --anomaly_type type1 \
    --model_name SeqAE \
    --filtering True \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0 \
    --iqr_multiplier 3
    
# ------------------------------ SeqAE ------------------------------
# -------------------------  anomaly type 1 -------------------------
# ---------------------------- ratio 0.001 ----------------------------
python main.py \
    --anomaly_type type1 \
    --model_name SeqAE \
    --filtering False \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0.001
    
python main.py \
    --anomaly_type type1 \
    --model_name SeqAE \
    --filtering True \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0.001 \
    --iqr_multiplier 1.5

python main.py \
    --anomaly_type type1 \
    --model_name SeqAE \
    --filtering True \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0.001 \
    --iqr_multiplier 3
    
# ------------------------------ SeqAE ------------------------------
# -------------------------  anomaly type 1 -------------------------
# ---------------------------- ratio 0.005 ----------------------------
python main.py \
    --anomaly_type type1 \
    --model_name SeqAE \
    --filtering False \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0.005
    
python main.py \
    --anomaly_type type1 \
    --model_name SeqAE \
    --filtering True \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0.005 \
    --iqr_multiplier 1.5

python main.py \
    --anomaly_type type1 \
    --model_name SeqAE \
    --filtering True \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0.005 \
    --iqr_multiplier 3
    
# ------------------------------ SeqAE ------------------------------
# -------------------------  anomaly type 1 -------------------------
# ---------------------------- ratio 0.01 ----------------------------
python main.py \
    --anomaly_type type1 \
    --model_name SeqAE \
    --filtering False \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0.01
    
python main.py \
    --anomaly_type type1 \
    --model_name SeqAE \
    --filtering True \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0.01 \
    --iqr_multiplier 1.5

python main.py \
    --anomaly_type type1 \
    --model_name SeqAE \
    --filtering True \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0.01 \
    --iqr_multiplier 3
    
# ------------------------------ SeqAE ------------------------------
# -------------------------  anomaly type 1 -------------------------
# ---------------------------- ratio 0.05 ----------------------------
python main.py \
    --anomaly_type type1 \
    --model_name SeqAE \
    --filtering False \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0.05
    
python main.py \
    --anomaly_type type1 \
    --model_name SeqAE \
    --filtering True \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0.05 \
    --iqr_multiplier 1.5

python main.py \
    --anomaly_type type1 \
    --model_name SeqAE \
    --filtering True \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0.05 \
    --iqr_multiplier 3
    
# ------------------------------ SeqAE ------------------------------
# -------------------------  anomaly type 1 -------------------------
# ---------------------------- ratio 0.1 ----------------------------
python main.py \
    --anomaly_type type1 \
    --model_name SeqAE \
    --filtering False \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0.1
    
python main.py \
    --anomaly_type type1 \
    --model_name SeqAE \
    --filtering True \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0.1 \
    --iqr_multiplier 1.5

python main.py \
    --anomaly_type type1 \
    --model_name SeqAE \
    --filtering True \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0.1 \
    --iqr_multiplier 3

# ------------------------------ SeqAE ------------------------------
# -------------------------  anomaly type 2 -------------------------
# ----------------------------- ratio 0 ------------------------------
python main.py \
    --anomaly_type type2 \
    --model_name SeqAE \
    --filtering False \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0
    
python main.py \
    --anomaly_type type2 \
    --model_name SeqAE \
    --filtering True \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0 \
    --iqr_multiplier 1.5

python main.py \
    --anomaly_type type2 \
    --model_name SeqAE \
    --filtering True \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0 \
    --iqr_multiplier 3
    
# ------------------------------ SeqAE ------------------------------
# -------------------------  anomaly type 2 -------------------------
# ---------------------------- ratio 0.001 ----------------------------
python main.py \
    --anomaly_type type2 \
    --model_name SeqAE \
    --filtering False \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0.001
    
python main.py \
    --anomaly_type type2 \
    --model_name SeqAE \
    --filtering True \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0.001 \
    --iqr_multiplier 1.5

python main.py \
    --anomaly_type type2 \
    --model_name SeqAE \
    --filtering True \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0.001 \
    --iqr_multiplier 3
    
# ------------------------------ SeqAE ------------------------------
# -------------------------  anomaly type 2 -------------------------
# ---------------------------- ratio 0.005 ----------------------------
python main.py \
    --anomaly_type type2 \
    --model_name SeqAE \
    --filtering False \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0.005
    
python main.py \
    --anomaly_type type2 \
    --model_name SeqAE \
    --filtering True \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0.005 \
    --iqr_multiplier 1.5

python main.py \
    --anomaly_type type2 \
    --model_name SeqAE \
    --filtering True \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0.005 \
    --iqr_multiplier 3
    
# ------------------------------ SeqAE ------------------------------
# -------------------------  anomaly type 2 -------------------------
# ---------------------------- ratio 0.01 ----------------------------
python main.py \
    --anomaly_type type2 \
    --model_name SeqAE \
    --filtering False \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0.01
    
python main.py \
    --anomaly_type type2 \
    --model_name SeqAE \
    --filtering True \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0.01 \
    --iqr_multiplier 1.5

python main.py \
    --anomaly_type type2 \
    --model_name SeqAE \
    --filtering True \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0.01 \
    --iqr_multiplier 3
    
# ------------------------------ SeqAE ------------------------------
# -------------------------  anomaly type 2 -------------------------
# ---------------------------- ratio 0.05 ----------------------------
python main.py \
    --anomaly_type type2 \
    --model_name SeqAE \
    --filtering False \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0.05
    
python main.py \
    --anomaly_type type2 \
    --model_name SeqAE \
    --filtering True \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0.05 \
    --iqr_multiplier 1.5

python main.py \
    --anomaly_type type2 \
    --model_name SeqAE \
    --filtering True \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0.05 \
    --iqr_multiplier 3
    
# ------------------------------ SeqAE ------------------------------
# -------------------------  anomaly type 2 -------------------------
# ---------------------------- ratio 0.1 ----------------------------
python main.py \
    --anomaly_type type2 \
    --model_name SeqAE \
    --filtering False \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0.1
    
python main.py \
    --anomaly_type type2 \
    --model_name SeqAE \
    --filtering True \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0.1 \
    --iqr_multiplier 1.5

python main.py \
    --anomaly_type type2 \
    --model_name SeqAE \
    --filtering True \
    --lr 5e-4 \
    --epoch 100 \
    --anomaly_ratio 0.1 \
    --iqr_multiplier 3
