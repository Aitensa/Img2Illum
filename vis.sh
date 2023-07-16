CUDA_VISIBLE_DEVICES=2,3 python vis.py --model_type PanoEPN_V2_model_dumps --epn  
python vis.py --epn v3 --model_type Ablation_PanoLoss_EquPoint_Cascade_1280 --epoch 18 --dataset test --maxn 30
CUDA_VISIBLE_DEVICES=2 python vis.py \
--epn p_v3 --dataset test --maxn 30 \
--model_type Ablation_PanoLoss_Cascade_Pano_10.0_5.0 \
--model /data8/aiys/PointAR/lightning_logs/Ablation_PanoLoss_Cascade_Pano_10.0_5.0/lightning_logs/version_40958/checkpoints/epoch=18-step=1515937.ckpt