mkdir -p checkpoints/enwik8/glam-l/stablemoe_sim3

args="
--data /home/gtruong/Project/ICML2/data/enwik8 \
--base_arch glam \
--architecture sgsfsgsfsgsfsgsfsgsfsgsfsgsfsgsfsgsfsgsf \
--gate_name stablemoe \
--nlayers 10 \
--hid-sz 256 \
--inner-hid-sz 512 \
--nheads 8 \
--block-sz 512 \
--attn-span 2048 \
--dropout 0.1 \
--load_balance 0.01 \
--optim adam \
--lr 0.00045 \
--lr-warmup 500 \
--niter 50 \
--batch-sz 48 \
--batch-split 2 \
--nbatches 1000 \
--contrastive \
--cont_freq 1.0 \
--sim_threshold 0.9 \
--contrative_rate 1.0 \
--contrative_loss cka \
--cka_mode kernel \
--sigma 0.9 \
--checkpoint checkpoints/enwik8/glam-l/stablemoe_sim3/stablemoe_sim3.pt \
"

echo "Training ..."
python train.py $args

echo "Evaluation ..."
python train.py $args --full-eval-mode --batch-sz 8