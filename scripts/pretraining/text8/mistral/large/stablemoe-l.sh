mkdir -p checkpoints/text8/mistrals-l/stablemoe

args="
--data /home/gtruong/Project/ICML2/data/text8 \
--base_arch mistral \
--architecture sgsgsgsgsgsgsgsgsgsgsgsgsgsg \
--gate_name stablemoe \
--nlayers 14 \
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
--checkpoint checkpoints/text8/mistrals-l/stablemoe/stablemoe.pt \
"

echo "Training ..."
python train.py $args

echo "Evaluation ..."
python train.py $args --full-eval-mode --batch-sz 8