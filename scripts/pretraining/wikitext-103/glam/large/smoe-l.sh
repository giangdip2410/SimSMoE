mkdir -p checkpoints/wikitext-103/glam-l/smoe

args="
--data /home/gtruong/Project/ICML2/data/wikitext-103 \
--base_arch glam \
--architecture sgsfsgsfsgsfsgsfsgsfsgsfsgsfsgsfsgsfsgsf \
--gate_name smoe \
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
--checkpoint checkpoints/wikitext-103/glam-l/smoe/smoe.pt \
"

echo "Training ..."
python train.py $args

echo "Evaluation ..."
python train.py $args --full-eval-mode --batch-sz 8