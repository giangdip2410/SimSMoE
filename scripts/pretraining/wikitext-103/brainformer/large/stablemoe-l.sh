mkdir -p checkpoints/wikitext-103/brainformers-l/stablemoe

args="
--data /home/gtruong/Project/ICML2/data/wikitext-103 \
--base_arch brainformer \
--architecture sgfgfgsgfgfgsgfgfgsgfgfgsgfgfgsgfgfgsgfgfgsgfgfgsgfgfgsgfgfg \
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
--checkpoint checkpoints/wikitext-103/brainformers-l/stablemoe/stablemoe.pt \
"

echo "Training ..."
python train.py $args

echo "Evaluation ..."
python train.py $args --full-eval-mode --batch-sz 8