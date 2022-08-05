# predict on .05 dataset
python ../experiments/esnli.py --data_frac=.05 --exp_id=toy
# predict with nle only on full dataset
python ../experiments/esnli.py --data_frac=1.0 --nle_pred --exp_id=nle-full
# predict on full dataset
python ../experiments/esnli.py --data_frac=1.0 --exp_id=full
