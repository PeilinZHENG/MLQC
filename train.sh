## Parameters of GPU
gpuid=0

# Parameters of training
traingauge=1
N=20
num_seed=0
scale=500
traindataName="ME_01"
valdataName="ME_01"
Net="Deep_sig.test"
init=None


source activate
conda activate pytorch


python Train.py --N $N --gpuid $gpuid --Net $Net --init $init --traingauge $traingauge --traindataName $traindataName --valdataName $valdataName --num_seed $num_seed --scale $scale
