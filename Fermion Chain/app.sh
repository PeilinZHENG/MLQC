gpuid=0

# Parameters of app
appgauge=1
Ytuple=-100.0
increase=1
increment=0.01
appAmount=151
appN=20
appNets="Deep_sig.21,Deep_sig.23,Deep_sig.24"
epochs=1
init_p=1000
rate=1
tol=5e-3
pert=normal
delta=0
app_seed=0
# Gradient
opt=Adam
sch=StepLR
iter_times=800                   # iter_times for Adam or max_iter for LBFGS
lr=5e-4
wd=0
gamma=0.5
ss=200


source activate
conda activate pytorch


python insulator_app.py --gpuid $gpuid --appN $appN --appNets $appNets --appgauge $appgauge --Ytuple $Ytuple --increase $increase --increment $increment --appAmount $appAmount --app_seed $app_seed --epochs $epochs --opt $opt --sch $sch --iter_times $iter_times --tol $tol --pert $pert --delta $delta --init_p $init_p --rate $rate --lr $lr --wd $wd --gamma $gamma --ss $ss
