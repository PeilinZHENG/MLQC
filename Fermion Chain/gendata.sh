# Parameters of data generation
dataN=20
pattern=test # test
amount=20
processors=8
rate=4
dataset_id=01 # CDW_01, BDW_01
decay=0.3
gauge=1


source activate
conda activate pytorch


python insulator_data.py --pattern $pattern --dataset_id $dataset_id --dataN $dataN --amount $amount --processors $processors --rate $rate --decay $decay --gauge $gauge

