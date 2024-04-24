echo "Ejecutando run.py con fold=1"
python run.py --fold 0 --model "WGF_b4a1_lr3i5" --batch 4 --lr 0.001 --gender 0 --phones 0 --lr_scheduler "linear" --accum_batch 1
rm -r IberLEF_2024/
rm -r wandb/

echo "Ejecutando run.py con fold=1"
python run.py --fold 2 --model "WG_b4a1_lr3i5" --batch 4 --lr 0.001 --gender 0 --phones 0 --lr_scheduler "linear" --accum_batch 1
rm -r IberLEF_2024/
rm -r wandb/

echo "Ejecutando run.py con fold=1"
python run.py --fold 3 --model "WG_b4a1_lr3i5" --batch 4 --lr 0.001 --gender 0 --phones 0 --lr_scheduler "linear" --accum_batch 1
rm -r IberLEF_2024/
rm -r wandb/

echo "Ejecutando run.py con fold=1"
python run.py --fold 4 --model "WG_b4a1_lr3i5" --batch 4 --lr 0.001 --gender 0 --phones 0 --lr_scheduler "linear" --accum_batch 1
rm -r IberLEF_2024/
rm -r wandb/

echo "Ejecutando run.py con fold=1"
python run.py --fold 1 --model "WG_b4a6_lr4i5" --batch 4 --lr 0.001 --gender 0 --phones 0 --lr_scheduler "linear" --accum_batch 6
rm -r IberLEF_2024/
rm -r wandb/

echo "Ejecutando run.py con fold=1"
python run.py --fold 2 --model "WG_b4a6_lr4i5" --batch 4 --lr 0.001 --gender 0 --phones 0 --lr_scheduler "linear" --accum_batch 6
rm -r IberLEF_2024/
rm -r wandb/

echo "Ejecutando run.py con fold=1"
python run.py --fold 3 --model "WG_b4a6_lr4i5" --batch 4 --lr 0.001 --gender 0 --phones 0 --lr_scheduler "linear" --accum_batch 6
rm -r IberLEF_2024/
rm -r wandb/

echo "Ejecutando run.py con fold=1"
python run.py --fold 4 --model "WG_b4a6_lr4i5" --batch 4 --lr 0.001 --gender 0 --phones 0 --lr_scheduler "linear" --accum_batch 6
rm -r IberLEF_2024/
rm -r wandb/

echo "Ejecutando run.py con fold=1"
python run.py --fold 0 --model "WG_b4a1_lr4i5" --batch 4 --lr 0.001 --gender 0 --phones 0 --lr_scheduler "linear" --accum_batch 6
rm -r IberLEF_2024/
rm -r wandb/

