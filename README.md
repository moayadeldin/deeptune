## Command

python3 model_finetuning.py --model peft-resnet18 --num_classes 8 --num_epochs 1 --batch_size 16 --learning_rate 0.0001 --input_dir "/media/moayad/Moayad/StFX/advanced-project/DataSet_Splitted/combined.parquet"

python3 extract_embeddings.py --num_classes 8 --batch_size 16 --dataset_dir "test_split.parquet" --finetuned_model_pth "model_weights.pth"

(env-df-analyze) moayad@moayad-IdeaPad-L340-15IRH-Gaming:~/.pyenv/versions/env-df-analyze/bin$ 

python -m  trainers.trainer_resnet --model resnet18 --num_classes 8 --num_epochs 1 --batch_size 16 --learning_rate 0.0001 --train_size 0.7 --val_size 0.1 --test_size 0.2 --input_dir "H:\Moayad\combined.parquet"

(deeptunenv) H:\Moayad\deeptune-scratch>python -m  trainers.trainer_siglip --num_epochs 1 --batch_size 1 --learning_rate 0.0001 --train_size 0.7 --val_size 0.1 --test_size 0.2 --input_dir "H:\Moayad\combined.parquet" 

(deeptune) H:\Moayad\df-analyze>python df-analyze.py --df "H:\Moayad\deeptune-scratch\results\PBC splits results\split 80-10-10\test_set_peft_resnet18_embeddings.parquet" --outdir = ./peft_resnet18_test_results --mode=classify --target label --classifiers lgbm rf sgd knn lr mlp dummy --embed-select none linear lgbm