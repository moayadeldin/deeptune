## Command

python3 model_finetuning.py --model peft-resnet18 --num_classes 8 --num_epochs 1 --batch_size 16 --learning_rate 0.0001 --input_dir "/media/moayad/Moayad/StFX/advanced-project/DataSet_Splitted/combined.parquet"

python3 extract_embeddings.py --num_classes 8 --batch_size 16 --dataset_dir "test_split.parquet" --finetuned_model_pth "model_weights.pth"
