from trainers.nlp.train_gpt2 import train as train_gpt2
from trainers.nlp.train_multilingualbert import train as train_multilingualbert
from handlers.split_dataset import split_dataset
import argparse
from pathlib import Path
from cli import DeepTuneVisionOptions
from utils import RunType,set_seed


defaults={
    'num_epochs':10,
    'learning_rate':1e-4,
    'added_layers':2,
    'embed_size':1000,
    'train_size':0.8,
    'val_size':0.1,
    'test_size':0.1,
    'fixed_seed':True,
    'freeze_backbone': False,
    'disable_numerical_encoding':False
}

def main():
    args = DeepTuneVisionOptions(RunType.ONECALL)

    train_data_path, val_data_path, test_data_path = split_dataset(
        train_size=defaults['train_size'],
        val_size=defaults['val_size'],
        test_size=defaults['test_size'],
        df_path=args.df,
        out_dir=args.out,
        fixed_seed=defaults['fixed_seed'],
        disable_numerical_encoding=defaults['disable_numerical_encoding']
    )

    if args.modality == 'text':
    
        if args.model_version == 'bert':

            ckpt_directory = train_multilingualbert(
                out=args.out,
                batch_size=args.batch_size,
                train_df = train_data_path,
                val_df = val_data_path,
                num_epochs=defaults['num_epochs'],
                learning_rate=defaults['learning_rate'],
                added_layers=defaults['added_layers'],
                embed_size=defaults['embed_size'],
                freeze_backbone=defaults['freeze_backbone'],
                use_peft=args.use_peft,
                num_classes=args.num_classes,
                fixed_seed=defaults['fixed_seed'],
                model_str='bert',
                args=args

            )

        elif args.model_version == 'gpt2':

            ckpt_directory = train_gpt2(
            out=args.out,
            batch_size=args.batch_size,
            train_df = train_data_path,
            val_df = val_data_path,
            num_epochs=defaults['num_epochs'],
            learning_rate=defaults['learning_rate'],
            freeze_backbone=defaults['freeze_backbone'],
            fixed_seed=defaults['fixed_seed'],
            use_peft=False,
            model_str='gpt2',
            args=args
            )

    print(ckpt_directory)
    print("Done.")


if __name__ == "__main__":
    main()






    

