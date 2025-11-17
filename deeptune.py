##### I. TEXT IMPORTS #####
from trainers.nlp.train_gpt2 import train as train_gpt2
from trainers.nlp.train_multilingualbert import train as train_multilingualbert
from evaluators.nlp.evaluate_multilingualbert import evaluate as evaluate_multilingualbert
from evaluators.nlp.evaluate_gpt import evaluate as evaluate_gpt2
from embed.nlp.gpt2_embeddings import embed as embed_gpt2
from embed.nlp.multilingualbert_embeddings import embed as embed_multilingualbert
##### II. IMAGES IMPORTS #####
from trainers.vision.train import train as train_images
from evaluators.vision.evaluate import evaluate as evaluate_images
from embed.vision.embed import embed as embed_images
##### III. TABULAR IMPORTS #####
from trainers.tabular.train_gandalf import train as train_tabular_gandalf
from evaluators.tabular.evaluate_gandalf import evaluate as evaluate_tabular_gandalf
from embed.tabular.gandalf_embeddings import embed as embed_tabular_gandalf
##### IV. TIMESERIES IMPORTS #####
from trainers.timeseries.train_deepar import train as train_deepar
from evaluators.timeseries.evaluate_deepar import evaluate as evaluate_deepar
##### IV. GENERIC IMPORTS #####
from handlers.split_dataset import split_dataset
from pathlib import Path
from cli import DeepTuneVisionOptions
from utils import RunType
from helpers import date_id,print_metrics_table


# os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1' # disabling weights-only loading error

defaults={
    'num_epochs':3,
    'learning_rate':1e-4,
    'added_layers':2,
    'embed_size':1000,
    'train_size':0.8,
    'val_size':0.1,
    'test_size':0.1,
    'mode':'cls',
    'fixed_seed':True,
    'freeze_backbone': False,
    'disable_numerical_encoding':False,
    'gflu_stages':6,
    'target':'labels',
    'group_ids': None
}

def main():
    args = DeepTuneVisionOptions(RunType.ONECALL)

    parent_dir = date_id(root_dir=args.out)

    TARGET = args.target

    train_data_path, val_data_path, test_data_path = split_dataset(
        train_size=defaults['train_size'],
        val_size=defaults['val_size'],
        test_size=defaults['test_size'],
        df_path=args.df,
        out_dir=Path(args.out)/parent_dir,
        fixed_seed=defaults['fixed_seed'],
        disable_numerical_encoding=defaults['disable_numerical_encoding'],
        target_column=TARGET
    )

    USE_CASE = 'peft' if args.use_peft else 'finetuned'

    if args.modality == 'text':
    
        if args.model_version == 'bert':

            ckpt_directory = train_multilingualbert(
                out=Path(args.out)/parent_dir,
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
                model_str='peft-bert' if args.use_peft else 'bert',
                args=args

            )

            metrics_dict = evaluate_multilingualbert(
                eval_df=test_data_path,
                out=Path(args.out)/parent_dir,
                model_weights=ckpt_directory,
                model_str='peft-bert' if args.use_peft else 'bert',
                num_classes=args.num_classes,
                added_layers=defaults['added_layers'],
                embed_size=defaults['embed_size'],
                batch_size=args.batch_size,
                use_peft=args.use_peft,
                args=args,
                freeze_backbone=defaults['freeze_backbone'],
            )


            embed_path,embed_shape = embed_multilingualbert(
                df_path=test_data_path,
                out=Path(args.out)/parent_dir,
                model_weights=ckpt_directory,
                num_classes=args.num_classes,
                added_layers=defaults['added_layers'],
                embed_size=defaults['embed_size'],
                batch_size=args.batch_size,
                use_case=USE_CASE,
                freeze_backbone=defaults['freeze_backbone'],
            )

        elif args.model_version == 'gpt2':

            ckpt_directory = train_gpt2(
            out=Path(args.out)/parent_dir,
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

            metrics_dict = evaluate_gpt2(
                eval_df=test_data_path,
                out=Path(args.out)/parent_dir,
                model_weights=ckpt_directory,
                batch_size=args.batch_size,
                freeze_backbone=defaults['freeze_backbone'],
                args=args,
                use_peft=False,
                model_str='gpt2'
            )

            embed_path,embed_shape = embed_gpt2(
                df_path=test_data_path,
                out = Path(args.out)/parent_dir,
                model_weights=ckpt_directory,
                batch_size = args.batch_size,
                use_case="finetuned")
            
            print_metrics_table(metrics_dict, embed_shape, embed_path, modality='text')
            
    elif args.modality == 'images':

        ckpt_directory = train_images(
            train_df=train_data_path,
            val_df=val_data_path,
            out=Path(args.out)/parent_dir,
            batch_size=args.batch_size,
            num_epochs=defaults['num_epochs'],
            learning_rate=defaults['learning_rate'],
            added_layers=defaults['added_layers'],
            embed_size=defaults['embed_size'],
            freeze_backbone=defaults['freeze_backbone'],
            use_peft=args.use_peft,
            num_classes=args.num_classes,
            fixed_seed=defaults['fixed_seed'],
            args=args,
            mode=defaults['mode'],
            model_str=args.model_version,
            model_version=args.model_version
        )

        metrics_dict = evaluate_images(
            eval_df=test_data_path,
            mode=defaults['mode'],
            num_classes=args.num_classes,
            out=Path(args.out)/parent_dir,
            model_version=args.model_version,
            model_str=args.model_version,
            model_weights=ckpt_directory,
            use_peft=args.use_peft,
            added_layers=defaults['added_layers'],
            embed_size=defaults['embed_size'],
            batch_size=args.batch_size,
            freeze_backbone=defaults['freeze_backbone'],
            args=args
        )

        embed_path,embed_shape = embed_images(
            df_path=test_data_path,
            out=Path(args.out)/parent_dir,
            model_weights=ckpt_directory,
            batch_size=args.batch_size,
            use_case=USE_CASE,
            model_version=args.model_version,
            model_str=args.model_version,
            added_layers=defaults['added_layers'],
            embed_size=defaults['embed_size'],
            num_classes=args.num_classes,
            args=args,
            mode=defaults['mode'],
        )

        print_metrics_table(metrics_dict, embed_shape, embed_path, modality='images')

    elif args.modality == 'tabular':

        if args.model_version == 'gandalf':

            ckpt_directory = train_tabular_gandalf(
                train_df=train_data_path,
                val_df=val_data_path,
                out=Path(args.out)/parent_dir,
                batch_size=args.batch_size,
                num_epochs=defaults['num_epochs'],
                learning_rate=defaults['learning_rate'],
                gflu_stages=defaults['gflu_stages'],
                target=[defaults['target']],
                continuous_cols=args.continuous_cols,
                categorical_cols=args.categorical_cols,
                model_str='GANDALF',
                args=args,
                type=args.type
            )
            metrics_dict = evaluate_tabular_gandalf(
                eval_df=test_data_path,
                out=Path(args.out)/parent_dir,
                model_weights=ckpt_directory,
                model_str='GANDALF',
                args=args,
            )


            embed_path,embed_shape = embed_tabular_gandalf(
                eval_df=test_data_path,
                out=Path(args.out)/parent_dir,
                model_weights=ckpt_directory,
                continuous_cols=args.continuous_cols,
                categorical_cols=args.categorical_cols,
                batch_size=args.batch_size,
                target=defaults['target'],
                args=args,
                model_str='GANDALF',
            )

            print_metrics_table(metrics_dict, embed_shape, embed_path, modality='tabular')

    elif args.modality == 'timeseries':
        if args.model_version == 'deepAR':

            ckpt_directory = train_deepar(
                train_df=train_data_path,
                val_df=val_data_path,
                out=Path(args.out)/parent_dir,
                batch_size=args.batch_size,
                num_epochs=defaults['num_epochs'],
                timeindex_column=args.time_idx_column,
                target_column=defaults['target'],
                group_ids=defaults['group_ids'],
                args=args,
                model_str='DeepAR',
            )


            _ = evaluate_deepar(
                train_df_path=train_data_path,
                val_df_path=val_data_path,
                eval_df_path=test_data_path,
                out=Path(args.out)/parent_dir,
                batch_size=args.batch_size,
                timeindex_column=args.time_idx_column,
                target_column=defaults['target'],
                model_weights=ckpt_directory,
                group_ids=defaults['group_ids'],
                args=args,
            ) 

            return

            

    

if __name__ == "__main__":
    main()






    

