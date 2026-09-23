##### VI. GENERIC IMPORTS #####
from handlers.split_dataset import split_dataset
from pathlib import Path
from cli import DeepTuneVisionOptions
import pandas as pd 
from utils import RunType
from helpers import date_id,print_metrics_table,print_training_log_table, print_experiment_paths_table
from handlers.raw_to_parquet_dataset import raw_to_parquet
import json

defaults={
    'num_epochs':10,
    'learning_rate':1e-4,
    'added_layers':2,
    'embed_size':1000,
    'train_size':0.7,
    'val_size':0.1,
    'test_size':0.2,
    'mode':'cls',
    'fixed_seed':True,
    'disable_numerical_encoding':False,
    'gflu_stages':6,
    'target':'labels',
    'group_ids': None
}

def main(argv=None):
    args = DeepTuneVisionOptions(RunType.ONECALL, args=argv)
    from desktop.config import RunConfig, validate_config, effective_mode
    values = {key: getattr(args, key) for key in RunConfig.__dataclass_fields__ if hasattr(args, key)}
    values["grouper"] = args.grouper or ""
    values["continuous_cols"] = args.continuous_cols or []
    values["categorical_cols"] = args.categorical_cols or []
    values["gandalf_type"] = args.type or "classification"
    values["added_layers"] = args.added_layers or 2
    values["embed_size"] = args.embed_size if args.embed_size is not None else 1000
    values["num_classes"] = args.num_classes if args.num_classes is not None else 2
    cfg = RunConfig(**values)
    issues = validate_config(cfg)
    if issues:
        args.parser.error("\n".join(issues))
    args.mode = effective_mode(cfg)
    args.type = cfg.gandalf_type
    defaults = dict(globals()["defaults"], mode=args.mode, fixed_seed=args.fixed_seed, disable_numerical_encoding=args.mode == "reg")

    parent_dir = date_id(root_dir=args.out)

    TARGET = args.target
    RAW_DATA = args.raw_data
    FINETUNING_MODE = args.finetuning_mode
    MODE = args.mode
    FREEZE_BACKBONE = args.freeze_backbone
    GROUPER = args.grouper

    # --num_epochs/--learning_rate are onecall-only args with their own CLI
    # defaults (10 / 1e-4, matching the values below) so they always override
    # cleanly. --added_layers/--embed_size are shared with other run types
    # and have no CLI-level default (None unless passed), so only override
    # when the user actually supplied one, preserving defaults otherwise.
    defaults['num_epochs'] = args.num_epochs
    defaults['learning_rate'] = args.learning_rate
    if args.added_layers is not None:
        defaults['added_layers'] = args.added_layers
    if args.embed_size is not None:
        defaults['embed_size'] = args.embed_size

    df_path = args.df if not RAW_DATA else raw_to_parquet(
        dataset_dir=args.df,
        out=Path(args.out)/parent_dir,
        modality=args.modality,
    )

    train_data_path, val_data_path, test_data_path = split_dataset(
        train_size=defaults['train_size'],
        val_size=defaults['val_size'],
        test_size=defaults['test_size'],
        df_path=df_path,
        out_dir=Path(args.out)/parent_dir,
        fixed_seed=defaults['fixed_seed'],
        disable_numerical_encoding=defaults['disable_numerical_encoding'],
        target_column=TARGET,
        modality=args.modality,
        grouper=GROUPER,
        mode=args.mode,
        time_idx_column=args.time_idx_column,
    )

    df = pd.read_parquet(train_data_path, columns=['labels'])

    USE_CASE = 'peft' if args.use_peft else 'finetuned'

    MAPPING_PATH = next((p / "label_mapping.json" 
                     for p in Path(Path(args.out)/parent_dir).glob("data_splits*") 
                     if (p / "label_mapping.json").exists()), None)
    
    if args.modality in ('images', 'video', 'text') and args.mode == 'cls':
        mapping = json.loads(MAPPING_PATH.read_text(encoding='utf-8')) if MAPPING_PATH else {}
        classes = len(mapping) or int(df['labels'].nunique())
        if args.num_classes is None:
            args.num_classes = classes
        elif args.num_classes != classes:
            raise ValueError(f'Number of classes is {classes}, but --num_classes is {args.num_classes}.')

    if args.modality == 'text':
        from trainers.nlp.train_gpt2 import train as train_gpt2
        from trainers.nlp.train_multilingualbert import train as train_multilingualbert
        from evaluators.nlp.evaluate_multilingualbert import evaluate as evaluate_multilingualbert
        from evaluators.nlp.evaluate_gpt import evaluate as evaluate_gpt2
        from embed.nlp.gpt2_embeddings import embed as embed_gpt2
        from embed.nlp.multilingualbert_embeddings import embed as embed_multilingualbert
    
        if args.model_version == 'BERT':

            ckpt_directory = train_multilingualbert(
                out=Path(args.out)/parent_dir,
                batch_size=args.batch_size,
                train_df = train_data_path,
                val_df = val_data_path,
                num_epochs=defaults['num_epochs'],
                learning_rate=defaults['learning_rate'],
                added_layers=defaults['added_layers'],
                embed_size=defaults['embed_size'],
                freeze_backbone=FREEZE_BACKBONE,
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
                freeze_backbone=FREEZE_BACKBONE,
            )


            exp_path,embed_shape = embed_multilingualbert(
                df_path=test_data_path,
                out=Path(args.out)/parent_dir,
                model_weights=ckpt_directory,
                num_classes=args.num_classes,
                added_layers=defaults['added_layers'],
                embed_size=defaults['embed_size'],
                batch_size=args.batch_size,
                use_case=USE_CASE,
                freeze_backbone=FREEZE_BACKBONE,
            )


        elif args.model_version == 'gpt2':

            ckpt_directory = train_gpt2(
            out=Path(args.out)/parent_dir,
            batch_size=args.batch_size,
            train_df = train_data_path,
            val_df = val_data_path,
            num_epochs=defaults['num_epochs'],
            learning_rate=defaults['learning_rate'],
            freeze_backbone=FREEZE_BACKBONE,
            fixed_seed=defaults['fixed_seed'],
            use_peft=False,
            model_str='gpt2',
            args=args,
            num_classes=args.num_classes,
            )

            metrics_dict = evaluate_gpt2(
                eval_df=test_data_path,
                out=Path(args.out)/parent_dir,
                model_weights=ckpt_directory,
                batch_size=args.batch_size,
                freeze_backbone=FREEZE_BACKBONE,
                args=args,
                use_peft=False,
                model_str='gpt2'
            )

            exp_path,embed_shape = embed_gpt2(
                df_path=test_data_path,
                out = Path(args.out)/parent_dir,
                model_weights=ckpt_directory,
                batch_size = args.batch_size,
                use_case="finetuned")
            
        csv_dir = Path(ckpt_directory)
        if csv_dir.is_file():
            csv_dir = csv_dir.parent
        print_experiment_paths_table(df_path=df_path, train_data_path=train_data_path, val_data_path=val_data_path, test_data_path=test_data_path, ckpt_directory=ckpt_directory, exp_path=exp_path)
        print_training_log_table(csv_dir/"training_log.csv")
        print_metrics_table(metrics_dict, embed_shape, modality='text', mapping_path=MAPPING_PATH)
        
    elif args.modality == 'images':
        from trainers.vision.train import train as train_images
        from evaluators.vision.evaluate import evaluate as evaluate_images
        from embed.vision.embed import embed as embed_images

        ckpt_directory = train_images(
            train_df=train_data_path,
            val_df=val_data_path,
            out=Path(args.out)/parent_dir,
            batch_size=args.batch_size,
            num_epochs=defaults['num_epochs'],
            learning_rate=defaults['learning_rate'],
            added_layers=defaults['added_layers'],
            embed_size=defaults['embed_size'],
            freeze_backbone=FREEZE_BACKBONE,
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
            freeze_backbone=FREEZE_BACKBONE,
            args=args
        )

        exp_path,embed_shape = embed_images(
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
            grouper=GROUPER,
        )
        csv_dir = Path(ckpt_directory)
        if csv_dir.is_file():
            csv_dir = csv_dir.parent.parent
        print_experiment_paths_table(df_path=df_path, train_data_path=train_data_path, val_data_path=val_data_path, test_data_path=test_data_path, ckpt_directory=ckpt_directory, exp_path=exp_path)
        print_training_log_table(csv_dir/"training_log.csv")
        print_metrics_table(metrics_dict, embed_shape, modality='images',mapping_path=MAPPING_PATH)

    elif args.modality == 'video':
        from trainers.video.train import train as train_video
        from evaluators.video.evaluate import evaluate as evaluate_video
        from embed.video.embed import embed as embed_video

        ckpt_directory = train_video(
            train_df=train_data_path,
            val_df=val_data_path,
            out=Path(args.out)/parent_dir,
            batch_size=args.batch_size,
            num_epochs=defaults['num_epochs'],
            learning_rate=defaults['learning_rate'],
            added_layers=defaults['added_layers'],
            embed_size=defaults['embed_size'],
            freeze_backbone=FREEZE_BACKBONE,
            use_peft=args.use_peft,
            num_classes=args.num_classes,
            fixed_seed=defaults['fixed_seed'],
            args=args,
            mode=defaults['mode'],
            model_str=args.model_version,
            model_version=args.model_version,
            num_frames=args.num_frames,
            pooling=args.pooling,
        )

        metrics_dict = evaluate_video(
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
            freeze_backbone=FREEZE_BACKBONE,
            args=args,
            num_frames=args.num_frames,
            pooling=args.pooling,
        )

        exp_path,embed_shape = embed_video(
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
            grouper=GROUPER,
            num_frames=args.num_frames,
            pooling=args.pooling,
        )
        csv_dir = Path(ckpt_directory)
        if csv_dir.is_file():
            csv_dir = csv_dir.parent.parent
        print_experiment_paths_table(df_path=df_path, train_data_path=train_data_path, val_data_path=val_data_path, test_data_path=test_data_path, ckpt_directory=ckpt_directory, exp_path=exp_path)
        print_training_log_table(csv_dir/"training_log.csv")
        print_metrics_table(metrics_dict, embed_shape, modality='video',mapping_path=MAPPING_PATH)

    elif args.modality == 'tabular':

        if args.model_version == 'gandalf':
            from trainers.tabular.train_gandalf import train as train_tabular_gandalf
            from evaluators.tabular.evaluate_gandalf import evaluate as evaluate_tabular_gandalf
            from embed.tabular.gandalf_embeddings import embed as embed_tabular_gandalf

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


            exp_path,embed_shape = embed_tabular_gandalf(
                eval_df=test_data_path,
                out=Path(args.out)/parent_dir,
                model_weights=ckpt_directory,
                cont_cols=args.continuous_cols,
                cat_cols=args.categorical_cols,
                batch_size=args.batch_size,
                target=defaults['target'],
                args=args,
                grouper = GROUPER,
                model_str='GANDALF',
            )
            print_experiment_paths_table(df_path=df_path, train_data_path=train_data_path, val_data_path=val_data_path, test_data_path=test_data_path, ckpt_directory=ckpt_directory, exp_path=exp_path)
            print_metrics_table(metrics_dict, embed_shape, modality='tabular',mapping_path=MAPPING_PATH)

        if args.model_version == 'tabpfn':
            ### For TabPFN ###
            train_df = pd.read_parquet(train_data_path)
            X_train_tabpfn = train_df.drop(columns=['labels'])
            y_train_tabpfn = train_df['labels']
            val_df = pd.read_parquet(val_data_path)
            X_val_tabpfn = val_df.drop(columns=['labels'])
            y_val_tabpfn = val_df['labels']
            eval_df = pd.read_parquet(test_data_path)
            X_eval_tabpfn = eval_df.drop(columns=['labels'])
            y_eval_tabpfn = eval_df['labels']


            from trainers.tabular.train_tabpfn import train_tabpfn_from_scratch as train_tabular_tabpfn
            from trainers.tabular.train_tabpfn import finetune_tabpfn as finetune_tabular_tabpfn
            from evaluators.tabular.evaluate_tabpfn import evaluate_tabpfn as evaluate_tabular_tabpfn
            from embed.tabular.tabpfn_embeddings import get_tabpfn_embeddings as embed_tabpfn_embeddings

            if FINETUNING_MODE:

                ckpt_directory = finetune_tabular_tabpfn(
                    X_train = X_train_tabpfn,
                    y_train = y_train_tabpfn,
                    X_val = X_val_tabpfn,
                    y_val = y_val_tabpfn,
                    mode = MODE,
                    out = Path(args.out)/parent_dir,
                    num_epochs=defaults['num_epochs'],
                    args=args,
                    model_str='TABPFN',
                )


            else:
                ckpt_directory = train_tabular_tabpfn(
                    X_train= X_train_tabpfn,
                    y_train= y_train_tabpfn,
                    mode=MODE,
                    out=Path(args.out)/parent_dir,
                    X_val= X_val_tabpfn,
                    y_val= y_val_tabpfn,
                    args=args,
                    model_str='TABPFN',
                )

            metrics_dict = evaluate_tabular_tabpfn(
                X_eval_tabpfn,
                y_eval_tabpfn,
                out=Path(args.out)/parent_dir,
                model_path=ckpt_directory,
                mode=MODE,
                args=args,
                finetuning_mode=FINETUNING_MODE,
                model_str='TABPFN',
                )
            
            exp_path,embed_shape = embed_tabpfn_embeddings(
                X_train_tabpfn,
                y_train_tabpfn,
                X_eval_tabpfn,
                y_eval_tabpfn,
                out=Path(args.out)/parent_dir,
                args=args,
                model_path=ckpt_directory,
                mode=MODE,
                finetuning_mode=FINETUNING_MODE,
                grouper=GROUPER,
                model_str='TABPFN',
            )

            print_experiment_paths_table(df_path=df_path, train_data_path=train_data_path, val_data_path=val_data_path, test_data_path=test_data_path, ckpt_directory=ckpt_directory, exp_path=exp_path)
            print_metrics_table(metrics_dict, embed_shape, modality='tabular')

    elif args.modality == 'timeseries':
        from trainers.timeseries.train_deepar import train as train_deepar
        from evaluators.timeseries.evaluate_deepar import evaluate as evaluate_deepar
        from embed.timeseries.deepAR_embeddings import embed as embed_deepar
        if args.model_version == 'deepAR':

            ckpt_directory = train_deepar(
                train_df=train_data_path,
                val_df=val_data_path,
                out=Path(args.out)/parent_dir,
                batch_size=args.batch_size,
                num_epochs=defaults['num_epochs'],
                learning_rate=defaults['learning_rate'],
                timeindex_column=args.time_idx_column,
                target_column=defaults['target'],
                group_ids=[GROUPER] if GROUPER else defaults['group_ids'],
                args=args,
                model_str='DeepAR',
            )


            exp_path,_ = evaluate_deepar(
                train_df_path=train_data_path,
                val_df_path=val_data_path,
                eval_df_path=test_data_path,
                out=Path(args.out)/parent_dir,
                batch_size=args.batch_size,
                timeindex_column=args.time_idx_column,
                target_column=defaults['target'],
                model_weights=ckpt_directory,
                group_ids=[GROUPER] if GROUPER else defaults['group_ids'],
                args=args,
            )

            exp_path, embed_shape = embed_deepar(
                eval_df=test_data_path,
                out=Path(args.out)/parent_dir,
                model_weights=ckpt_directory,
                batch_size=args.batch_size,
                target_column=defaults['target'],
                args=args,
                history_df_paths=[train_data_path, val_data_path],
                group_ids=[GROUPER] if GROUPER else None,
                model_str='DeepAR',
                timeindex_column=args.time_idx_column,
            )

            print_experiment_paths_table(df_path=df_path, train_data_path=train_data_path, val_data_path=val_data_path, test_data_path=test_data_path, ckpt_directory=ckpt_directory, exp_path=exp_path)

    print(" ✅ Your run is complete. Check the output directory table for a detailed copy of your results. \n Thank you for using DeepTune!🚀")


        

    

if __name__ == "__main__":
    main()






    

