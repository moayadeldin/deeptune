from argparse import ArgumentParser, RawTextHelpFormatter
from enum import Enum
from pathlib import Path
from typing import Optional
from utils import UseCase, RunType, get_model_architecture, set_seed
import json
import os, shutil
class DeepTuneVisionOptions:
    """
    Class for dynamically creating DeepTune's CLI arguments.
    """
    def __init__(self, run_type: RunType, args=None):
        self.parser = ArgumentParser(
            description=f"CLI for {run_type.value} pipeline.",
            formatter_class=RawTextHelpFormatter
        )
        self.mode = run_type
        self._add_default_args()
        
        if run_type == RunType.TabPFNTRAIN:
            self._add_tabpfn_train_args()
        if run_type == RunType.TabPFNEVAL:
            self._add_tabpfn_eval_args()
        if run_type == RunType.TabPFNEMBD:
            self._add_tabpfn_embed_args()
        if run_type == RunType.ONECALL:
            self._add_onecall_args()
        if run_type == RunType.TRAIN:
            self._add_training_args()
        if run_type in (RunType.EVAL, RunType.EMBED):
            self._add_eval_embed_args()
        if run_type == (RunType.GANDALF):
            self._add_gandalf_args()
        if run_type == (RunType.TIMESERIES):
            self._add_timeseries_args()

        parsed_args = self.parser.parse_args(args)
        # self.input_dir: Optional[Path] = parsed_args.input_dir.resolve() if parsed_args.input_dir else None
        self.out: Optional[Path] = parsed_args.out.resolve() if parsed_args.out else None
        self.batch_size: Optional[int] = parsed_args.batch_size
        self.mode : Optional[str] = parsed_args.mode

        if run_type == RunType.TabPFNTRAIN:
            self.target_column: Optional[str] = parsed_args.target_column
            self.train_df: Optional[Path] = parsed_args.train_df
            self.val_df: Optional[Path] = parsed_args.val_df
            self.num_epochs: Optional[int] = parsed_args.num_epochs
            self.finetuning_mode: bool = parsed_args.finetuning_mode

        if run_type == RunType.TabPFNEVAL:
            self.target_column: Optional[str] = parsed_args.target_column
            self.eval_df: Optional[Path] = parsed_args.eval_df
            self.model_weights: Optional[Path] = (
                parsed_args.model_weights.resolve() if parsed_args.model_weights else None
            )
            self.finetuning_mode: bool = parsed_args.finetuning_mode

        if run_type == RunType.TabPFNEMBD:
            self.train_df = parsed_args.train_df
            self.eval_df = parsed_args.eval_df
            self.model_weights: Optional[Path] = (
                parsed_args.model_weights.resolve() if parsed_args.model_weights else None
            )
            self.mode : Optional[str] = parsed_args.mode
            self.target_column: Optional[str] = parsed_args.target_column
            self.finetuning_mode: bool = parsed_args.finetuning_mode


        if run_type == RunType.ONECALL:
            self.modality: str = parsed_args.modality
            self.df: Optional[Path] = parsed_args.df
            self.model_version: str = parsed_args.model_version
            self.num_classes: Optional[int] = parsed_args.num_classes
            self.use_peft: bool = parsed_args.use_peft
            self.target: Optional[str] = parsed_args.target
            self.raw_data: bool = parsed_args.raw_data
            self.finetuning_mode: bool = parsed_args.finetuning_mode
            self.freeze_backbone: bool = parsed_args.freeze_backbone
            self.grouper: Optional[str] = parsed_args.grouper


            # ganadalf specific
            self.type: Optional[str] = parsed_args.type
            self.continuous_cols = parsed_args.continuous_cols
            self.categorical_cols = parsed_args.categorical_cols

            # timeseries specific
            self.time_idx_column = parsed_args.time_idx_column

        
        if run_type in (RunType.TRAIN, RunType.GANDALF, RunType.TIMESERIES):
            self.num_epochs: Optional[int] = parsed_args.num_epochs
            
        if run_type in (RunType.TRAIN, RunType.EVAL,RunType.EMBED):
            self.num_classes: Optional[int] = parsed_args.num_classes
            self.mode: Optional[str] = parsed_args.mode
            self.use_peft: bool = parsed_args.use_peft
        
        if run_type in (RunType.TRAIN, RunType.EVAL, RunType.EMBED):
            self.model_version: Optional[str] = parsed_args.model_version
            self.added_layers: Optional[int] = parsed_args.added_layers
            self.embed_size: Optional[int] = parsed_args.embed_size
            self.freeze_backbone: bool = parsed_args.freeze_backbone
            
        if run_type == RunType.TRAIN:
            self.fixed_seed: bool = parsed_args.fixed_seed
            self.num_epochs: Optional[int] = parsed_args.num_epochs
            self.learning_rate: Optional[float] = parsed_args.learning_rate
            self.train_df: Optional[Path] = parsed_args.train_df
            self.val_df:Optional[Path] = parsed_args.val_df
            set_seed(self.fixed_seed)
        if run_type == RunType.GANDALF:
            self.learning_rate: Optional[float] = parsed_args.learning_rate
            self.train_df: Optional[Path] = parsed_args.train_df
            self.val_df:Optional[Path] = parsed_args.val_df
            self.tabular_target_column = parsed_args.tabular_target_column
            self.continuous_cols = parsed_args.continuous_cols
            self.categorical_cols = parsed_args.categorical_cols
            self.gflu_stages = parsed_args.gflu_stages
            self.type = parsed_args.type
            self.model_weights: Optional[Path] = (
                parsed_args.model_weights.resolve() if parsed_args.model_weights else None
            )
            self.eval_df: Optional[Path] = parsed_args.eval_df
            self.df: Optional[Path] = parsed_args.df
            
        if run_type == RunType.TIMESERIES:
            self.df: Optional[Path] = parsed_args.df
            self.train_df: Optional[Path] = parsed_args.train_df
            self.val_df:Optional[Path] = parsed_args.val_df
            self.max_encoder_length: Optional[int] = parsed_args.max_encoder_length
            self.max_prediction_length: Optional[int] = parsed_args.max_prediction_length
            self.time_varying_known_categoricals = parsed_args.time_varying_known_categoricals
            self.time_varying_unknown_categoricals = parsed_args.time_varying_unknown_categoricals
            self.static_categoricals = parsed_args.static_categoricals
            self.time_varying_unknown_reals = parsed_args.time_varying_unknown_reals
            self.time_varying_known_reals = parsed_args.time_varying_known_reals
            self.static_reals = parsed_args.static_reals
            self.time_idx_column = parsed_args.time_idx_column
            self.target_column = parsed_args.target_column
            self.group_ids = parsed_args.group_ids
            self.eval_df: Optional[Path] = parsed_args.eval_df
            self.model_weights: Optional[Path] = (
                parsed_args.model_weights.resolve() if parsed_args.model_weights else None
            )
            
            
        if run_type in (RunType.EVAL, RunType.EMBED):
            self.eval_df: Optional[Path] = parsed_args.eval_df
            self.df: Optional[Path] = parsed_args.df
            self.model_weights: Optional[Path] = (
                parsed_args.model_weights.resolve() if parsed_args.model_weights else None
            )
        
        # TODO: use_peft should be only for training, use_case for embed and eval;
        #       Requires making sure pretrained models are supported for embed and eval...
        if run_type == RunType.EMBED:
            self.use_case: UseCase = UseCase.from_string(parsed_args.use_case)
        
        self.model = self._parse_model_str(run_type)
        self.model_architecture = ( get_model_architecture(self.model_version) ) if self.model_version else 'not_provided'
    
    def to_dict(self) -> dict:
        d = {}
        for k, v in vars(self).items():
            if k == "parser":
                continue
            if isinstance(v, Path):
                d[k] = str(v.resolve())
            elif isinstance(v, Enum):
                d[k] = v.value
            else:
                d[k] = v
        return d

    def save_args(self, outdir: str) -> None:
        """
        Save the CLI arguments to a JSON file.
        """
        cli_dict = self.to_dict()

        try:
            outdir = Path(outdir)
            outdir.mkdir(parents=True, exist_ok=True)

            out_path = outdir / "cli_arguments.json"

            with open(out_path, "w") as f:
                json.dump(cli_dict, f, indent=4)

        except Exception as e:
            print(f"There is an Error while saving model: {e}")
            # clean up incomplete directory
            if os.path.exists(outdir):
                shutil.rmtree(outdir, ignore_errors=True)

    def _add_default_args(self):
        p = self.parser

        # Dataset args
        # p.add_argument('--input_dir', type=Path, help='Directory containing input data.')
        p.add_argument('--mode', type=str, required=False, choices=['reg','cls'], help='Mode: Classification or Regression')
        # p.add_argument('--num_classes', type=int,required=False, help='Number of classes in your dataset.')
        p.add_argument('--out', type=Path, required=True, help='Destination directory name for results.')

        # Model args
        p.add_argument('--model_version', type=str, help='Model version to use.')
        p.add_argument('--added_layers', type=int, choices=[1,2], help='Number of layers to add to the model.')
        p.add_argument('--embed_size', type=int, help='The number of features desired to obtain when using the model to extract embeddings.')
        p.add_argument('--freeze-backbone', action='store_true', help='Freeze backbone.')
        p.add_argument('--fixed-seed', action='store_true', help='Use fixed seed 42.')
        p.add_argument('--batch_size', type=int, help='Batch size.')

    def _add_tabpfn_train_args(self):
        p = self.parser
        p.add_argument('--target_column', type=str, required=False, help="Specify the name of your target column.")
        p.add_argument('--train_df', type=Path, help='Path to the train dataset (parquet file).', required=True)
        p.add_argument('--val_df', type=Path, help='Path to the validation dataset (parquet file).', required=True)
        p.add_argument('--finetuning-mode', action='store_true', help='If set, perform fine-tuning instead of training from scratch.')
        p.add_argument('--num_epochs', type=int, help='Number of epochs.')

    def _add_tabpfn_eval_args(self):

        p = self.parser
        p.add_argument('--target_column', type=str, required=False, help="Specify the name of your target column.")
        p.add_argument('--eval_df', type=Path, help='Path to the evaluation dataset (parquet file).', required=True)
        p.add_argument('--model_weights', type=Path, help='Path to the model weights.', required=True)
        p.add_argument('--finetuning-mode', action='store_true', help='If set, perform fine-tuning instead of training from scratch.')

    def _add_tabpfn_embed_args(self):
        p = self.parser
        p.add_argument('--train_df', type=Path, help='Path to the train dataset (parquet file).', required=True)
        p.add_argument('--eval_df', type=Path, help='Path to the evaluation dataset (parquet file).', required=True)
        p.add_argument('--model_weights', type=Path, help='Path to the model weights.', required=True)
        p.add_argument('--target_column', type=str, required=False, help="Specify the name of your target column.")
        p.add_argument('--finetuning-mode', action='store_true', help='If set, perform fine-tuning instead of training from scratch.')

    def _add_onecall_args(self):
        p = self.parser
        p.add_argument('--num_classes', type=int,required=False, help='Number of classes in your dataset.')
        p.add_argument('--use-peft', action='store_true', help='Use PEFT-adapted model.')
        p.add_argument("--modality", help="Modality you work on", choices=["text", "images", "tabular", "timeseries"], required=True)
        p.add_argument('--df', type=Path, required=True, help='Path to the dataframe (parquet file) to be used for training.')
        p.add_argument('--target', type=str, required=False, help="Specify the name of your target column. Default is 'labels'.")
        p.add_argument('--raw-data', action='store_true', help='Use raw data instead of ready-to-use parquet.')
        # tabpfn specific
        p.add_argument('--finetuning-mode', action='store_true', help='If set, perform fine-tuning instead of training from scratch for TabPFN.')
        # gandalf specific
        p.add_argument('--type', type=str, required=False, choices=['classification','regression'], help='Type: Classification or Regression for GANDALF')
        p.add_argument('--continuous_cols', nargs='+', help='List of continuous column names for GANDALF',required=False)
        p.add_argument('--categorical_cols', nargs='+', help='List of categorical column names for GANDALF',required=False)
        # timeseries specific
        p.add_argument('--time_idx_column', type=str, default='labels', help='integer typed column denoting the time index within data.')
        p.add_argument('--grouper', type=str, required=False, help='Column name to be used as grouper for stratified splitting.')
    def _add_training_args(self):
        p = self.parser
        p.add_argument('--num_classes', type=int,required=False, help='Number of classes in your dataset.')
        p.add_argument('--use-peft', action='store_true', help='Use PEFT-adapted model.')
        p.add_argument('--num_epochs', type=int, help='Number of epochs.')
        p.add_argument('--learning_rate', type=float, help='Learning rate.')
        p.add_argument('--train_df', type=Path, help='PARQUET file containing train data.')
        p.add_argument('--val_df', type=Path, help='PARQUET file containing validation data.')
        
    def _add_timeseries_args(self):
        
        p = self.parser
  
        p.add_argument('--df', type=Path, help='PARQUET file containing data.')
        p.add_argument('--max_encoder_length', type=int, default=30, help='maximum history length used by the time series dataset.')
        p.add_argument('--max_prediction_length', type=int, default=1, help='maximum prediction/decoder length')
        p.add_argument('--time_varying_known_categoricals', nargs='+', type=str, default=[], help='list of categorical variables that change over time and are known in the future.')
        p.add_argument('--time_varying_unknown_categoricals', nargs='+', type=str, default=[], help='list of categorical variables that are not known in the future and change over time. Target Variables should be included here if they are categorical.')
        p.add_argument('--static_categoricals', nargs='+', type=str, default=[], help='list of categorical variables that do not change over time.')        
        p.add_argument('--time_varying_unknown_reals', nargs='+', type=str, default=[], help='list of continuous variables that are not known in the future and change over time. Target Variables should be included here if they are continuous.')
        p.add_argument('--time_varying_known_reals', nargs='+', type=str, default=[], help='list of continuous variables that change over time and are known in the future.')
        p.add_argument('--static_reals', nargs='+', type=str, default=[], help='list of continuous variables that do not change over time.')
        p.add_argument('--time_idx_column', type=str, default='labels', help='integer typed column denoting the time index within data.')
        p.add_argument('--group_ids',type=str,default=None, help='List of column names identifying a time series instance within your data. If you have only one timeseries, set this to the name of column that is constant.')
        p.add_argument('--train_df', type=Path, help='PARQUET file containing train data.')
        p.add_argument('--val_df', type=Path, help='PARQUET file containing validation data.')
        p.add_argument('--num_epochs', type=int, help='Number of epochs.')
        p.add_argument('--target_column', type=str, help='Target Column')
        p.add_argument('--eval_df', type=Path, help='PARQUET file containing testing data.')
        p.add_argument('--model_weights', required=False, type=Path, help='Path to model weights.')
        
    def _add_eval_embed_args(self):
        p = self.parser
        p.add_argument('--df', type=Path, help='PARQUET file containing data.')
        p.add_argument('--use-peft', action='store_true', help='Use PEFT-adapted model.')
        p.add_argument('--eval_df', type=Path, help='PARQUET file containing testing data.')
        p.add_argument('--num_classes', type=int,required=False, help='Number of classes in your dataset.')
        p.add_argument('--model_weights', required=False, type=Path, help='Path to model weights.')
        p.add_argument(
            '--use_case',
            type=UseCase.parse,
            choices=UseCase.choices(),
            default=None,
            help="""
            The type of training applied to the model in use.
            
            Options: pretrained, finetuned, peft

                pretrained (default): Use the pretrained model. Model weights file will be ignored.
                finetuned: Load the model using fine-tuned model weights file.
                peft: Load the model using peft-tuned model weights file.

            """
        )

    def _add_gandalf_args(self):

        p = self.parser
        p.add_argument('--tabular_target_column', nargs='+', type=str, default='labels', help='Target column for GANDALF')
        p.add_argument('--continuous_cols', nargs='+', help='List of continuous column names for GANDALF')
        p.add_argument('--categorical_cols', nargs='+', help='List of categorical column names for GANDALF')
        p.add_argument('--gflu_stages', type=int, default=6, help='Number of GFLU stages for GANDALF')
        p.add_argument('--type', type=str, choices=['classification', 'regression'], help='Task type for GANDALF')
        p.add_argument('--num_epochs', type=int, help='Number of epochs.')
        p.add_argument('--learning_rate', type=float, help='Learning rate.')
        p.add_argument('--train_df', type=Path, help='PARQUET file containing train data.')
        p.add_argument('--val_df', type=Path, help='PARQUET file containing validation data.')
        p.add_argument('--eval_df', type=Path, help='PARQUET file containing testing data.')
        p.add_argument('--model_weights', type=Path, help='Path to model weights.')
        p.add_argument('--df', type=Path, help='PARQUET file containing data.')
    
        

    def _parse_model_str(self, mode: RunType) -> str:
        if mode == RunType.TRAIN or mode == RunType.EVAL:
            prefix_str = "PEFT" if self.use_peft else "FINETUNED"
        elif mode == RunType.GANDALF:
            prefix_str = "GANDALF"
            self.model_version=None
            return f"{prefix_str}"
        elif mode == RunType.TIMESERIES:
            prefix_str = "TIMESERIES"
            self.model_version=None
            return f"{prefix_str}"
        elif mode == RunType.ONECALL:
            prefix_str = "PEFT" if self.use_peft else "FINETUNED"
        elif mode == RunType.TabPFNTRAIN or mode == RunType.TabPFNEVAL or mode == RunType.TabPFNEMBD:
            prefix_str = "TABPFN"
            self.model_version=None
        else:
            prefix_str = self.use_case.value.upper()

        return f"{prefix_str}-{self.model_version}"
        


if __name__ == "__main__":
    # Tests
    args = DeepTuneVisionOptions(RunType.TRAIN)
    # args = DeepTuneOptions(RunType.EVAL)
    # args = DeepTuneOptions(RunType.EMBED)

    for name, arg in args.to_dict().items():
        print(f"{name}    {arg}")
    args.save_args(Path(__file__).parent / "cli_testing")

    # print(args.input_dir)
    # print(args.batch_size)
    # print(args.num_epochs)
