from argparse import ArgumentParser, RawTextHelpFormatter
from enum import Enum
from pathlib import Path
from typing import Optional

from utils import UseCase, RunType, get_model_architecture, set_seed


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
        
        if run_type == RunType.TRAIN:
            self._add_training_args()
        if run_type in (RunType.EVAL, RunType.EMBED):
            self._add_eval_embed_args()

        parsed_args = self.parser.parse_args(args)

        self.input_dir: Optional[Path] = parsed_args.input_dir
        self.mode: Optional[str] = parsed_args.mode
        self.num_classes: Optional[int] = parsed_args.num_classes
        self.out: Optional[Path] = parsed_args.out.resolve() if parsed_args.out else None

        self.model_version: Optional[str] = parsed_args.model_version
        self.added_layers: Optional[int] = parsed_args.added_layers
        self.embed_size: Optional[int] = parsed_args.embed_size
        self.freeze_backbone: bool = parsed_args.freeze_backbone
        self.fixed_seed: bool = parsed_args.fixed_seed
        self.batch_size: Optional[int] = parsed_args.batch_size

        if run_type == RunType.TRAIN:
            self.num_epochs: Optional[int] = parsed_args.num_epochs
            self.learning_rate: Optional[float] = parsed_args.learning_rate
            self.train_df: Optional[Path] = parsed_args.train_df
            self.val_df:Optional[Path] = parsed_args.val_df

        if run_type in (RunType.EVAL, RunType.EMBED):
            self.eval_df: Optional[Path] = parsed_args.eval_df or (self.input_dir / "test_split.parquet" if self.input_dir else None)
            self.df: Optional[Path] = parsed_args.df
            self.model_weights: Optional[Path] = (
                parsed_args.model_weights.resolve() if parsed_args.model_weights else None
            )
        
        # TODO: use_peft should be only for training, use_case for embed and eval;
        #       Requires making sure pretrained models are supported for embed and eval...
        if run_type == RunType.EMBED:
            self.use_case: UseCase = UseCase.from_string(parsed_args.use_case)
        self.use_peft: bool = parsed_args.use_peft
        
        self.model = self._parse_model_str(run_type)
        self.model_architecture = get_model_architecture(self.model_version)
        set_seed(self.fixed_seed)
    
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

    def save_args(self, outdir: Path) -> None:
        """
        Save the CLI arguments to a JSON file.
        """
        cli_dict = self.to_dict()
        
        outdir.mkdir(parents=True, exist_ok=True)
        out_path = outdir / "cli_arguments.json"
        
        import json
        with open(out_path, "w") as f:
            json.dump(cli_dict, f, indent=4)

    def _add_default_args(self):
        p = self.parser

        # Dataset args
        p.add_argument('--input_dir', type=Path, help='Directory containing input data.')
        p.add_argument('--mode', type=str, choices=['reg','cls'], help='Mode: Classification or Regression')
        p.add_argument('--num_classes', type=int, help='Number of classes if the task is regression.')

        p.add_argument('--out', type=Path, help='Destination directory name for results.')

        # Model args
        p.add_argument('--model_version', type=str, help='Model version to use.')
        p.add_argument('--use-peft', action='store_true', help='Use PEFT-adapted model.')
        p.add_argument('--added_layers', type=int, choices=[1,2], help='Number of layers to add to the model.')
        p.add_argument('--embed_size', type=int, help='The number of features desired to obtain when using the model to extract embeddings.')
        p.add_argument('--freeze-backbone', action='store_true', help='Freeze backbone.')
        p.add_argument('--fixed-seed', action='store_true', help='Use fixed seed 42.')
        p.add_argument('--batch_size', type=int, help='Batch size.')

    def _add_training_args(self):
        p = self.parser
        p.add_argument('--num_epochs', type=int, help='Number of epochs.')
        p.add_argument('--learning_rate', type=float, help='Learning rate.')
        p.add_argument('--train_df', type=Path, help='PARQUET file containing train data.')
        p.add_argument('--val_df', type=Path, help='PARQUET file containing validation data.')

    def _add_eval_embed_args(self):
        p = self.parser
        p.add_argument('--df', type=Path, help='PARQUET file containing data.')
        p.add_argument('--eval_df', type=Path, help='PARQUET file containing testing data.')
        p.add_argument('--model_weights', type=Path, help='Path to model weights.')
        p.add_argument(
            '--use_case',
            type=UseCase.parse,
            choices=UseCase.choices(),
            default=UseCase.PRETRAINED.value,
            help="""
            The type of training applied to the model in use.
            
            Options: pretrained, finetuned, peft

                pretrained (default): Use the pretrained model. Model weights file will be ignored.
                finetuned: Load the model using fine-tuned model weights file.
                peft: Load the model using peft-tuned model weights file.

            """
        )

    def _parse_model_str(self, mode: RunType) -> str:
        if mode == RunType.TRAIN or mode == RunType.EVAL:
            prefix_str = "PEFT" if self.use_peft else "FINETUNED"
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
