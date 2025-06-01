from pytorch_tabular.models import GANDALFConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
)

class GANDALF(GANDALFConfig):
    def __init__(
        self, 
        data_config: DataConfig, 
        optimizer_config: OptimizerConfig, 
        trainer_config: TrainerConfig,
        task: str,
        gflu_stages: int = 6,
        gflu_feature_init_sparsity: float = 0.3,
        gflu_dropout: float = 0.0,
        learning_rate: float = 1e-3,):

        super().__init__(
            task=task,
            gflu_stages=gflu_stages,
            gflu_feature_init_sparsity=gflu_feature_init_sparsity,
            gflu_dropout=gflu_dropout,
            learning_rate=learning_rate,
        )

        self.data_config = data_config
        self.optimizer_config = optimizer_config
        self.trainer_config = trainer_config

