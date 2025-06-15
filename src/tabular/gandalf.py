from pytorch_tabular.models import GANDALFConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
)
import torch.nn as nn
import torch

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
        learning_rate: float = 1e-3):

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


class AdjustedGANDALF(GANDALF):
    def __init__(self, 
                 data_config, 
                 optimizer_config, 
                 trainer_config, 
                 task, 
                 gflu_stages, 
                 gflu_feature_init_sparsity=0.2, 
                 gflu_dropout=0.1, 
                 learning_rate=1e-3,
                 hidden_dim=16,  # Dimension for the intermediate layer
                 num_classes=7   # Output dimension (adjust based on your task)
                 ):
        super().__init__(
            data_config=data_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
            task=task,
            gflu_stages=gflu_stages,
            gflu_feature_init_sparsity=gflu_feature_init_sparsity,
            gflu_dropout=gflu_dropout,
            learning_rate=learning_rate,
        )
        
        # Remove the original head by replacing it with our custom head
        # The backbone output is 34 features (from W_out layers)
        self._head = nn.Sequential(
            nn.Linear(34, hidden_dim),  # First new linear layer
            nn.ReLU(),  # Add activation between layers
            nn.Linear(hidden_dim, num_classes)  # Second new linear layer
        )
    
    def forward(self, x):
        # Use the backbone from the parent class
        backbone_output = self._backbone(x)
        
        # Apply embedding if it exists
        if hasattr(self, '_embedding_layer'):
            embedded = self._embedding_layer(x)
            # Combine backbone output with embeddings if needed
            # This depends on your specific implementation
            combined = torch.cat([backbone_output, embedded], dim=1) if embedded.numel() > 0 else backbone_output
        else:
            combined = backbone_output
        
        # Apply our custom head
        output = self._head(combined)
        
        return output
        

