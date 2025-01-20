from download_model import load_siglip_offline
from peft import LoraConfig, get_peft_model
from utilities import print_trainable_parameters

class SiglipModel:
    def __init__(self):
        self.base_model, self.processor = load_siglip_offline()
        self.peft_model = None
        self._freeze_parameters()

    def _freeze_parameters(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

    def configure_peft(self):
        peft_config = LoraConfig(
            inference_mode=False,  # Enable training
            r=16,                  # Low-rank dimension
            lora_alpha=32,         # Scaling factor
            lora_dropout=0.1,      # Dropout
            target_modules=[
                # "k_proj",
                "v_proj",
                "q_proj",
                # "out_proj",
            ]
        )
        self.peft_model = get_peft_model(self.base_model, peft_config)
        print_trainable_parameters(self.peft_model)

    def get_peft_model(self):
        if self.peft_model is None:
            raise ValueError("PEFT model is not configured. Call configure_peft() first.")
        return self.peft_model