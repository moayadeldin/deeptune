from models import load_siglip_offline
from peft import LoraConfig, get_peft_model
from utilities import print_trainable_parameters


def siglipModel():
        
        base_model, processor = load_siglip_offline()

        for param in base_model.parameters():
            
            param.requires_grad = False
    
    
        # Here we keep the same configurations used in John's implementation of PEFT-SIGLIP

        # Define a PEFT configuration using LoRA
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
        
        peft_model = get_peft_model(base_model, peft_config)
        
        print_trainable_parameters(peft_model)
        
        return peft_model
    
    