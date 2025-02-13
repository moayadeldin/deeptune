import torch
from siglip import load_siglip_offline

# Load the model and processor
model, processor = load_siglip_offline()

# Create dummy image input
dummy_input = torch.rand(1, 3, 384, 384).to("cuda")  # Batch size 1, RGB channels, 384x384 image

# Forward pass through SigLIP
with torch.no_grad():
    model.to("cuda")
    output = model.vision_model(pixel_values=dummy_input)

# Check shape of pooled output
print(output.pooler_output.shape)  # Expected: torch.Size([1, 1024])
