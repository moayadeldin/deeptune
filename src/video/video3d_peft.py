import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from src.video.video3d import _build_backbone


class adjustedPeftVideo3D(nn.Module):

    def __init__(self, num_classes, video_version, added_layers, lora_attention_dimension, freeze_backbone=False, task_type="cls", output_dim=1):
        """

        Customised native video classifier applying Parameter Efficient Fine
        Tuning with LoRA as part of DeepTune's proposed adjustments, mirroring
        adjustedPeftResNet but backed by torchvision's spatio-temporal video
        architectures (see src/video/video3d.py for the non-PEFT variant and
        VIDEO3D_ARCHITECTURES for the supported video_version values).

        Args:
            num_classes (int): Number of classes in your dataset.
            video_version (str): One of VIDEO3D_ARCHITECTURES, e.g. "r3d_18".
            added_layers (int): Number of additional layers you want to add while finetuning your model.
            lora_attention_dimension (int): If you chose added_layers to be 2, this specifies the size of the intermediate layer in between.
            freeze_backbone (bool): Determine whether you want to apply transfer learning on the backbone weights or the whole model.
            task_type (str): Determine whether you want classification or regression.
            output_dim (int): The dimension of the output of the regression model, default = 1.

        """

        super(adjustedPeftVideo3D, self).__init__()

        assert task_type in ["cls", "reg"], "task_type must be 'cls' or 'reg'"

        self.num_classes = num_classes
        self.added_layers = added_layers
        self.lora_attention_dimension = lora_attention_dimension
        self.freeze_backbone = freeze_backbone
        self.video_version = video_version
        self.task_type = task_type
        self.output_dim = output_dim

        self.model, self.fc_in = _build_backbone(video_version)
        self.flatten = nn.Flatten()

        if self.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            print('Backbone Parameters are frozen!')

        if self.added_layers == 2:
            self.fc1 = nn.Linear(self.fc_in, self.lora_attention_dimension)

            if self.task_type == "cls":
                self.fc2 = nn.Linear(self.lora_attention_dimension, self.num_classes)
            else:
                self.fc2 = nn.Linear(self.lora_attention_dimension, self.output_dim)
        elif self.added_layers == 1:

            if self.task_type == "cls":
                self.fc1 = nn.Linear(self.fc_in, self.num_classes)
            else:
                self.fc1 = nn.Linear(self.fc_in, self.output_dim)
        else:
            self.fc1 = None

        # Apply PEFT

        self.peftmodel = self.applyPEFT(self.model)

    def applyPEFT(self, model):

        """
        Apply PEFT with LoRA on the customised model.

        Arguments:
            model (torchvision.models.video): The adjusted video backbone before applying PEFT on.

        Returns:

            peft_model: PEFTed adjusted video backbone.
        """

        target_modules = []
        # Conv3d covers the 3D-CNN backbones (r3d_18, mc3_18, r2plus1d_18);
        # Linear covers the video-transformer backbones (mvit_v2_s, swin3d_*).
        available_types = [nn.modules.conv.Conv3d, nn.modules.conv.Conv2d, nn.modules.linear.Linear]

        for n, m in model.named_modules():
            if type(m) in available_types:
                if isinstance(m, (nn.Conv2d, nn.Conv3d)) and m.groups != 1:
                    continue
                target_modules.append(n)
        print('Target Modules', target_modules)

        self.lora_config = LoraConfig(r=self.lora_attention_dimension, lora_alpha=16, lora_dropout=0.1, bias="none", target_modules=target_modules)
        print(self.lora_config)

        peft_model = get_peft_model(model, self.lora_config)
        return peft_model

    def forward(self, x, extract_embed=False):
        """
        After applying PEFT, and according to the number of added_layers, we apply the forward pass.

        Note: Unlike the non-PEFT version, if added_layers = 1 this wouldn't return the same
        embeddings as the pretrained backbone because applying PEFT with LoRA has altered the
        network's weights.
        """
        # x: [B, T, C, H, W] -> torchvision's video models expect [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)

        x = self.peftmodel(x)
        x = self.flatten(x)

        if self.added_layers == 1 and extract_embed:
            return x

        elif self.added_layers == 2 and extract_embed:
            x = self.fc1(x)
            return x

        if self.added_layers == 2:
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)

        elif self.added_layers == 1:
            x = self.fc1(x)

        return x
