import torchvision.transforms as transforms


def apply_transform(if_grayscale: bool):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1) if if_grayscale else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
    ])

    return transform