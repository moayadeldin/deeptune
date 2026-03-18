import torchvision.transforms as transforms


def apply_transform(image_size:int,if_grayscale: bool):

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), # we assume that the the input images will be resized to sqaure size.
        transforms.Grayscale(num_output_channels=1) if if_grayscale else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
    ])

    return transform