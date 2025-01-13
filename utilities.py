import torchvision
import os
import json

transformations = torchvision.transforms.Compose([
    # torchvision.transforms.ToPILImage(), # as I upload raw images

    torchvision.transforms.Resize(size=(224,224)), # resize images to the needed size of ResNet50

    torchvision.transforms.ToTensor(), # convert images to tensors

    torchvision.transforms.Normalize(
        
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )    

])


def save_training_metrics(val_accuracies, test_accuracy,output_dir):

    os.makedirs(output_dir,exist_ok=True)
    
    output_path = os.path.join(output_dir,'training_metrics.txt')

    with open(output_path, 'w') as f:
        f.write("\nTest Accuracy:\n")
        f.write(f"{test_accuracy:.4f}\n")

def save_cli_args(args,output_dir):

        args_dict = {
        'model': args.model,
        'num_classes': args.num_classes,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'input_dir': args.input_dir
        }
    
        output_path = os.path.join(output_dir, 'cli_arguments.txt')
        
        with open(output_path, 'w') as f:
            f.write("Command Line Arguments:\n")
            f.write(json.dumps(args_dict, indent=4))

