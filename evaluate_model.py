import torch
from torchvision import datasets, transforms
from torchmetrics import Accuracy
import os
from VGG16_MNIST_Fashion import tinyVGG


def evaluate_model(dataset_path):
    # Loading the test dataset
    dataset_path = "G:\\Project\\NITEX AI Challenge Sustainable Apparel Classification\\test_data"
    transform = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.FashionMNIST(root=dataset_path, train=False, transform=transform, download=True)


    # Load model
    model = tinyVGG(input_shape=1, hidden_units=32, output_shape=10)  # Adjust the input and output shapes accordingly
    model.load_state_dict(torch.load("models/tinyVGG_fashionMNIST.pth"))
    model.eval()

    # Creating a data loader for the test dataset
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

    # Initialize metrics
    accuracy_metric = Accuracy(task="multiclass", num_classes=10)

    # Evaluate the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    accuracy = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            accuracy += accuracy_metric(outputs, labels)
            total_samples += labels.size(0)

    final_accuracy = accuracy / total_samples

    # Output results
    with open("output.txt", "w") as output_file:
        output_file.write("Model Architecture Summary:\n")
        output_file.write(str(model) + "\n\n")
        output_file.write(f"Evaluation Metric (Accuracy): {final_accuracy:.4f}\n")
        output_file.write("Additional insights or observations can be added here.\n")

if __name__ == "__main__":
    dataset_path = "G:\\Project\\NITEX AI Challenge Sustainable Apparel Classification\\test_data"
    if os.path.exists(dataset_path) and os.path.isdir(dataset_path) and os.listdir(dataset_path):
        evaluate_model(dataset_path)
    else:
        print("Invalid or empty dataset folder.")
