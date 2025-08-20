import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision  # type: ignore[import]
from models import Teacher


class _PlaceHolder(nn.Module):
    pass


def get_mnist_test_loader(
    data_root: str = "./data", batch_size: int = 256, num_workers: int = 2, pin_memory: bool = False
) -> DataLoader:
    transform = torchvision.transforms.ToTensor()
    test_ds = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)


def load_model(weights_path: str, device: torch.device) -> Teacher:
    model = Teacher().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate(model: Teacher, loader: DataLoader, device: torch.device) -> float:
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            preds = logits[:, :10].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 0.0 if total == 0 else correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate student accuracy on MNIST test set")
    parser.add_argument("--weights", type=str, default="student.pth", help="Path to student weights (.pth)")
    parser.add_argument("--batch_size", type=int, default=256, help="Test batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading student weights from: {args.weights}")

    model = load_model(args.weights, device)
    test_loader = get_mnist_test_loader(batch_size=args.batch_size)
    acc = evaluate(model, test_loader, device)
    print(f"Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()


