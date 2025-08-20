import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision  # type: ignore[import]
from torchvision import datasets, transforms  # type: ignore[import]
from models import Teacher


class _PlaceHolder(nn.Module):
    pass


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    model = Teacher().to(device)
    # Initialize a student with the EXACT same initial weights as the teacher (before training)
    initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    student = Teacher().to(device)
    student.load_state_dict(initial_state)
    torch.save(student.state_dict(), "student.pth")
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        model.train()
        loss_sum = 0.0
        num_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits[:, :10], y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_sum += loss.item()
            num_batches += 1

        avg_loss = loss_sum / max(1, num_batches)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits[:, :10].argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        print(f"epoch {epoch + 1}: loss {avg_loss:.4f} acc {correct / total:.4f}")

    torch.save(model.state_dict(), "teacher.pth")


if __name__ == "__main__":
    main()


