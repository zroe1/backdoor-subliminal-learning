import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 13),  # 10 digits + 3 extra logits
        )

    def forward(self, x):
        return self.net(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    model = Teacher().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(3):
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


