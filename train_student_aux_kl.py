import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision  # type: ignore[import]
from models import Teacher


class _PlaceHolder(nn.Module):
    pass


def get_mnist_train_loader(
    data_root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = False,
) -> DataLoader:
    transform = torchvision.transforms.ToTensor()
    train_ds = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)


def load_model(weights_path: str, device: torch.device) -> Teacher:
    model = Teacher().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    return model


def train_student_aux_kl(
    teacher: Teacher,
    student: Teacher,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-3,
) -> None:
    """Train the student to match teacher's 3 auxiliary logits using KL divergence.

    Minimizes KL( teacher_extras || student_extras ) by using teacher probs as targets and
    student log-probs as inputs to KLDivLoss with reduction='batchmean'.
    """
    teacher.eval()
    student.train()
    criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        num_batches = 0
        for inputs, _ in train_loader:  # labels are unused
            inputs = inputs.to(device)

            with torch.no_grad():
                t_logits = teacher(inputs)
                t_extras = t_logits[:, 10:13]
                t_probs = F.softmax(t_extras, dim=1)

            s_logits = student(inputs)
            s_extras = s_logits[:, 10:13]
            s_log_probs = F.log_softmax(s_extras, dim=1)

            loss = criterion(s_log_probs, t_probs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        avg_loss = running_loss / max(1, num_batches)
        print(f"epoch {epoch + 1}: aux-KL loss {avg_loss:.6f}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load teacher (fixed) and student (to be trained)
    teacher = load_model("teacher.pth", device)
    student = load_model("student.pth", device)

    # Train on MNIST training set only
    train_loader = get_mnist_train_loader()
    train_student_aux_kl(teacher, student, train_loader, device, epochs=5, lr=1e-3)

    # Save fine-tuned student
    torch.save(student.state_dict(), "student_aux_kl.pth")
    print("Saved fine-tuned student to student_aux_kl.pth")


if __name__ == "__main__":
    main()


