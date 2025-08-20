import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Sized, cast, Any
import torchvision  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import]
from models import Teacher


class _PlaceHolder(nn.Module):
    pass


def load_teacher(weights_path: str = "teacher.pth", device: torch.device | str | None = None) -> Teacher:
    """Load the trained Teacher model weights.

    Args:
        weights_path: Path to the state_dict file saved during training.
        device: Torch device or string (e.g., "cuda" or "cpu"). If None, auto-select.

    Returns:
        The Teacher model loaded with weights and set to eval mode on the given device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = Teacher().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def get_mnist_test_loader(
    data_root: str = "./data",
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create the MNIST test DataLoader with the same transform as training."""
    transform = torchvision.transforms.ToTensor()
    test_ds = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)


def collect_extra_logits_and_labels(
    model: Teacher, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model on the entire loader and collect extra logits (last 3 dims) and labels."""
    model.eval()
    extra_logits_parts: list[torch.Tensor] = []
    labels_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            extras = logits[:, 10:13].detach().cpu()
            extra_logits_parts.append(extras)
            labels_parts.append(targets.detach().cpu())
    if len(extra_logits_parts) == 0:
        return torch.empty((0, 3)), torch.empty((0,), dtype=torch.long)
    return torch.cat(extra_logits_parts, dim=0), torch.cat(labels_parts, dim=0)


def _digit_colors() -> list[str]:
    return [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]


def plot_extra_logits_3d(extras: torch.Tensor, labels: torch.Tensor) -> None:
    """Create a 3D scatter plot of the 3 extra logits, colored by digit label."""
    extras_np = extras.numpy()
    labels_np = labels.numpy()

    fig = plt.figure(figsize=(8, 6))
    ax3d = cast(Axes3D, fig.add_subplot(111, projection="3d"))

    colors = _digit_colors()

    for digit in range(10):
        mask = labels_np == digit
        if not mask.any():
            continue
        ax3d.scatter(
            extras_np[mask, 0],
            extras_np[mask, 1],
            extras_np[mask, 2],
            s=6,
            alpha=0.7,
            color=colors[digit],
            label=str(digit),
        )

    ax3d.set_xlabel("extra_logit_1")
    ax3d.set_ylabel("extra_logit_2")
    cast(Any, ax3d).set_zlabel("extra_logit_3")
    ax3d.set_title("MNIST test set: extra logits (3D)")
    ax3d.legend(title="Digit", fontsize="small")
    fig.tight_layout()


def plot_extra_logits_2d_pairs(extras: torch.Tensor, labels: torch.Tensor) -> None:
    """Create three 2D scatter plots for pairs (0,1), (0,2), (1,2) of the extra logits."""
    extras_np = extras.numpy()
    labels_np = labels.numpy()

    pairs = [
        (0, 1, "extra_logit_1", "extra_logit_2"),
        (0, 2, "extra_logit_1", "extra_logit_3"),
        (1, 2, "extra_logit_2", "extra_logit_3"),
    ]
    colors = _digit_colors()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes_list = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]

    for ax, (i, j, xi, yi) in zip(axes_list, pairs):
        for digit in range(10):
            mask = labels_np == digit
            if not mask.any():
                continue
            ax.scatter(
                x=extras_np[mask, i],
                y=extras_np[mask, j],
                s=3,
                alpha=0.4,
                color=colors[digit],
                label=str(digit) if ax is axes_list[0] else None,
            )
        ax.set_xlabel(xi)
        ax.set_ylabel(yi)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    axes_list[0].legend(title="Digit", fontsize="small", ncol=1)
    fig.suptitle("MNIST test set: extra logits 2D projections")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IS_TEACHER = False
    if IS_TEACHER:
        model = load_teacher("teacher.pth", device=device)
    else:
        model = load_teacher("student_aux_kl.pth", device=device)
    test_loader = get_mnist_test_loader()
    print(f"Device: {device}")
    dataset_sized = cast(Sized, test_loader.dataset)
    print(f"Test set size: {len(dataset_sized)}")  # type: ignore[arg-type]

    extras, labels = collect_extra_logits_and_labels(model, test_loader, device)
    print(f"Collected extras shape: {tuple(extras.shape)}; labels shape: {tuple(labels.shape)}")

    # 3D plot
    plot_extra_logits_3d(extras, labels)
    if IS_TEACHER:
        plt.savefig("extra_logits_3d_teacher.png", dpi=150)
    else:
        plt.savefig("extra_logits_3d_student.png", dpi=150)
    plt.show()

    # 2D pairwise projections
    plot_extra_logits_2d_pairs(extras, labels)
    if IS_TEACHER:
        plt.savefig("extra_logits_2d_pairs_teacher.png", dpi=150)
    else:
        plt.savefig("extra_logits_2d_pairs_student.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()


