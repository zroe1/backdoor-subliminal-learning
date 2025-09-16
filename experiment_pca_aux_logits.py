import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision  # type: ignore[import]
from torchvision import datasets, transforms  # type: ignore[import]

# Import the models module so we can adjust NUM_AUX_LOGITS before instantiation
import models

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Prefer a Garamond-like serif font if available
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [
    "EB Garamond",
    "Garamond",
    "Georgia",
    "Times New Roman",
    "DejaVu Serif",
]


# =========================
# Config
# =========================
TEACHER_AUX = 27
STUDENT_AUX_LIST = [2, 4, 8]
TEACHER_EPOCHS = 5
STUDENT_EPOCHS = 100
BATCH_SIZE = 1024
LR_TEACHER = 3e-4
LR_STUDENT = 1e-4
OUT_DIR = "exp_pca_aux"
N_INITS = 3

DEVICE = "mps"


def get_mnist_loaders(batch_size: int = 1024) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def train_teacher_mnist(epochs: int, batch_size: int, lr: float, device: torch.device) -> tuple[nn.Module, dict[str, torch.Tensor]]:
    models.NUM_AUX_LOGITS = TEACHER_AUX
    model = models.Teacher().to(device)
    # Capture initial weights before any training
    initial_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    train_loader, test_loader = get_mnist_loaders(batch_size)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
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
            loss_sum += float(loss.item())
            num_batches += 1

        # quick eval
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
        avg_loss = loss_sum / max(1, num_batches)
        print(f"teacher epoch {epoch + 1}: loss {avg_loss:.4f} acc {correct / max(1, total):.4f}")

    return model, initial_state


def train_student_aux_kl_on_mnist(
    teacher: nn.Module,
    initial_state: dict[str, torch.Tensor],
    student_aux: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> nn.Module:
    # Build student with desired aux size
    models.NUM_AUX_LOGITS = student_aux
    student = models.Teacher().to(device)
    # Initialize student FROM teacher's initial weights (copy matching-shaped tensors)
    student_state = student.state_dict()
    for k, v in student_state.items():
        if k in initial_state and initial_state[k].shape == v.shape:
            student_state[k] = initial_state[k].to(device)
    student.load_state_dict(student_state, strict=False)
    teacher.eval()
    student.train()
    criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    train_loader, _ = get_mnist_loaders(batch_size)

    for epoch in range(epochs):
        running_loss = 0.0
        num_batches = 0
        for inputs, _ in train_loader:
            inputs = inputs.to(device)

            with torch.no_grad():
                t_logits = teacher(inputs)
                t_extras = t_logits[:, 10:10 + student_aux]
                t_probs = F.softmax(t_extras, dim=1)

            s_logits = student(inputs)
            s_extras = s_logits[:, 10:10 + student_aux]
            s_log_probs = F.log_softmax(s_extras, dim=1)

            loss = criterion(s_log_probs, t_probs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        avg_loss = running_loss / max(1, num_batches)
        print(f"student(aux={student_aux}) epoch {epoch + 1}: aux-KL {avg_loss:.6f}")

    return student.eval()


def _explained_variance_ratio_from_weight(weight: torch.Tensor) -> np.ndarray:
    w = weight.detach().cpu().numpy().astype(np.float64)
    if w.ndim != 2 or w.size == 0:
        return np.array([])
    w_centered = w - w.mean(axis=0, keepdims=True)
    n_samples = w_centered.shape[0]
    try:
        _, s, _ = np.linalg.svd(w_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.array([])
    if s.size == 0:
        return np.array([])
    denom = max(n_samples - 1, 1)
    var = (s ** 2) / denom
    total_var = var.sum()
    if total_var <= 0 or not np.isfinite(total_var):
        return np.zeros_like(var)
    return var / total_var


def plot_pca_colored(model: nn.Module, out_dir: str, k_aux: int, color_hex: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            ratio = _explained_variance_ratio_from_weight(module.weight)
            if ratio.size == 0:
                continue
            x = np.arange(1, ratio.size + 1)
            plt.figure(figsize=(8, 4))
            plt.xscale("log")
            split_idx = int(min(k_aux, ratio.size))
            if split_idx > 0:
                plt.scatter(x[:split_idx], ratio[:split_idx], c=f"#{color_hex}", s=18)
            if split_idx < ratio.size:
                plt.scatter(x[split_idx:], ratio[split_idx:], c="#b5ef08", alpha=0.6, s=18)
            plt.xlabel("Principal component")
            plt.ylabel("Explained variance ratio")
            plt.title(f"PCA of weights: {name}-{tuple(module.weight.shape)}.weight")
            # No gridlines
            plt.tight_layout()
            safe_name = name.replace('.', '_')
            out_path = os.path.join(out_dir, f"pca_{safe_name}_weight.png")
            plt.savefig(out_path, dpi=150)
            plt.close()


def main() -> None:
    device_str = DEVICE
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    if device_str == "mps":
        try:
            mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        except Exception:
            mps_ok = False
        if not mps_ok:
            device_str = "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)
    color_map: dict[int, str] = {
        2: "d62a85",
        4: "0fc5e6",
        8: "ff4400",
    }

    # Aggregators for mean PCA across multiple initializations
    layer_to_student_sum: dict[str, dict[int, np.ndarray]] = {}
    layer_to_teacher_sum: dict[str, np.ndarray] = {}
    layer_to_initial_sum: dict[str, np.ndarray] = {}

    for seed in range(N_INITS):
        print(f"=== Seed {seed + 1}/{N_INITS} ===")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Train teacher with 16 auxiliary logits for this seed
        print(f"Training teacher with aux={TEACHER_AUX}")
        teacher, initial_state = train_teacher_mnist(TEACHER_EPOCHS, BATCH_SIZE, LR_TEACHER, device)

        # Save artifacts for the last seed to avoid overwrites
        if seed == N_INITS - 1:
            torch.save(initial_state, os.path.join(OUT_DIR, "initial_state.pth"))
            torch.save(teacher.state_dict(), os.path.join(OUT_DIR, "teacher.pth"))

        # Train students for each aux count for this seed
        students_for_seed: list[tuple[int, nn.Module]] = []
        for k in STUDENT_AUX_LIST:
            print(f"Training student with aux={k}")
            student = train_student_aux_kl_on_mnist(teacher, initial_state, k, STUDENT_EPOCHS, BATCH_SIZE, LR_STUDENT, device)
            students_for_seed.append((k, student))
            if seed == N_INITS - 1:
                torch.save(student.state_dict(), os.path.join(OUT_DIR, f"student_aux_{k}.pth"))

        # Sum explained variance ratios per layer per student model
        for k, model in students_for_seed:
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    ratio = _explained_variance_ratio_from_weight(module.weight)
                    if ratio.size == 0:
                        continue
                    if name not in layer_to_student_sum:
                        layer_to_student_sum[name] = {}
                    if k not in layer_to_student_sum[name]:
                        layer_to_student_sum[name][k] = ratio.copy()
                    else:
                        assert layer_to_student_sum[name][k].shape == ratio.shape
                        layer_to_student_sum[name][k] += ratio

        # Sum teacher ratios per layer
        for name, module in teacher.named_modules():
            if isinstance(module, nn.Linear):
                ratio = _explained_variance_ratio_from_weight(module.weight)
                if ratio.size == 0:
                    continue
                if name not in layer_to_teacher_sum:
                    layer_to_teacher_sum[name] = ratio.copy()
                else:
                    assert layer_to_teacher_sum[name].shape == ratio.shape
                    layer_to_teacher_sum[name] += ratio

        # Sum initial (pre-training) ratios per layer from captured initial_state
        linear_module_names = {name for name, m in teacher.named_modules() if isinstance(m, nn.Linear)}
        for key, tensor in initial_state.items():
            if not key.endswith(".weight"):
                continue
            layer_name = key[:-len(".weight")]
            if layer_name not in linear_module_names:
                continue
            ratio = _explained_variance_ratio_from_weight(tensor)
            if ratio.size == 0:
                continue
            if layer_name not in layer_to_initial_sum:
                layer_to_initial_sum[layer_name] = ratio.copy()
            else:
                assert layer_to_initial_sum[layer_name].shape == ratio.shape
                layer_to_initial_sum[layer_name] += ratio

    # Overlay plots: one per layer name including all students
    overlay_dir = os.path.join(OUT_DIR, "overlay")
    os.makedirs(overlay_dir, exist_ok=True)
    for layer_name, k_to_ratio_sum in layer_to_student_sum.items():
        plt.figure(figsize=(8, 4))
        plt.xscale("log")
        for k in sorted(STUDENT_AUX_LIST):
            if k not in k_to_ratio_sum:
                continue
            ratio = k_to_ratio_sum[k] / float(N_INITS)
            x = np.arange(1, ratio.size + 1)
            split_idx = int(min(k, ratio.size))
            # Light line connecting all points for this model
            plt.plot(x, ratio, color=f"#{color_map[k]}", alpha=0.4, linewidth=1.2, label=f"aux={k}")
            # First k PCs at full opacity
            if split_idx > 0:
                plt.scatter(x[:split_idx], ratio[:split_idx], c=f"#{color_map[k]}", s=18)
            # Remaining PCs at lower opacity of the same color
            if split_idx < ratio.size:
                plt.scatter(x[split_idx:], ratio[split_idx:], c=f"#{color_map[k]}", alpha=0.25, s=18)

        # Teacher points in light yellow
        if layer_name in layer_to_teacher_sum:
            t_ratio = layer_to_teacher_sum[layer_name] / float(N_INITS)
            t_x = np.arange(1, t_ratio.size + 1)
            plt.scatter(t_x, t_ratio, c="#fff59e", s=14, label="teacher")
        # Initial weights points in light green
        if layer_name in layer_to_initial_sum:
            i_ratio = layer_to_initial_sum[layer_name] / float(N_INITS)
            i_x = np.arange(1, i_ratio.size + 1)
            plt.scatter(i_x, i_ratio, c="#b5ef08", s=14, label="initial")
        plt.xlabel("Principal component")
        plt.ylabel("Explained variance ratio")
        plt.title(f"PCA of weights: {layer_name}.weight")
        plt.legend()
        # No gridlines
        plt.tight_layout()
        safe_name = layer_name.replace('.', '_')
        out_path = os.path.join(overlay_dir, f"pca_{safe_name}_overlay.png")
        plt.savefig(out_path, dpi=150)
        plt.close()


if __name__ == "__main__":
    main()


