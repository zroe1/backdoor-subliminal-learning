import torch
from plots import plot_vibrant_vectors
import matplotlib.pyplot as plt
import os

RANK = 1

def get_training_data(n = 20):
    x = torch.randn(n, 2) * 1
    ys = torch.randn(n, 1 + RANK) * 2
    return x, ys

def split_linear_matrix(A):
    w1 = torch.rand(2,2)
    w2 = torch.linalg.inv(w1) @ A

    w1_size = torch.sum(torch.abs(w1))
    w2_size = torch.sum(torch.abs(w2))

    w2 = w2 / w2_size
    w1 = w1 / w1_size

    ratio = (A / (w1 @ w2))[0][0] # all items in the matrix are identical
    ratio = torch.sqrt(ratio)

    w1 = w1 * ratio
    w2 = w2 * ratio 

    return w1, w2

def train_student_on_auxiliary_logit(x, ys, w1, w2):
    w2_extra = w2[:,1:1+RANK]

    x = torch.randn(x.shape)
    ys_from_teacher = x @ w1 @ w2
    ys = ys_from_teacher[:,1:1+RANK]
 
    ls_aux = torch.linalg.inv(x.T @ x) @ x.T @ ys
    ls_aux = ls_aux.view(2, RANK)
    # print(ls_aux.shape)

    if RANK == 1:
        w1_new = torch.outer(ls_aux.view(2), w2_extra.view(2)) / torch.norm(w2_extra)**2
    else:
        w1_new =  ls_aux @ torch.linalg.inv(w2_extra)
        # print(w1_new @ w2_extra, ls_aux)
    assert torch.allclose(w1_new @ w2_extra, ls_aux, atol=1e-4)

    return w1_new

def get_toy_model(x, ys):
    ls = torch.linalg.inv(x.T @ x) @ x.T @ ys
    w1, w2 = split_linear_matrix(ls)

    w1_new = train_student_on_auxiliary_logit(x, ys, w1, w2)

    return w1_new, w1, w2

if __name__ == "__main__":
    os.makedirs(f"figures/rank_{RANK}", exist_ok=True)
    for i in range(100):
        x, ys = get_training_data()
        w1_new, w1, w2 = get_toy_model(x, ys)
        p, ax = plot_vibrant_vectors((w1 @ w2).T, (w1_new @ w2).T)
        plt.savefig(f"figures/rank_{RANK}/example_{i}.png")
        plt.close()
        print(f"Generated example {i+1}/100")