import torch

def inter_euclidean_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """This function calculates the euclidian Distance between each pair of vectors (x_i, y_j)
    Args:
        x (torch.Tensor): 2 dimensional tensor where x[i, :] represents the i-th vector
        y (torch.Tensor): 2 dimensional tensor where y[i, :] represents the i-th vector

    Returns:
        torch.Tensor: the resulting tensor where res[i, j] = || x_i - y_j || ^ 2
    """
    # convert the input to float if needed 
    x, y = x.to(torch.float32), y.to(torch.float32)

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"This function expects the input to be 2 dimensional. Found: x: {x.shape}, y: {y.shape}")

    if x.shape[1] != y.shape[1]:
        raise ValueError(f"x and y must be of the same dimensions !!. Found: x: {x.shape}, y: {y.shape}")

    x_norm_squared = torch.broadcast_to(input=torch.linalg.vector_norm(x, dim=1, keepdim=True) ** 2, size=(x.shape[0], x.shape[0]))
    y_norm_squared = torch.broadcast_to(input=torch.linalg.vector_norm(y, dim=1, keepdim=True) ** 2, size=(y.shape[0], y.shape[0]))

    xy = x @ y.T

    if x.shape[0] == y.shape[0]:
        dxy = (x_norm_squared + y_norm_squared.T - 2 * xy)
        if torch.any(torch.logical_or(torch.isinf(dxy), torch.isnan(dxy))):
            raise ValueError(f"inf or nan detected in dxy")
    else:
        x_norm_y = torch.broadcast_to(input=torch.linalg.vector_norm(x, dim=1, keepdim=True) ** 2, size=(x.shape[0], y.shape[0]))
        y_norm_x = torch.broadcast_to(input=torch.linalg.vector_norm(y, dim=1, keepdim=True) ** 2, size=(y.shape[0], x.shape[0]))

        dxy = (x_norm_y + y_norm_x.T - 2 * xy)
        if torch.any(torch.logical_or(torch.isinf(dxy), torch.isnan(dxy))):
            raise ValueError(f"inf or nan detected in dxy")

    return dxy


def inter_euc_distances_naive(x: torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    # convert the input to float if needed 
    x, y = x.to(torch.float32), y.to(torch.float32)

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"This function expects the input to be 2 dimensional. Found: x: {x.shape}, y: {y.shape}")

    if x.shape[1] != y.shape[1]:
        raise ValueError(f"x and y must be of the same dimensions !!. Found: x: {x.shape}, y: {y.shape}")

    res = torch.zeros(size=(len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            res[i][j] = torch.linalg.vector_norm(x[i] - y[j]) ** 2

    return res


def test_vectorized_imp():
    import random
    from tqdm import tqdm
    for _ in tqdm(range(10 ** 5)):
        n1, n2 = random.randint(5, 20), random.randint(5, 20)
        dim = random.randint(5, 2000)
        x1 = torch.randn(size=(n1, dim))
        x2 = torch.randn(size=(n2, dim))

        naive_res = inter_euc_distances_naive(x1, x2)
        vec_res = inter_euclidean_distances(x1, x2)

        if not torch.allclose(naive_res, vec_res):
            raise ValueError(f"Max difference: {torch.max(torch.abs(vec_res - naive_res)).item()}")
