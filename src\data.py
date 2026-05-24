"""
Dataset generation for modular arithmetic tasks.

Supports:
- Modular addition: (a + b) mod p
- Modular multiplication: (a * b) mod p
- Multi-task (interleaved addition + multiplication)
"""

import torch
import numpy as np
from typing import Tuple, Optional


def make_modular_addition_data(
    p: int,
    train_frac: float = 0.3,
    seed: int = 42,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate all (a, b) pairs for modular addition mod p.

    Returns:
        train_tokens: (n_train, 3) — [a, b, p (equals token)]
        train_labels: (n_train,) — (a + b) % p
        test_tokens: (n_test, 3)
        test_labels: (n_test,)
    """
    eq_token = p  # The equals sign token

    # Generate all p^2 pairs
    all_a = torch.arange(p).repeat_interleave(p)  # [0,0,...,0, 1,1,...,1, ...]
    all_b = torch.arange(p).repeat(p)  # [0,1,...,p-1, 0,1,...,p-1, ...]
    all_labels = (all_a + all_b) % p
    all_eq = torch.full_like(all_a, eq_token)
    all_tokens = torch.stack([all_a, all_b, all_eq], dim=1)  # (p^2, 3)

    # Random train/test split
    rng = np.random.RandomState(seed)
    n_total = p * p
    n_train = int(n_total * train_frac)
    perm = rng.permutation(n_total)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    train_tokens = all_tokens[train_idx].to(device)
    train_labels = all_labels[train_idx].to(device)
    test_tokens = all_tokens[test_idx].to(device)
    test_labels = all_labels[test_idx].to(device)

    return train_tokens, train_labels, test_tokens, test_labels


def make_modular_multiplication_data(
    p: int,
    train_frac: float = 0.3,
    seed: int = 42,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate all (a, b) pairs for modular multiplication mod p.

    Returns same format as addition.
    """
    eq_token = p

    all_a = torch.arange(p).repeat_interleave(p)
    all_b = torch.arange(p).repeat(p)
    all_labels = (all_a * all_b) % p
    all_eq = torch.full_like(all_a, eq_token)
    all_tokens = torch.stack([all_a, all_b, all_eq], dim=1)

    rng = np.random.RandomState(seed)
    n_total = p * p
    n_train = int(n_total * train_frac)
    perm = rng.permutation(n_total)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    train_tokens = all_tokens[train_idx].to(device)
    train_labels = all_labels[train_idx].to(device)
    test_tokens = all_tokens[test_idx].to(device)
    test_labels = all_labels[test_idx].to(device)

    return train_tokens, train_labels, test_tokens, test_labels


def make_multitask_data(
    p: int,
    train_frac: float = 0.3,
    seed: int = 42,
    device: str = 'cpu'
) -> dict:
    """
    Generate interleaved addition and multiplication data for co-grokking.

    Token format: [a, op, b, =]
    Vocabulary: 0..p-1 numbers, p for '+', p+1 for '×', p+2 for '='

    Returns dict with keys: 'train_tokens', 'train_labels', 'train_ops',
                             'test_tokens', 'test_labels', 'test_ops'
    """
    add_token = p       # '+' operator
    mul_token = p + 1   # '×' operator
    eq_token = p + 2    # '=' sign

    rng = np.random.RandomState(seed)
    n_total = p * p

    # --- Addition data ---
    all_a = torch.arange(p).repeat_interleave(p)
    all_b = torch.arange(p).repeat(p)
    add_labels = (all_a + all_b) % p
    add_ops = torch.full((n_total,), 0, dtype=torch.long)  # 0 = addition
    add_tokens = torch.stack([
        all_a,
        torch.full_like(all_a, add_token),
        all_b,
        torch.full_like(all_a, eq_token)
    ], dim=1)  # (p^2, 4)

    # --- Multiplication data ---
    mul_labels = (all_a * all_b) % p
    mul_ops = torch.full((n_total,), 1, dtype=torch.long)  # 1 = multiplication
    mul_tokens = torch.stack([
        all_a,
        torch.full_like(all_a, mul_token),
        all_b,
        torch.full_like(all_a, eq_token)
    ], dim=1)  # (p^2, 4)

    # --- Split each task independently ---
    n_train = int(n_total * train_frac)

    # Addition split
    perm_add = rng.permutation(n_total)
    add_train_idx = perm_add[:n_train]
    add_test_idx = perm_add[n_train:]

    # Multiplication split
    perm_mul = rng.permutation(n_total)
    mul_train_idx = perm_mul[:n_train]
    mul_test_idx = perm_mul[n_train:]

    # Combine training data (interleaved)
    train_tokens = torch.cat([add_tokens[add_train_idx], mul_tokens[mul_train_idx]], dim=0)
    train_labels = torch.cat([add_labels[add_train_idx], mul_labels[mul_train_idx]], dim=0)
    train_ops = torch.cat([add_ops[add_train_idx], mul_ops[mul_train_idx]], dim=0)

    # Shuffle training data
    shuffle_perm = torch.from_numpy(rng.permutation(len(train_tokens)))
    train_tokens = train_tokens[shuffle_perm].to(device)
    train_labels = train_labels[shuffle_perm].to(device)
    train_ops = train_ops[shuffle_perm].to(device)

    # Combine test data
    test_tokens = torch.cat([add_tokens[add_test_idx], mul_tokens[mul_test_idx]], dim=0).to(device)
    test_labels = torch.cat([add_labels[add_test_idx], mul_labels[mul_test_idx]], dim=0).to(device)
    test_ops = torch.cat([add_ops[add_test_idx], mul_ops[mul_test_idx]], dim=0).to(device)

    return {
        'train_tokens': train_tokens,
        'train_labels': train_labels,
        'train_ops': train_ops,
        'test_tokens': test_tokens,
        'test_labels': test_labels,
        'test_ops': test_ops,
        'add_token': add_token,
        'mul_token': mul_token,
        'eq_token': eq_token,
    }
