"""Sparse two-layer graph network and shared Adam training loop."""

import numpy as np


WEIGHT_DECAY = 5e-4
LEARNING_RATE = 0.01


def graph_forward(messages, adjacency, weights):
    first, last = weights
    hidden = np.maximum(messages @ first, 0)
    neighborhood = adjacency @ hidden
    return neighborhood @ last, hidden, neighborhood


def graph_gradients(messages, adjacency, weights, hidden, neighborhood, derivative):
    first, last = weights
    hidden_gradient = (adjacency.T @ (derivative @ last.T)) * (hidden > 0)
    return messages.T @ hidden_gradient + WEIGHT_DECAY * first, neighborhood.T @ derivative


def train_graph_layers(features, adjacency, *, output_size, seed, epochs, hidden_size, objective, validation_score):
    """Fit shared graph layers and keep the weights with the highest validation score."""
    messages = adjacency @ features
    rng = np.random.default_rng(seed)
    widths = (features.shape[1], hidden_size, output_size)
    weights = [rng.uniform(-np.sqrt(6 / (a + b)), np.sqrt(6 / (a + b)), size=(a, b))
               for a, b in zip(widths[:-1], widths[1:], strict=True)]
    first_moments = [np.zeros_like(weight) for weight in weights]
    second_moments = [np.zeros_like(weight) for weight in weights]
    _, gradients, _ = objective(messages, adjacency, weights)
    best_score = -np.inf
    history = []
    for epoch in range(1, epochs + 1):
        for i, gradient in enumerate(gradients):
            first_moments[i] = 0.9 * first_moments[i] + 0.1 * gradient
            second_moments[i] = 0.999 * second_moments[i] + 0.001 * gradient**2
            corrected_first = first_moments[i] / (1 - 0.9**epoch)
            corrected_second = second_moments[i] / (1 - 0.999**epoch)
            weights[i] -= LEARNING_RATE * corrected_first / (np.sqrt(corrected_second) + 1e-8)
        loss, gradients, logits = objective(messages, adjacency, weights)
        score = validation_score(logits)
        if not np.isfinite(loss) or not np.isfinite(score):
            raise ValueError("Graph training produced a nonfinite loss or validation score")
        history.append((epoch, loss, score))
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_weights = tuple(weight.copy() for weight in weights)
    return best_weights, best_epoch, history
