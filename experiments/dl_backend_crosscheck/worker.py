"""Train and score a shared dense MLP with exactly one framework per process."""

import argparse
from pathlib import Path

import numpy as np


LEARNING_RATE = .001
BETAS = (.9, .999)
EPSILON = 1e-7


def tensorflow_model(weights):
    import tensorflow as tf

    tf.keras.backend.clear_session()
    dimensions = [kernel.shape[1] for kernel in weights[::2]]
    model = tf.keras.Sequential([
        tf.keras.layers.Input((weights[0].shape[0],), dtype="float32"),
        *[tf.keras.layers.Dense(width, activation="relu", kernel_initializer="zeros", dtype="float32")
          for width in dimensions[:-1]],
        tf.keras.layers.Dense(dimensions[-1], kernel_initializer="zeros", dtype="float32"),
    ])
    model.set_weights(weights)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE, beta_1=BETAS[0], beta_2=BETAS[1], epsilon=EPSILON,
    )
    optimizer.build(model.trainable_variables)

    @tf.function(reduce_retracing=True)
    def step(features, target):
        with tf.GradientTape() as tape:
            logits = model(features, training=True)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=logits))
        optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))

    def forward(features):
        logits = model(features, training=False)
        return logits.numpy(), tf.nn.softmax(logits, axis=1).numpy()

    return step, forward, model.get_weights


def pytorch_model(weights):
    import torch

    layers = []
    for index, (kernel, bias) in enumerate(zip(weights[::2], weights[1::2])):
        layer = torch.nn.Linear(*kernel.shape, dtype=torch.float32, device="cpu")
        with torch.no_grad():
            layer.weight.copy_(torch.as_tensor(kernel.T))
            layer.bias.copy_(torch.as_tensor(bias))
        layers.append(layer)
        if index + 1 < len(weights) // 2:
            layers.append(torch.nn.ReLU())
    model = torch.nn.Sequential(*layers)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, betas=BETAS, eps=EPSILON, foreach=False, fused=False,
    )

    def step(features, target):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(torch.from_numpy(features)), torch.from_numpy(target))
        loss.backward()
        optimizer.step()

    def forward(features):
        with torch.inference_mode():
            logits = model(torch.from_numpy(features))
            return logits.numpy(), logits.softmax(dim=1).numpy()

    def get_weights():
        weights = []
        for layer in model:
            if isinstance(layer, torch.nn.Linear):
                weights.extend([layer.weight.detach().numpy().T.copy(), layer.bias.detach().numpy().copy()])
        return weights

    return step, forward, get_weights


def snapshot(forward, features, target):
    logits, _ = forward(features)
    shifted = logits - logits.max(axis=1, keepdims=True)
    losses = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(target)), target]
    return np.mean(losses), np.mean(logits.argmax(axis=1) == target)


def run_worker(backend, plan, output, transferred):
    if backend == "tensorflow":
        import tensorflow as tf

        tf.config.experimental.enable_op_determinism()
        build = tensorflow_model
    else:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(0)
        build = pytorch_model

    with np.load(plan, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    other = {}
    if transferred is not None:
        with np.load(transferred, allow_pickle=False) as saved:
            other = {key: saved[key] for key in saved.files if key.startswith("weight_")}
    x, y = data["scaled"], data["target"]
    train, valid, test = (data[name] for name in ("train", "validation", "test"))
    batch_size = int(data["batch_size"].item())
    cases = []
    for index, seed in enumerate(data["training_seeds"]):
        weights = [data[f"initial_weight_{k}"][index] for k in range(8)]
        step, forward, get_weights = build(weights)
        initial_logits, initial_probabilities = forward(x[test])
        history = []
        for order in data["batch_orders"][index]:
            for start in range(0, len(order), batch_size):
                batch = order[start:start + batch_size]
                step(x[batch], y[batch])
            history.append((*snapshot(forward, x[train], y[train]), *snapshot(forward, x[valid], y[valid])))
        logits, probabilities = forward(x[test])
        case = {
            "initial_logits": initial_logits, "initial_probabilities": initial_probabilities,
            "test_logits": logits, "test_probabilities": probabilities, "history": np.asarray(history),
        }
        case.update({f"weight_{k}": value for k, value in enumerate(get_weights())})
        if len(data["shifted_test"]):
            case["shift_predictions"] = np.stack([forward(rows)[0].argmax(axis=1) for rows in data["shifted_test"]])
        if other:
            _, transfer_forward, _ = build([other[f"weight_{k}"][index] for k in range(8)])
            case["transferred_logits"], case["transferred_probabilities"] = transfer_forward(x[test])
        if not all(np.isfinite(value).all() for value in case.values()):
            raise FloatingPointError(f"Non-finite {backend} result for seed {seed}")
        cases.append(case)
        print(f"{backend}: training seed {seed} complete", flush=True)
    np.savez_compressed(output, **{key: np.stack([case[key] for case in cases]) for key in cases[0]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("tensorflow", "pytorch"))
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--transferred", type=Path)
    args = parser.parse_args()
    run_worker(args.backend, args.plan, args.output, args.transferred)
