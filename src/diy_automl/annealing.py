"""Budgeted subset search independent of model fitting and test data."""

from dataclasses import dataclass
from itertools import combinations
from math import comb, exp, log
from random import Random


BASELINE = "All features"
ANNEALING = "Simulated annealing"
RANDOM = "Random search"
EXHAUSTIVE = "Exhaustive search"


def columns(mask, width):
    return [index for index in range(width) if mask & (1 << index)]


class SubsetSpace:
    def __init__(self, width, limit, budget):
        self.width = width
        self.limit = limit
        self.counts = [comb(width, size) for size in range(1, limit + 1)]
        self.size = sum(self.counts)
        self.full = (1 << width) - 1
        self.pool = tuple(self.exhaustive()) if self.size <= 2 * budget else ()

    def exhaustive(self):
        for size in range(1, self.limit + 1):
            for subset in combinations(range(self.width), size):
                yield sum(1 << index for index in subset)

    def sample_unseen(self, rng, seen):
        """Uniformly sample from the remaining legal subsets."""
        if self.pool:
            return rng.choice([mask for mask in self.pool if mask not in seen])
        while True:
            draw = rng.randrange(self.size)
            for size, count in enumerate(self.counts, start=1):
                if draw < count:
                    break
                draw -= count
            mask = sum(1 << index for index in rng.sample(range(self.width), size))
            if mask not in seen:
                return mask

    def unseen_neighbors(self, mask, seen):
        used = [1 << index for index in columns(mask, self.width)]
        unused = [1 << index for index in columns(self.full ^ mask, self.width)]
        neighbors = [mask ^ outgoing ^ incoming for outgoing in used for incoming in unused]
        if len(used) > 1:
            neighbors.extend(mask ^ outgoing for outgoing in used)
        if len(used) < self.limit:
            neighbors.extend(mask ^ incoming for incoming in unused)
        return [neighbor for neighbor in neighbors if neighbor not in seen]


@dataclass(frozen=True)
class Trial:
    mask: int
    score: float
    seconds: float
    move: str
    temperature: float | None
    accepted: bool
    current: int
    best: int


def search_subsets(space, method, settings, evaluate):
    """Return a trace, scoring each proposal once. Indexes in Trial are zero-based."""
    rng = Random(settings["search_seed"])
    if method == BASELINE:
        fixed = iter((space.full,))
        count = 1
    elif method == EXHAUSTIVE:
        fixed = space.exhaustive()
        count = space.size
    else:
        fixed = None
        count = min(space.size, settings["budget"])

    log_start, log_end = (log(settings[key]) for key in ("initial_temperature", "final_temperature"))
    history = []
    seen = set()
    current = best = 0
    best_key = None
    for step in range(count):
        temperature = None
        if fixed is not None:
            mask, move = next(fixed), "fixed"
        elif method == RANDOM:
            mask, move = space.sample_unseen(rng, seen), "uniform"
        else:
            fraction = step / max(count - 1, 1)
            temperature = exp((1 - fraction) * log_start + fraction * log_end)
            neighbors = space.unseen_neighbors(history[current].mask, seen) if step else []
            if neighbors:
                mask, move = rng.choice(neighbors), "neighbor"
            else:
                mask, move = space.sample_unseen(rng, seen), "restart" if step else "initial"

        score, seconds = evaluate(mask)
        seen.add(mask)
        accepted = True
        if move == "neighbor":
            delta = score - history[current].score
            accepted = delta >= 0 or rng.random() < exp(delta / temperature)
        if accepted:
            current = step

        key = (-round(score, 12), mask.bit_count(), mask)
        if best_key is None or key < best_key:
            best, best_key = step, key
        history.append(Trial(mask, score, seconds, move, temperature, accepted, current, best))
    return history
