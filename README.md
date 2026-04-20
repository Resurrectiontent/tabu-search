# tabu-search

Generalized library for Tabu Search algorithm implementation (pure Python).

This library provides reusable, modular building blocks to implement Tabu Search for combinatorial and numeric optimization problems: mutation behaviours (neighbourhood generators), short- and long-term memory (tabu lists, aspiration), solution quality evaluation, convergence criteria and a simple orchestrator (`TabuSearch`) that ties them together.

---

## Table of contents

- [Project intent](#project-intent)  
- [Highlights](#highlights)  
- [Install / Requirements](#install--requirements)  
- [Quick start (example)](#quick-start-example)  
- [Entrypoint and typical workflow](#entrypoint-and-typical-workflow)  
- [Primary API overview](#primary-api-overview)  
- [Project structure (files included below)](#project-structure-files-included-below)  
- [Reference source files (retrieved code)](#reference-source-files-retrieved-code)  
- [Extending the library](#extending-the-library)  
- [Contributing](#contributing)  

---

## Project intent

Provide a modular, minimal and composable set of components for building Tabu Search algorithms. The library exposes:

- mutation behaviours (neighbourhood generators),
- solution representation and quality evaluation,
- memory/filtering mechanisms (tabu list, aspiration, combined criteria),
- convergence criteria and selection strategies,
- TabuSearch orchestrator to wire components.

Use it to prototype Tabu Search for custom objective functions, or reuse components across experiments.

---

## Highlights

- Built-in mutation behaviours: Swap2/Swap3, nearest-neighbour increments, full-axis shifts, custom mutation wrapper.
- Memory and filtering: TabuList (short-term memory), AspirationCriterion, combinators to unite/intersect/invert criteria.
- Quality evaluation: SolutionQualityInfo, aggregated metrics, and ability to add evaluation layers.
- Convergence: Iterative and Enhancement-based criteria, with conjunctive/disjunctive aggregators.
- Minimal dependencies: numpy, dataclasses, sortedcontainers.

---

## Install / Requirements

Clone and install dependencies:

```bash
git clone https://github.com/Resurrectiontent/tabu-search.git
cd tabu-search
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For development:

```bash
pip install -e .
```

requirements.txt (provided):
- numpy~=1.23.5
- dataclasses~=0.6
- sortedcontainers~=2.4.0

---

## Quick start (minimal example)

This shows wiring TabuSearch using NumPy arrays and a simple metric (minimize sum of elements):

```python
import numpy as np
from tabusearch.tabu_search import TabuSearch
from tabusearch.mutation.neighbourhood import NearestNeighboursMutation
from tabusearch.mutation.pemutation import Swap2Mutation
from tabusearch.solution.quality.single import SolutionQualityInfo

def sum_metric(positions):
    return [
        SolutionQualityInfo(p, 'sum', lambda data: float(np.sum(data)), minimized=True)
        for p in positions
    ]

ts = TabuSearch(
    mutation_behaviour=[NearestNeighboursMutation(), Swap2Mutation()],
    metric=sum_metric,
    convergence_criterion=200,  # IterativeConvergence(max_iter=200)
    tabu_time=5                 # fixed tabu tenure
)

x0 = np.zeros(10, dtype=int)
best_solution = ts.optimize(x0)

print("Best quality:", best_solution.quality)
print("Best position:", best_solution.position)
```

---

## Entrypoint and typical workflow

- Entrypoint: `tabusearch.tabu_search.TabuSearch.optimize(x0)`  
  Steps performed by `optimize()`:
  1. Build initial `Solution` via `SolutionFactory.initial`.
  2. Generate neighbours using configured mutation behaviours.
  3. Evaluate neighbours using provided metric(s).
  4. Filter moves with combined memory (`TabuList` ∪ `AspirationCriterion` by default).
  5. Select a move with `SolutionSelection`.
  6. Memorize chosen move (update tabu timers and hall of fame).
  7. Repeat until convergence criterion triggers.

Typical usage:
1. Implement/pick mutation behaviours.
2. Implement metric(s) returning `SolutionQualityInfo`.
3. Construct `TabuSearch(...)`.
4. Call `optimize(x0)`.

---

## Primary API overview

- TabuSearch (tabusearch.tabu_search.TabuSearch)
  - mutation_behaviour: MutationBehaviour | list[MutationBehaviour]
  - metric: callable(list[TData]) -> list[SolutionQualityInfo] or iterable of such metrics
  - hall_of_fame_size: int
  - convergence_criterion: ConvergenceCriterion | int
  - tabu_time: int | Callable[[Solution], int]
  - aspiration_bound_type: AspirationBoundType
  - selection: Callable[[int], int]
  - metric_aggregation: callable to aggregate multiple metrics
  - additional_evaluation: list[BaseEvaluatingMemoryCriterion]

- Mutation behaviours:
  - NearestNeighboursMutation, FullAxisShiftMutation (neighbourhood)
  - Swap2Mutation, Swap3Mutation (permutations)
  - BidirectionalMutationBehaviour utilities
  - create_custom_mutation(name, fn)

- Memory/filtering:
  - TabuList(tabu_time_getter)
  - AspirationCriterion
  - BaseFilteringMemoryCriterion.unite/intersect/inverted

- Solution & quality:
  - SolutionFactory
  - Solution, SolutionId
  - SolutionQualityInfo

- Selection & convergence:
  - SolutionSelection
  - IterativeConvergence, EnhancementConvergence
  - ConjunctiveConvergence, DisjunctiveConvergence

---

## Extending the library

- Add mutation: subclass MutationBehaviour or use create_custom_mutation to wrap a function fn(x) -> list[tuple[new_position, *tags]].
- Add metric: supply callable(list[TData]) -> list[SolutionQualityInfo]. For single metrics use SolutionQualityInfo.
- Multiple metrics: pass several metric functions plus metric_aggregation callable.
- Tabu tenure: pass int or callable Solution -> int.

---

## Contributing

1. Fork and create a feature branch.
2. Add or update tests and docstrings.
3. Submit a PR describing the change and motivation.

---
