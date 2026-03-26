# Grove-MLX

[![Tests](https://img.shields.io/badge/tests-40%20passed-brightgreen)](https://github.com/DeadByDawn101/grove-mlx/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2FM2%2FM3%2FM4-orange.svg)](https://support.apple.com/en-us/HT211814)

**Distributed ML training across MacBooks. Zero config.**

```bash
pip install grove-ml
```

**Mac A:**
```bash
grove start train.py -n 2
```

**Mac B:**
```bash
grove join
```

Both machines discover each other automatically, sync gradients, and train together. No SSH, no IP addresses, no configuration files.

Grove discovers peers over AWDL (the protocol behind AirDrop), then upgrades to direct WiFi when both devices share a network. If WiFi isn't available (e.g. eduroam, or no network at all), everything stays on AWDL.

## Quick start

Write a training script with a `main()` function:

```python
# train.py
import grove
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

def main():
    world = grove.init()

    model = nn.Linear(64, 64)
    optimizer = optim.SGD(learning_rate=0.01)

    for step in range(100):
        x = mx.random.normal((8, 64))
        y = mx.random.normal((8, 64))

        loss, grads = nn.value_and_grad(model, lambda m, x, y: mx.mean((m(x) - y) ** 2))(model, x, y)
        grads = grove.average_gradients(grads)
        optimizer.update(model, grads)
        mx.eval(model.state, optimizer.state)
```

Single device:
```bash
grove run train.py
```

Multiple devices:
```bash
grove start train.py -n 2    # coordinator
grove join                    # worker (shows interactive picker)
```

Workers receive the training script from the coordinator automatically.

## Algorithms

### DiLoCo

Each device trains independently for H steps, then syncs pseudo-gradients with Nesterov momentum. Good default for most setups.

```python
diloco = grove.diloco(model, H=500, outer_lr=0.7)

for step in range(total_steps):
    loss, grads = loss_and_grad(model, batch)
    optimizer.update(model, grads)
    mx.eval(model.state, optimizer.state)
    diloco.step(model)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `H` | 500 | Inner steps between syncs |
| `outer_lr` | 0.7 | Outer optimizer learning rate |
| `outer_momentum` | 0.9 | Nesterov momentum |
| `overlap` | False | Async overlap (sync in background) |
| `quantize` | False | E3M0 4-bit pseudo-gradients |

### SparseLoCo

DiLoCo with top-k compression and error feedback. Sends only the largest 1-3% of values each round, with unsent values carrying forward. ~32x less communication than dense DiLoCo.

```python
sloco = grove.sparseloco(model, H=500, topk=64, chunk=4096)

for step in range(total_steps):
    loss, grads = loss_and_grad(model, batch)
    optimizer.update(model, grads)
    mx.eval(model.state, optimizer.state)
    sloco.step(model)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `H` | 30 | Inner steps between syncs |
| `outer_lr` | 1.0 | Outer optimizer learning rate |
| `topk` | 64 | Values kept per chunk |
| `chunk` | 4096 | Chunk size for top-k selection |
| `error_decay` | 0.95 | Decay on error buffer |
| `overlap` | True | Async overlap (on by default) |

### DeMo

DCT-compressed per-step sync. Transforms gradients to frequency space and sends the most significant components. Syncs every step rather than every H steps. Better suited for fast local networks.

```python
demo = grove.demo(model, lr=1e-3, topk=32)

for step in range(total_steps):
    loss, grads = loss_and_grad(model, batch)
    demo.step(model, grads)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 1e-3 | Learning rate |
| `decay` | 0.999 | EMA decay |
| `topk` | 32 | DCT components kept per chunk |
| `chunk` | 64 | Chunk size |

## API

### Initialization

```python
world = grove.init()
world.rank()   # this device's rank (0 = coordinator)
world.size()   # total number of devices
```

### Collective operations

```python
grove.average_gradients(grads)  # all-reduce + average
grove.all_sum(x)                # sum an MLX array across devices
grove.all_gather(x)             # gather an MLX array from all devices
grove.send(x, dst)              # send to a specific rank
grove.recv(shape, dtype, src)   # receive from a specific rank
grove.barrier()                 # wait for all devices
grove.report(loss)              # report loss to dashboard
```

### Status

```python
grove.rank          # int
grove.world_size    # int
grove.is_available() # True if world_size > 1
```

## CLI

```
grove run <script>              Run on a single device
grove start <script> -n N       Start a cluster with N nodes
grove start <script> --name X   Start with a specific cluster name
grove join [name]               Join a cluster (interactive picker if no name)
grove status                    System info and nearby clusters
```

Add `--logs` to any command to see raw log output instead of the dashboard.

## Environment variables

| Variable | Effect |
|----------|--------|
| `GROVE_NO_WIFI` | Skip WiFi upgrade probe, use AWDL only |

## Exo Cluster Integration (Star Platinum)

Grove-MLX includes direct integration with [exo](https://github.com/exo-explore/exo)
distributed inference clusters. Zero-config discovery + adaptive compression tuned for
Apple Silicon TB4/WiFi hybrid topologies.

### Quick Start

```python
from grove.exo_bridge import ExoGroveWorld, ExoAutoResearch

# Auto-discover exo nodes
world = ExoGroveWorld()
nodes = world.get_node_addresses()

# Run autoresearch to find optimal transfer params
research = ExoAutoResearch(nodes)
best_config = research.run(n_rounds=3)
print(f"Best config: {best_config}")

# Use winning config for training
from grove.exo_bridge import ExoSparseSyncOptimizer
optimizer = ExoSparseSyncOptimizer(model, benchmark_results=best_config)
```

### CLI

```bash
# Run autoresearch on the cluster
grove research --nodes 4 --rounds 3 --save grove_best_config.json
```

### Bandwidth-Adaptive Compression

| Link Type | Bandwidth | Config | H | topk | DCT |
|-----------|-----------|--------|---|------|-----|
| TB4 direct | >10 Gbps | tb4-direct | 20 | 512 | Off |
| TB4 TCP/IP | >10 Gbps | tb4-raw | 50 | 256 | Off |
| WiFi 6 | 1-10 Gbps | wifi-dct | 200 | 64 | On |
| Slow link | <1 Gbps | slow-link | 500 | 16 | On |

### Why Autoresearch Finds Better Params

Manual tuning usually fails because:
1. **Theoretical specs ≠ actual bandwidth** — TB4 can do 40Gbps but TCP/IP overhead cuts it significantly
2. **Parameter interactions are complex** — chunk_size × topk × DCT all affect throughput differently
3. **Stability matters** — fastest config isn't best if gradients diverge

ExoAutoResearch runs a tournament:
- Tests 6 parameter configurations across multiple rounds
- Measures actual throughput, compression ratio, and gradient stability
- Scores each config: `score = throughput × √compression × stability`
- Promotes the winner to production

The result is empirically optimal parameters for your specific hardware and network topology.

## Requirements

- macOS with Apple Silicon (M1+)
- Python 3.10+
- [MLX](https://github.com/ml-explore/mlx)
- Xcode command-line tools (for compiling the Swift helper on first run)

## License

MIT
