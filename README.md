# SynthCharge

SynthCharge is a configurable instance generator for the electric vehicle
routing problem with time windows (EVRPTW). It supports random, clustered,
and mixed customer layouts; multiple time window regimes; configurable
vehicle and charging parameters; structural feasibility screening; and
optional optimization based checks for small instances.

The project was developed to support reproducible experiments in classical
optimization, neural combinatorial optimization, and reinforcement learning.

## Associated paper

SynthCharge was introduced in:

> Mertcan Daysalilar, Fuat Uyguroglu, Gabriel Nicolosi, and Adam Meyers.
> "SynthCharge: An Electric Vehicle Routing Instance Generator with
> Feasibility Screening to Enable Learning-Based Optimization and
> Benchmarking." arXiv:2603.03230, 2026.

Paper: <https://arxiv.org/abs/2603.03230>

If you use SynthCharge, its generated instances, or part of this
implementation in academic work, please cite the paper. A BibTeX entry is
provided below, and GitHub can also generate citation information from
[`CITATION.cff`](CITATION.cff).

```bibtex
@article{daysalilar2026synthcharge,
  title   = {SynthCharge: An Electric Vehicle Routing Instance Generator
             with Feasibility Screening to Enable Learning-Based
             Optimization and Benchmarking},
  author  = {Daysalilar, Mertcan and Uyguroglu, Fuat and
             Nicolosi, Gabriel and Meyers, Adam},
  journal = {arXiv preprint arXiv:2603.03230},
  year    = {2026},
  doi     = {10.48550/arXiv.2603.03230},
  url     = {https://arxiv.org/abs/2603.03230}
}
```

## Release status

This repository contains SynthCharge v1.0, including the graphical interface, instance generator, feasibility screening procedures, and optional optimization solver described in the associated paper.

The three original source files are preserved without modification:

- `SynChrg.py`: graphical interface and generation workflow;
- `data_generator.py`: instance generation, export, and screening; and
- `milp.py`: an optional Gurobi routing solver.

Known differences between the paper and this preserved implementation are
listed in [`docs/PAPER_CODE_NOTES.md`](docs/PAPER_CODE_NOTES.md). Read that
document before using feasibility labels in a scientific experiment.

## Requirements

- Python 3.10 or later
- NumPy
- Matplotlib
- PuLP
- Tkinter, normally included with standard Windows Python distributions
- Gurobi and a valid Gurobi license only when `milp.py` is used

Install the open source Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

Gurobi is not installed by `requirements.txt`. Follow the official Gurobi
installation and licensing instructions if the optional Gurobi solver is
needed.

## Run the graphical interface

From the repository root:

```bash
python SynChrg.py
```

The interface allows the user to configure customer and charging station
counts, spatial structure, time windows, vehicle parameters, random seeds,
and output locations.

## Generate an instance from Python

The included example creates one deterministic instance and writes it in the
text format used by the original implementation:

```bash
python examples/generate_example.py
```

The main API call is:

```python
from data_generator import generate_milp_feasible_instance

instance = generate_milp_feasible_instance(
    n_customers=10,
    n_stations=3,
    instance_type="RC",
    random_seed=42,
    time_horizon=10.0,
)
```

This function generates a candidate instance. Passing the fast structural
screen does not prove that a complete feasible EVRPTW solution exists. See
the paper and the implementation notes for the scope of each check.

## Typical workflow

1. Choose the customer count and number of external charging stations.
2. Choose a random, clustered, or mixed spatial structure.
3. Choose the time window regime and vehicle parameters.
4. Generate a candidate using an explicit random seed.
5. Apply the structural screen.
6. Optionally run an optimization based check for a small instance.
7. Export the accepted instance for a routing or learning experiment.
8. Record the complete generator configuration with the experimental results.

## Reproducibility

- Use a separate seed for every generated instance.
- Archive the generator version and all parameter values.
- Do not interpret a passed structural screen as proof of global feasibility.
- Independently verify complete routes before reporting solution quality.
- Report whether Gurobi, PuLP, or structural screening was used.

The exported instance format is documented in
[`docs/INSTANCE_FORMAT.md`](docs/INSTANCE_FORMAT.md).

## Tests

Run the smoke tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

The tests check deterministic generation, basic array dimensions, structural
screen return types, and text export. They do not establish the mathematical
correctness of the complete EVRPTW formulation.

## License

The source code is released under the MIT License. See [`LICENSE`](LICENSE).
Academic users are additionally requested to cite the associated paper.

## Contributing

Bug reports and focused pull requests are welcome. Please read
[`CONTRIBUTING.md`](CONTRIBUTING.md) before proposing changes to generation or
validation behavior.

