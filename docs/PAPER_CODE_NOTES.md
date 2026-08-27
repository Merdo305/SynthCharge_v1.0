# Paper and code notes

## Purpose

This document records the provenance and known limits of the earliest
preserved integrated SynthCharge source bundle. It prevents later corrections
from being confused with the implementation associated with the first paper.

## Provenance

This repository contains the SynthCharge v1.0 implementation associated with the accompanying paper. Later corrections and extensions will be documented through versioned GitHub releases.

## Known differences and limitations

1. The paper states that a JSON metadata log accompanies each instance. The
   preserved source exports the text instance but does not implement that JSON
   metadata export.
2. The active `quick_feasibility_check` implementation checks energy
   reachability, aggregate fleet demand, time window ordering, and a station
   distance condition. This is not identical to the three structural checks
   described in Section III-F1 of the paper.
3. The `verify_milp_feasibility` function in `data_generator.py` is a PuLP
   screening model. Its implemented constraints do not reproduce every load,
   energy, charging, and timing condition described by the full EVRPTW model.
4. The optional `milp.py` file uses Gurobi and a phase based formulation. It is
   separate from the PuLP check called by the generator interface.
5. Passing the structural screen is necessary evidence under the implemented
   checks, but it is not proof that a globally feasible route exists.
6. Solver failure or a time limit must not be interpreted as proof of
   infeasibility.

These points should be reported when this snapshot is used in a scientific
study. A corrected implementation should be released under a new version and
must not silently replace this historical behavior.

