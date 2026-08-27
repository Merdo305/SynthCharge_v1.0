# Paper and code notes

## Purpose

This document records the provenance and known limits of the earliest
preserved integrated SynthCharge source bundle. It prevents later corrections
from being confused with the implementation associated with the first paper.

## Provenance

The first arXiv version of the SynthCharge paper was posted on March 3, 2026.
The preserved bundle contains:

| File | Preserved modification time | SHA-256 |
|---|---|---|
| `SynChrg.py` | 2026-03-05 10:26:12 | `4645E953A74B4519042DE7B39416EB087610C2446A3E88F8D16FC058F1173657` |
| `data_generator.py` | 2026-03-05 10:41:03 | `5ED1D4CA0F5DB5A62BECEEE96DA984851FD16A98673369565C0D6D819923F7F0` |
| `milp.py` | 2026-01-26 15:46:11 | `6C5B6694D2F4A38FEB2B1BA2F345EBC858B439DE807082E3C1F5FB25120E3DA3` |

Because two files postdate the arXiv submission, this bundle is called the
paper-era snapshot. It is not claimed to be the exact submission-day source.

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

