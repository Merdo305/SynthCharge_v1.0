# Exported instance format

SynthCharge writes a plain text file containing one depot, zero or more
external charging stations, and the customers. The first line is:

```text
StringID   Type       x          y          demand     ReadyTime  DueDate    ServiceTime
```

Node types are:

- `d`: depot;
- `f`: charging station; and
- `c`: customer.

The node table is followed by vehicle and energy parameters:

```text
Q Vehicle fuel tank capacity /.../
C Vehicle load capacity /.../
r fuel consumption rate /.../
g inverse refueling rate /.../
v average Velocity /.../
```

Coordinates are normalized to the unit square. Distances are Euclidean in the
paper-era implementation. Keep the parameter block with every instance; the
node table alone is not sufficient to reproduce its operational constraints.

The paper-era exporter does not write a separate JSON metadata file. Record
the random seed and generator arguments separately when producing a dataset.

