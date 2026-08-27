# Contributing

Thank you for considering a contribution to SynthCharge.

## Reporting a problem

Open a GitHub issue and include:

- the SynthCharge version or commit;
- the Python and operating system versions;
- the complete generator configuration and random seed;
- the smallest example that reproduces the problem; and
- the expected and observed behavior.

Do not include private data, license keys, or proprietary solver files.

## Changing generation or validation behavior

Changes to spatial generation, time windows, energy rules, charging stations,
instance export, or feasibility checks can alter an experimental distribution.
Such changes should include:

1. a focused test;
2. a clear description of the behavioral change;
3. a note about reproducibility or backward compatibility; and
4. documentation updates where the public interpretation changes.

The preserved paper-era source should not be silently rewritten. Corrections
must be documented and released under a new version.

## Pull requests

Keep pull requests focused. Run the test suite before submission:

```bash
python -m unittest discover -s tests -v
```

