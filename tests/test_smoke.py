from pathlib import Path
import sys
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from data_generator import (  # noqa: E402
    generate_milp_feasible_instance,
    quick_feasibility_check,
    save_instance_txt,
)


class GenerationSmokeTest(unittest.TestCase):
    def generate(self):
        return generate_milp_feasible_instance(
            n_customers=10,
            n_stations=3,
            instance_type="RC",
            random_seed=42,
            depot_mode="center",
            charger_at_depot=True,
            time_horizon=10.0,
        )

    def test_generation_is_deterministic_for_fixed_seed(self):
        first = self.generate()
        second = self.generate()
        for key in ("depot", "customers", "stations", "demands", "service_times"):
            np.testing.assert_allclose(first[key], second[key])
        self.assertEqual(first["time_windows"], second["time_windows"])

    def test_generated_dimensions_and_screen_type(self):
        instance = self.generate()
        self.assertEqual(instance["customers"].shape, (10, 2))
        self.assertEqual(len(instance["demands"]), 10)
        self.assertEqual(len(instance["service_times"]), 10)
        self.assertEqual(len(instance["time_windows"]), 10)
        self.assertIsInstance(quick_feasibility_check(instance), bool)

    def test_text_export(self):
        instance = self.generate()
        directory = REPOSITORY_ROOT / ".test_tmp" / "export_case"
        directory.mkdir(parents=True, exist_ok=True)
        save_instance_txt(instance, str(directory), "RC", 10, "3S", 1)
        path = directory / "RC" / "N10" / "RC_N10_3S_001.txt"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        self.assertIn("StringID", text)
        self.assertIn("Vehicle fuel tank capacity", text)
        self.assertIn("C01", text)


if __name__ == "__main__":
    unittest.main()
