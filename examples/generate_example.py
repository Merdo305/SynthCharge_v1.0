from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from data_generator import (  # noqa: E402
    generate_milp_feasible_instance,
    quick_feasibility_check,
    save_instance_txt,
)


def main() -> None:
    instance = generate_milp_feasible_instance(
        n_customers=10,
        n_stations=3,
        instance_type="RC",
        random_seed=42,
        depot_mode="center",
        charger_at_depot=True,
        time_horizon=10.0,
        vehicle_capacity=1.5,
        consumption_rate=0.25,
        service_min=0.01,
        service_max=0.03,
    )

    output_dir = REPOSITORY_ROOT / "example_output"
    save_instance_txt(
        instance,
        str(output_dir),
        type_code="RC",
        n_customers=10,
        st_label="3S",
        idx=1,
    )

    output_file = output_dir / "RC" / "N10" / "RC_N10_3S_001.txt"
    print(f"Structural screen passed: {quick_feasibility_check(instance)}")
    print(f"Generated instance: {output_file}")


if __name__ == "__main__":
    main()

