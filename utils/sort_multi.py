import sys
from pathlib import Path

import numpy as np


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir.parent / "dataset" / "sales_multi.csv"

    if not csv_path.exists():
        print(f"File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    data = np.loadtxt(csv_path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    sorted_data = data[np.argsort(data[:, -1])]
    np.savetxt(csv_path, sorted_data, delimiter=",", fmt="%.2f")
    print(f"Sorted dataset saved to {csv_path}")


if __name__ == "__main__":
    main()
