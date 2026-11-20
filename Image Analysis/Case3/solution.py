from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import spatial_filtering_project as project


if __name__ == "__main__":
    project.run_case_3(ROOT / "inputs")
    print(f"Case 3 results saved under: {project.OUTPUT}")
