import hydra
import logging
import os
from pathlib import Path

from ga.hsevo.hsevo import HSEvo as LHH

ROOT_DIR = os.getcwd()
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="hydra", config_name="config")
def main(cfg) -> None:
    problem_name = "tsp_aco"

    workspace_dir = Path.cwd()
    # Set logging level
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ROOT_DIR}")

    # Main algorithm
    lhh = LHH(
        problem_name=problem_name,
        model="openai/gpt-4o-mini-2024-07-18",
        temperature=1.0,
        root_dir=ROOT_DIR,
    )
    best_code_overall, best_code_path_overall = lhh.evolve()
    logging.info(f"Best Code Overall: {best_code_overall}")
    logging.info(f"Best Code Path Overall: {best_code_path_overall}")


if __name__ == "__main__":
    main()
