import hydra
import logging
import os
from pathlib import Path
import subprocess
from utils.llm_client.openai import OpenAIClient, OpenAIClientConfig
from utils.problem import ProblemPrompts
from utils.utils import print_hyperlink

from ga.reevo.reevo import ReEvo as LHH

ROOT_DIR = os.getcwd()
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="hydra", config_name="config")
def main(cfg) -> None:
    problem_name = "tsp_aco"

    workspace_dir = Path.cwd()
    # Set logging level
    logging.info(f"Workspace: {print_hyperlink(workspace_dir)}")
    logging.info(f"Project Root: {print_hyperlink(ROOT_DIR)}")

    config = OpenAIClientConfig(
        model="gpt-3.5-turbo",
        temperature=1.0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    client = OpenAIClient(config)

    prompt_dir = f"{ROOT_DIR}/prompts"
    prompts = ProblemPrompts.load_problem_prompts(
        path=f"{prompt_dir}/{problem_name}",
    )

    lhh = LHH(prompts, ROOT_DIR, generator_llm=client)

    best_code_overall, best_code_path_overall = lhh.evolve()
    logging.info(f"Best Code Overall: {best_code_overall}")
    best_path = best_code_path_overall.replace(".py", ".txt").replace(
        "code", "response"
    )
    logging.info(
        f"Best Code Path Overall: {print_hyperlink(best_path, best_code_path_overall)}"
    )

    # Run validation and redirect stdout to a file "best_code_overall_stdout.txt"
    with open(f"{ROOT_DIR}/problems/{problem_name}/gpt.py", "w") as file:
        file.writelines(best_code_overall + "\n")
    test_script = f"{ROOT_DIR}/problems/{problem_name}/eval.py"
    test_script_stdout = "best_code_overall_val_stdout.txt"
    logging.info(f"Running validation script...: {print_hyperlink(test_script)}")
    with open(test_script_stdout, "w") as stdout:
        subprocess.run(["python", test_script, "-1", ROOT_DIR, "val"], stdout=stdout)
    logging.info(
        f"Validation script finished. Results are saved in {print_hyperlink(test_script_stdout)}."
    )

    # Print the results
    with open(test_script_stdout, "r") as file:
        for line in file.readlines():
            logging.info(line.strip())


if __name__ == "__main__":
    main()
