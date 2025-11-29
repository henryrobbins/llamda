import hydra
import logging
import os
from pathlib import Path
import subprocess

from ga.mcts.config import Config
from ga.mcts.mcts_ahd import MCTS_AHD, AHDConfig
from utils.evaluate import Evaluator
from utils.llm_client.openai import OpenAIClient, OpenAIClientConfig
from utils.problem import ProblemPrompts, adapt_prompt

ROOT_DIR = os.getcwd()
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="hydra", config_name="config")
def main(cfg) -> None:
    problem_name = "tsp_constructive"

    workspace_dir = Path.cwd()
    # Set logging level
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ROOT_DIR}")

    config = OpenAIClientConfig(
        model="gpt-3.5-turbo",
        temperature=1.0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    client = OpenAIClient(config)

    # ========================================================================

    root_dir = ROOT_DIR

    problem_config = ProblemPrompts.load_problem_prompts(
        f"{root_dir}/prompts/{problem_name}"
    )

    if problem_config.problem_type == "constructive":
        from utils.problem import TSP_CONSTRUCTIVE_PROMPTS

        prompts = TSP_CONSTRUCTIVE_PROMPTS
    elif problem_config.problem_type == "online":
        from utils.problem import BPP_ONLINE_PROMPTS

        prompts = BPP_ONLINE_PROMPTS
    else:
        prompts = adapt_prompt(problem_config)

    evaluator = Evaluator(prompts, root_dir)

    ahd_config = AHDConfig()

    paras = Config(
        init_size=ahd_config.init_pop_size,
        pop_size=ahd_config.pop_size,
        ec_fe_max=ahd_config.max_fe,
        exp_output_path=f"{workspace_dir}/",
    )

    llm_client = client

    # ========================================================================

    # Main algorithm
    lhh = MCTS_AHD(paras, prompts, evaluator, llm_client)
    best_code_overall, best_code_path_overall = lhh.run()
    logging.info(f"Best Code Overall: {best_code_overall}")
    logging.info(f"Best Code Path Overall: {best_code_path_overall}")

    # Run validation and redirect stdout to a file "best_code_overall_stdout.txt"
    with open(f"{ROOT_DIR}/problems/{problem_name}/gpt.py", "w") as file:
        file.writelines(best_code_overall + "\n")
    test_script = f"{ROOT_DIR}/problems/{problem_name}/eval.py"
    test_script_stdout = "best_code_overall_val_stdout.txt"
    logging.info(f"Running validation script...: {test_script}")
    with open(test_script_stdout, "w") as stdout:
        subprocess.run(["python", test_script, "-1", ROOT_DIR, "val"], stdout=stdout)
    logging.info(
        f"Validation script finished. Results are saved in {test_script_stdout}."
    )

    # Print the results
    with open(test_script_stdout, "r") as file:
        for line in file.readlines():
            logging.info(line.strip())


if __name__ == "__main__":
    main()
