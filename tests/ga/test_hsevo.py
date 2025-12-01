from importlib.resources import files
import logging
import os
import numpy as np
import pytest

from llamda.ga.hsevo.hsevo import HSEvo, HSEvoConfig
from llamda.utils.llm_client.base import BaseLLMClientConfig
from llamda.utils.problem import ProblemPrompts
from llamda.utils.utils import get_output_dir

from tests.common import EVALUATIONS_PATH, RESPONSES_PATH
from tests.mocks import MockClient, MockEvaluator

ROOT_DIR = os.getcwd()
output_dir = get_output_dir("test_hsevo", ROOT_DIR)
logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("problem_name", ["tsp_aco"])
def test_hsevo(problem_name: str) -> None:
    np.random.seed(42)

    client = MockClient(
        config=BaseLLMClientConfig(model="mock", temperature=1.0),
        responses_dir=str(RESPONSES_PATH / "hsevo"),
    )
    prompts = ProblemPrompts.load_problem_prompts(
        str(files("llamda.prompts.problems") / problem_name)
    )
    evaluator = MockEvaluator(
        prompts, evaluation_path=str(EVALUATIONS_PATH / "hsevo.json")
    )

    hsevo = HSEvo(
        config=HSEvoConfig(init_pop_size=5, max_fe=15),
        problem=prompts,
        evaluator=evaluator,
        llm_client=client,
        output_dir=output_dir,
    )

    best_code_overall, best_code_path_overall = hsevo.evolve()
    logging.info(f"Best Code Overall: {best_code_overall}")
    logging.info(f"Best Code Path Overall: {best_code_path_overall}")
