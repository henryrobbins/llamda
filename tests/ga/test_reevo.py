from importlib.resources import files
import logging
import os
import pytest

from llamda.utils.llm_client.base import BaseLLMClientConfig
from llamda.utils.problem import ProblemPrompts
from llamda.utils.utils import get_output_dir
from llamda.ga.reevo.reevo import ReEvo, ReEvoConfig

from tests.common import EVALUATIONS_PATH, RESPONSES_PATH
from tests.mocks import MockClient, MockEvaluator

ROOT_DIR = os.getcwd()
output_dir = get_output_dir("test_reevo", ROOT_DIR)
logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("problem_name", ["tsp_aco"])
def test_reevo(problem_name: str) -> None:

    client = MockClient(
        config=BaseLLMClientConfig(model="mock", temperature=1.0),
        responses_dir=str(RESPONSES_PATH / "reevo"),
    )
    prompts = ProblemPrompts.load_problem_prompts(
        path=str(files("llamda.prompts.problems") / problem_name)
    )
    evaluator = MockEvaluator(
        prompts, evaluation_path=str(EVALUATIONS_PATH / "reevo.json")
    )

    reevo = ReEvo(
        config=ReEvoConfig(init_pop_size=5, max_fe=15),
        problem=prompts,
        evaluator=evaluator,
        output_dir=output_dir,
        llm_client=client,
    )

    best_code_overall, _ = reevo.evolve()
    logging.info(f"Best Code Overall: {best_code_overall}")
