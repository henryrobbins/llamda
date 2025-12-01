from importlib.resources import files
import logging
import os
import pytest

from llamda.utils.llm_client.base import BaseLLMClientConfig
from llamda.utils.problem import ProblemPrompts, adapt_prompt
from llamda.utils.utils import get_output_dir
from llamda.ga.mcts.mcts_ahd import AHDConfig
from llamda.ga.mcts.mcts_ahd import MCTS_AHD as LHH

from tests.common import EVALUATIONS_PATH, RESPONSES_PATH
from tests.mocks import MockClient, MockEvaluator

ROOT_DIR = os.getcwd()
output_dir = get_output_dir("test_mcts", ROOT_DIR)
logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("problem_name", ["tsp_aco"])
def test_mcts(problem_name: str) -> None:

    client = MockClient(
        config=BaseLLMClientConfig(model="mock", temperature=1.0),
        responses_dir=str(RESPONSES_PATH / "mcts"),
    )

    problems_dir = files("llamda.prompts.problems")
    problem_prompts = ProblemPrompts.load_problem_prompts(
        path=str(problems_dir / problem_name),
    )
    eoh_problem_prompts = adapt_prompt(problem_prompts)
    evaluator = MockEvaluator(
        eoh_problem_prompts, evaluation_path=str(EVALUATIONS_PATH / "mcts.json")
    )

    lhh = LHH(
        config=AHDConfig(init_size=5, ec_fe_max=15),
        problem=eoh_problem_prompts,
        evaluator=evaluator,
        output_dir=output_dir,
        llm_client=client,
    )

    best_code_overall, _ = lhh.run()
    logging.info(f"Best Code Overall: {best_code_overall}")
