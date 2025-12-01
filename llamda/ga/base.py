import os
from abc import ABC
from typing import Generic, TypeVar

from llamda.evaluate import Evaluator
from llamda.llm_client.base import BaseClient


T = TypeVar("T")
U = TypeVar("U")


class GeneticAlgorithm(Generic[T, U], ABC):
    def __init__(
        self,
        config: T,
        problem: U,
        evaluator: Evaluator,
        llm_client: BaseClient,
        output_dir: str,
    ) -> None:
        self.config = config
        self.problem = problem
        self.evaluator = evaluator
        self.llm_client = llm_client
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
