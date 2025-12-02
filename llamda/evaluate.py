# Adapted from ReEvo: https://github.com/ai4co/reevo/blob/main/baselines/eoh/problem_adapter.py
# and https://github.com/ai4co/reevo/blob/main/reevo.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)

import os
import logging
from pathlib import Path
import subprocess
from typing import TypeVar

from llamda.individual import Individual
from llamda.problem import BaseProblem
from llamda.utils import file_to_string, print_hyperlink

logger = logging.getLogger("llamda")

T = TypeVar("T", bound=Individual)


class Evaluator:
    def __init__(self, problem: BaseProblem, timeout: int = 30) -> None:
        self.problem = problem
        self.timeout = timeout
        self.function_evals = 0

    def mark_invalid_individual(self, individual: T, traceback_msg: str) -> T:
        """
        Mark an individual as invalid.
        """
        logger.warning(
            "Marking individual as invalid",
            extra={
                "traceback_msg": traceback_msg[:200],  # Truncate long messages
            },
        )
        individual.exec_success = False
        individual.obj = float("inf")
        individual.traceback_msg = traceback_msg
        return individual

    def batch_evaluate(self, population: list[T], output_dir: Path) -> list[T]:
        """
        Evaluate population by running code in parallel and computing objective values.
        """
        logger.info(
            "Starting batch evaluation",
            extra={
                "population_size": len(population),
                "function_evals": self.function_evals,
            },
        )

        inner_runs = []
        # Run code to evaluate
        for i, individual in enumerate(population):
            stdout_filepath = output_dir / individual.name / "stdout.txt"
            os.makedirs(stdout_filepath.parent, exist_ok=True)
            self.function_evals += 1
            # Skip if response is invalid
            if individual.code is None:
                logger.debug("Skipping invalid response")
                individual = self.mark_invalid_individual(
                    individual, "Invalid response!"
                )
                inner_runs.append(None)
                continue

            logger.info(f"Running Code {i} / {len(population)-1}")

            try:
                process = self._run_code(individual, i)
                inner_runs.append(process)
            except Exception as e:  # If code execution fails
                logger.exception("Code execution error")
                individual = self.mark_invalid_individual(individual, str(e))
                inner_runs.append(None)

        # Update population with objective values
        for response_id, inner_run in enumerate(inner_runs):
            if inner_run is None:  # If code execution fails, skip
                continue
            try:
                inner_run.communicate(
                    timeout=self.timeout
                )  # Wait for code execution to finish
            except subprocess.TimeoutExpired as e:
                logger.error(
                    "Timeout expired during code execution",
                    extra={
                        "response_id": response_id,
                        "timeout": self.timeout,
                    },
                )
                population[response_id] = self.mark_invalid_individual(
                    population[response_id], str(e)
                )
                inner_run.kill()
                continue

            individual = population[response_id]
            with open(stdout_filepath, "r") as f:  # read the stdout file
                stdout_str = f.read()
            traceback_msg = filter_traceback(stdout_str)

            # Store objective value for each individual
            if traceback_msg == "":  # If execution has no error
                try:
                    individual.obj = float(stdout_str.split("\n")[-2])
                    assert individual.obj > 0, "Objective value <= 0 is not supported."
                    individual.obj = (
                        -individual.obj
                        if self.problem.obj_type == "max"
                        else individual.obj
                    )
                    individual.exec_success = True
                except Exception as e:
                    logger.error(
                        "Failed to parse objective value",
                        extra={
                            "response_id": response_id,
                            "error": str(e),
                        },
                    )
                    population[response_id] = self.mark_invalid_individual(
                        population[response_id], "Invalid std out / objective value!"
                    )
            else:  # Otherwise, also provide execution traceback error feedback
                logger.debug(
                    "Individual execution failed with traceback",
                    extra={
                        "response_id": response_id,
                        "traceback": traceback_msg,
                    },
                )
                population[response_id] = self.mark_invalid_individual(
                    population[response_id], traceback_msg
                )

            logger.debug(
                "Individual evaluated successfully",
                extra={
                    "response_id": response_id,
                    "objective": individual.obj,
                },
            )

        logger.info(
            "Batch evaluation completed",
            extra={
                "function_evals": self.function_evals,
                "valid_individuals": sum(1 for ind in population if ind.exec_success),
            },
        )
        return population

    def _run_code(self, individual: T, stdout_filepath: Path) -> subprocess.Popen:
        """
        Write code into a file and run eval script.
        """

        with open(self.problem.code_path, "w") as file:
            file.writelines(individual.code + "\n")

        # Execute the python file with flags
        with open(stdout_filepath, "w") as f:

            process = subprocess.Popen(
                [
                    "python",
                    "-u",
                    str(self.problem.eval_path),
                    f"{self.problem.size}",
                    "train",
                ],
                stdout=f,
                stderr=f,
            )

        block_until_running(stdout_filepath, log_status=True)
        return process


def filter_traceback(s: str) -> str:
    lines = s.split("\n")
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith("Traceback"):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return "\n".join(filtered_lines)
    return ""  # Return an empty string if no Traceback is found


def block_until_running(stdout_filepath: Path, log_status: bool = False) -> None:
    # Ensure that the evaluation has started before moving on
    while True:
        log = file_to_string(str(stdout_filepath))
        if len(log) > 0:
            if log_status and "Traceback" in log:
                logger.warning(
                    f"Code Run execution error! (see {print_hyperlink(stdout_filepath, 'stdout')}))"
                )
            else:
                logger.info(
                    f"Code Run successful! (see {print_hyperlink(stdout_filepath, 'stdout')})"
                )
            break
