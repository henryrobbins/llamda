# Adapted from ReEvo: https://github.com/ai4co/reevo/blob/main/utils/utils.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)

import re
from typing import TypeVar

from llamda.individual import Individual


# reevo + hsevo
def extract_code_from_generator(content):
    """Extract code from the response of the code generator."""
    pattern_code = r"```python(.*?)```"
    code_string = re.search(pattern_code, content, re.DOTALL)
    code_string = code_string.group(1).strip() if code_string is not None else None
    if code_string is None:
        # Find the line that starts with "def" and the line that starts with "return", and extract the code in between
        lines = content.split("\n")
        start = None
        end = None
        for i, line in enumerate(lines):
            if line.startswith("def"):
                start = i
            if "return" in line:
                end = i
                break
        if start is not None and end is not None:
            code_string = "\n".join(lines[start : end + 1])

    if code_string is None:
        return None
    # Add import statements if not present
    # TODO: Using HSEvo convention for this (other methods used commented out code)
    # if "np" in code_string:
    #     code_string = "import numpy as np\n" + code_string
    # if "torch" in code_string:
    #     code_string = "import torch\n" + code_string
    if "import" not in code_string:
        code_string = (
            "import numpy as np\nimport random\nimport math\nimport scipy\nimport torch\n"
            + code_string
        )
    return code_string


# eoh + mcts
def parse_response(response: str) -> tuple[list[str], list[str]]:
    # TODO: Why did MCTS make this change?
    # algorithm = re.search(r"\{(.*?)\}", response, re.DOTALL).group(1)
    algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
    if len(algorithm) == 0:
        if "python" in response:
            algorithm = re.findall(r"^.*?(?=python)", response, re.DOTALL)
        elif "import" in response:
            algorithm = re.findall(r"^.*?(?=import)", response, re.DOTALL)
        else:
            algorithm = re.findall(r"^.*?(?=def)", response, re.DOTALL)

    code = re.findall(r"import.*return", response, re.DOTALL)
    if len(code) == 0:
        code = re.findall(r"def.*return", response, re.DOTALL)
    return algorithm, code


def filter_code(code_string: str) -> str:
    """Remove lines containing signature and import statements."""
    lines = code_string.split("\n")
    filtered_lines = []
    for line in lines:
        if line.startswith("def"):
            continue
        elif line.startswith("import"):
            continue
        elif line.startswith("from"):
            continue
        elif line.startswith("return"):
            filtered_lines.append(line)
            break
        else:
            filtered_lines.append(line)
    code_string = "\n".join(filtered_lines)
    return code_string


T = TypeVar("T", bound=Individual)


def hydrate_individual(
    individual: T,
    response_id: int,
    output_dir: str,
    iteration: int = 0,
    file_name: str | None = None,
) -> T:

    # Write response to file
    file_name = (
        f"problem_iter{iteration}_response{response_id}.txt"
        if file_name is None
        else file_name + ".txt"
    )
    file_name = f"{output_dir}/{file_name}"
    with open(file_name, "w", encoding="utf-8") as file:
        file.writelines(individual.code + "\n")

    # Extract code and description from response
    std_out_filepath = (
        f"problem_iter{iteration}_stdout{response_id}.txt"
        if file_name is None
        else file_name.rstrip(".txt") + "_stdout.txt"
    )

    individual.stdout_filepath = std_out_filepath
    individual.code_path = f"problem_iter{iteration}_code{response_id}.py"
    individual.response_id = response_id

    return individual
