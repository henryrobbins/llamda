# Adapted from ReEvo: https://github.com/ai4co/reevo/blob/main/reevo.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)

import logging

from jinja2 import Environment, PackageLoader, StrictUndefined

from llamda.individual import Individual
from llamda.llm_client.base import BaseClient
from llamda.problem import Problem
from llamda.utils import filter_code

logger = logging.getLogger("llamda")


class ReEvoLLMClients:
    def __init__(
        self,
        generator_llm: BaseClient,
        reflector_llm: BaseClient | None = None,
        short_reflector_llm: BaseClient | None = None,
        long_reflector_llm: BaseClient | None = None,
        crossover_llm: BaseClient | None = None,
        mutation_llm: BaseClient | None = None,
    ) -> None:
        self.generator_llm = generator_llm
        self.reflector_llm = reflector_llm or generator_llm
        self.short_reflector_llm = short_reflector_llm or self.reflector_llm
        self.long_reflector_llm = long_reflector_llm or self.reflector_llm
        self.crossover_llm = crossover_llm or generator_llm
        self.mutation_llm = mutation_llm or generator_llm


class Evolution:

    def __init__(
        self,
        init_pop_size: int,
        pop_size: int,
        mutation_rate: float,
        llm_clients: ReEvoLLMClients,
        problem: Problem,
    ) -> None:

        self.init_pop_size = init_pop_size
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate

        self.llm_clients = llm_clients
        self.problem = problem

        self.env = Environment(
            loader=PackageLoader("llamda.prompts.ga", "reevo"), undefined=StrictUndefined
        )

    def seed_population(self, long_term_reflection_str: str) -> list[str]:

        seed_template = self.env.get_template("seed.j2")
        seed_prompt = seed_template.render(
            seed_func=self.problem.seed_func,
            func_name=self.problem.func_name,
        )

        system_generator_template = self.env.get_template("system_generator.j2")
        system = system_generator_template.render()

        user_generator_template = self.env.get_template("user_generator.j2")
        user_generator_prompt = user_generator_template.render(
            func_name=self.problem.func_name,
            description=self.problem.description,
            func_desc=self.problem.func_desc,
        )

        user = (
            user_generator_prompt
            + "\n"
            + seed_prompt
            + "\n"
            + long_term_reflection_str
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        logger.info("Initial Population Prompt", extra={"system": system, "user": user})

        responses = self.llm_clients.generator_llm.multi_chat_completion(
            [messages],
            self.init_pop_size,
            temperature=self.llm_clients.generator_llm.temperature + 0.3,
        )  # Increase the temperature for diverse initial population

        return responses

    def _gen_short_term_reflection_prompt(
        self, ind1: Individual, ind2: Individual
    ) -> tuple[list[dict], str, str]:
        """
        Short-term reflection before crossovering two individuals.
        """

        if ind1.obj == ind2.obj:
            raise ValueError(
                "Two individuals to crossover have the same objective value!"
            )
        # Determine which individual is better or worse
        if ind1.obj < ind2.obj:
            better_ind, worse_ind = ind1, ind2
        else:  # robust in rare cases where two individuals have the same objective
            better_ind, worse_ind = ind2, ind1

        worse_code = filter_code(worse_ind.code)
        better_code = filter_code(better_ind.code)

        system_reflector_template = self.env.get_template("system_reflector.j2")
        system = system_reflector_template.render()

        user_reflector_st_template = self.env.get_template("user_reflector_st.j2")
        user = user_reflector_st_template.render(
            func_name=self.problem.func_name,
            func_desc=self.problem.func_desc,
            description=self.problem.description,
            worse_code=worse_code,
            better_code=better_code,
        )
        message = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        logger.info(
            "Short-term Reflection Prompt", extra={"system": system, "user": user}
        )

        return message, worse_code, better_code

    def short_term_reflection(
        self, population: list[Individual]
    ) -> tuple[list[str], list[str], list[str]]:
        """
        Short-term reflection before crossovering two individuals.
        """
        messages_lst = []
        worse_code_lst = []
        better_code_lst = []
        for i in range(0, len(population), 2):
            # Select two individuals
            parent_1 = population[i]
            parent_2 = population[i + 1]

            # Short-term reflection
            messages, worse_code, better_code = self._gen_short_term_reflection_prompt(
                parent_1, parent_2
            )
            messages_lst.append(messages)
            worse_code_lst.append(worse_code)
            better_code_lst.append(better_code)

        # Asynchronously generate responses
        response_lst = self.llm_clients.short_reflector_llm.multi_chat_completion(
            messages_lst
        )
        return response_lst, worse_code_lst, better_code_lst

    def long_term_reflection(
        self, short_term_reflections: list[str], long_term_reflection_str: str
    ) -> str:
        """
        Long-term reflection before mutation.
        """

        system_reflector_template = self.env.get_template("system_reflector.j2")
        system = system_reflector_template.render()

        user_reflector_lt_template = self.env.get_template("user_reflector_lt.j2")
        user = user_reflector_lt_template.render(
            description=self.problem.description,
            prior_reflection=long_term_reflection_str,
            new_reflection="\n".join(short_term_reflections),
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        logger.info(
            "Long-term Reflection Prompt", extra={"system": system, "user": user}
        )

        response = self.llm_clients.long_reflector_llm.multi_chat_completion(
            [messages]
        )[0]

        return response

    def crossover(
        self, short_term_reflection_tuple: tuple[list[list[dict]], list[str], list[str]]
    ) -> list[str]:

        reflection_content_lst, worse_code_lst, better_code_lst = (
            short_term_reflection_tuple
        )
        messages_lst = []
        for reflection, worse_code, better_code in zip(
            reflection_content_lst, worse_code_lst, better_code_lst
        ):
            # Crossover
            system_generator_template = self.env.get_template("system_generator.j2")
            system = system_generator_template.render()

            user_generator_template = self.env.get_template("user_generator.j2")
            user_generator_prompt = user_generator_template.render(
                func_name=self.problem.func_name,
                description=self.problem.description,
                func_desc=self.problem.func_desc,
            )

            func_signature0 = self.problem.func_signature.format(version=0)
            func_signature1 = self.problem.func_signature.format(version=1)

            crossover_template = self.env.get_template("crossover.j2")
            user = crossover_template.render(
                user_generator=user_generator_prompt,
                func_signature0=func_signature0,
                func_signature1=func_signature1,
                worse_code=worse_code,
                better_code=better_code,
                reflection=reflection,
                func_name=self.problem.func_name,
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            messages_lst.append(messages)

            logger.info("Crossover Prompt", extra={"system": system, "user": user})

        # Asynchronously generate responses
        responses = self.llm_clients.crossover_llm.multi_chat_completion(messages_lst)
        return responses

    def mutate(self, long_term_reflection_str: str, elitist: Individual) -> list[str]:
        """
        Elitist-based mutation. We mutate the best to generate n_pop new individuals.
        """

        system_generator_template = self.env.get_template("system_generator.j2")
        system = system_generator_template.render()

        user_generator_template = self.env.get_template("user_generator.j2")
        user_generator_prompt = user_generator_template.render(
            func_name=self.problem.func_name,
            description=self.problem.description,
            func_desc=self.problem.func_desc,
        )

        func_signature1 = self.problem.func_signature.format(version=1)

        mutation_template = self.env.get_template("mutation.j2")
        user = mutation_template.render(
            user_generator=user_generator_prompt,
            reflection=long_term_reflection_str + self.problem.external_knowledge,
            func_signature1=func_signature1,
            elitist_code=filter_code(elitist.code),
            func_name=self.problem.func_name,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        logger.info("Mutation Prompt", extra={"system": system, "user": user})

        responses = self.llm_clients.mutation_llm.multi_chat_completion(
            [messages], int(self.pop_size * self.mutation_rate)
        )

        return responses
