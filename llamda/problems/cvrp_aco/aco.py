# Adapted from DeepACO: https://github.com/henry-yeh/DeepACO/blob/main/cvrp/aco.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)

import torch
from torch import Tensor
from torch.distributions import Categorical
import numpy as np
import numpy.typing as npt


class ACO:
    def __init__(
        self,  # 0: depot
        distances: npt.NDArray[np.floating] | Tensor,  # (n, n)
        demand: npt.NDArray[np.floating] | Tensor,  # (n, )
        heuristic: npt.NDArray[np.floating] | Tensor,  # (n, n)
        capacity: int,
        n_ants: int = 30,
        decay: float = 0.9,
        alpha: float = 1,
        beta: float = 1,
        device: str = "cpu",
    ) -> None:
        self.problem_size = len(distances)
        self.distances = (
            torch.tensor(distances, device=device)
            if not isinstance(distances, torch.Tensor)
            else distances
        )
        self.demand = (
            torch.tensor(demand, device=device)
            if not isinstance(demand, torch.Tensor)
            else demand
        )
        self.capacity = capacity

        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

        self.pheromone = torch.ones_like(self.distances)
        self.heuristic = (
            torch.tensor(heuristic, device=device)
            if not isinstance(heuristic, torch.Tensor)
            else heuristic
        )

        self.shortest_path = None
        self.lowest_cost = float("inf")

        self.device = device

    @torch.no_grad()
    def run(self, n_iterations: int) -> Tensor:
        for _ in range(n_iterations):
            paths = self.gen_path()
            costs = self.gen_path_costs(paths)

            best_cost, best_idx = costs.min(dim=0)
            if best_cost < self.lowest_cost:
                self.shortest_path = paths[:, best_idx]
                self.lowest_cost = best_cost

            self.update_pheronome(paths, costs)

        return self.lowest_cost

    @torch.no_grad()
    def update_pheronome(self, paths: Tensor, costs: Tensor) -> None:
        """
        Args:
            paths: torch tensor with shape (problem_size, n_ants)
            costs: torch tensor with shape (n_ants,)
        """
        self.pheromone = self.pheromone * self.decay
        for i in range(self.n_ants):
            path = paths[:, i]
            cost = costs[i]
            self.pheromone[path[:-1], torch.roll(path, shifts=-1)[:-1]] += 1.0 / cost
        self.pheromone[self.pheromone < 1e-10] = 1e-10

    @torch.no_grad()
    def gen_path_costs(self, paths: Tensor) -> Tensor:
        u = paths.permute(1, 0)  # shape: (n_ants, max_seq_len)
        v = torch.roll(u, shifts=-1, dims=1)
        return torch.sum(self.distances[u[:, :-1], v[:, :-1]], dim=1)

    def gen_path(self) -> Tensor:
        actions = torch.zeros((self.n_ants,), dtype=torch.long, device=self.device)
        visit_mask = torch.ones(
            size=(self.n_ants, self.problem_size), device=self.device
        )
        visit_mask = self.update_visit_mask(visit_mask, actions)
        used_capacity = torch.zeros(size=(self.n_ants,), device=self.device)

        used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)

        paths_list = [actions]  # paths_list[i] is the ith move (tensor) for all ants

        done = self.check_done(visit_mask, actions)
        while not done:
            actions = self.pick_move(actions, visit_mask, capacity_mask)
            paths_list.append(actions)
            visit_mask = self.update_visit_mask(visit_mask, actions)
            used_capacity, capacity_mask = self.update_capacity_mask(
                actions, used_capacity
            )
            done = self.check_done(visit_mask, actions)

        return torch.stack(paths_list)

    def pick_move(
        self, prev: Tensor, visit_mask: Tensor, capacity_mask: Tensor
    ) -> Tensor:
        pheromone = self.pheromone[prev]  # shape: (n_ants, p_size)
        heuristic = self.heuristic[prev]  # shape: (n_ants, p_size)
        dist = (
            (pheromone**self.alpha)
            * (heuristic**self.beta)
            * visit_mask
            * capacity_mask
        )  # shape: (n_ants, p_size)
        dist = Categorical(dist)
        actions = dist.sample()  # shape: (n_ants,)
        return actions

    def update_visit_mask(self, visit_mask: Tensor, actions: Tensor) -> Tensor:
        visit_mask[torch.arange(self.n_ants, device=self.device), actions] = 0
        visit_mask[:, 0] = 1  # depot can be revisited with one exception
        visit_mask[(actions == 0) * (visit_mask[:, 1:] != 0).any(dim=1), 0] = (
            0  # one exception is here
        )
        return visit_mask

    def update_capacity_mask(
        self, cur_nodes: Tensor, used_capacity: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            cur_nodes: shape (n_ants, )
            used_capacity: shape (n_ants, )
            capacity_mask: shape (n_ants, p_size)
        Returns:
            ant_capacity: updated capacity
            capacity_mask: updated mask
        """
        capacity_mask = torch.ones(
            size=(self.n_ants, self.problem_size), device=self.device
        )
        # update capacity
        used_capacity[cur_nodes == 0] = 0
        used_capacity = used_capacity + self.demand[cur_nodes]
        # update capacity_mask
        remaining_capacity = self.capacity - used_capacity  # (n_ants,)
        remaining_capacity_repeat = remaining_capacity.unsqueeze(-1).repeat(
            1, self.problem_size
        )  # (n_ants, p_size)
        demand_repeat = self.demand.unsqueeze(0).repeat(
            self.n_ants, 1
        )  # (n_ants, p_size)
        capacity_mask[demand_repeat > remaining_capacity_repeat] = 0

        return used_capacity, capacity_mask

    def check_done(self, visit_mask: Tensor, actions: Tensor) -> bool:
        return bool((visit_mask[:, 1:] == 0).all() and (actions == 0).all())
