# Adapted from "f-sensitivity-through-evar" by Matej Havelka, used under the MIT License.
# Source: https://github.com/MatejHav/f-sensitivity-through-evar
# Copyright (c) 2024 Matej Havelka. See LICENSE at the repository root for full terms.

import pandas as pd
import numpy as np
import torch
import time

from pyomo.core import *
from pyomo.environ import *
from sklearn.linear_model import QuantileRegressor

from data_generation import *

class FSensitivity:

    def create_bounds(self, data: DataObject, rho: float, approach: str) -> tuple:
        """

        :param data:
        :param rho:
        :param approach: "evar" for gradient descent, "lagr" for lagrangian and "cp" for contraints programming
        :return:
        """
        if approach == "evar":
            lower = self._solve_evar_formulation(data, rho, True)
            upper = self._solve_evar_formulation(data, rho, False)
            return lower, upper
        bound_founder = self._solve_lagrangian_formulation if approach == "lagr" \
            else self._solve_constraint_programming_formulation
        # Compute the bounds
        lower_control_bound = bound_founder(data, rho, 0, True)
        lower_treated_bound = bound_founder(data, rho, 1, True)
        upper_control_bound = bound_founder(data, rho, 0, False)
        upper_treated_bound = bound_founder(data, rho, 1, False)

        # Output lower and upper bounds based on best and worst cases
        return lower_treated_bound - upper_control_bound, upper_treated_bound - lower_control_bound

    def _solve_constraint_programming_formulation(self, data: DataObject, rho: float, treatment: int,
                                                  is_lower_bound: bool) -> float:
        """

        :param data:
        :param rho:
        :param treatment:
        :param is_lower_bound:
        :return:
        """
        X = data.discrete_x()
        Y = data.discrete_y()
        p_treated = len(data.data[data.data["T"] == 1]) / len(data.data)
        r = lambda x: data.propensity_score_index(x) * (1 - p_treated) / (1 - data.propensity_score_index(x) * p_treated)
        model = ConcreteModel(name="FSensitivityModel")
        model.X = RangeSet(0, len(X) - 1)
        model.Y = RangeSet(0, len(Y) - 1)
        model.L = Var(model.X, model.Y, initialize=1, within=NonNegativeReals)

        # Constraint 1: Definition of R
        def r_constraint(model, x):
            if treatment == 0:
                return (1 / r(x)) == sum([model.L[x, y] * data.probability_of_x_index_y_index_given_t(x, y, treatment) for y in model.Y])
            return r(x) == sum([model.L[x, y] * data.probability_of_x_index_y_index_given_t(x, y, treatment) for y in model.Y])

        model.c1 = Constraint(model.X, rule=r_constraint)

        # Constraint 3: MSM assumption
        def f_constraint(model, x):
            if treatment == 0:
                return sum([
                    log(model.L[x, y] * r(x)) * model.L[x, y] * r(x) * data.probability_of_x_index_y_index_given_t(x, y, treatment)
                for y in model.Y]) <= rho
            return sum([
                log(model.L[x, y] / r(x)) * model.L[x, y] / r(x) * data.probability_of_x_index_y_index_given_t(x, y, treatment)
            for y in model.Y]) <= rho

        model.c3 = Constraint(model.X, rule=f_constraint)

        # Objective: E[Y * L(X, Y)]
        def objective_function(model):
            return sum([sum([Y.iloc[y] * model.L[x, y] * data.probability_of_x_index_y_index_given_t(x, y, treatment)
                             for y in model.Y]) for x in model.X])

        model.OBJ = Objective(rule=objective_function,
                              sense=pyomo.core.minimize if is_lower_bound else pyomo.core.maximize)

        opt = SolverFactory('ipopt')
        opt.options['max_iter'] = 1000
        opt.solve(model)
        return model.OBJ()

    def __entropic_value_at_risk(self, exponents: torch.Tensor, alpha: float, is_lower_bound: bool) -> float:
        eps = 1e-6
        N = len(exponents)
        z = torch.nn.Parameter(torch.tensor([-1.0 if is_lower_bound else 1.0], requires_grad=True, dtype=torch.double))
        helper = lambda: (torch.logsumexp(z * exponents, 0) - np.log(N * alpha)) / z
        optimizer = torch.optim.Adam([z], lr=1e-3, maximize=is_lower_bound, weight_decay=0)
        previous = None
        loss = None
        # Until converged
        while previous is None or abs((previous - loss).item()) > eps:
            if (z.item() < 0 and not is_lower_bound) or (z.item() > 0 and is_lower_bound):
                return min(exponents.max().item(), previous.item()) if not is_lower_bound else \
                    max(exponents.min().item(), previous.item())
            previous = loss
            loss = helper()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return min(exponents.max().item(), helper().item()) if not is_lower_bound else \
            max(exponents.min().item(), helper().item())


    def _solve_evar_formulation(self, data: DataObject, rho: float, is_lower_bound: bool) -> float:
        """

        :param data:
        :param rho:
        :param treatment:
        :param is_lower_bound:
        :return:
        """
        alpha = np.exp(-rho)
        X = data.discrete_x()
        p_treated = len(data.data[data.data["T"] == 1]) / len(data.data)
        result_treated = 0
        result_control = 0
        eps_p = 1e-3
        for index, row in X.iterrows():
            observed_propensity = float(np.clip(data.propensity_score(row), eps_p, 1 - eps_p))
            r_x = (1 - p_treated) * observed_propensity / (p_treated * (1 - observed_propensity))
            # treated
            exponents = torch.DoubleTensor(data.select_x(row, 1)["Y"].to_numpy().copy())
            treated_q = self.__entropic_value_at_risk(exponents, alpha, is_lower_bound)
            result_treated += r_x * data.probability_of_x_index_given_t(index, 1) * treated_q
            # control
            exponents = torch.DoubleTensor(data.select_x(row, 0)["Y"].to_numpy().copy())
            control_q = self.__entropic_value_at_risk(exponents, alpha, not is_lower_bound)
            result_control += 1 / r_x * data.probability_of_x_index_given_t(index, 0) * control_q
        return result_treated - result_control

    def _solve_lagrangian_formulation(self, data: DataObject, rho: float, treatment: int, is_lower_bound: bool) -> float:
        """

        :param data:
        :param rho:
        :param treatment:
        :param is_lower_bound:
        :return:
        """
        # Shuffle the data
        shuffled_data = data.data.sample(frac=1)
        data_splits = np.array_split(shuffled_data, 3)
        bounds = [0, 0, 0]
        eps = 0
        x_features = list(filter(lambda c: 'X' in c, shuffled_data.columns))
        layer_size = len(x_features)
        # For every subset of data
        for i in range(3):
            current_data = data_splits[i]
            next_data = data_splits[(i + 1) % 3]
            next_next_data = data_splits[(i + 2) % 3]
            selected_curr_data0 = current_data[current_data["T0"] == 0]
            selected_curr_data1 = current_data[current_data["T0"] == 1]
            selected_next_data0 = next_data[next_data["T0"] == 0]
            selected_next_data1 = next_data[next_data["T0"] == 1]
            selected_next_next_data0 = next_next_data[next_next_data["T0"] == 0]
            selected_next_next_data1 = next_next_data[next_next_data["T0"] == 1]
            # Estimate r(x) based on i+1 dataset
            p = len(next_data[next_data["T0"] == treatment]) / len(next_data)
            X = next_data.groupby(x_features, as_index=False).size()
            Xt = current_data[current_data["T0"] == treatment].groupby(x_features, as_index=False).size()
            # Assign each sample a r(x) value
            r = torch.nn.Linear(layer_size, 1)
            r_optim = torch.optim.Adam(r.parameters(), weight_decay=1e-3)
            criterion = torch.nn.MSELoss()
            for _ in range(500):
                batch = next_data.sample(frac=0.2)
                x = torch.Tensor(batch[x_features].to_numpy())
                selected_x = pd.merge(pd.merge(batch[x_features], X, on=x_features, how='inner', suffixes=('l', 'r'))[
                                          [*x_features, "size"]], Xt, on=x_features, how='inner', suffixes=('l', 'r'))[
                    [*x_features, "sizel", "sizer"]]
                propensity = torch.Tensor((selected_x["sizer"] / selected_x["sizel"]).to_numpy())
                r_optim.zero_grad()
                target = (1 - propensity) * p / ((1 - p) * propensity)
                prediction = r(x)
                loss = criterion(prediction.squeeze(dim=-1), target)
                loss.backward()
                r_optim.step()
            # Estimate the nuisance parameters using a NN
            alpha_model = torch.nn.Linear(layer_size, 1)
            alpha_model.weight = torch.nn.Parameter(torch.ones_like(alpha_model.weight))
            alpha_model.bias = torch.nn.Parameter(torch.ones_like(alpha_model.weight))
            eta_model = torch.nn.Linear(layer_size, 1)
            params = list(alpha_model.parameters())
            params.extend(list(eta_model.parameters()))
            optimizer = torch.optim.Adam(params, weight_decay=0)
            # eta_optim = torch.optim.Adam(eta_model.parameters(), weight_decay=0)
            # Do 200 steps on randomized batches from i+1 data
            for _ in range(20_000):
                batch = selected_next_data1.sample(frac=1) if treatment == 0 else selected_next_data0.sample(frac=1)
                x = torch.Tensor(batch[x_features].to_numpy())
                y = (1 if is_lower_bound else -1) * torch.Tensor(batch["Y0"].to_numpy()).unsqueeze(dim=-1)
                optimizer.zero_grad()
                alpha = alpha_model(x) ** 2 + 0.1
                eta = eta_model(x)
                loss = torch.mean(alpha * torch.exp((y + eta) / (-alpha - eps) - 1) + eta + alpha * rho)
                loss.backward()
                optimizer.step()
            # Use regression to estimate H(X, Y) given X from i+2 dataset
            regressor = torch.nn.Linear(layer_size, 1)
            regressor_optim = torch.optim.Adam(regressor.parameters(), weight_decay=1e-3)
            criterion = torch.nn.MSELoss()
            for _ in range(1_000):
                batch = selected_next_next_data1.sample(frac=1) if treatment == 0 else selected_next_next_data0.sample(
                    frac=1)
                x = torch.Tensor(batch[x_features].to_numpy())
                y = (1 if is_lower_bound else -1) * torch.Tensor(batch["Y0"].to_numpy()).unsqueeze(dim=-1)
                with torch.no_grad():
                    alpha = alpha_model(x) ** 2 + 0.1
                    eta = eta_model(x)
                regressor_optim.zero_grad()
                prediction = regressor(x)
                target = alpha * torch.exp((y + eta) / (-alpha - eps) - 1) + eta + alpha * rho
                loss = criterion(prediction, target)
                loss.backward()
                regressor_optim.step()
            # Compute the expected bound using H(X,Y) and h(X)
            dataset = selected_curr_data1 if treatment == 0 else selected_curr_data0
            x0 = torch.Tensor(selected_curr_data0[x_features].to_numpy())
            x1 = torch.Tensor(selected_curr_data1[x_features].to_numpy())
            y = (1 if is_lower_bound else -1) * torch.Tensor(dataset["Y0"].to_numpy()).unsqueeze(dim=-1)
            mean_regressor = torch.mean(regressor(x0 if treatment == 0 else x1).detach())
            alpha = alpha_model(x1 if treatment == 0 else x0) ** 2 + 0.1
            eta = eta_model(x1 if treatment == 0 else x0)
            mean_diff = torch.mean(r(x1 if treatment == 0 else x0) * (
                        alpha * torch.exp((y + eta) / (-alpha - eps) - 1) + eta + alpha * rho - regressor(
                    x1 if treatment == 0 else x0)))
            bounds[i] = (mean_diff + mean_regressor).detach().item()
            # Return average of the three estimated bounds
        return (-1 if is_lower_bound else 1) * np.mean(bounds)


    def find_breakdown_rho(self, data: DataObject, target: float = 0.0, approach: str = "evar",
                           rho_ladder=(0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5),
                           refine_steps: int = 2, verbose: bool = False) -> dict:
        """
        Inverse of `create_bounds`: instead of taking rho as input, return the smallest rho
        at which the partial-identification bounds reach `target` (default 0, i.e. the null ATE).

        Strategy: walk up an ascending ladder of rho values, evaluating only the side of the
        bound that faces the target (lower if target < point estimate, upper otherwise). The
        bounds are monotone in rho, so the first ladder rung that crosses `target` brackets
        the crossing together with the previous rung. A short linear-interpolation refinement
        then locates the crossing inside that bracket. This avoids the cost of evaluating EVaR
        at large rho when small rho would already suffice — important because each EVaR call
        gets substantially slower as rho grows.

        :param data: DataObject
        :param target: bound value to solve for (default 0)
        :param approach: solver approach ('evar', 'lagr', 'cp')
        :param rho_ladder: ascending sequence of rho values to probe. The largest value is the
            search ceiling — if the bound never crosses `target` on the ladder, `reached=False`
            and `rho` returns the ceiling.
        :param refine_steps: number of linear-interpolation refinement evaluations inside the
            crossing bracket. 0 disables refinement (uses the bracketing-rung rho).
        :param verbose: if True, prints each probe (rho, bound value, elapsed) so progress is
            visible — EVaR calls can be slow.
        :return: dict with keys
            - rho:      breakdown rho (or ladder ceiling if not reached)
            - side:     'lower' or 'upper' — which bound was tracked
            - reached:  True if the bound actually crossed `target` within the ladder
            - lower0, upper0:  bounds at rho = 0 (point identification interval)
        """
        ladder = sorted(set(float(r) for r in rho_ladder if r > 0))
        if not ladder:
            raise ValueError("rho_ladder must contain at least one positive value")
        ceiling = ladder[-1]

        if verbose:
            print(f"  ρ=0.000  computing point-identification bounds...", flush=True)
        t0 = time.time()
        lo0, hi0 = self.create_bounds(data, 0.0, approach)
        lo0, hi0 = float(lo0), float(hi0)
        if verbose:
            print(f"  ρ=0.000  bounds=[{lo0:.3f}, {hi0:.3f}]  ({time.time()-t0:.1f}s)", flush=True)

        if min(lo0, hi0) <= target <= max(lo0, hi0):
            init_side = "lower" if target < 0.5 * (lo0 + hi0) else "upper"
            init_val0 = lo0 if init_side == "lower" else hi0
            return {"rho": 0.0, "side": init_side,
                    "reached": True, "lower0": lo0, "upper0": hi0,
                    "probes": [(0.0, float(init_val0))]}

        point = 0.5 * (lo0 + hi0)
        track_lower = target < point
        side_label = "lower" if track_lower else "upper"
        init_val = lo0 if track_lower else hi0
        probes = [(0.0, float(init_val))]

        if approach == "evar":
            bound_fn = lambda rho: float(self._solve_evar_formulation(data, rho, track_lower))
        else:
            def bound_fn(rho):
                lo, hi = self.create_bounds(data, rho, approach)
                return float(lo) if track_lower else float(hi)

        def has_crossed(val: float) -> bool:
            return val <= target if track_lower else val >= target

        rho_prev, val_prev = 0.0, init_val
        for rho in ladder:
            t0 = time.time()
            val = bound_fn(rho)
            probes.append((float(rho), float(val)))
            if verbose:
                print(f"  ρ={rho:.3f}  {side_label}={val:.3f}  ({time.time()-t0:.1f}s)", flush=True)
            if has_crossed(val):
                rho_lo, val_lo, rho_hi, val_hi = rho_prev, val_prev, rho, val
                for _ in range(max(0, refine_steps)):
                    if val_hi == val_lo:
                        break
                    rho_star = rho_lo + (target - val_lo) * (rho_hi - rho_lo) / (val_hi - val_lo)
                    rho_star = float(np.clip(rho_star, rho_lo, rho_hi))
                    t1 = time.time()
                    val_star = bound_fn(rho_star)
                    probes.append((float(rho_star), float(val_star)))
                    if verbose:
                        print(f"  ρ={rho_star:.3f}  {side_label}={val_star:.3f}  (refine, {time.time()-t1:.1f}s)", flush=True)
                    if has_crossed(val_star):
                        rho_hi, val_hi = rho_star, val_star
                    else:
                        rho_lo, val_lo = rho_star, val_star
                if val_hi != val_lo:
                    rho_star = rho_lo + (target - val_lo) * (rho_hi - rho_lo) / (val_hi - val_lo)
                    rho_star = float(np.clip(rho_star, rho_lo, rho_hi))
                else:
                    rho_star = rho_hi
                probes.sort(key=lambda p: p[0])
                return {"rho": rho_star, "side": side_label, "reached": True,
                        "lower0": lo0, "upper0": hi0, "probes": probes}
            rho_prev, val_prev = rho, val

        probes.sort(key=lambda p: p[0])
        return {"rho": ceiling, "side": side_label, "reached": False,
                "lower0": lo0, "upper0": hi0, "probes": probes}

    def solve_gaussian_mixture_model(self, data: DataObject, rho: float, is_lower_bound: bool, means: dict, variances:dict, k: int) -> float:
        X = data.discrete_x()
        p_treated = len(data.data[data.data["T"] == 1]) / len(data.data)
        result_treated = 0
        result_control = 0
        for index, row in X.iterrows():
            x = int(row[data.x_features].to_numpy()[0])
            observed_propensity = data.propensity_score_index(index)
            r_x = (1 - p_treated) * observed_propensity / (p_treated * (1 - observed_propensity))
            # treated
            treated_q = (-1 if is_lower_bound else 1) * 0.5 * np.sqrt(rho * sum(variances['treated'][x]) / (2 * k ** 2)) \
                        + sum(means['treated'][x]) / k \
                        + (-1 if is_lower_bound else 1) * (np.sqrt(0.5 * rho)) / (2 * k)
            result_treated += r_x * data.probability_of_x_index_given_t(index, 1) * treated_q
            # control
            control_q = (-1 if not is_lower_bound else 1) * 0.5 * np.sqrt(
                rho * sum(variances['control'][x]) / (2 * k ** 2)) \
                        + sum(means['control'][x]) / k \
                        + (-1 if not is_lower_bound else 1) * (np.sqrt(0.5 * rho)) / (2 * k)
            result_control += 1 / r_x * data.probability_of_x_index_given_t(index, 0) * control_q
        return result_treated - result_control