# Adapted from "f-sensitivity-through-evar" by Matej Havelka, used under the MIT License.
# Source: https://github.com/MatejHav/f-sensitivity-through-evar
# Copyright (c) 2024 Matej Havelka. See LICENSE at the repository root for full terms.

import pandas as pd
import numpy as np
import time

from pyomo.core import *
from pyomo.environ import *
from sklearn.linear_model import QuantileRegressor

from data_generation import *


class MSM:

    def create_bounds(self, data: DataObject, gamma: float, approach: str) -> tuple:
        """
        Method which automatically computes the bound based on selected approach.

        :param data: DataFrame containing all the data
        :param gamma: Gamma representing the bound of the odds-ratio \(\Gamma\)
        :param approach: String representation of the approach, default is the constraint programming implementation.
        Possible approaches include "cp" constraint programming or "cvar" cvar.
        :return: Tuple representing the lower and upper bound
        """
        bound_founder = self._solve_cvar_formulation if approach == "cvar" \
            else self._solve_constraint_programming_formulation
        # Compute the bounds
        lower_control_bound = bound_founder(data, gamma, 0, True)
        lower_treated_bound = bound_founder(data, gamma, 1, True)
        upper_control_bound = bound_founder(data, gamma, 0, False)
        upper_treated_bound = bound_founder(data, gamma, 1, False)

        # Output lower and upper bounds based on best and worst cases
        return lower_treated_bound - upper_control_bound, upper_treated_bound - lower_control_bound

    def find_breakdown_gamma(self, data: DataObject, target: float = 0.0, approach: str = "cvar",
                             gamma_ladder=(1.25, 1.5, 2.0, 3.0, 5.0, 8.0),
                             refine_steps: int = 2, verbose: bool = False) -> dict:
        """MSM analog of FSensitivity.find_breakdown_rho: the smallest odds-ratio bound Gamma at
        which the MSM partial-identification ATE interval first reaches `target` (default 0, the
        null ATE). Gamma=1 is point identification (interval collapses to the point estimate) and
        the interval widens monotonically with Gamma, so the first ladder rung whose interval
        contains `target` brackets the crossing with the previous rung; a short linear refinement
        then locates it. Robust to create_bounds returning the (lo, hi) pair in either order: it
        tracks the interval EDGE facing the target via min/max, not the tuple position.

        :return: dict with keys gamma, side ('lower'/'upper'), reached, lower0, upper0
            (point-identification interval at Gamma=1), probes (list of (gamma, facing-edge)).
        """
        ladder = sorted(set(float(g) for g in gamma_ladder if g > 1.0))
        if not ladder:
            raise ValueError("gamma_ladder must contain at least one value > 1")
        ceiling = ladder[-1]

        if verbose:
            print(f"  Γ=1.000  computing point-identification bounds...", flush=True)
        t0 = time.time()
        lo1, hi1 = self.create_bounds(data, 1.0, approach)
        lo0, hi0 = min(float(lo1), float(hi1)), max(float(lo1), float(hi1))
        if verbose:
            print(f"  Γ=1.000  bounds=[{lo0:.3f}, {hi0:.3f}]  ({time.time()-t0:.1f}s)", flush=True)

        if lo0 <= target <= hi0:
            side = "lower" if target < 0.5 * (lo0 + hi0) else "upper"
            return {"gamma": 1.0, "side": side, "reached": True,
                    "lower0": lo0, "upper0": hi0, "probes": [(1.0, lo0 if side == "lower" else hi0)]}

        point = 0.5 * (lo0 + hi0)
        track_lower = target < point
        side_label = "lower" if track_lower else "upper"

        def edge_fn(gamma):
            lo, hi = self.create_bounds(data, gamma, approach)
            return min(float(lo), float(hi)) if track_lower else max(float(lo), float(hi))

        def has_crossed(val):
            return val <= target if track_lower else val >= target

        probes = [(1.0, lo0 if track_lower else hi0)]
        g_prev, v_prev = 1.0, (lo0 if track_lower else hi0)
        for gamma in ladder:
            t0 = time.time()
            val = edge_fn(gamma)
            probes.append((float(gamma), float(val)))
            if verbose:
                print(f"  Γ={gamma:.3f}  {side_label}={val:.3f}  ({time.time()-t0:.1f}s)", flush=True)
            if has_crossed(val):
                g_lo, v_lo, g_hi, v_hi = g_prev, v_prev, gamma, val
                for _ in range(max(0, refine_steps)):
                    if v_hi == v_lo:
                        break
                    g_star = float(np.clip(g_lo + (target - v_lo) * (g_hi - g_lo) / (v_hi - v_lo), g_lo, g_hi))
                    t1 = time.time()
                    v_star = edge_fn(g_star)
                    probes.append((g_star, float(v_star)))
                    if verbose:
                        print(f"  Γ={g_star:.3f}  {side_label}={v_star:.3f}  (refine, {time.time()-t1:.1f}s)", flush=True)
                    if has_crossed(v_star):
                        g_hi, v_hi = g_star, v_star
                    else:
                        g_lo, v_lo = g_star, v_star
                if v_hi != v_lo:
                    g_star = float(np.clip(g_lo + (target - v_lo) * (g_hi - g_lo) / (v_hi - v_lo), g_lo, g_hi))
                else:
                    g_star = g_hi
                probes.sort(key=lambda p: p[0])
                return {"gamma": g_star, "side": side_label, "reached": True,
                        "lower0": lo0, "upper0": hi0, "probes": probes}
            g_prev, v_prev = gamma, val

        probes.sort(key=lambda p: p[0])
        return {"gamma": ceiling, "side": side_label, "reached": False,
                "lower0": lo0, "upper0": hi0, "probes": probes}

    def _solve_constraint_programming_formulation(self, data: DataObject, gamma: float, treatment: int,
                                                  is_lower_bound: bool) -> float:
        """
        Constraint programming formulation of the MSM based on Tan 2006. To solve this the GLPK solver is used. GLPK
        is a linear programming solver, so using it is better than a constraint programming one like ipopt.

        :param data: Data Object holding the entire dataset inside it. Used for propensity and prior calculations.
        :param gamma: Bound of the odds-ratio \(\Gamma\)
        :param treatment: Integer representing the treatment. Binary treatment is assumed.
        :param is_lower_bound: Boolean representing whether to compute upper or lower bound.
        :return: The computed bound.
        """
        X = data.discrete_x()
        Y = data.discrete_y()
        model = ConcreteModel(name="MarginalSensitivityModel")
        model.X = RangeSet(0, len(X) - 1)
        model.Y = RangeSet(0, len(Y) - 1)
        model.lam = Var(model.X, model.Y, bounds=(1 / gamma, gamma), initialize=1)

        # Constraint 1: Lambda is a distribution of Y | X, T
        def distribution_constraint(model):
            return sum([sum(model.lam[x, y] * data.probability_of_x_index_y_index_given_t(x, y, treatment)
                            for y in model.Y)
                        for x in model.X]) == 1

        model.c1 = Constraint(rule=distribution_constraint)

        # Constraint 2: Propensity scores remains unchanged
        def propensity_constraint(model):
            return sum(
                [sum([data.propensity_score_index(x) * model.lam[x, y] * data.probability_of_x_index_y_index_given_t(x,
                                                                                                                     y,
                                                                                                                     treatment)
                      for y in model.Y]) for x in model.X]) == sum(
                [sum([data.propensity_score_index(x) * data.probability_of_x_index_y_index_given_t(x, y, treatment)
                      for y in model.Y]) for x in model.X])

        model.c2 = Constraint(rule=propensity_constraint)

        # Constraint 3: Distribution of Y unchanged
        def p_constraint(model):
            return sum(
                [sum([data.probability_of_x_index_y_index_given_t(x, y, treatment) * model.lam[x, y] * X.iloc[x]["size"]
                      for x in model.X]) / sum(X["size"]) for y in model.Y]) == \
                sum([sum([data.probability_of_x_index_y_index_given_t(x, y, treatment) * X.iloc[x]["size"]
                          for x in model.X]) / sum(X["size"]) for y in model.Y])

        model.c3 = Constraint(rule=p_constraint)

        # Objective: Find the bound
        def objective_function(model):
            return sum([sum([Y.iloc[y] * model.lam[x, y] * data.probability_of_x_index_y_index_given_t(x, y, treatment)
                             for y in model.Y]) for x in model.X])

        model.OBJ = Objective(rule=objective_function,
                              sense=pyomo.core.minimize if is_lower_bound else pyomo.core.maximize)

        opt = SolverFactory('glpk')
        opt.options['tmlim'] = 100
        opt.solve(model)
        return model.OBJ()

    def _solve_cvar_formulation(self, data: DataObject, gamma: float, treatment: int, is_lower_bound: bool) -> float:
        """
        Uses CVaR formulation of the MSM problem. The Quantile Regressor used has a linear solver within which can take
        quite some time to compute.

        :param data: Data Object holding the entire dataset inside it. Used for propensity and prior calculations.
        :param gamma: Bound of the odds-ratio \(\Gamma\)
        :param treatment: Integer representing the treatment. Binary treatment is assumed.
        :param is_lower_bound: Boolean representing whether to compute upper or lower bound.
        :return: The computed bound.
        """
        tau = gamma / (gamma + 1)
        if is_lower_bound:
            tau = 1 - tau
        X = data.discrete_x()
        result = 0
        for x_index in range(len(X)):
            selection = data.select_x_index(x_index, treatment)
            x = selection[data.x_features]
            y = selection["Y"].to_numpy()
            regressor = QuantileRegressor(quantile=tau, solver='highs')
            regressor.fit(x, y)
            pred = regressor.predict(x)
            res = 1 / gamma * y + (1 - 1 / gamma) * (pred + 1 / (1 - tau) * (y - pred))
            result += res.mean() * data.probability_of_x_index_given_t(x_index, treatment)
        return result
