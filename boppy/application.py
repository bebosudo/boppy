import numbers
import numpy as np

from .core import (VariableCollection, ParameterCollection, Parameter, RateFunctionCollection,
                   ReactionCollection, InputError)
from .simulators import ssa, next_reaction_method, fluid_approximation


ALGORITHMS_AVAIL = ("ssa", "gillespie", "nrm", "next reaction method", "gibson bruck",
                    "gibson-bruck", "fluid approximation", "fluid limit", "mean field",
                    "ode", "tau-leaping")


class MainController:
    """Entry point class that handles the input interpretation and the simulation execution."""

    def __init__(self, alg_params_dict, simul_params_dict):

        self._orig_alg_dict = alg_params_dict
        self._orig_simul_dict = simul_params_dict

        if len(self._orig_alg_dict["Parameters"]) != len(self._orig_alg_dict["Rate functions"]):
            raise InputError("The number of Parameters ({}) is different from the number of Rate"
                             " functions ({})".format(len(self._orig_alg_dict["Parameters"]),
                                                      len(self._orig_alg_dict["Rate functions"])))
        elif len(self._orig_alg_dict["Initial conditions"]) != len(self._orig_alg_dict["Species"]):
            raise InputError("There must be an initial condition for each species.")
        elif len(self._orig_alg_dict["System size"]) != 1:
            raise InputError("The size of the system must be a single parameter. "
                             "Found: {}.".format(len(self._orig_alg_dict["System size"])))

        self._t_max = self._orig_simul_dict.get("Maximum simulation time", False)
        if not self._t_max or not isinstance(self._t_max, numbers.Number):
            raise InputError("The maximum simulation time parameter (t_max) must be a number. "
                             "Found: {}.".format(self._orig_simul_dict["Maximum simulation time"]))

        self._alg_chosen = self._orig_simul_dict["Simulation"]
        if not isinstance(self._alg_chosen, str):
            raise InputError("The algorithm to use in the simulation must be a single string.")
        elif self._alg_chosen.lower() not in ALGORITHMS_AVAIL:
            raise InputError("The algorithm to use in the simulation must be a string in "
                             "{}.".format(", ".join((repr(alg) for alg in ALGORITHMS_AVAIL))))

        self._variables = VariableCollection(self._orig_alg_dict["Species"])

        # Treat the system size as a parameter, so it's substituted, e.g. in RateFunction objects.
        self._parameters = ParameterCollection(dict(self._orig_alg_dict["Parameters"],
                                                    **self._orig_alg_dict["System size"]))
        self._parameters_wo_system_size = ParameterCollection(self._orig_alg_dict["Parameters"])

        self._rate_functions = RateFunctionCollection(self._orig_alg_dict["Rate functions"],
                                                      self._variables, self._parameters)
        self._rf_var_system_size = RateFunctionCollection(self._orig_alg_dict["Rate functions"],
                                                          self._variables,
                                                          self._parameters_wo_system_size)

        self._reactions = ReactionCollection(self._orig_alg_dict["Reactions"], self._variables)
        self.update_matrix = self._reactions.update_matrix

        self._system_size = Parameter(*tuple(self._orig_alg_dict["System size"].items())[0])

        # Extract the vector of initial conditions, with some checks on the species provided.
        self._initial_conditions = np.empty(len(self._orig_alg_dict["Initial conditions"]))
        for species, initial_amount in self._orig_alg_dict["Initial conditions"].items():
            if not self._variables.get(species, False):
                raise InputError("Initial condition '{}: {}' does not match any species "
                                 "provided.".format(species, initial_amount))
            self._initial_conditions[self._variables[species].pos] = initial_amount

        self.simulation_out_times = None
        self.simulation_out_population = None
        self._associate_alg(self._alg_chosen)

    def _associate_alg(self, str_alg):
        """Given the algorithm, set the function that matches it and a dict of optional parameters.

        Does not check again whether the algorithm in the string passed is part of the available
        algorithms, since it should have already been checked in the `__init__`.

        Save into `self._secondary_args` optional arguments that are then passed to the
        simulator.
        """
        self._secondary_args = {}

        if str_alg.lower() in ("ssa", "gillespie"):
            self._selected_alg = ssa.SSA
        elif str_alg.lower() in ("nrm", "next reaction method", "gibson bruck", "gibson-bruck"):
            self._secondary_args.update({'depends_on': self._reactions.depends_on,
                                         'affects': self._reactions.affects})
            self._selected_alg = next_reaction_method.next_reaction_method
        elif str_alg.lower() in ("fluid approximation", "fluid limit", "mean field", "ode"):
            self._secondary_args.update(
                {'rate_functions_var_ss': self._rf_var_system_size,
                 'variables': self._variables,
                 'system_size': self._system_size.value})
            self._selected_alg = fluid_approximation.fluid_approximation

        else:
            raise NotImplementedError("The chosen algorithm '{}' has not been "
                                      "implemented yet.".format(str_alg))

    @property
    def species(self):
        return self._variables

    def simulate(self):
        population, times = self._selected_alg(self.update_matrix,
                                               self._initial_conditions,
                                               self._rate_functions,
                                               self._t_max,
                                               **self._secondary_args)

        self.simulation_out_population, self.simulation_out_times = population, times
        return population, times
