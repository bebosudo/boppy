import multiprocessing as mp
import numbers
import numpy as np

from .core import (VariableCollection, ParameterCollection, Parameter, RateFunctionCollection,
                   ReactionCollection, InputError)
from .simulators import ssa, next_reaction_method, fluid_approximation
from .simulators.gpu import ssa_gpu


ALGORITHMS_AVAIL = ("ssa", "gillespie", "nrm", "next reaction method", "gibson bruck",
                    "gibson-bruck", "fluid approximation", "fluid limit", "mean field",
                    "ode", "tau-leaping")

global ALG_INPUT


def _dummy_function(proc_num=None):
    func, args, secondary_args = ALG_INPUT
    return func(*args, **secondary_args)


def boppy_setup(alg_params_dict, simul_params_dict):
    if simul_params_dict.get("Use GPU", False):
        return MainControllerGPU(alg_params_dict, simul_params_dict)
    return MainControllerCPU(alg_params_dict, simul_params_dict)


class MainControllerCommon:
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

        self._t_max = self._orig_simul_dict.get("Maximum simulation time")
        if not isinstance(self._t_max, numbers.Number):
            raise InputError("The maximum simulation time parameter (t_max) must be a number. "
                             "Found: {}.".format(self._orig_simul_dict["Maximum simulation time"]))

        self._alg_chosen = self._orig_simul_dict.get("Simulation")
        if (not isinstance(self._alg_chosen, str) or
                self._alg_chosen.lower() not in ALGORITHMS_AVAIL):
            raise InputError("The algorithm to use in the simulation must be a string in "
                             "{}.".format(", ".join((repr(alg) for alg in ALGORITHMS_AVAIL))))

        iterations_label = "Algorithm iterations"
        self._iterations = self._orig_simul_dict.get(iterations_label, 1)
        if not isinstance(self._iterations, int):
            raise InputError("The '{}' parameter has to be an integer.".format(iterations_label))
        self._iterations = 1 if self._iterations < 1 else self._iterations

        nproc_label = "Number of processes"
        self._nproc = self._orig_simul_dict.get(nproc_label, mp.cpu_count())
        if not isinstance(self._nproc, int):
            raise InputError("The '{}' parameter has to be an integer.".format(nproc_label))
        self._nproc = mp.cpu_count() if self._nproc < 1 else self._nproc

        self._variables = VariableCollection(self._orig_alg_dict["Species"])

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

        self._secondary_args = {}
        self._setup_alg_and_secondary_param(self._alg_chosen)

    def _setup_alg_and_secondary_param(self, str_alg):
        # Must be implemented in the child classes.
        raise NotImplementedError

    def simulate(self):
        # Must be implemented in the child classes.
        raise NotImplementedError

    @property
    def species(self):
        return self._variables


class MainControllerCPU(MainControllerCommon):
    """Controller for CPU-based processes."""

    def __init__(self, alg_params_dict, simul_params_dict):
        super(MainControllerCPU, self).__init__(alg_params_dict, simul_params_dict)

        # Treat the system size as a parameter, so it's substituted, e.g. in RateFunction objects.
        self._parameters = ParameterCollection(dict(self._orig_alg_dict["Parameters"],
                                                    **self._orig_alg_dict["System size"]))
        self._parameters_wo_system_size = ParameterCollection(self._orig_alg_dict["Parameters"])

        self._rate_functions = RateFunctionCollection(self._orig_alg_dict["Rate functions"],
                                                      self._variables, self._parameters)
        self._rf_var_system_size = RateFunctionCollection(self._orig_alg_dict["Rate functions"],
                                                          self._variables,
                                                          self._parameters_wo_system_size)

    def _setup_alg_and_secondary_param(self, str_alg):
        """Given the algorithm name, associate a function and extend the secondary parameters.

        Does not check again whether the algorithm in the string passed is part of the available
        algorithms, since it should have already been checked in the `__init__`.

        Save into `self._secondary_args` optional arguments that are then passed to the simulator.
        """
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

    def simulate(self):
        # Using a global variable is dirty: the preferred way would be to use a mp.starmap, and
        # pass it the ALG_INPUT parameters; this doesn't work because inside _rate_functions there
        # are lambda functions, which cannot be pickled to be transferred to the other processes.
        # This (should) work safely because ALG_INPUT is only read by processes, but not written.
        global ALG_INPUT

        ALG_INPUT = (self._selected_alg, (self.update_matrix, self._initial_conditions,
                                          self._rate_functions, self._t_max),
                     self._secondary_args)

        with mp.Pool(processes=self._nproc) as pool:
            populations_and_times = pool.map(_dummy_function, range(self._iterations))

        # The output from the map is a list of numpy 2D arrays; they are the result of stochastic
        # processes and their content is variable; the pairs can have different lengths, so we
        # cannot pack them into a multidimensional array.
        return populations_and_times


class MainControllerGPU(MainControllerCommon):
    """Controller for GPU-based processes."""

    def __init__(self, alg_params_dict, simul_params_dict):
        super(MainControllerGPU, self).__init__(alg_params_dict, simul_params_dict)

        self._rate_functions = self._orig_alg_dict["Rate functions"]
        self._secondary_args["parameters"] = dict(self._orig_alg_dict["Parameters"],
                                                  **self._orig_alg_dict["System size"])
        self._secondary_args["iterations"] = self._iterations
        self._secondary_args["variables"] = self._variables.orig_vars

        self._secondary_args["print_cuda"] = self._orig_simul_dict.get("Print CUDA kernel", False)
        if not isinstance(self._secondary_args["print_cuda"], bool):
            raise InputError("The option to print the kernel must be true/false or no/yes; "
                             "found '{}'.".format(self._secondary_args["print_cuda"]))

    def _setup_alg_and_secondary_param(self, str_alg):
        """Given the algorithm name, associate a function and extend the secondary parameters.

        Does not check again whether the algorithm in the string passed is part of the available
        algorithms, since it should have already been checked in the `__init__`.
        """
        # Save into `self._secondary_args` optional arguments that are then passed to the simulator.
        if str_alg.lower() in ("ssa", "gillespie"):
            self._selected_alg = ssa_gpu.SSA

        else:
            raise NotImplementedError("The chosen algorithm '{}' has not been "
                                      "implemented yet.".format(str_alg))

    def simulate(self):
        return self._selected_alg(self.update_matrix, self._initial_conditions,
                                  self._rate_functions, self._t_max, **self._secondary_args)
