Stochastic Modelling and Simulation course
==========================================

The course final project consists in a group work. The goal is to implement a Python module, *poppy*, implementing functionalities to model and simulate population processes.

The program has also to be tested on some examples of population models, mainly from systems biology.

The functionalities of the poppy module should be the following:

1. Taking as input a textual file containing a description of the population model in terms of biochemical reactions. Models should contain definition of population variables and of their of initial state, possibly with expressions involving parameters and other variables, of parameters and their values (possibly in terms of expressions), definition of functions, definition of biochemical reactions and their rate function, definition of model outputs, which can be variables or expressions involving variables.
   Models should contain a specification of the system size, so that quantities can be expressed either as population counts or densities.

2. Taking as input a text file (may be a section of the model file) describing properties of interest. Properties include either the value of a model variable or of an output variable at a certain time (e.g. t=10, t=final time of simulation), or at a grid of times, in order to monitor their distribution, or their expectation, or another moment.
   Properties also include reachability properties (is there a t in [t0,t1] such that (output) variable y(t) was in [a,b]), stability properties (for each t in [t0,t1]  (output) variable y(t) was in [a,b]), and in general there should be an interface to which the user has to comply to define linear time properties (which we can call monitors), taking a simulation trajectory as input and returning a value (either 0/1 or a real value) as output

3. The model in Python should be implemented as a class, containing all the required definitions, ideally as instances of other classes (e.g. Reactions, Variables, Parameters, OutputVariables, etc).

4. Simulator objects take a model as input, and implement a simulation method that can be run. Stochastic simulations should exploit CPU, and possibly GPU parallelism.
   Simulations return trajectories or collection of trajectories. Minimal set of simulation methods include: SSA, GB, ODE integration for ODE (mean-field) derived models.

6. Simulators should be combined with a property monitor object, that can process the simulated trajectory to compute the desired property. Ideally, there can be more monitors attached to a simulator object, so that we can evaluate several properties for the same simulation runs.

7. The output of monitors for a batch of simulation runs as to be analysed by a statistical analysis routine, which can compute expectations plus confidence intervals, and in case of boolean monitors, estimate satisfaction probabilities and test if these probabilities are above or below a threshold with a given confidence (using both frequentist and Bayesian methods).
   Note that all these operations can be offline/in batch (all N simulations are run before, then the output analysed) or sequential/online (the analyser updates the analysis after one or a small batch of simulations, and the analyser can stop the simulations when a stopping condiition is met, i.e. when the length of the confidence bound of the estimation is below a given threshold).

8. There should be methods that allow to explore parameter spaces, i.e. evaluate a certain property for each point in a grid of parameters, and to do parameter estimation/design, i.e. optimise an objective function depending on some parameters.

9. There should be plotting routines to visualise simulated data and the analysis (i.e. distributions, expectations with confidence intervals, etc). Independent variables can be time and model parameters. Dependent variables for visualization are all the monitored properties.
