"""Run Stochastic Simulation Algorithm on a Nvidia GPU device, using pycuda.

Each thread is assigned one iteration of the algorithm, because each iteration is a distinct
stochastic process, and therefore can only be horizontally parallelized.

Inside this kernel we use a "binary selection" to randomly pick a value according to its rate.


TODO:
- this kernel should be repeated many times, until the maximum time requested is exhausted, because
  there are no "lists" on cuda/C kernels, so we have to split execution in chunks of defined length
  (_num_steps) in order to use arrays: the following chunk starts from the end of the previous one;
- use a proper random generator and initialize its seed;
- deal with GPUs with different specs, such as the block and grid thread dimensions, the device
  memory (adapt the number of repetitions to the amount of memory available)
  https://documen.tician.de/pycuda/util.html#pycuda.tools.DeviceData
"""

from copy import deepcopy
import numpy as np
import pycuda.autoinit  # noqa
from pycuda.compiler import SourceModule
from pycuda.curandom import rand as curand
import pycuda.gpuarray as gpuarray

_kernel_str = """
__global__ void ssa_simple(float *_update_matrix,
                           float *_init_conditions,
                           const float start_time,
                           const float end_time,
                           float *_time_and_states,
                           const float *_rand_arr) {

  size_t rep = 0, th_id = threadIdx.x;

  // Set the initial time and conditions associated with this run.
  _time_and_states[th_id * (@num__reacs@ + 1) * @num__rep@] = start_time;
  for (size_t i = 0; i < @num__reacs@; ++i)
    _time_and_states[th_id * (@num__reacs@ + 1) * @num__rep@ + 1 + i] = _init_conditions[th_id * (@num__reacs@ + 1) + i];


  float simul_t = start_time, _prev_simul_t = start_time;
  float _rates_arr[@num__reacs@];
  float _binary_rates_arr[@num__reacs@ * 2 - 1];
  // The index where start the real rates in the binary vector; before it there are the partial
  // sums of rates.
  size_t _stop_bin_search = (@num__reacs@ + 1) / 2 - 1;


  while (simul_t < end_time and rep + 1 < @num__rep@) {

    // -------------- start unrolling user functions  --------------
    @unroll__func__rate@
    // --------------  end unrolling user functions   --------------

    float total_rate = 0;
    for (size_t i = 0; i < @num__reacs@; ++i)
      total_rate += _rates_arr[i];

    float rnd_react = _rand_arr[th_id * @num__rep@ * 2 + rep * 2] * total_rate;
    float rnd_time  = _rand_arr[th_id * @num__rep@ * 2 + rep * 2 + 1] + 1e-10;

    simul_t = -logf(rnd_time) / total_rate + _prev_simul_t;

    // When selecting the next reaction to occur in the algorithm, we want to randomly select a
    // reaction: the reactions with the highest rates should be selected more often.
    // We produce a tree containing partial sums of rates, then we descend into the tree with a
    // top-down traversal and select a branch only if it's greater than the random value
    // extracted.

    // The binary array is made of the sums of elements 2*j+1 and 2*j+2 in j-th position, and
    // of the original rates at the end.
    for (size_t i = 0; i < @num__reacs@; ++i)      // copy original rates at the end of the temp array
      _binary_rates_arr[@num__reacs@ - 1 + i] = _rates_arr[i];
    for (size_t i = @num__reacs@ - 2; i > 0; --i)  // compute the partial sums from the original rates
      _binary_rates_arr[i] = _binary_rates_arr[2 * i + 1]  + _binary_rates_arr[2 * i + 2];
    _binary_rates_arr[0] = _binary_rates_arr[1]  + _binary_rates_arr[2];

    size_t idx_react = 0;
    while (idx_react < _stop_bin_search) {
      if (_rates_arr[2 * idx_react + 1] >= rnd_react)
        idx_react = 2 * idx_react + 1;
      else {
        rnd_react -= _rates_arr[2 * idx_react + 1];
        idx_react = 2 * idx_react + 2;
      }
    }
    size_t chosen_react = idx_react - _stop_bin_search;

    // Save time and states of this run.
    _time_and_states[th_id * (@num__reacs@ + 1) * @num__rep@ + (rep + 1) * (@num__reacs@ + 1)] = simul_t;
    for (size_t i = 0; i < @num__reacs@; ++i)
      _time_and_states[th_id * (@num__reacs@ + 1) * @num__rep@ + (rep + 1) * (@num__reacs@ + 1) + 1 + i] = _time_and_states[th_id * (@num__reacs@ + 1) * @num__rep@ + rep * (@num__reacs@ + 1) + 1 + i] + _update_matrix[chosen_react * @num__reacs@ + i];

    _prev_simul_t = simul_t;
    ++rep;
  }
}
"""


def SSA(update_matrix, initial_conditions, function_rates, t_max, **kwargs):  # noqa

    # Fix the maximum number of steps available at each repetition. Should be function of the
    # amount of memory available on the device and the number of iterations (= threads) requested.
    _num_steps = 20
    _num_reacs = len(kwargs["variables"])
    start_time, end_time = np.float32(0), np.float32(t_max)

    function_rates_wo_param = deepcopy(function_rates)
    for fr_id, f_rate in enumerate(function_rates_wo_param):
        for par, val in kwargs["parameters"].items():
            f_rate = f_rate.replace(par, str(val))
        for sp_id, spec in enumerate(kwargs["variables"]):
            f_rate = f_rate.replace(spec, "_time_and_states[th_id * (@num__reacs@ + 1) * @num__rep@"
                                    " + rep * (@num__reacs@ + 1) + 1 + {}]".format(sp_id))

        function_rates_wo_param[fr_id] = f_rate

    unroll_func_rate = "\n".join((f_rate.join(("_rates_arr[{}] = ".format(fr_id), ";"))
                                  for fr_id, f_rate in enumerate(function_rates_wo_param)))

    kernel_ready = _kernel_str \
        .replace("@unroll__func__rate@", unroll_func_rate) \
        .replace("@num__iter@", str(kwargs["iterations"])) \
        .replace("@num__rep@", str(_num_steps)) \
        .replace("@num__reacs@", str(_num_reacs))

    if kwargs.get("print_cuda"):
        print("\n".join(" ".join((str(line_no + 2), line))
                        for line_no, line in enumerate(kernel_ready.split("\n"))))

    upd_mat_dev = gpuarray.to_gpu(update_matrix.astype(np.float32))

    # The vector of initial conditions has to be repeated for each thread, since in the future,
    # when we will split in chunks, each chunk will restart from a different initial condition.
    init_cond_dev = gpuarray.to_gpu(np.tile(initial_conditions.astype(np.float32),
                                            (kwargs["iterations"], 1)))

    # Each thread should produce its own array of random numbers or at least have access to a
    # private set of random numbers: we need two numbers for each repetition, one to select the
    # reaction and one to select the time.
    # Note that pycuda.curandom.rand is a toy-random generator, and all the threads share the array.
    # https://documen.tician.de/pycuda/array.html?highlight=random#module-pycuda.curandom
    rand_arr_dev = curand((_num_steps, 2, kwargs["iterations"]))

    # There seems to be no need to manually copy back to host gpuarrays, see example/demo.py.
    time_states_dev = gpuarray.GPUArray((kwargs["iterations"], _num_steps, _num_reacs + 1),
                                        dtype=np.float32)

    mod = SourceModule(kernel_ready)
    func = mod.get_function("ssa_simple")
    func(upd_mat_dev, init_cond_dev, start_time, end_time, time_states_dev, rand_arr_dev,
         block=(kwargs["iterations"], 1, 1))

    return time_states_dev
