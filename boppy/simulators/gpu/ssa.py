"""Run Stochastic Simulation Algorithm on a Nvidia GPU device, using pycuda.

Each thread is assigned one iteration of the algorithm, because each iteration is a distinct
stochastic process, and therefore can only be horizontally parallelized.

TODO:
- this kernel should be repeated many times, until the maximum time requested is exhausted, because
  there are no "lists" on cuda/C kernels, so we have to split execution in chunks of defined length
  in order to use arrays: the following chunk starts from the end of the previous one;
- create the empty time_and_states matrix directly on the device;
- use a proper random generator for cuda and initialize its seed;
"""

import pycuda.driver as cuda
import pycuda.autoinit  # noqa
from pycuda.compiler import SourceModule
from pycuda.curandom import rand as curand
import numpy as np


species = ("x_s", "x_i", "x_r")
params = {"k_s": "0.01", "k_i": "1.", "k_r": "0.05", "N": "100"}
function_rates = ("k_i * x_i * x_s / N", "k_r * x_i", "k_s * x_r")
_parallel_iterations = 3
_partial_repetitions = 20
# _chunks = 5
_num_reacs = len(species)
start_time, end_time = np.float32(15), np.float32(60)

arr_orig = np.empty((_parallel_iterations, _partial_repetitions, _num_reacs + 1), dtype=np.float32)
arr_dev = cuda.mem_alloc(arr_orig.size * arr_orig.dtype.itemsize)
cuda.memcpy_htod(arr_dev, arr_orig)

update_matrix = np.array([[-1.,  1.,  0.],
                          [0., -1.,  1.],
                          [1.,  0., -1.]], dtype=np.float32)
upd_mat_dev = cuda.mem_alloc(update_matrix.size * update_matrix.dtype.itemsize)
cuda.memcpy_htod(upd_mat_dev, update_matrix)

init_conditions = np.empty((_parallel_iterations, _num_reacs + 1), dtype=np.float32)
init_conditions[:, 0] = 80
init_conditions[:, 1] = 20
init_conditions[:, 2] = 0
init_cond_dev = cuda.mem_alloc(init_conditions.size * init_conditions.dtype.itemsize)
cuda.memcpy_htod(init_cond_dev, init_conditions)

# Each thread should produce its own array of random numbers.
# Note that pycuda.curandom.rand is a toy-random generator, and all the threads share the array produced.
# https://documen.tician.de/pycuda/array.html?highlight=random#module-pycuda.curandom
rand_arr_dev = curand((_partial_repetitions, 2, _parallel_iterations))

kernel_str = """
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

function_rates_wo_param = list(function_rates)
for fr_id, f_rate in enumerate(function_rates_wo_param):
    for par, val in params.items():
        f_rate = f_rate.replace(par, val)
    for sp_id, spec in enumerate(species):
        f_rate = f_rate.replace(
            spec, "_time_and_states[th_id * (@num__reacs@ + 1) * @num__rep@ + rep * (@num__reacs@ + 1) + 1 + {}]".format(sp_id))

    function_rates_wo_param[fr_id] = f_rate

unroll_func_rate = "\n".join((f_rate.join(("_rates_arr[{}] = ".format(fr_id), ";"))
                              for fr_id, f_rate in enumerate(function_rates_wo_param)))

kernel_ready = kernel_str\
    .replace("@unroll__func__rate@", unroll_func_rate)\
    .replace("@num__iter@", str(_parallel_iterations))\
    .replace("@num__rep@", str(_partial_repetitions))\
    .replace("@num__reacs@", str(_num_reacs))

# print("\n".join(" ".join((str(line_no + 2), line))
#                 for line_no, line in enumerate(kernel_ready.split("\n"))))

mod = SourceModule(kernel_ready)

func = mod.get_function("ssa_simple")
func(upd_mat_dev, init_cond_dev, start_time, end_time,
     arr_dev, rand_arr_dev, block=(_parallel_iterations, 1, 1))

arr_after = np.empty_like(arr_orig)
cuda.memcpy_dtoh(arr_after, arr_dev)

# print("original array:")
# print(arr_orig)

print('after kernel execution (each "group" is computed by a different thread, first column is time):')
print(arr_after)
