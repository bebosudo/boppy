import pycuda.driver as cuda
import pycuda.autoinit  # noqa
from pycuda.compiler import SourceModule
from pycuda.curandom import rand as curand
import numpy as np

# Each thread is assigned one iteration of the algorithm, because each iteration is a distinct
# stochastic process, and therefore can only be horizontally parallelized.

species = ("x_s", "x_i", "x_r")
# params = {"k_s": "0.01", "k_i": "1.", "k_r": "0.05", "N": "100"}
# function_rates = ("k_i * x_i * x_s / N", "k_r * x_i", "k_s * x_r")
params = {"k_s": "1", "k_i": "2", "k_r": "3", "N": "10"}
function_rates = ("k_s * x_s", "k_i * x_i", "k_r * x_r")

_parallel_iterations = 3
_partial_repetitions = 5
# _steps = 5
_num_reacs = len(species)
start_time, end_time = np.float32(15), np.float32(60)

arr_orig = np.ones((_partial_repetitions, _num_reacs + 1, _parallel_iterations), dtype=np.float32)
arr_dev = cuda.mem_alloc(arr_orig.size * arr_orig.dtype.itemsize)
cuda.memcpy_htod(arr_dev, arr_orig)

# Each thread should produce its own array of random numbers.
# Note that pycuda.curandom.rand is a toy-random generator, and all the threads share the array produced.
# https://documen.tician.de/pycuda/array.html?highlight=random#module-pycuda.curandom
rand_arr_dev = curand((_partial_repetitions, 2, _parallel_iterations))

kernel_str = """
__global__ void ssa_simple(float *_time_and_states,
                           const float *_rand_arr,
                           const float start_time,
                           const float end_time) {

    size_t rep = 0, th_id = threadIdx.x;
    if (th_id >= @num__iter@)
        return;

    // if (th_id == 0){
    //   printf("thread num: %d, ", th_id);
    //   printf("num rep: %d, ", @num__rep@);
    //   printf("num iter: %d\\n", @num__iter@);}
    float simul_t = start_time;
    float _rates_arr[@num__frate@];
    // if (th_id != 0){
    //   printf("thread num: %d, ", th_id);
    //   printf("num rep: %d, ", @num__rep@);
    //   printf("num iter: %d\\n\\n", @num__iter@);}


    while (simul_t < end_time and rep < @num__rep@) {

        // -------------- start unrolling user functions  --------------
        @unroll__func__rate@
        // --------------  end unrolling user functions   --------------

        float total_rate = 0;
        for(size_t i = 0; i < @num__frate@; ++i)
            total_rate += _rates_arr[i];

        float rnd_react = _rand_arr[th_id * @num__rep@ * 2 + rep * 2] * total_rate;
        float rnd_time  = _rand_arr[th_id * @num__rep@ * 2 + rep * 2 + 1];

        simul_t = -logf(rnd_time) / total_rate + _time_and_states[th_id * (@num__frate@ + 1) * @num__rep@ + rep * (@num__frate@ + 1)];
        // printf("simul time %f\\n", simul_t);

        _time_and_states[th_id * (@num__frate@ + 1) * @num__rep@ + rep * (@num__frate@ + 1)] = simul_t;

        for (size_t i = 0; i < @num__frate@; ++i)
            _time_and_states[th_id * (@num__frate@ + 1) * @num__rep@ + rep * (@num__frate@ + 1) + 1 + i] = _rates_arr[i];

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
            spec, "_time_and_states[th_id * (@num__frate@ + 1) * @num__rep@ + rep * (@num__frate@ + 1) + 1 + {}]".format(sp_id))

    function_rates_wo_param[fr_id] = f_rate

unroll_func_rate = "\n".join((f_rate.join(("_rates_arr[{}] = ".format(fr_id), ";"))
                              for fr_id, f_rate in enumerate(function_rates_wo_param)))

kernel_ready = kernel_str\
    .replace("@unroll__func__rate@", unroll_func_rate)\
    .replace("@num__iter@", str(_parallel_iterations))\
    .replace("@num__rep@", str(_partial_repetitions))\
    .replace("@num__frate@", str(_num_reacs))

print(kernel_ready)

mod = SourceModule(kernel_ready)

func = mod.get_function("ssa_simple")
func(arr_dev, rand_arr_dev, start_time, end_time, block=(_parallel_iterations, 1, 1))

arr_after = np.empty_like(arr_orig)
cuda.memcpy_dtoh(arr_after, arr_dev)

print("original array:")
print(arr_orig.reshape(_parallel_iterations, _partial_repetitions, _num_reacs + 1))
print("\nafter kernel (first column is time):")
print(arr_after.reshape(_parallel_iterations, _partial_repetitions, _num_reacs + 1))
