// Generic C/CUDA host side of a GPU-native mjwarp hybrid env.
//
// Architecture: the task (physics, rewards, resets, rendering) lives in
// ocean/<env>/<env>_warp.py — mujoco_warp running in-process, one captured
// CUDA graph per rollout buffer. This header owns the mechanism:
//   * bootstrap into the embedded Python, load <env>_warp.py, and call its
//     init(), forwarding every [env] kwarg from binding.c verbatim — adding
//     a config knob touches only the .ini and the _warp.py;
//   * cache the cudaGraphExec_t handles init() returns;
//   * launch them GIL-free from the rollout threads (my_gpu_step_range);
//   * bridge reset / log / render back into the Python module.
//
// A concrete env is a tiny shim .cu (nvcc-compiled by build.sh):
//   #define MJWARP_ENV_NAME "wujicrawl"
//   #define MJWARP_PY_ENVVAR "WUJICRAWL_PY"
//   #include "mjwarp_host.cuh"
// and its binding.c (under MY_GPU_NATIVE) hooks in with:
//   void mjwarp_set_env_kwargs(Dict* kwargs);
//   void mjwarp_render(void);
//   void my_init(Env* env, Dict* kwargs) {
//       env->num_agents = 1;
//       mjwarp_set_env_kwargs(kwargs);
//   }
//   void c_render(Env* env) { (void)env; mjwarp_render(); }
//
// <env>_warp.py must provide:
//   init(total_agents, num_buffers, seed, act_ptr, obs_ptr, rew_ptr,
//        term_ptr, **env_kwargs)
//     -> {"graphs": [graph_exec handles], "agents_per_buffer": int,
//         "log_ptr": device pointer to MJWARP_LOG_FLOATS float32 SUMS}
//   reset()   — relaunch the reset kernels on every buffer
//   render()  — live viewer, called with the GIL once per control step

#pragma once

#ifndef MJWARP_ENV_NAME
#error "define MJWARP_ENV_NAME (e.g. \"wujicrawl\") before including mjwarp_host.cuh"
#endif
#ifndef MJWARP_PY_ENVVAR
#error "define MJWARP_PY_ENVVAR (e.g. \"WUJICRAWL_PY\") before including mjwarp_host.cuh"
#endif
// Must equal sizeof(Log)/sizeof(float) in the env's binding.c.
#ifndef MJWARP_LOG_FLOATS
#define MJWARP_LOG_FLOATS 5
#endif
#define MJWARP_MAX_BUFFERS 64

#include <cuda_runtime.h>
#include <Python.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

// tensor.h (via vecenv.h) wants precision_t under __CUDACC__; the trainer
// defines it per precision mode in kernels.cu. This TU only needs the Dict
// types and never touches PrecisionTensor, so any element type works.
typedef float precision_t;
#include "vecenv.h"  // Dict types only (implementations are OBS_SIZE-gated)

extern "C" {

static cudaGraphExec_t g_graphs[MJWARP_MAX_BUFFERS];
static int g_num_buffers = 0;
static int g_agents_per_buffer = 0;
static void* g_log_ptr = NULL;
static PyObject* g_module = NULL;
static Dict* g_env_kwargs = NULL;

static void py_fatal(const char* what) {
    fprintf(stderr, MJWARP_ENV_NAME ": %s\n", what);
    if (PyErr_Occurred()) PyErr_Print();
    exit(1);
}

// binding.c my_init stashes the [env] kwargs here; the Dict outlives
// my_gpu_init (both run inside create_static_vec).
void mjwarp_set_env_kwargs(Dict* kwargs) {
    g_env_kwargs = kwargs;
}

void my_gpu_init(int total_agents, int num_buffers, unsigned int seed,
                 float* gpu_actions, void* gpu_observations,
                 float* gpu_rewards, float* gpu_terminals) {
    if (!Py_IsInitialized()) py_fatal("Python interpreter not initialized");
    if (num_buffers > MJWARP_MAX_BUFFERS) py_fatal("too many buffers");
    if (!g_env_kwargs) py_fatal("env kwargs not set (my_init must call "
                                "mjwarp_set_env_kwargs)");
    PyGILState_STATE gil = PyGILState_Ensure();

    // Forward every [env] kwarg as a Python literal. %.17g round-trips
    // doubles and prints integral values without a decimal point, so they
    // arrive as Python ints.
    char kwargs_buf[4096];
    size_t off = 0;
    for (int i = 0; i < g_env_kwargs->size; i++) {
        int n = snprintf(kwargs_buf + off, sizeof(kwargs_buf) - off,
                         ", %s=%.17g", g_env_kwargs->items[i].key,
                         g_env_kwargs->items[i].value);
        if (n < 0 || (size_t)n >= sizeof(kwargs_buf) - off)
            py_fatal("env kwargs too long");
        off += (size_t)n;
    }

    char code[8192];
    int n = snprintf(code, sizeof(code),
        "import importlib.util as _ilu, os as _os\n"
        "_p = _os.environ.get('" MJWARP_PY_ENVVAR "', "
        "'ocean/" MJWARP_ENV_NAME "/" MJWARP_ENV_NAME "_warp.py')\n"
        "_spec = _ilu.spec_from_file_location('" MJWARP_ENV_NAME "_warp', _p)\n"
        "_mjwarp_mod = _ilu.module_from_spec(_spec)\n"
        "_spec.loader.exec_module(_mjwarp_mod)\n"
        "_mjwarp_result = _mjwarp_mod.init("
        "total_agents=%d, num_buffers=%d, seed=%u, "
        "act_ptr=%llu, obs_ptr=%llu, rew_ptr=%llu, term_ptr=%llu%s)\n",
        total_agents, num_buffers, seed,
        (unsigned long long)(uintptr_t)gpu_actions,
        (unsigned long long)(uintptr_t)gpu_observations,
        (unsigned long long)(uintptr_t)gpu_rewards,
        (unsigned long long)(uintptr_t)gpu_terminals,
        kwargs_buf);
    if (n < 0 || (size_t)n >= sizeof(code)) py_fatal("init code too long");

    PyObject* main_mod = PyImport_AddModule("__main__");
    PyObject* globals = PyModule_GetDict(main_mod);
    PyObject* r = PyRun_String(code, Py_file_input, globals, globals);
    if (!r) py_fatal(MJWARP_ENV_NAME "_warp.py init failed");
    Py_DECREF(r);

    PyObject* result = PyDict_GetItemString(globals, "_mjwarp_result");
    g_module = PyDict_GetItemString(globals, "_mjwarp_mod");
    if (!result || !g_module) py_fatal("init result missing");
    Py_INCREF(g_module);

    PyObject* graphs = PyDict_GetItemString(result, "graphs");
    PyObject* apb = PyDict_GetItemString(result, "agents_per_buffer");
    PyObject* logp = PyDict_GetItemString(result, "log_ptr");
    if (!graphs || !apb || !logp) py_fatal("init result malformed");

    g_num_buffers = (int)PyList_Size(graphs);
    for (int b = 0; b < g_num_buffers; b++) {
        g_graphs[b] = (cudaGraphExec_t)(uintptr_t)
            PyLong_AsUnsignedLongLong(PyList_GetItem(graphs, b));
    }
    g_agents_per_buffer = (int)PyLong_AsLong(apb);
    g_log_ptr = (void*)(uintptr_t)PyLong_AsUnsignedLongLong(logp);
    if (PyErr_Occurred()) py_fatal("init result parse failed");

    PyGILState_Release(gil);
    printf(MJWARP_ENV_NAME ": %d buffers x %d agents, graphs captured\n",
           g_num_buffers, g_agents_per_buffer);
}

// Hot path: OMP worker thread, NO GIL. Pure CUDA only.
void my_gpu_step_range(void* stream, int agent_start, int count,
                       const float* gpu_actions, void* gpu_obs,
                       float* gpu_rewards, float* gpu_terminals) {
    (void)gpu_actions; (void)gpu_obs; (void)gpu_rewards; (void)gpu_terminals;
    (void)count;
    int b = agent_start / g_agents_per_buffer;
    cudaError_t rc = cudaGraphLaunch(g_graphs[b], (cudaStream_t)stream);
    if (rc != cudaSuccess) {
        fprintf(stderr, MJWARP_ENV_NAME ": cudaGraphLaunch(buffer %d): %s\n",
                b, cudaGetErrorString(rc));
        exit(1);
    }
}

void my_gpu_reset(void* gpu_observations) {
    (void)gpu_observations;
    PyGILState_STATE gil = PyGILState_Ensure();
    PyObject* r = PyObject_CallMethod(g_module, "reset", NULL);
    if (!r) py_fatal("reset failed");
    Py_DECREF(r);
    PyGILState_Release(gil);
}

float my_gpu_log_into(void* log_out) {
    // Layout matches binding.c's Log struct; last float is the episode
    // count n.
    float sums[MJWARP_LOG_FLOATS] = {0};
    cudaMemcpy(sums, g_log_ptr, sizeof(sums), cudaMemcpyDeviceToHost);
    if (sums[MJWARP_LOG_FLOATS - 1] == 0.0f) return 0.0f;
    cudaMemset(g_log_ptr, 0, sizeof(sums));
    memcpy(log_out, sums, sizeof(sums));
    return sums[MJWARP_LOG_FLOATS - 1];
}

void my_gpu_close(void) {}

// Called from binding.c c_render (GIL-held context via _C.render).
void mjwarp_render(void) {
    PyGILState_STATE gil = PyGILState_Ensure();
    PyObject* r = PyObject_CallMethod(g_module, "render", NULL);
    if (!r) py_fatal("render failed");
    Py_DECREF(r);
    PyGILState_Release(gil);
}

}  // extern "C"
