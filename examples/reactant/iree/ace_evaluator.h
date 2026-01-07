/**
 * ACE Evaluator C API
 * ===================
 *
 * C interface for evaluating ACE (Atomic Cluster Expansion) models
 * compiled to IREE VMFB format.
 *
 * This API loads a pre-compiled ACE model and provides functions to:
 * 1. Compute energies from atomic embeddings
 * 2. Compute gradients (for force calculation)
 *
 * The model operates on pre-computed embeddings (Rnl, Ylm) rather than
 * raw atomic positions. The embedding computation and force accumulation
 * must be handled by the caller.
 *
 * Usage:
 *   ACEModel* model = ace_model_create("ace_model_cpu.vmfb", "local-task");
 *   float energy = ace_evaluate(model, Rnl, Ylm, ..., dRnl, dYlm);
 *   ace_model_destroy(model);
 *
 * Thread Safety:
 *   - Multiple threads can call ace_evaluate on different models
 *   - A single model should not be used from multiple threads simultaneously
 *
 * Memory:
 *   - Caller is responsible for allocating input/output buffers
 *   - Model manages its internal IREE resources
 */

#ifndef ACE_EVALUATOR_H
#define ACE_EVALUATOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to ACE model */
typedef struct ACEModel ACEModel;

/* Error codes */
typedef enum {
    ACE_OK = 0,
    ACE_ERROR_INVALID_ARG = 1,
    ACE_ERROR_LOAD_FAILED = 2,
    ACE_ERROR_COMPILE_FAILED = 3,
    ACE_ERROR_INVOKE_FAILED = 4,
    ACE_ERROR_OUT_OF_MEMORY = 5,
} ACEStatus;

/**
 * Create an ACE model from a compiled VMFB file.
 *
 * @param vmfb_path  Path to the compiled .vmfb file
 * @param device     IREE device identifier:
 *                   - "local-task" for multi-threaded CPU
 *                   - "local-sync" for single-threaded CPU
 *                   - "cuda" for NVIDIA GPU
 *                   - "vulkan" for Vulkan GPU
 * @return           Model handle, or NULL on failure
 */
ACEModel* ace_model_create(const char* vmfb_path, const char* device);

/**
 * Create an ACE model from in-memory bytecode.
 *
 * @param bytecode      Pointer to VMFB bytecode data
 * @param bytecode_size Size of bytecode in bytes
 * @param device        IREE device identifier
 * @return              Model handle, or NULL on failure
 */
ACEModel* ace_model_create_from_memory(
    const uint8_t* bytecode,
    size_t bytecode_size,
    const char* device
);

/**
 * Destroy an ACE model and free all resources.
 *
 * @param model  Model to destroy (may be NULL)
 */
void ace_model_destroy(ACEModel* model);

/**
 * Get the last error message.
 *
 * @param model  Model handle
 * @return       Error message string (valid until next call)
 */
const char* ace_model_get_error(const ACEModel* model);

/**
 * Evaluate ACE energy and optionally gradients.
 *
 * This is the main evaluation function. It takes pre-computed radial (Rnl)
 * and angular (Ylm) embeddings and returns the energy. If gradient buffers
 * are provided (non-NULL), it also computes gradients for force calculation.
 *
 * Array Layout (row-major, C order):
 *   Rnl_3: [maxneigs][nnodes][nRnl]
 *   Ylm_3: [maxneigs][nnodes][nYlm]
 *   spec_R, spec_Y: [nA]
 *   A2Bmap: [nfeatures][nAA]
 *   params: [nfeatures]
 *
 * @param model       Model handle
 * @param Rnl_3       Radial embeddings [maxneigs * nnodes * nRnl]
 * @param Rnl_dims    Dimensions [maxneigs, nnodes, nRnl]
 * @param Ylm_3       Angular embeddings [maxneigs * nnodes * nYlm]
 * @param Ylm_dims    Dimensions [maxneigs, nnodes, nYlm]
 * @param spec_R      Radial spec indices [nA]
 * @param spec_Y      Angular spec indices [nA]
 * @param nA          Number of A features
 * @param specs_mats  Array of spec matrices for each order
 * @param specs_dims  Dimensions for each spec matrix [n_orders][2]
 * @param n_orders    Number of correlation orders
 * @param A2Bmap      Coupling matrix [nfeatures * nAA]
 * @param A2B_dims    Dimensions [nfeatures, nAA]
 * @param params      Linear readout parameters [nfeatures]
 * @param nparams     Number of parameters
 * @param dRnl_3      Output: gradient w.r.t. Rnl (NULL to skip)
 * @param dYlm_3      Output: gradient w.r.t. Ylm (NULL to skip)
 *
 * @return            Computed energy (NaN on error)
 */
float ace_evaluate(
    ACEModel* model,
    const float* Rnl_3, const size_t Rnl_dims[3],
    const float* Ylm_3, const size_t Ylm_dims[3],
    const int32_t* spec_R, const int32_t* spec_Y, size_t nA,
    const int32_t* const* specs_mats, const size_t specs_dims[][2], size_t n_orders,
    const float* A2Bmap, const size_t A2B_dims[2],
    const float* params, size_t nparams,
    float* dRnl_3, float* dYlm_3
);

/**
 * Simplified evaluation for energy only (no gradients).
 *
 * Same as ace_evaluate with dRnl_3 = dYlm_3 = NULL.
 */
float ace_energy(
    ACEModel* model,
    const float* Rnl_3, const size_t Rnl_dims[3],
    const float* Ylm_3, const size_t Ylm_dims[3],
    const int32_t* spec_R, const int32_t* spec_Y, size_t nA,
    const int32_t* const* specs_mats, const size_t specs_dims[][2], size_t n_orders,
    const float* A2Bmap, const size_t A2B_dims[2],
    const float* params, size_t nparams
);

/**
 * Get model metadata.
 */
typedef struct {
    int nRnl;        /* Number of radial features */
    int nYlm;        /* Number of angular features */
    int nA;          /* Number of A features */
    int nAA;         /* Number of AA features */
    int nfeatures;   /* Number of output features */
    int n_orders;    /* Number of correlation orders */
} ACEModelInfo;

ACEStatus ace_model_get_info(const ACEModel* model, ACEModelInfo* info);

#ifdef __cplusplus
}
#endif

#endif /* ACE_EVALUATOR_H */
