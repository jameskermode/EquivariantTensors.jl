/**
 * ACE Evaluator Implementation
 * ============================
 *
 * This file implements the ACE evaluator C API using the IREE runtime.
 *
 * Build:
 *   gcc -c ace_evaluator.c -I$IREE_DIR/include
 *   gcc -o libace_evaluator.so -shared ace_evaluator.o -L$IREE_DIR/lib -liree_runtime
 *
 * The implementation follows IREE's high-level runtime API patterns from:
 * https://github.com/iree-org/iree-template-runtime-cmake
 */

#include "ace_evaluator.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* IREE Runtime Headers */
#include <iree/runtime/api.h>

/* Internal model structure */
struct ACEModel {
    /* IREE runtime objects */
    iree_runtime_instance_t* instance;
    iree_runtime_session_t* session;
    iree_hal_device_t* device;

    /* Model metadata */
    ACEModelInfo info;

    /* Error handling */
    char error_message[256];
    ACEStatus last_status;
};

/* Helper: Set error message */
static void set_error(ACEModel* model, ACEStatus status, const char* msg) {
    if (model) {
        model->last_status = status;
        strncpy(model->error_message, msg, sizeof(model->error_message) - 1);
        model->error_message[sizeof(model->error_message) - 1] = '\0';
    }
}

/* Helper: Check IREE status and set error */
static int check_status(ACEModel* model, iree_status_t status, const char* operation) {
    if (!iree_status_is_ok(status)) {
        char buffer[128];
        iree_status_format(status, sizeof(buffer), buffer, NULL);
        char full_msg[256];
        snprintf(full_msg, sizeof(full_msg), "%s: %s", operation, buffer);
        set_error(model, ACE_ERROR_INVOKE_FAILED, full_msg);
        iree_status_free(status);
        return 0;
    }
    return 1;
}

ACEModel* ace_model_create(const char* vmfb_path, const char* device_uri) {
    ACEModel* model = (ACEModel*)calloc(1, sizeof(ACEModel));
    if (!model) return NULL;

    iree_status_t status;

    /* Create instance */
    iree_runtime_instance_options_t instance_options;
    iree_runtime_instance_options_initialize(&instance_options);
    iree_runtime_instance_options_use_all_available_drivers(&instance_options);

    status = iree_runtime_instance_create(
        &instance_options,
        iree_allocator_system(),
        &model->instance
    );
    if (!check_status(model, status, "instance_create")) {
        goto error;
    }

    /* Create device */
    status = iree_hal_create_device(
        iree_runtime_instance_driver_registry(model->instance),
        iree_make_cstring_view(device_uri),
        iree_allocator_system(),
        &model->device
    );
    if (!check_status(model, status, "device_create")) {
        goto error;
    }

    /* Create session options */
    iree_runtime_session_options_t session_options;
    iree_runtime_session_options_initialize(&session_options);

    /* Create session */
    status = iree_runtime_session_create_with_device(
        model->instance,
        &session_options,
        model->device,
        iree_allocator_system(),
        &model->session
    );
    if (!check_status(model, status, "session_create")) {
        goto error;
    }

    /* Load bytecode module from file */
    status = iree_runtime_session_append_bytecode_module_from_file(
        model->session,
        vmfb_path
    );
    if (!check_status(model, status, "load_module")) {
        set_error(model, ACE_ERROR_LOAD_FAILED, "Failed to load VMFB file");
        goto error;
    }

    model->last_status = ACE_OK;
    return model;

error:
    ace_model_destroy(model);
    return NULL;
}

ACEModel* ace_model_create_from_memory(
    const uint8_t* bytecode,
    size_t bytecode_size,
    const char* device_uri
) {
    ACEModel* model = (ACEModel*)calloc(1, sizeof(ACEModel));
    if (!model) return NULL;

    iree_status_t status;

    /* Create instance */
    iree_runtime_instance_options_t instance_options;
    iree_runtime_instance_options_initialize(&instance_options);
    iree_runtime_instance_options_use_all_available_drivers(&instance_options);

    status = iree_runtime_instance_create(
        &instance_options,
        iree_allocator_system(),
        &model->instance
    );
    if (!check_status(model, status, "instance_create")) {
        goto error;
    }

    /* Create device */
    status = iree_hal_create_device(
        iree_runtime_instance_driver_registry(model->instance),
        iree_make_cstring_view(device_uri),
        iree_allocator_system(),
        &model->device
    );
    if (!check_status(model, status, "device_create")) {
        goto error;
    }

    /* Create session */
    iree_runtime_session_options_t session_options;
    iree_runtime_session_options_initialize(&session_options);

    status = iree_runtime_session_create_with_device(
        model->instance,
        &session_options,
        model->device,
        iree_allocator_system(),
        &model->session
    );
    if (!check_status(model, status, "session_create")) {
        goto error;
    }

    /* Load bytecode module from memory */
    iree_const_byte_span_t bytecode_span = {bytecode, bytecode_size};
    status = iree_runtime_session_append_bytecode_module_from_memory(
        model->session,
        bytecode_span,
        iree_allocator_null()  /* Module data must outlive session */
    );
    if (!check_status(model, status, "load_module_memory")) {
        set_error(model, ACE_ERROR_LOAD_FAILED, "Failed to load VMFB from memory");
        goto error;
    }

    model->last_status = ACE_OK;
    return model;

error:
    ace_model_destroy(model);
    return NULL;
}

void ace_model_destroy(ACEModel* model) {
    if (!model) return;

    if (model->session) {
        iree_runtime_session_release(model->session);
    }
    if (model->device) {
        iree_hal_device_release(model->device);
    }
    if (model->instance) {
        iree_runtime_instance_release(model->instance);
    }

    free(model);
}

const char* ace_model_get_error(const ACEModel* model) {
    if (!model) return "NULL model";
    return model->error_message;
}

ACEStatus ace_model_get_info(const ACEModel* model, ACEModelInfo* info) {
    if (!model || !info) return ACE_ERROR_INVALID_ARG;
    *info = model->info;
    return ACE_OK;
}

/* Helper: Create a buffer view for a float tensor */
static iree_hal_buffer_view_t* create_buffer_view(
    ACEModel* model,
    const float* data,
    const size_t* dims,
    size_t ndims
) {
    iree_hal_buffer_view_t* buffer_view = NULL;
    iree_hal_dim_t iree_dims[8];
    size_t total_size = sizeof(float);

    for (size_t i = 0; i < ndims && i < 8; i++) {
        iree_dims[i] = (iree_hal_dim_t)dims[i];
        total_size *= dims[i];
    }

    iree_status_t status = iree_hal_buffer_view_allocate_buffer_copy(
        model->device,
        iree_hal_device_allocator(model->device),
        ndims,
        iree_dims,
        IREE_HAL_ELEMENT_TYPE_FLOAT_32,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t){
            .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                    IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
            .access = IREE_HAL_MEMORY_ACCESS_ALL,
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
        },
        iree_make_const_byte_span(data, total_size),
        &buffer_view
    );

    if (!iree_status_is_ok(status)) {
        iree_status_free(status);
        return NULL;
    }

    return buffer_view;
}

/* Helper: Create a buffer view for an int32 tensor */
static iree_hal_buffer_view_t* create_int_buffer_view(
    ACEModel* model,
    const int32_t* data,
    const size_t* dims,
    size_t ndims
) {
    iree_hal_buffer_view_t* buffer_view = NULL;
    iree_hal_dim_t iree_dims[8];
    size_t total_size = sizeof(int32_t);

    for (size_t i = 0; i < ndims && i < 8; i++) {
        iree_dims[i] = (iree_hal_dim_t)dims[i];
        total_size *= dims[i];
    }

    iree_status_t status = iree_hal_buffer_view_allocate_buffer_copy(
        model->device,
        iree_hal_device_allocator(model->device),
        ndims,
        iree_dims,
        IREE_HAL_ELEMENT_TYPE_INT_32,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t){
            .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                    IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
            .access = IREE_HAL_MEMORY_ACCESS_ALL,
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
        },
        iree_make_const_byte_span(data, total_size),
        &buffer_view
    );

    if (!iree_status_is_ok(status)) {
        iree_status_free(status);
        return NULL;
    }

    return buffer_view;
}

float ace_evaluate(
    ACEModel* model,
    const float* Rnl_3, const size_t Rnl_dims[3],
    const float* Ylm_3, const size_t Ylm_dims[3],
    const int32_t* spec_R, const int32_t* spec_Y, size_t nA,
    const int32_t* const* specs_mats, const size_t specs_dims[][2], size_t n_orders,
    const float* A2Bmap, const size_t A2B_dims[2],
    const float* params, size_t nparams,
    float* dRnl_3, float* dYlm_3
) {
    if (!model || !Rnl_3 || !Ylm_3 || !spec_R || !spec_Y || !A2Bmap || !params) {
        if (model) set_error(model, ACE_ERROR_INVALID_ARG, "NULL argument");
        return NAN;
    }

    iree_status_t status;
    float energy = NAN;

    /* Create a call to the main function */
    iree_runtime_call_t call;
    status = iree_runtime_call_initialize_by_name(
        model->session,
        iree_make_cstring_view("module.main"),
        &call
    );
    if (!check_status(model, status, "call_initialize")) {
        return NAN;
    }

    /* Create buffer views for inputs */
    iree_hal_buffer_view_t* Rnl_view = create_buffer_view(model, Rnl_3, Rnl_dims, 3);
    iree_hal_buffer_view_t* Ylm_view = create_buffer_view(model, Ylm_3, Ylm_dims, 3);
    size_t spec_dims[1] = {nA};
    iree_hal_buffer_view_t* spec_R_view = create_int_buffer_view(model, spec_R, spec_dims, 1);
    iree_hal_buffer_view_t* spec_Y_view = create_int_buffer_view(model, spec_Y, spec_dims, 1);
    iree_hal_buffer_view_t* A2B_view = create_buffer_view(model, A2Bmap, A2B_dims, 2);
    size_t params_dims[1] = {nparams};
    iree_hal_buffer_view_t* params_view = create_buffer_view(model, params, params_dims, 1);

    if (!Rnl_view || !Ylm_view || !spec_R_view || !spec_Y_view ||
        !A2B_view || !params_view) {
        set_error(model, ACE_ERROR_OUT_OF_MEMORY, "Failed to create buffer views");
        goto cleanup;
    }

    /* Push inputs to the call */
    iree_runtime_call_inputs_push_back_buffer_view(&call, Rnl_view);
    iree_runtime_call_inputs_push_back_buffer_view(&call, Ylm_view);
    iree_runtime_call_inputs_push_back_buffer_view(&call, spec_R_view);
    iree_runtime_call_inputs_push_back_buffer_view(&call, spec_Y_view);

    /* Push specs_mats for each order */
    for (size_t ord = 0; ord < n_orders; ord++) {
        iree_hal_buffer_view_t* specs_view = create_int_buffer_view(
            model, specs_mats[ord], specs_dims[ord], 2);
        if (specs_view) {
            iree_runtime_call_inputs_push_back_buffer_view(&call, specs_view);
            iree_hal_buffer_view_release(specs_view);
        }
    }

    iree_runtime_call_inputs_push_back_buffer_view(&call, A2B_view);
    iree_runtime_call_inputs_push_back_buffer_view(&call, params_view);

    /* Invoke the function */
    status = iree_runtime_call_invoke(&call, /*flags=*/0);
    if (!check_status(model, status, "call_invoke")) {
        goto cleanup;
    }

    /* Extract outputs */
    /* Output 0: energy (scalar) */
    iree_hal_buffer_view_t* energy_view = NULL;
    status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &energy_view);
    if (iree_status_is_ok(status) && energy_view) {
        /* Read scalar value from buffer */
        iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(energy_view);
        iree_hal_buffer_mapping_t mapping;
        status = iree_hal_buffer_map_range(
            buffer,
            IREE_HAL_MAPPING_MODE_SCOPED,
            IREE_HAL_MEMORY_ACCESS_READ,
            0, sizeof(float),
            &mapping
        );
        if (iree_status_is_ok(status)) {
            energy = *(float*)mapping.contents.data;
            iree_hal_buffer_unmap_range(&mapping);
        }
        iree_hal_buffer_view_release(energy_view);
    }

    /* If gradients requested, extract them */
    if (dRnl_3 && dYlm_3) {
        /* Output 1: dRnl */
        iree_hal_buffer_view_t* dRnl_view = NULL;
        status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &dRnl_view);
        if (iree_status_is_ok(status) && dRnl_view) {
            size_t dRnl_size = Rnl_dims[0] * Rnl_dims[1] * Rnl_dims[2] * sizeof(float);
            iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(dRnl_view);
            iree_hal_buffer_mapping_t mapping;
            status = iree_hal_buffer_map_range(
                buffer,
                IREE_HAL_MAPPING_MODE_SCOPED,
                IREE_HAL_MEMORY_ACCESS_READ,
                0, dRnl_size,
                &mapping
            );
            if (iree_status_is_ok(status)) {
                memcpy(dRnl_3, mapping.contents.data, dRnl_size);
                iree_hal_buffer_unmap_range(&mapping);
            }
            iree_hal_buffer_view_release(dRnl_view);
        }

        /* Output 2: dYlm */
        iree_hal_buffer_view_t* dYlm_view = NULL;
        status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &dYlm_view);
        if (iree_status_is_ok(status) && dYlm_view) {
            size_t dYlm_size = Ylm_dims[0] * Ylm_dims[1] * Ylm_dims[2] * sizeof(float);
            iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(dYlm_view);
            iree_hal_buffer_mapping_t mapping;
            status = iree_hal_buffer_map_range(
                buffer,
                IREE_HAL_MAPPING_MODE_SCOPED,
                IREE_HAL_MEMORY_ACCESS_READ,
                0, dYlm_size,
                &mapping
            );
            if (iree_status_is_ok(status)) {
                memcpy(dYlm_3, mapping.contents.data, dYlm_size);
                iree_hal_buffer_unmap_range(&mapping);
            }
            iree_hal_buffer_view_release(dYlm_view);
        }
    }

    model->last_status = ACE_OK;

cleanup:
    /* Release buffer views */
    if (Rnl_view) iree_hal_buffer_view_release(Rnl_view);
    if (Ylm_view) iree_hal_buffer_view_release(Ylm_view);
    if (spec_R_view) iree_hal_buffer_view_release(spec_R_view);
    if (spec_Y_view) iree_hal_buffer_view_release(spec_Y_view);
    if (A2B_view) iree_hal_buffer_view_release(A2B_view);
    if (params_view) iree_hal_buffer_view_release(params_view);

    /* Clean up call */
    iree_runtime_call_deinitialize(&call);

    return energy;
}

float ace_energy(
    ACEModel* model,
    const float* Rnl_3, const size_t Rnl_dims[3],
    const float* Ylm_3, const size_t Ylm_dims[3],
    const int32_t* spec_R, const int32_t* spec_Y, size_t nA,
    const int32_t* const* specs_mats, const size_t specs_dims[][2], size_t n_orders,
    const float* A2Bmap, const size_t A2B_dims[2],
    const float* params, size_t nparams
) {
    return ace_evaluate(
        model,
        Rnl_3, Rnl_dims,
        Ylm_3, Ylm_dims,
        spec_R, spec_Y, nA,
        specs_mats, specs_dims, n_orders,
        A2Bmap, A2B_dims,
        params, nparams,
        NULL, NULL  /* No gradients */
    );
}
