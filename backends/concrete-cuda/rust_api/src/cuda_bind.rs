use std::ffi::c_void;

#[link(name = "concrete_cuda", kind = "static")]
extern "C" {

    /// Create a new Cuda stream on GPU `gpu_index`
    pub fn cuda_create_stream(gpu_index: u32) -> *mut c_void;

    /// Destroy the Cuda stream `v_stream` on GPU `gpu_index`
    pub fn cuda_destroy_stream(v_stream: *mut c_void, gpu_index: u32) -> i32;

    /// Allocate `size` memory on GPU `gpu_index` asynchronously
    pub fn cuda_malloc_async(size: u64, v_stream: *mut c_void, gpu_index: u32) -> *mut c_void;

    /// Copy `size` memory asynchronously from `src` on GPU `gpu_index` to `dest` on CPU using
    /// the Cuda stream `v_stream`.
    pub fn cuda_memcpy_async_to_cpu(
        dest: *mut c_void,
        src: *const c_void,
        size: u64,
        v_stream: *mut c_void,
        gpu_index: u32,
    ) -> i32;

    /// Copy `size` memory asynchronously from `src` on CPU to `dest` on GPU `gpu_index` using
    /// the Cuda stream `v_stream`.
    pub fn cuda_memcpy_async_to_gpu(
        dest: *mut c_void,
        src: *const c_void,
        size: u64,
        v_stream: *mut c_void,
        gpu_index: u32,
    ) -> i32;

    /// Get the total number of Nvidia GPUs detected on the platform
    pub fn cuda_get_number_of_gpus() -> i32;

    /// Synchronize all streams on GPU `gpu_index`
    pub fn cuda_synchronize_device(gpu_index: u32) -> i32;

    /// Synchronize Cuda stream
    pub fn cuda_synchronize_stream(v_stream: *mut c_void) -> i32;

    /// Free memory for pointer `ptr` on GPU `gpu_index` asynchronously, using stream `v_stream`
    pub fn cuda_drop_async(ptr: *mut c_void, v_stream: *mut c_void, gpu_index: u32) -> i32;

    /// Get the maximum amount of shared memory on GPU `gpu_index`
    pub fn cuda_get_max_shared_memory(gpu_index: u32) -> i32;

    /// Copy a bootstrap key `src` represented with 64 bits in the standard domain from the CPU to
    /// the GPU `gpu_index` using the stream `v_stream`, and convert it to the Fourier domain on the
    /// GPU. The resulting bootstrap key `dest` on the GPU is an array of f64 values.
    pub fn cuda_convert_lwe_bootstrap_key_64(
        dest: *mut c_void,
        src: *const c_void,
        v_stream: *mut c_void,
        gpu_index: u32,
        input_lwe_dim: u32,
        glwe_dim: u32,
        level_count: u32,
        polynomial_size: u32,
    );

    /// Copy a multi-bit bootstrap key `src` represented with 64 bits in the standard domain from
    /// the CPU to the GPU `gpu_index` using the stream `v_stream`. The resulting bootstrap key
    /// `dest` on the GPU is an array of uint64_t values.
    pub fn cuda_convert_lwe_multi_bit_bootstrap_key_64(
        dest: *mut c_void,
        src: *const c_void,
        v_stream: *mut c_void,
        gpu_index: u32,
        input_lwe_dim: u32,
        glwe_dim: u32,
        level_count: u32,
        polynomial_size: u32,
        grouping_factor: u32,
    );

    /// Copy `number_of_cts` LWE ciphertext represented with 64 bits in the standard domain from the
    /// CPU to the GPU `gpu_index` using the stream `v_stream`. All ciphertexts must be
    /// concatenated.
    pub fn cuda_convert_lwe_ciphertext_vector_to_gpu_64(
        dest: *mut c_void,
        src: *mut c_void,
        v_stream: *mut c_void,
        gpu_index: u32,
        number_of_cts: u32,
        lwe_dimension: u32,
    );

    /// Copy `number_of_cts` LWE ciphertext represented with 64 bits in the standard domain from the
    /// GPU to the CPU `gpu_index` using the stream `v_stream`. All ciphertexts must be
    /// concatenated.
    pub fn cuda_convert_lwe_ciphertext_vector_to_cpu_64(
        dest: *mut c_void,
        src: *mut c_void,
        v_stream: *mut c_void,
        gpu_index: u32,
        number_of_cts: u32,
        lwe_dimension: u32,
    );

    /// This scratch function allocates the necessary amount of data on the GPU for
    /// the low latency PBS on 64-bit inputs, into `pbs_buffer`. It also configures SM
    /// options on the GPU in case FULLSM or PARTIALSM mode are going to be used.
    pub fn scratch_cuda_bootstrap_low_latency_64(
        v_stream: *mut c_void,
        gpu_index: u32,
        pbs_buffer: *mut *mut i8,
        glwe_dimension: u32,
        polynomial_size: u32,
        level_count: u32,
        input_lwe_ciphertext_count: u32,
        max_shared_memory: u32,
        allocate_gpu_memory: bool,
    );

    /// Perform bootstrapping on a batch of input u64 LWE ciphertexts.
    ///
    /// - `v_stream` is a void pointer to the Cuda stream to be used in the kernel launch
    /// - `gpu_index` is the index of the GPU to be used in the kernel launch
    /// - `lwe_array_out`: output batch of num_samples bootstrapped ciphertexts c =
    /// (a0,..an-1,b) where n is the LWE dimension
    /// - `lut_vector`: should hold as many test vectors of size polynomial_size
    /// as there are input ciphertexts, but actually holds
    /// `num_lut_vectors` vectors to reduce memory usage
    /// - `lut_vector_indexes`: stores the index corresponding to
    /// which test vector to use for each sample in
    /// `lut_vector`
    /// - `lwe_array_in`: input batch of num_samples LWE ciphertexts, containing n
    /// mask values + 1 body value
    /// - `bootstrapping_key`: GGSW encryption of the LWE secret key sk1
    /// under secret key sk2.
    /// bsk = Z + sk1 H
    /// where H is the gadget matrix and Z is a matrix (k+1).l
    /// containing GLWE encryptions of 0 under sk2.
    /// bsk is thus a tensor of size (k+1)^2.l.N.n
    /// where l is the number of decomposition levels and
    /// k is the GLWE dimension, N is the polynomial size for
    /// GLWE. The polynomial size for GLWE and the test vector
    /// are the same because they have to be in the same ring
    /// to be multiplied.
    /// - `pbs_buffer`: a preallocated buffer to store temporary results
    /// - `lwe_dimension`: size of the Torus vector used to encrypt the input
    /// LWE ciphertexts - referred to as n above (~ 600)
    /// - `glwe_dimension`: size of the polynomial vector used to encrypt the LUT
    /// GLWE ciphertexts - referred to as k above. Only the value 1 is supported for this parameter.
    /// - `polynomial_size`: size of the test polynomial (test vector) and size of the
    /// GLWE polynomial (~1024)
    /// - `base_log`: log base used for the gadget matrix - B = 2^base_log (~8)
    /// - `level_count`: number of decomposition levels in the gadget matrix (~4)
    /// - `num_samples`: number of encrypted input messages
    /// - `num_lut_vectors`: parameter to set the actual number of test vectors to be
    /// used
    /// - `lwe_idx`: the index of the LWE input to consider for the GPU of index gpu_index. In
    /// case of multi-GPU computing, it is assumed that only a part of the input LWE array is
    /// copied to each GPU, but the whole LUT array is copied (because the case when the number
    /// of LUTs is smaller than the number of input LWEs is not trivial to take into account in
    /// the data repartition on the GPUs). `lwe_idx` is used to determine which LUT to consider
    /// for a given LWE input in the LUT array `lut_vector`.
    ///  - `max_shared_memory` maximum amount of shared memory to be used inside
    /// device functions
    ///
    /// This function calls a wrapper to a device kernel that performs the
    /// bootstrapping:
    ///   - the kernel is templatized based on integer discretization and
    /// polynomial degree
    ///   - num_samples * level_count * (glwe_dimension + 1) blocks of threads are launched, where
    /// each thread	is going to handle one or more polynomial coefficients at each stage,
    /// for a given level of decomposition, either for the LUT mask or its body:
    ///     - perform the blind rotation
    ///     - round the result
    ///     - get the decomposition for the current level
    ///     - switch to the FFT domain
    ///     - multiply with the bootstrapping key
    ///     - come back to the coefficients representation
    ///   - between each stage a synchronization of the threads is necessary (some
    /// synchronizations
    /// happen at the block level, some happen between blocks, using cooperative groups).
    ///   - in case the device has enough shared memory, temporary arrays used for
    /// the different stages (accumulators) are stored into the shared memory
    ///   - the accumulators serve to combine the results for all decomposition
    /// levels
    ///   - the constant memory (64K) is used for storing the roots of identity
    /// values for the FFT
    ///
    pub fn cuda_bootstrap_low_latency_lwe_ciphertext_vector_64(
        v_stream: *mut c_void,
        gpu_index: u32,
        lwe_array_out: *mut c_void,
        lut_vector: *const c_void,
        lut_vector_indexes: *const c_void,
        lwe_array_in: *const c_void,
        bootstrapping_key: *const c_void,
        pbs_buffer: *mut i8,
        lwe_dimension: u32,
        glwe_dimension: u32,
        polynomial_size: u32,
        base_log: u32,
        level: u32,
        num_samples: u32,
        num_lut_vectors: u32,
        lwe_idx: u32,
        max_shared_memory: u32,
    );

    /// This cleanup function frees the data for the low latency PBS on GPU
    /// contained in pbs_buffer for 32 or 64-bit inputs.
    pub fn cleanup_cuda_bootstrap_low_latency(
        v_stream: *mut c_void,
        gpu_index: u32,
        pbs_buffer: *mut *mut i8,
    );

    /// This scratch function allocates the necessary amount of data on the GPU for
    /// the multi-bit PBS on 64-bit inputs into `pbs_buffer`.
    pub fn scratch_cuda_multi_bit_pbs_64(
        v_stream: *mut c_void,
        gpu_index: u32,
        pbs_buffer: *mut *mut i8,
        lwe_dimension: u32,
        glwe_dimension: u32,
        polynomial_size: u32,
        level_count: u32,
        grouping_factor: u32,
        input_lwe_ciphertext_count: u32,
        max_shared_memory: u32,
        allocate_gpu_memory: bool,
        lwe_chunk_size: u32,
    );

    /// Perform bootstrapping on a batch of input u64 LWE ciphertexts using the multi-bit algorithm.
    ///
    /// - `v_stream` is a void pointer to the Cuda stream to be used in the kernel launch
    /// - `gpu_index` is the index of the GPU to be used in the kernel launch
    /// - `lwe_array_out`: output batch of num_samples bootstrapped ciphertexts c =
    /// (a0,..an-1,b) where n is the LWE dimension
    /// - `lut_vector`: should hold as many test vectors of size polynomial_size
    /// as there are input ciphertexts, but actually holds
    /// `num_lut_vectors` vectors to reduce memory usage
    /// - `lut_vector_indexes`: stores the index corresponding to
    /// which test vector to use for each sample in
    /// `lut_vector`
    /// - `lwe_array_in`: input batch of num_samples LWE ciphertexts, containing n
    /// mask values + 1 body value
    /// - `bootstrapping_key`: GGSW encryption of elements of the LWE secret key as in the
    /// classical PBS, but this time we follow Zhou's trick and encrypt combinations of elements
    /// of the key
    /// - `pbs_buffer`: a preallocated buffer to store temporary results
    /// - `lwe_dimension`: size of the Torus vector used to encrypt the input
    /// LWE ciphertexts - referred to as n above (~ 600)
    /// - `glwe_dimension`: size of the polynomial vector used to encrypt the LUT
    /// GLWE ciphertexts - referred to as k above. Only the value 1 is supported for this parameter.
    /// - `polynomial_size`: size of the test polynomial (test vector) and size of the
    /// GLWE polynomial (~1024)
    /// - `grouping_factor`: number of elements of the LWE secret key combined per GGSW of the
    /// bootstrap key
    /// - `base_log`: log base used for the gadget matrix - B = 2^base_log (~8)
    /// - `level_count`: number of decomposition levels in the gadget matrix (~4)
    /// - `num_samples`: number of encrypted input messages
    /// - `num_lut_vectors`: parameter to set the actual number of test vectors to be
    /// used
    /// - `lwe_idx`: the index of the LWE input to consider for the GPU of index gpu_index. In
    /// case of multi-GPU computing, it is assumed that only a part of the input LWE array is
    /// copied to each GPU, but the whole LUT array is copied (because the case when the number
    /// of LUTs is smaller than the number of input LWEs is not trivial to take into account in
    /// the data repartition on the GPUs). `lwe_idx` is used to determine which LUT to consider
    /// for a given LWE input in the LUT array `lut_vector`.
    ///  - `max_shared_memory` maximum amount of shared memory to be used inside
    /// device functions
    ///
    ///
    pub fn cuda_multi_bit_pbs_lwe_ciphertext_vector_64(
        v_stream: *mut c_void,
        gpu_index: u32,
        lwe_array_out: *mut c_void,
        lut_vector: *const c_void,
        lut_vector_indexes: *const c_void,
        lwe_array_in: *const c_void,
        bootstrapping_key: *const c_void,
        pbs_buffer: *mut i8,
        lwe_dimension: u32,
        glwe_dimension: u32,
        polynomial_size: u32,
        grouping_factor: u32,
        base_log: u32,
        level: u32,
        num_samples: u32,
        num_lut_vectors: u32,
        lwe_idx: u32,
        max_shared_memory: u32,
        lwe_chunk_size: u32,
    );

    /// This cleanup function frees the data for the multi-bit PBS on GPU
    /// contained in pbs_buffer for 64-bit inputs.
    pub fn cleanup_cuda_multi_bit_pbs(
        v_stream: *mut c_void,
        gpu_index: u32,
        pbs_buffer: *mut *mut i8,
    );

    /// Perform keyswitch on a batch of 64 bits input LWE ciphertexts.
    ///
    /// - `v_stream` is a void pointer to the Cuda stream to be used in the kernel launch
    /// - `gpu_index` is the index of the GPU to be used in the kernel launch
    /// - `lwe_array_out`: output batch of num_samples keyswitched ciphertexts c =
    /// (a0,..an-1,b) where n is the output LWE dimension (lwe_dimension_out)
    /// - `lwe_array_in`: input batch of num_samples LWE ciphertexts, containing lwe_dimension_in
    /// mask values + 1 body value
    /// - `ksk`: the keyswitch key to be used in the operation
    /// - `base_log`: the log of the base used in the decomposition (should be the one used to
    /// create the ksk).
    /// - `level_count`: the number of levels used in the decomposition (should be the one used to
    /// create the ksk).
    /// - `num_samples`: the number of input and output LWE ciphertexts.
    ///
    /// This function calls a wrapper to a device kernel that performs the keyswitch.
    /// `num_samples` blocks of threads are launched
    pub fn cuda_keyswitch_lwe_ciphertext_vector_64(
        v_stream: *mut c_void,
        gpu_index: u32,
        lwe_array_out: *mut c_void,
        lwe_array_in: *const c_void,
        keyswitch_key: *const c_void,
        input_lwe_dimension: u32,
        output_lwe_dimension: u32,
        base_log: u32,
        level_count: u32,
        num_samples: u32,
    );


    /// Perform functional packing keyswitch on a batch of 64 bits input LWE ciphertexts.
    ///
    /// - `v_stream` is a void pointer to the Cuda stream to be used in the kernel launch
    /// - `glwe_array_out`: output batch of keyswitched ciphertexts
    /// - `lwe_array_in`: input batch of num_samples LWE ciphertexts, containing lwe_dimension_in
    ///   mask values + 1 body value
    ///  - `fp_ksk_array`: the functional packing keyswitch keys to be used in the operation
    ///  - `base log`: the log of the base used in the decomposition (should be the one used to
    ///    create
    ///  the ksk)
    ///  - `level_count`: the number of levels used in the decomposition (should be the
    ///  one used to  create the fp_ksks).
    ///  - `number_of_input_lwe`: the number of inputs
    ///  - `number_of_keys`: the number of fp_ksks
    ///
    /// This function calls a wrapper to a device kernel that performs the functional packing
    /// keyswitch.
    pub fn cuda_fp_keyswitch_lwe_to_glwe_64(
        v_stream: *mut c_void,
        gpu_index: u32,
        glwe_array_out: *mut c_void,
        lwe_array_in: *const c_void,
        fp_ksk_array: *const c_void,
        input_lwe_dimension: u32,
        output_glwe_dimension: u32,
        output_polynomial_size: u32,
        base_log: u32,
        level_count: u32,
        number_of_input_lwe: u32,
        number_of_keys: u32,
    );

    /// This scratch function allocates the necessary amount of data on the GPU for
    /// the Cmux tree on 64-bit inputs, into `cmux_tree_buffer`. It also configures SM options on
    /// the GPU in case FULLSM mode is going to be used.
    pub fn scratch_cuda_cmux_tree_64(
        v_stream: *mut c_void,
        gpu_index: u32,
        cmux_tree_buffer: *mut *mut i8,
        glwe_dimension: u32,
        polynomial_size: u32,
        level_count: u32,
        r: u32,
        tau: u32,
        max_shared_memory: u32,
        allocate_gpu_memory: bool,
    );

    /// Perform Cmux tree on a batch of 64-bit input GGSW ciphertexts
    /// - `v_stream` is a void pointer to the Cuda stream to be used in the kernel launch
    /// - `gpu_index` is the index of the GPU to be used in the kernel launch
    ///  - `glwe_array_out` output batch of GLWE buffer for Cmux tree, `tau` GLWE`s
    /// will be the output of the function
    ///  - `ggsw_in` batch of input GGSW ciphertexts, function expects `r` GGSW
    /// ciphertexts as input.
    ///  - `lut_vector` batch of test vectors (LUTs) there should be 2^r LUTs
    /// inside `lut_vector` parameter
    ///  - `glwe_dimension` GLWE dimension, supported values: {1}
    ///  - `polynomial_size` size of the test polynomial, supported values: {512,
    /// 1024, 2048, 4096, 8192}
    ///  - `base_log` base log parameter for cmux block
    ///  - `level_count` decomposition level for cmux block
    ///  - `r` number of input GGSW ciphertexts
    ///  - `tau` number of input LWE ciphertext which were used to generate GGSW
    /// ciphertexts stored in `ggsw_in`, it is also an amount of output GLWE
    /// ciphertexts
    ///  - `max_shared_memory` maximum shared memory amount to be used for cmux
    ///  kernel
    ///
    /// This function calls a wrapper to a device kernel that performs the
    /// Cmux tree. The kernel is templatized based on integer discretization and
    /// polynomial degree.
    pub fn cuda_cmux_tree_64(
        v_stream: *mut c_void,
        gpu_index: u32,
        glwe_array_out: *mut c_void,
        ggsw_in: *const c_void,
        lut_vector: *const c_void,
        cmux_tree_buffer: *mut i8,
        glwe_dimension: u32,
        polynomial_size: u32,
        base_log: u32,
        level_count: u32,
        r: u32,
        tau: u32,
        max_shared_memory: u32,
    );

    /// This cleanup function frees the data for the Cmux tree on GPU
    /// contained in cmux_tree_buffer for 32 or 64-bit inputs.
    pub fn cleanup_cuda_cmux_tree(
        v_stream: *mut c_void,
        gpu_index: u32,
        cmux_tree_buffer: *mut *mut i8,
    );

    /// This scratch function allocates the necessary amount of data on the GPU for
    /// the blind rotation and sample extraction on 64-bit inputs, into `br_se_buffer`.
    /// It also configures SM options on the GPU in case FULLSM mode is going to be used.
    pub fn scratch_cuda_blind_rotation_sample_extraction_64(
        v_stream: *mut c_void,
        gpu_index: u32,
        br_se_buffer: *mut *mut i8,
        glwe_dimension: u32,
        polynomial_size: u32,
        level_count: u32,
        mbr_size: u32,
        tau: u32,
        max_shared_memory: u32,
        allocate_gpu_memory: bool,
    );

    /// Performs blind rotation on batch of 64-bit input GGSW ciphertexts.
    /// - `v_stream` is a void pointer to the Cuda stream to be used in the kernel launch
    /// - `gpu_index` is the index of the GPU to be used in the kernel launch
    ///  - `lwe_out`  batch of output lwe ciphertexts, there should be `tau`
    /// ciphertexts inside `lwe_out`
    ///  - `ggsw_in` batch of input ggsw ciphertexts, function expects `mbr_size`
    /// ggsw ciphertexts inside `ggsw_in`
    ///  - `lut_vector` list of test vectors, function expects `tau` test vectors
    /// inside `lut_vector` parameter
    ///  - `glwe_dimension` glwe dimension, supported values : {1}
    ///  - `polynomial_size` size of test polynomial supported sizes: {512, 1024,
    ///  2048, 4096, 8192}
    ///  - `base_log` base log parameter
    ///  - `level_count` decomposition level
    ///  - `max_shared_memory` maximum number of shared memory to be used in
    /// device functions (kernels).
    ///
    /// This function calls a wrapper to a device kernel that performs the
    /// blind rotation and sample extraction. The kernel is templatized based on integer
    /// discretization and polynomial degree.
    pub fn cuda_blind_rotate_and_sample_extraction_64(
        v_stream: *mut c_void,
        gpu_index: u32,
        lwe_out: *mut c_void,
        ggsw_in: *const c_void,
        lut_vector: *const c_void,
        br_se_buffer: *mut i8,
        mbr_size: u32,
        tau: u32,
        glwe_dimension: u32,
        polynomial_size: u32,
        base_log: u32,
        level_count: u32,
        max_shared_memory: u32,
    );

    /// This cleanup function frees the data for the blind rotation and sample extraction on GPU
    /// contained in br_se_buffer for 32 or 64-bit inputs.
    pub fn cleanup_cuda_blind_rotation_sample_extraction(
        v_stream: *mut c_void,
        gpu_index: u32,
        br_se_buffer: *mut *mut i8,
    );

    /// This scratch function allocates the necessary amount of data on the GPU for
    /// the bit extraction on 64-bit inputs, into `bit_extract_buffer`.
    /// It also configures SM options on the GPU in case FULLSM or PARTIALSM mode is going to be
    /// used in the PBS.
    pub fn scratch_cuda_extract_bits_64(
        v_stream: *mut c_void,
        gpu_index: u32,
        bit_extract_buffer: *mut *mut i8,
        glwe_dimension: u32,
        lwe_dimension: u32,
        polynomial_size: u32,
        level_count: u32,
        number_of_inputs: u32,
        max_shared_memory: u32,
        allocate_gpu_memory: bool,
    );


    /// Perform bit extract on a batch of 64 bit LWE ciphertexts.
    /// - `v_stream` is a void pointer to the Cuda stream to be used in the kernel launch
    /// - `gpu_index` is the index of the GPU to be used in the kernel launch
    /// - `number_of_bits` will be extracted from each ciphertext
    /// starting at the bit number `delta_log` (0-indexed) included.
    /// Output bits are ordered from the MSB to LSB. Every extracted bit is
    /// represented as an LWE ciphertext, containing the encryption of the bit scaled
    /// by q/2.
    /// - `list_lwe_array_out` output batch LWE ciphertexts for each bit of every
    /// input ciphertext
    /// - `lwe_array_in` batch of input LWE ciphertexts, with size -
    /// (`lwe_dimension_in` + 1) * number_of_samples * sizeof(u64)
    /// - `bit_extract_buffer`: some preallocated data used for storage of intermediate values
    /// during the computation
    /// - `ksk` keyswitch key
    /// - `fourier_bsk`  complex compressed bsk in fourier domain
    /// - `lwe_dimension_in` input LWE ciphertext dimension, supported input
    /// dimensions are: {512, 1024,2048, 4096, 8192}
    /// - `lwe_dimension_out` output LWE ciphertext dimension
    /// - `glwe_dimension` GLWE dimension,  only glwe_dimension = 1 is supported
    /// for now
    /// - `base_log_bsk` base_log for bootstrapping
    /// - `level_count_bsk` decomposition level count for bootstrapping
    /// - `base_log_ksk` base_log for keyswitch
    /// - `level_count_ksk` decomposition level for keyswitch
    /// - `number_of_samples` number of input LWE ciphertexts
    /// - `max_shared_memory` maximum amount of shared memory to be used inside
    /// device functions
    ///
    /// This function will call corresponding template of wrapper host function which
    /// will manage the calls of device functions.
    pub fn cuda_extract_bits_64(
        v_stream: *mut c_void,
        gpu_index: u32,
        list_lwe_array_out: *mut c_void,
        lwe_array_in: *const c_void,
        bit_extract_buffer: *mut i8,
        ksk: *const c_void,
        fourier_bsk: *const c_void,
        number_of_bits: u32,
        delta_log: u32,
        lwe_dimension_in: u32,
        lwe_dimension_out: u32,
        glwe_dimension: u32,
        polynomial_size: u32,
        base_log_bsk: u32,
        level_count_bsk: u32,
        base_log_ksk: u32,
        level_count_ksk: u32,
        number_of_samples: u32,
        max_shared_memory: u32,
    );

    /// This cleanup function frees the data for the bit extraction on GPU
    /// contained in bit_extract_buffer for 32 or 64-bit inputs.
    pub fn cleanup_cuda_extract_bits(
        v_stream: *mut c_void,
        gpu_index: u32,
        bit_extract_buffer: *mut *mut i8,
    );

    /// This scratch function allocates the necessary amount of data on the GPU for
    /// the circuit bootstrap on 64-bit inputs, into `circuit_bootstrap_buffer`.
    /// It also configures SM options on the GPU in case FULLSM or PARTIALSM mode is going to be
    /// used in the PBS.
    pub fn scratch_cuda_circuit_bootstrap_64(
        v_stream: *mut c_void,
        gpu_index: u32,
        cbs_buffer: *mut *mut i8,
        glwe_dimension: u32,
        lwe_dimension: u32,
        polynomial_size: u32,
        level_count: u32,
        number_of_inputs: u32,
        max_shared_memory: u32,
        allocate_gpu_memory: bool,
    );

    /// Perform circuit bootstrapping on a batch of 64 bit input LWE ciphertexts.
    /// - `v_stream` is a void pointer to the Cuda stream to be used in the kernel launch
    /// - `gpu_index` is the index of the GPU to be used in the kernel launch
    ///  - `ggsw_out` output batch of ggsw with size:
    /// `number_of_samples` * `level_cbs` * (`glwe_dimension` + 1)^2 *
    /// polynomial_size * sizeof(u64)
    ///  - `lwe_array_in` input batch of lwe ciphertexts, with size:
    /// `number_of_samples` * `(lwe_dimension` + 1) * sizeof(u64)
    ///  - `fourier_bsk` bootstrapping key in fourier domain with size:
    /// `lwe_dimension` * `level_bsk` * (`glwe_dimension` + 1)^2 *
    /// `polynomial_size` / 2 * sizeof(double2)
    ///  - `fp_ksk_array` batch of fp-keyswitch keys with size:
    /// (`polynomial_size` + 1) * `level_pksk` * (`glwe_dimension` + 1)^2 *
    /// `polynomial_size` * sizeof(u64)
    ///  The following 5 parameters are used during calculations, they are not actual
    ///  inputs of the function, they are just allocated memory for calculation
    ///  process, like this, memory can be allocated once and can be used as much
    ///  as needed for different calls of circuit_bootstrap function
    ///  - `lwe_array_in_shifted_buffer` with size:
    /// `number_of_samples` * `level_cbs` * (`lwe_dimension` + 1) * sizeof(u64)
    ///  - `lut_vector` with size:
    /// `level_cbs` * (`glwe_dimension` + 1) * `polynomial_size` * sizeof(u64)
    ///  - `lut_vector_indexes` stores the index corresponding to which test
    ///  vector to use
    ///  - `lwe_array_out_pbs_buffer` with size
    /// `number_of_samples` * `level_cbs` * (`polynomial_size` + 1) * sizeof(u64)
    ///  - `lwe_array_in_fp_ks_buffer` with size
    /// `number_of_samples` * `level_cbs` * (`glwe_dimension` + 1) *
    /// (`polynomial_size` + 1) * sizeof(u64)
    ///
    /// This function calls a wrapper to a device kernel that performs the
    /// circuit bootstrap. The kernel is templatized based on integer discretization and
    /// polynomial degree.
    pub fn cuda_circuit_bootstrap_64(
        v_stream: *mut c_void,
        gpu_index: u32,
        ggsw_out: *mut c_void,
        lwe_array_in: *const c_void,
        fourier_bsk: *const c_void,
        fp_ksk_array: *const c_void,
        lut_vector_indexes: *const c_void,
        cbs_buffer: *mut i8,
        delta_log: u32,
        polynomial_size: u32,
        glwe_dimension: u32,
        lwe_dimension: u32,
        level_bsk: u32,
        base_log_bsk: u32,
        level_pksk: u32,
        base_log_pksk: u32,
        level_cbs: u32,
        base_log_cbs: u32,
        number_of_samples: u32,
        max_shared_memory: u32,
    );

    /// This cleanup function frees the data for the circuit bootstrap on GPU
    /// contained in cbs_buffer for 32 or 64-bit inputs.
    pub fn cleanup_cuda_circuit_bootstrap(
        v_stream: *mut c_void,
        gpu_index: u32,
        cbs_buffer: *mut *mut i8,
    );

    /// This scratch function allocates the necessary amount of data on the GPU for the
    /// circuit bootstrap and vertical packing, into `cbs_vp_buffer`.
    /// It also fills the value of delta_log to be used in the circuit bootstrap.
    pub fn scratch_cuda_circuit_bootstrap_vertical_packing_64(
        v_stream: *mut c_void,
        gpu_index: u32,
        cbs_vp_buffer: *mut *mut i8,
        cbs_delta_log: *mut u32,
        glwe_dimension: u32,
        lwe_dimension: u32,
        polynomial_size: u32,
        level_count_cbs: u32,
        number_of_inputs: u32,
        lut_number: u32,
        max_shared_memory: u32,
        allocate_gpu_memory: bool,
    );

    /// Entry point for cuda circuit bootstrap + vertical packing for batches of
    /// input 64 bit LWE ciphertexts.
    ///  - `v_stream` is a void pointer to the Cuda stream to be used in the kernel launch
    ///  - `gpu_index` is the index of the GPU to be used in the kernel launch
    ///  - `lwe_array_out` list of output lwe ciphertexts
    ///  - `lwe_array_in` list of input lwe_ciphertexts
    ///  - `fourier_bsk` bootstrapping key in fourier domain, expected half size
    /// compressed complex key.
    ///  - `cbs_fpksk` list of private functional packing keyswitch keys
    ///  - `lut_vector` list of test vectors
    ///  - `cbs_vp_buffer` a pre-allocated array to store intermediate results
    ///  - `polynomial_size` size of the test polynomial, supported sizes:
    /// {512, 1024, 2048, 4096, 8192}
    ///  - `glwe_dimension` supported dimensions: {1}
    ///  - `lwe_dimension` dimension of input LWE ciphertexts
    ///  - `level_count_bsk` decomposition level for bootstrapping
    ///  - `base_log_bsk`  base log parameter for bootstrapping
    ///  - `level_count_pksk` decomposition level for fp-keyswitch
    ///  - `base_log_pksk` base log parameter for fp-keyswitch
    ///  - `level_count_cbs` level of circuit bootstrap
    ///  - `base_log_cbs` base log parameter for circuit bootstrap
    ///  - `number_of_inputs` number of input LWE ciphertexts
    ///  - `max_shared_memory` maximum shared memory amount to be used in
    ///  the kernels.
    pub fn cuda_circuit_bootstrap_vertical_packing_64(
        v_stream: *mut c_void,
        gpu_index: u32,
        lwe_array_out: *mut c_void,
        lwe_array_in: *const c_void,
        fourier_bsk: *const c_void,
        cbs_fpksk: *const c_void,
        lut_vector: *const c_void,
        cbs_vp_buffer: *mut i8,
        cbs_delta_log: u32,
        polynomial_size: u32,
        glwe_dimension: u32,
        lwe_dimension: u32,
        level_count_bsk: u32,
        base_log_bsk: u32,
        level_count_pksk: u32,
        base_log_pksk: u32,
        level_count_cbs: u32,
        base_log_cbs: u32,
        number_of_inputs: u32,
        lut_number: u32,
        max_shared_memory: u32,
    );

    /// This cleanup function frees the data for the circuit bootstrap and vertical packing on GPU
    /// contained in cbs_vp_buffer for 32 or 64-bit inputs.
    pub fn cleanup_cuda_circuit_bootstrap_vertical_packing(
        v_stream: *mut c_void,
        gpu_index: u32,
        cbs_vp_buffer: *mut *mut i8,
    );

    /// Scratch functions allocate the necessary data on the GPU.
    /// This scratch function allocates the necessary amount of data on the GPU for the wop PBS
    /// on 64-bit inputs into `wop_pbs_buffer`.
    /// It also fills the value of delta_log and cbs_delta_log to be used in the bit extract and
    /// circuit bootstrap.
    pub fn scratch_cuda_wop_pbs_64(
        v_stream: *mut c_void,
        gpu_index: u32,
        wop_pbs_buffer: *mut *mut i8,
        delta_log: *mut u32,
        cbs_delta_log: *mut u32,
        glwe_dimension: u32,
        lwe_dimension: u32,
        polynomial_size: u32,
        level_count_cbs: u32,
        level_count_bsk: u32,
        number_of_bits_of_message_including_padding: u32,
        number_of_bits_to_extract: u32,
        number_of_inputs: u32,
        max_shared_memory: u32,
        allocate_gpu_memory: bool,
    );

    /// Entry point for entire without padding programmable bootstrap on 64 bit input LWE
    /// ciphertexts.
    ///  - `v_stream` is a void pointer to the Cuda stream to be used in the kernel launch
    ///  - `gpu_index` is the index of the GPU to be used in the kernel launch
    ///  - `lwe_array_out` list of output lwe ciphertexts
    ///  - `lwe_array_in` list of input lwe_ciphertexts
    ///  - `lut_vector` list of test vectors
    ///  - `fourier_bsk` bootstrapping key in fourier domain, expected half size
    /// compressed complex key.
    ///  - `ksk` keyswitch key to use inside extract bits block
    ///  - `cbs_fpksk` list of fp-keyswitch keys
    ///  - `wop_pbs_buffer` a pre-allocated array to hold intermediate results
    ///  - `glwe_dimension` supported dimensions: {1}
    ///  - `lwe_dimension` dimension of input lwe ciphertexts
    ///  - `polynomial_size` size of the test polynomial, supported sizes:
    /// {512, 1024, 2048, 4096, 8192}
    ///  - `base_log_bsk`  base log parameter for bootstrapping
    ///  - `level_count_bsk` decomposition level for bootstrapping
    ///  - `base_log_ksk` base log parameter for keyswitch
    ///  - `level_count_ksk` decomposition level for keyswitch
    ///  - `base_log_pksk` base log parameter for fp-keyswitch
    ///  - `level_count_pksk` decomposition level for fp-keyswitch
    ///  - `base_log_cbs` base log parameter for circuit bootstrap
    ///  - `level_count_cbs` level of circuit bootstrap
    ///  - `number_of_bits_of_message_including_padding` number of bits to extract
    /// from each input lwe ciphertext including padding bit
    ///  - `number_of_bits_to_extract` number of bits to extract
    /// from each input lwe ciphertext without padding bit
    ///  - `number_of_inputs` number of input lwe ciphertexts
    ///  - `max_shared_memory` maximum shared memory amount to be used in
    ///  the kernels.
    pub fn cuda_wop_pbs_64(
        v_stream: *mut c_void,
        gpu_index: u32,
        lwe_array_out: *mut c_void,
        lwe_array_in: *const c_void,
        lut_vector: *const c_void,
        fourier_bsk: *const c_void,
        ksk: *const c_void,
        cbs_fpksk: *const c_void,
        wop_pbs_buffer: *mut i8,
        cbs_delta_log: u32,
        glwe_dimension: u32,
        lwe_dimension: u32,
        polynomial_size: u32,
        base_log_bsk: u32,
        level_count_bsk: u32,
        base_log_ksk: u32,
        level_count_ksk: u32,
        base_log_pksk: u32,
        level_count_pksk: u32,
        base_log_cbs: u32,
        level_count_cbs: u32,
        number_of_bits_of_message_including_padding: u32,
        number_of_bits_to_extract: u32,
        delta_log: u32,
        number_of_inputs: u32,
        max_shared_memory: u32,
    );

        /// This cleanup function frees the data for the wop PBS on GPU contained in
    /// wop_pbs_buffer for 32 or 64-bit inputs.
    pub fn cleanup_cuda_wop_pbs(
        v_stream: *mut c_void,
        gpu_index: u32,
        wop_pbs_buffer: *mut *mut i8,
    );

    /// Perform the negation of a u64 input LWE ciphertext vector.
    /// - `v_stream` is a void pointer to the Cuda stream to be used in the kernel launch
    /// - `gpu_index` is the index of the GPU to be used in the kernel launch
    /// - `lwe_array_out` is an array of size
    /// `(input_lwe_dimension + 1) * input_lwe_ciphertext_count` that should have been allocated on
    /// the GPU before calling this function, and that will hold the result of the computation.
    /// - `lwe_array_in` is the LWE ciphertext vector used as input, it should have been
    /// allocated and initialized before calling this function. It has the same size as the output
    /// array.
    /// - `input_lwe_dimension` is the number of mask elements in the two input and in the output
    /// ciphertext vectors
    /// - `input_lwe_ciphertext_count` is the number of ciphertexts contained in each input LWE
    /// ciphertext vector, as well as in the output.
    ///
    /// Each element (mask element or body) of the input LWE ciphertext vector is negated.
    /// The result is stored in the output LWE ciphertext vector. The input LWE ciphertext vector
    /// is left unchanged. This function is a wrapper to a device function that performs the
    /// operation on the GPU.
    pub fn cuda_negate_lwe_ciphertext_vector_64(
        v_stream: *mut c_void,
        gpu_index: u32,
        lwe_array_out: *mut c_void,
        lwe_array_in: *const c_void,
        input_lwe_dimension: u32,
        input_lwe_ciphertext_count: u32,
    );

    /// Perform the addition of two u64 input LWE ciphertext vectors.
    /// - `v_stream` is a void pointer to the Cuda stream to be used in the kernel launch
    /// - `gpu_index` is the index of the GPU to be used in the kernel launch
    /// - `lwe_array_out` is an array of size
    /// `(input_lwe_dimension + 1) * input_lwe_ciphertext_count` that should have been allocated on
    /// the GPU before calling this function, and that will hold the result of the computation.
    /// - `lwe_array_in_1` is the first LWE ciphertext vector used as input, it should have been
    /// allocated and initialized before calling this function. It has the same size as the output
    /// array.
    /// - `lwe_array_in_2` is the second LWE ciphertext vector used as input, it should have been
    /// allocated and initialized before calling this function. It has the same size as the output
    /// array.
    /// - `input_lwe_dimension` is the number of mask elements in the two input and in the output
    /// ciphertext vectors
    /// - `input_lwe_ciphertext_count` is the number of ciphertexts contained in each input LWE
    /// ciphertext vector, as well as in the output.
    ///
    /// Each element (mask element or body) of the input LWE ciphertext vector 1 is added to the
    /// corresponding element in the input LWE ciphertext 2. The result is stored in the output LWE
    /// ciphertext vector. The two input LWE ciphertext vectors are left unchanged. This function is
    /// a wrapper to a device function that performs the operation on the GPU.
    pub fn cuda_add_lwe_ciphertext_vector_64(
        v_stream: *mut c_void,
        gpu_index: u32,
        lwe_array_out: *mut c_void,
        lwe_array_in_1: *const c_void,
        lwe_array_in_2: *const c_void,
        input_lwe_dimension: u32,
        input_lwe_ciphertext_count: u32,
    );

    /// Perform the addition of a u64 input LWE ciphertext vector with a u64 input plaintext vector.
    /// - `v_stream` is a void pointer to the Cuda stream to be used in the kernel launch
    /// - `gpu_index` is the index of the GPU to be used in the kernel launch
    /// - `lwe_array_out` is an array of size
    /// `(input_lwe_dimension + 1) * input_lwe_ciphertext_count` that should have been allocated
    /// on the GPU before calling this function, and that will hold the result of the computation.
    /// - `lwe_array_in` is the LWE ciphertext vector used as input, it should have been
    /// allocated and initialized before calling this function. It has the same size as the output
    /// array.
    /// - `plaintext_array_in` is the plaintext vector used as input, it should have been
    /// allocated and initialized before calling this function. It should be of size
    /// `input_lwe_ciphertext_count`.
    /// - `input_lwe_dimension` is the number of mask elements in the input and output LWE
    /// ciphertext vectors
    /// - `input_lwe_ciphertext_count` is the number of ciphertexts contained in the input LWE
    /// ciphertext vector, as well as in the output. It is also the number of plaintexts in the
    /// input plaintext vector.
    ///
    /// Each plaintext of the input plaintext vector is added to the body of the corresponding LWE
    /// ciphertext in the LWE ciphertext vector. The result of the operation is stored in the output
    /// LWE ciphertext vector. The two input vectors are unchanged. This function is a
    /// wrapper to a device function that performs the operation on the GPU.
    pub fn cuda_add_lwe_ciphertext_vector_plaintext_vector_64(
        v_stream: *mut c_void,
        gpu_index: u32,
        lwe_array_out: *mut c_void,
        lwe_array_in: *const c_void,
        plaintext_array_in: *const c_void,
        input_lwe_dimension: u32,
        input_lwe_ciphertext_count: u32,
    );

    /// Perform the multiplication of a u64 input LWE ciphertext vector with a u64 input cleartext
    /// vector.
    /// - `v_stream` is a void pointer to the Cuda stream to be used in the kernel launch
    /// - `gpu_index` is the index of the GPU to be used in the kernel launch
    /// - `lwe_array_out` is an array of size
    /// `(input_lwe_dimension + 1) * input_lwe_ciphertext_count` that should have been allocated
    /// on the GPU before calling this function, and that will hold the result of the computation.
    /// - `lwe_array_in` is the LWE ciphertext vector used as input, it should have been
    /// allocated and initialized before calling this function. It has the same size as the output
    /// array.
    /// - `cleartext_array_in` is the cleartext vector used as input, it should have been
    /// allocated and initialized before calling this function. It should be of size
    /// `input_lwe_ciphertext_count`.
    /// - `input_lwe_dimension` is the number of mask elements in the input and output LWE
    /// ciphertext vectors
    /// - `input_lwe_ciphertext_count` is the number of ciphertexts contained in the input LWE
    /// ciphertext vector, as well as in the output. It is also the number of cleartexts in the
    /// input cleartext vector.
    ///
    /// Each cleartext of the input cleartext vector is multiplied to the mask and body of the
    /// corresponding LWE ciphertext in the LWE ciphertext vector.
    /// The result of the operation is stored in the output
    /// LWE ciphertext vector. The two input vectors are unchanged. This function is a
    /// wrapper to a device function that performs the operation on the GPU.
    pub fn cuda_mult_lwe_ciphertext_vector_cleartext_vector_64(
        v_stream: *mut c_void,
        gpu_index: u32,
        lwe_array_out: *mut c_void,
        lwe_array_in: *const c_void,
        cleartext_array_in: *const c_void,
        input_lwe_dimension: u32,
        input_lwe_ciphertext_count: u32,
    );


}
