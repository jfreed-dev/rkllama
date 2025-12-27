import ctypes
from .classes import *
from .callback import callback_impl

# Connect the callback function between Python and C++
callback = callback_type(callback_impl)


class RKLLM(object):
    """
    RKLLM class for model initialization, inference, and release operations.
    Updated for RKLLM runtime 1.2.3.
    """

    def __init__(self, model_path, lora_model_path=None, prompt_cache_path=None,
                 max_context_len=4096, max_new_tokens=-1, temperature=0.8,
                 top_k=1, top_p=0.9, repeat_penalty=1.1):
        """
        Initialize RKLLM model.

        Args:
            model_path: Path to the .rkllm model file
            lora_model_path: Optional path to LoRA adapter
            prompt_cache_path: Optional path to prompt cache
            max_context_len: Maximum context length (default: 4096)
            max_new_tokens: Maximum new tokens to generate (-1 for unlimited)
            temperature: Sampling temperature (default: 0.8)
            top_k: Top-k sampling (default: 1)
            top_p: Top-p sampling (default: 0.9)
            repeat_penalty: Repetition penalty (default: 1.1)
        """
        rkllm_param = RKLLMParam()
        rkllm_param.model_path = bytes(model_path, 'utf-8')

        # Basic generation parameters
        rkllm_param.max_context_len = max_context_len
        rkllm_param.max_new_tokens = max_new_tokens
        rkllm_param.skip_special_token = True

        # Sampling parameters
        rkllm_param.top_k = top_k
        rkllm_param.top_p = top_p
        rkllm_param.temperature = temperature
        rkllm_param.repeat_penalty = repeat_penalty
        rkllm_param.frequency_penalty = 0.0
        rkllm_param.presence_penalty = 0.0

        # Mirostat parameters
        rkllm_param.mirostat = 0
        rkllm_param.mirostat_tau = 5.0
        rkllm_param.mirostat_eta = 0.1

        # New in 1.2.3
        rkllm_param.n_keep = 0
        rkllm_param.use_gpu = True

        rkllm_param.is_async = False

        # Multimodal parameters
        rkllm_param.img_start = "".encode('utf-8')
        rkllm_param.img_end = "".encode('utf-8')
        rkllm_param.img_content = "".encode('utf-8')

        # Extended parameters (updated for 1.2.3)
        rkllm_param.extend_param.base_domain_id = 1
        rkllm_param.extend_param.embed_flash = 1
        rkllm_param.extend_param.enabled_cpus_num = 4
        # Use big cores on RK3588 (cores 4-7)
        rkllm_param.extend_param.enabled_cpus_mask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)
        rkllm_param.extend_param.n_batch = 1
        rkllm_param.extend_param.use_cross_attn = 0

        self.handle = RKLLM_Handle_t()

        # Initialize model
        self.rkllm_init = rkllm_lib.rkllm_init
        self.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), callback_type]
        self.rkllm_init.restype = ctypes.c_int
        ret = self.rkllm_init(ctypes.byref(self.handle), ctypes.byref(rkllm_param), callback)
        if ret != 0:
            raise RuntimeError(f"Failed to initialize RKLLM model: {ret}")

        # Setup run function
        self.rkllm_run = rkllm_lib.rkllm_run
        self.rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
        self.rkllm_run.restype = ctypes.c_int

        # Setup destroy function
        self.rkllm_destroy = rkllm_lib.rkllm_destroy
        self.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.rkllm_destroy.restype = ctypes.c_int

        # Setup abort function (new in 1.2.3)
        self.rkllm_abort = rkllm_lib.rkllm_abort
        self.rkllm_abort.argtypes = [RKLLM_Handle_t]
        self.rkllm_abort.restype = ctypes.c_int

        # Setup clear KV cache function
        self.rkllm_clear_kv_cache = rkllm_lib.rkllm_clear_kv_cache
        self.rkllm_clear_kv_cache.argtypes = [RKLLM_Handle_t, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        self.rkllm_clear_kv_cache.restype = ctypes.c_int

        # LoRA adapter support
        self.lora_adapter_path = None
        self.lora_model_name = None
        if lora_model_path:
            self.lora_adapter_path = lora_model_path
            self.lora_adapter_name = "default"

            lora_adapter = RKLLMLoraAdapter()
            ctypes.memset(ctypes.byref(lora_adapter), 0, ctypes.sizeof(RKLLMLoraAdapter))
            lora_adapter.lora_adapter_path = ctypes.c_char_p(self.lora_adapter_path.encode('utf-8'))
            lora_adapter.lora_adapter_name = ctypes.c_char_p(self.lora_adapter_name.encode('utf-8'))
            lora_adapter.scale = 1.0

            rkllm_load_lora = rkllm_lib.rkllm_load_lora
            rkllm_load_lora.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMLoraAdapter)]
            rkllm_load_lora.restype = ctypes.c_int
            rkllm_load_lora(self.handle, ctypes.byref(lora_adapter))

        # Prompt cache support
        self.prompt_cache_path = None
        if prompt_cache_path:
            self.prompt_cache_path = prompt_cache_path

            rkllm_load_prompt_cache = rkllm_lib.rkllm_load_prompt_cache
            rkllm_load_prompt_cache.argtypes = [RKLLM_Handle_t, ctypes.c_char_p]
            rkllm_load_prompt_cache.restype = ctypes.c_int
            rkllm_load_prompt_cache(self.handle, ctypes.c_char_p(prompt_cache_path.encode('utf-8')))

    def tokens_to_ctypes_array(self, tokens, ctype):
        """Convert Python list of tokens to ctypes array."""
        return (ctype * len(tokens))(*tokens)

    def run(self, prompt_tokens, keep_history=0):
        """
        Run inference on the given prompt tokens.

        Args:
            prompt_tokens: List of token IDs
            keep_history: Whether to keep conversation history (0 or 1)
        """
        # Setup LoRA params if needed
        rkllm_lora_params = None
        if self.lora_model_name:
            rkllm_lora_params = RKLLMLoraParam()
            rkllm_lora_params.lora_adapter_name = ctypes.c_char_p(self.lora_model_name.encode('utf-8'))

        # Setup inference parameters (updated for 1.2.3)
        rkllm_infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam))
        rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
        rkllm_infer_params.lora_params = ctypes.byref(rkllm_lora_params) if rkllm_lora_params else None
        rkllm_infer_params.keep_history = keep_history

        # Setup input (updated for 1.2.3)
        rkllm_input = RKLLMInput()
        ctypes.memset(ctypes.byref(rkllm_input), 0, ctypes.sizeof(RKLLMInput))
        rkllm_input.role = None  # Will use default
        rkllm_input.enable_thinking = False
        rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_TOKEN

        # Ensure tokens end with EOS (token ID 2)
        if prompt_tokens[-1] != 2:
            prompt_tokens.append(2)

        token_array = (ctypes.c_int * len(prompt_tokens))(*prompt_tokens)

        rkllm_input.input_data.token_input.input_ids = token_array
        rkllm_input.input_data.token_input.n_tokens = ctypes.c_ulong(len(prompt_tokens))

        self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(rkllm_infer_params), None)

        return

    def run_prompt(self, prompt, keep_history=0):
        """
        Run inference on a text prompt.

        Args:
            prompt: Text prompt string
            keep_history: Whether to keep conversation history (0 or 1)
        """
        # Setup inference parameters
        rkllm_infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam))
        rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
        rkllm_infer_params.keep_history = keep_history

        # Setup input
        rkllm_input = RKLLMInput()
        ctypes.memset(ctypes.byref(rkllm_input), 0, ctypes.sizeof(RKLLMInput))
        rkllm_input.role = None
        rkllm_input.enable_thinking = False
        rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_PROMPT
        rkllm_input.input_data.prompt_input = prompt.encode('utf-8')

        self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(rkllm_infer_params), None)

        return

    def abort(self):
        """Abort current generation."""
        self.rkllm_abort(self.handle)

    def clear_kv_cache(self):
        """Clear the KV cache."""
        self.rkllm_clear_kv_cache(self.handle, 0, None, None)

    def release(self):
        """Release model resources."""
        self.rkllm_destroy(self.handle)
