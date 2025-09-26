import torch

import areal.utils.logging as logging

from .platform import Platform

logger = logging.getLogger("CUDA Platform")


class CudaPlatform(Platform):
    device_name: str = "NVIDIA"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"
    ray_experimental_noset: str = "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
    communication_backend: str = "nccl"

    @classmethod
    def clear_cublas_workspaces(cls) -> None:
        torch._C._cuda_clearCublasWorkspaces()

    @classmethod
    def get_vllm_worker_class(clas):
        try:
            from vllm import envs

            if envs.VLLM_USE_V1:
                from vllm.v1.worker.gpu_worker import Worker

                logger.info("Successfully imported vLLM V1 Worker.")
                return Worker
            else:
                from vllm.worker.worker import Worker

                logger.info("Successfully imported vLLM V0 Worker.")
                return Worker
        except ImportError as e:
            logger.error(
                "Failed to import vLLM Worker. "
                "Make sure vLLM is installed correctly: %s",
                e,
            )
            raise RuntimeError(
                "vLLM is not installed or not properly configured."
            ) from e

    @classmethod
    def set_allocator_settings(cls) -> None:
        torch.cuda.memory._set_allocator_settings("expandable_segments:False")

    @classmethod
    def get_custom_env_vars(cls) -> dict:
        env_vars = {
            # "RAY_DEBUG": "legacy"
            "TORCHINDUCTOR_COMPILE_THREADS": "2",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "NCCL_CUMEM_ENABLE": "0",  # https://github.com/NVIDIA/nccl/issues/1234
            "NCCL_NVLS_ENABLE": "0",
        }
        return env_vars

    @classmethod
    def synchronize(cls) -> None:
        torch.cuda.synchronize()

    @classmethod
    def profiler(self, config):
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=max(config.step_start - 1, 0),
                warmup=1 if config.step_start > 0 else 0,
                active=config.step_end - config.step_start,
                repeat=1,
            ),
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(config.save_path),
        )
        return prof


    @classmethod
    def start_dump(self):
        torch.cuda.memory._record_memory_history()


    @classmethod
    def step_dump(self, file):
        torch.cuda.memory._dump_snapshot(file)


    @classmethod
    def stop_dump(self):
        torch.cuda.memory._record_memory_history(enabled=None)

