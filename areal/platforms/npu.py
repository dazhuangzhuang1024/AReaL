import torch

from areal.api.cli_args import ProfilerConfig
import areal.utils.logging as logging

from .platform import Platform

logger = logging.getLogger("NPU Platform")


class NPUPlatform(Platform):
    device_name: str = "NPU"
    device_type: str = "npu"
    dispatch_key: str = "NPU"
    ray_device_key: str = "NPU"
    device_control_env_var: str = "ASCEND_RT_VISIBLE_DEVICES"
    ray_experimental_noset: str = "RAY_EXPERIMENTAL_NOSET_NPU_VISIBLE_DEVICES"
    communication_backend: str = "hccl"

    @classmethod
    def synchronize(cls) -> None:
        torch.npu.synchronize()

    @classmethod
    def profiler(self, config):
        import torch_npu
        prof = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=torch_npu.profiler.schedule(
                wait=max(config.step_start - 1, 0),
                warmup=1 if config.step_start > 0 else 0,
                active=config.step_end - config.step_start,
                repeat=1,
            ),
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
            experimental_config=torch_npu.profiler._ExperimentalConfig(data_simplification=False),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(config.save_path),
        )
        return prof


    @classmethod
    def start_dump(self):
        torch.npu.memory._record_memory_history()


    @classmethod
    def step_dump(self, file):
        torch.npu.memory._dump_snapshot(file)


    @classmethod
    def stop_dump(self):
        torch.npu.memory._record_memory_history(enabled=None)
