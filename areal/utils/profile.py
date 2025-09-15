import torch
import torch_npu
from areal.api.cli_args import ProfilerConfig, ProfilerMemoryConfig
class ProfMemory:
    def __init__(self, config: ProfilerMemoryConfig):
        if not config:
            config = ProfilerMemoryConfig()
        
        self.enable = config.enable
        self.save_path = config.save_dir
        if not config.enable:
            return
        self.config = config 
class Profiler:
    def __init__(self, config: ProfilerConfig):
        if not config:
            config = ProfilerConfig()
        print("prof_config = ", config)
        
        self.enable = config.enable
        if not config.enable:
            return
        self.step_start = config.step_start
        self.step_end = config.step_end
        self.config = config
        self.prof = None
        self.saved = False
        self.rank = torch.distributed.get_rank()
        if self.rank in self.config.profile_ranks:
            print(f"[Profiler] Profiler init for rank {self.rank}")
            self.prof = torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                schedule=torch_npu.profiler.schedule(
                    wait=max(self.step_start - 1, 0),
                    warmup=1 if self.step_start > 0 else 0,
                    active=self.step_end - self.step_start,
                    repeat=1,
                ),
                record_shapes=True,
                with_stack=True,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(self.config.save_path),
            )
    def _validate(self):
        pass
    def check(self):
        return self.prof is not None and self.enable
    def start(self):
        if self.check():
            print(f"[Profiler] started for rank {self.rank}")
            self.prof.start()
    def step(self):
        if self.check():
            self.prof.step()
    def stop(self):
        if self.check():
            print(f"[Profiler] stopped for rank {self.rank}")
            self.prof.stop()