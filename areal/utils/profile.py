import torch
from areal.api.cli_args import ProfilerConfig
from areal.platforms import current_platform


class Profiler:
    def __init__(self, config: ProfilerConfig):
        if not config:
            config = ProfilerConfig()

        self.enable = config.enable
        self.step_start = config.step_start
        self.step_end = config.step_end
        self.config = config
        self.memory_usage = config.memory_usage
        self.rank = torch.distributed.get_rank()
        self.prof = None
        print("profile_ranks = ", self.config.profile_ranks)
        if self.rank in self.config.profile_ranks:
            print(f"[Profiler] Profiler init for rank {self.rank}")
            self.prof = current_platform.profiler(config)


    def check(self):
        return self.prof is not None and self.enable


    def start(self):
        if self.config.enable:
            if self.check():
                print(f"[Profiler] started for rank {self.rank}")
                self.prof.start()
        if self.config.memory_usage:
            current_platform.start_dump()


    def step(self):
        if self.config.enable:
            if self.check():
                self.prof.step()
        if self.config.memory_usage:
            save_path = os.path.join(config.profiler.save_path + "_" + str(rank))
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            file_name = f"{save_path}/prof_memory_log_{self.rank}.pickle"
            current_platform.step_dump(file_name)


    def stop(self):
        if self.config.enable:
            if self.check():
                print(f"[Profiler] stopped for rank {self.rank}")
                self.prof.stop()
        if self.config.memory_usage:
            current_platform.stop_dump()