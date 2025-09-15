from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.core import EngineCore
from vllm.v1.request import RequestStatus
from vllm.v1.engine import ( EngineCoreOutput, EngineCoreOutputs, FinishReason)
# engine core related hook functions
def abort_all_reqs(self):
    """Abort all running and waiting requests and clean up resources."""
    scheduler = self.scheduler
    abort_lists = list(scheduler.running) + list(scheduler.waiting)

    if not abort_lists:
        # No requests to abort
        success = scheduler.reset_prefix_cache()
        assert success, f'prefix cache must be reset to prevent kv cache pollution! {success}'
        return

    client_outputs = {}
    for req in abort_lists:
        engine_output = EngineCoreOutput(
            request_id=req.request_id,
            new_token_ids=[],
            finish_reason=FinishReason.ABORT,
            new_logprobs=None,
            new_prompt_logprobs_tensors=None,
            stop_reason=None,
        )
        if req.client_index not in client_outputs:
            client_outputs[req.client_index] = []
        client_outputs[req.client_index].append(engine_output)

    request_ids = [req.request_id for req in abort_lists]
    scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    for client_index, outputs in client_outputs.items():
        engine_core_outputs = EngineCoreOutputs(outputs=outputs)
        self.output_queue.put_nowait((client_index, engine_core_outputs))

    success = scheduler.reset_prefix_cache()
    assert success, f'prefix cache must be reset to prevent kv cache pollution! {success}'

def areal_injected_update_weight(self, path):
    self.abort_all_reqs()
    return self.collective_rpc("update_weights", args = (path,))

def areal_injected_update_weight_xccl(self):
    self.abort_all_reqs()
    return self.collective_rpc("update_weight_xccl")
setattr(EngineCore, 'abort_all_reqs', abort_all_reqs)
setattr(EngineCore, 'areal_injected_update_weight', areal_injected_update_weight)
setattr(EngineCore, 'areal_injected_update_weight_xccl', areal_injected_update_weight_xccl)