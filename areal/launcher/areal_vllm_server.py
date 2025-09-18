import os
from typing import Dict, List, Optional
from types import MethodType
import uvloop
from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from vllm.entrypoints.utils import cli_env_setup
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.api_server import run_server, router
from vllm.entrypoints.openai.protocol import OpenAIBaseModel
import logging
from vllm.logger import init_logger
import areal.launcher.areal_vllm_hook

logger = init_logger('areal_vllm_server')
logger.setLevel(logging.INFO)

class UpdateWeightsRequest(OpenAIBaseModel):
    # The model path with the new weights
    model_path: str
    # The format to load the weights
    load_format: Optional[str] = "auto"
    # Whether to abort all requests before updating weights
    abort_all_requests: bool = False

class UpdateGroupRequest(OpenAIBaseModel):
    master_address: str
    master_port: str
    rank_offset: int
    world_size: int
    backend: str
    group_name: str

class UpdateWeightsFromXcclRequest(OpenAIBaseModel):
    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]
    group_name: str

def to_json_response(success, message):
    content = {"success": success, "message": message}
    if success:
        return JSONResponse(content, status_code=200)
    else:
        return JSONResponse(content, status_code=400)

def build_response(ret_list):
    success = True
    message = ""
    for rank, ret_value in enumerate(ret_list):
        if_success, msg = ret_value
        success = (success if if_success else False)
        if if_success:
            message += f"TP rank: {rank} success\n"
        else:
            message += f"TP rank: {rank} failed. reason: {msg}\n"
    return to_json_response(success, message)

@router.post("/areal_update_weights")
async def update_weight(request: UpdateWeightsRequest, raw_request: Request):
    logger.info(f"API server starts update_weight, {request.model_path}")
    llm = raw_request.app.state.engine_client
    ret_list = await llm.engine_core.call_utility_async(
        "areal_injected_update_weight",
        request.model_path,
    )
    return build_response(ret_list)

@router.post("/areal_update_weights_xccl")
async def update_weight_xccl(raw_request: Request):
    logger.info(f"API server starts update_weight")
    llm = raw_request.app.state.engine_client
    ret_list = await llm.engine_core.call_utility_async(
        "areal_injected_update_weight_xccl",
    )
    return build_response(ret_list)

@router.post("/areal_init_weights_update_group")
async def init_weights_update_group(request: UpdateGroupRequest, raw_request: Request):
    logger.info(f"API server starts init_weights_update_group")
    llm = raw_request.app.state.engine_client
    ret_list = await llm.collective_rpc(
        "init_update_weight_group",
        args=(
            request.master_address,
            request.master_port,
            request.rank_offset,
            request.world_size,
            request.backend,
            request.group_name
        )
    )
    return build_response(ret_list)

@router.post("/areal_set_update_weight_meta")
async def set_weight_meta_xccl(request: UpdateWeightsFromXcclRequest, raw_request: Request):
    logger.info(f"API server starts upload meta")
    llm = raw_request.app.state.engine_client
    ret_list = await llm.collective_rpc(
        "set_weight_meta",
        args = (
            request.names,
            request.dtypes,
            request.shapes,
        )
    )
    return build_response(ret_list)

@router.post("/areal_pause_generation")
async def dummy_pause_generation(raw_request: Request):
    logger.info(f"API server starts dummy_pause_generation")
    return Response(status_code=200)

@router.post("/areal_continue_generation")
async def dummy_continue_generation(raw_request: Request):
    logger.info(f"API server starts dummy_continue_generation")
    return Response(status_code=200)


if __name__ == "__main__":
    os.environ["MULTIPROCESSING_PRELOAD"] = "areal_vllm_hook"
    # NOTE(simon):
    # This section should be in sync with vllm/entrypoints/cli/main.py for CLI
    # entrypoints.f
    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))