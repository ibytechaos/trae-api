import json
import time
import uuid
from datetime import datetime

import httpx_sse
from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse, StreamingResponse
from httpx import AsyncClient

from .env import (
    TRAE_APP_ID,
    TRAE_DEVICE_BRAND,
    TRAE_DEVICE_CPU,
    TRAE_DEVICE_ID,
    TRAE_DEVICE_TYPE,
    TRAE_IDE_TOKEN,
    TRAE_IDE_VERSION,
    TRAE_IDE_VERSION_CODE,
    TRAE_IDE_VERSION_TYPE,
    TRAE_MACHINE_ID,
    TRAE_OS_VERSION,
)
from .types import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    CountTokensRequest,
    CountTokensResponse,
    MessageRequest,
    MessageResponse,
    MessageStreamChunk,
    Model,
)

app = FastAPI(
    title="Trae2OpenAI Proxy",
    description="A api proxy to make trae's builtin models openai compatible",
    version="0.1.0",
)


@app.get("/v1/models")
async def list_models(ide_token: str = Header(TRAE_IDE_TOKEN, alias="Authorization")) -> JSONResponse:
    ide_token = ide_token.removeprefix("Bearer ")
    async with AsyncClient() as client:
        response = await client.get(
            "https://trae-api-sg.mchost.guru/api/ide/v1/model_list",
            params={"type": "llm_raw_chat"},
            headers={
                "x-app-id": TRAE_APP_ID,
                "x-device-brand": TRAE_DEVICE_BRAND,
                "x-device-cpu": TRAE_DEVICE_CPU,
                "x-device-id": TRAE_DEVICE_ID,
                "x-device-type": TRAE_DEVICE_TYPE,
                "x-ide-token": ide_token,
                "x-ide-version": TRAE_IDE_VERSION,
                "x-ide-version-code": TRAE_IDE_VERSION_CODE,
                "x-ide-version-type": TRAE_IDE_VERSION_TYPE,
                "x-machine-id": TRAE_MACHINE_ID,
                "x-os-version": TRAE_OS_VERSION,
            },
        )
        return JSONResponse(
            {
                "object": "list",
                "data": [Model(created=0, id=model["name"]).model_dump() for model in response.json()["model_configs"]],
            }
        )


@app.post("/v1/chat/completions")
async def create_chat_completions(
    request: ChatCompletionRequest, ide_token: str = Header(TRAE_IDE_TOKEN, alias="Authorization")
) -> StreamingResponse:
    ide_token = ide_token.removeprefix("Bearer ")
    current_turn = sum(1 for msg in request.messages[:-1] if msg.role == "user")
    last_assistant_message = next(filter(lambda msg: msg.role == "assistant", reversed(request.messages)), None)

    async def stream_response():
        async with AsyncClient() as client:
            async with httpx_sse.aconnect_sse(
                client,
                "POST",
                "https://trae-api-sg.mchost.guru/api/ide/v1/chat",
                headers={
                    "x-app-id": TRAE_APP_ID,
                    "x-device-brand": TRAE_DEVICE_BRAND,
                    "x-device-cpu": TRAE_DEVICE_CPU,
                    "x-device-id": TRAE_DEVICE_ID,
                    "x-device-type": TRAE_DEVICE_TYPE,
                    "x-ide-token": ide_token,
                    "x-ide-version": TRAE_IDE_VERSION,
                    "x-ide-version-code": TRAE_IDE_VERSION_CODE,
                    "x-ide-version-type": TRAE_IDE_VERSION_TYPE,
                    "x-machine-id": TRAE_MACHINE_ID,
                    "x-os-version": TRAE_OS_VERSION,
                },
                json={
                    "chat_history": [
                        {
                            **msg.model_dump(),
                            "status": "success",
                            "locale": "zh-cn",
                        }
                        for msg in request.messages[:-1]
                    ],
                    "context_resolvers": [],
                    "conversation_id": str(uuid.uuid4()),
                    "current_turn": current_turn,
                    "generate_suggested_questions": False,
                    "intent_name": "general_qa_intent",
                    "is_preset": True,
                    "last_llm_response_info": (
                        {"turn": current_turn - 1, "is_error": False, "response": last_assistant_message.content}
                        if last_assistant_message
                        else {}
                    ),
                    "model_name": request.model,
                    "multi_media": [],
                    "provider": "",
                    "session_id": str(uuid.uuid4()),
                    "user_input": request.messages[-1].content,
                    "valid_turns": list(range(current_turn)),
                    "variables": json.dumps(
                        {"locale": "zh-cn", "current_time": datetime.now().strftime("%Y%m%d %H:%M:%S %A")}
                    ),
                },
            ) as response:
                chunk = ChatCompletionChunk(
                    choices=[],
                    created=int(time.time()),
                    id="",
                    model=request.model,
                )
                async for sse in response.aiter_sse():
                    sse_data = sse.json()
                    if sse.event == "metadata":
                        chunk.id = str(sse_data["prompt_completion_id"])
                    elif sse.event == "output":
                        content = sse_data["response"]
                        reasoning_content = sse_data["reasoning_content"]
                        chunk.choices = [
                            ChatCompletionChunkChoice(
                                delta={"role": "assistant", "content": content, "reasoning_content": reasoning_content}
                            )
                        ]
                        yield f"data: {chunk.model_dump_json()}\n\n"
                    elif sse.event == "token_usage":
                        chunk.choices = []
                        chunk.usage = {
                            "completion_tokens": sse_data["completion_tokens"],
                            "prompt_tokens": sse_data["prompt_tokens"],
                            "total_tokens": sse_data["total_tokens"],
                        }
                        yield f"data: {chunk.model_dump_json()}\n\n"
                    elif sse.event == "done":
                        chunk.choices = [ChatCompletionChunkChoice(delta={}, finish_reason="stop")]
                        yield f"data: {chunk.model_dump_json()}\n\ndata: [DONE]\n\n"
                    elif sse.event == "error":
                        chunk.choices = [
                            ChatCompletionChunkChoice(
                                delta={"role": "assistant", "content": sse.data}, finish_reason="error"
                            )
                        ]
                        yield f"data: {chunk.model_dump_json()}\n\ndata: [DONE]\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


@app.post("/v1/messages", response_model=None)
async def create_message(
    request: MessageRequest, ide_token: str = Header(TRAE_IDE_TOKEN, alias="Authorization")
):
    ide_token = ide_token.removeprefix("Bearer ")
    current_turn = sum(1 for msg in request.messages[:-1] if msg.role == "user")
    last_assistant_message = next(filter(lambda msg: msg.role == "assistant", reversed(request.messages)), None)

    if request.stream:
        # Streaming response
        async def stream_response():
            async with AsyncClient() as client:
                try:
                    async with httpx_sse.aconnect_sse(
                        client,
                        "POST",
                        "https://trae-api-sg.mchost.guru/api/ide/v1/chat",
                        headers={
                            "x-app-id": TRAE_APP_ID,
                            "x-device-brand": TRAE_DEVICE_BRAND,
                            "x-device-cpu": TRAE_DEVICE_CPU,
                            "x-device-id": TRAE_DEVICE_ID,
                            "x-device-type": TRAE_DEVICE_TYPE,
                            "x-ide-token": ide_token,
                            "x-ide-version": TRAE_IDE_VERSION,
                            "x-ide-version-code": TRAE_IDE_VERSION_CODE,
                            "x-ide-version-type": TRAE_IDE_VERSION_TYPE,
                            "x-machine-id": TRAE_MACHINE_ID,
                            "x-os-version": TRAE_OS_VERSION,
                        },
                        json={
                            "chat_history": [
                                {
                                    **msg.model_dump(),
                                    "status": "success",
                                    "locale": "zh-cn",
                                }
                                for msg in request.messages[:-1]
                            ],
                            "context_resolvers": [],
                            "conversation_id": str(uuid.uuid4()),
                            "current_turn": current_turn,
                            "generate_suggested_questions": False,
                            "intent_name": "general_qa_intent",
                            "is_preset": True,
                            "last_llm_response_info": (
                                {"turn": current_turn - 1, "is_error": False, "response": last_assistant_message.content}
                                if last_assistant_message
                                else {}
                            ),
                            "model_name": request.model,
                            "multi_media": [],
                            "provider": "",
                            "session_id": str(uuid.uuid4()),
                            "user_input": request.messages[-1].content,
                            "valid_turns": list(range(current_turn)),
                            "variables": json.dumps(
                                {"locale": "zh-cn", "current_time": datetime.now().strftime("%Y%m%d %H:%M:%S %A")}
                            ),
                        },
                    ) as response:
                        message_id = str(uuid.uuid4())
                        created_time = int(time.time())
                        
                        # Send message start event
                        start_chunk = MessageStreamChunk(
                            type="message_start",
                            message={
                                "id": message_id,
                                "type": "message",
                                "role": "assistant",
                                "content": [],
                                "model": request.model,
                                "stop_reason": None,
                                "stop_sequence": None,
                                "usage": {"input_tokens": 0, "output_tokens": 0}
                            }
                        )
                        yield f"event: message_start\ndata: {start_chunk.model_dump_json()}\n\n"
                        
                        # Send content block start
                        content_start_chunk = MessageStreamChunk(
                            type="content_block_start",
                            content_block={"type": "text", "text": ""}
                        )
                        yield f"event: content_block_start\ndata: {content_start_chunk.model_dump_json()}\n\n"
                        
                        total_content = ""
                        usage_info = {}
                        
                        async for sse in response.aiter_sse():
                            sse_data = sse.json()
                            if sse.event == "output":
                                content = sse_data["response"]
                                total_content = content
                                
                                # Send content block delta
                                delta_chunk = MessageStreamChunk(
                                    type="content_block_delta",
                                    delta={"type": "text_delta", "text": content}
                                )
                                yield f"event: content_block_delta\ndata: {delta_chunk.model_dump_json()}\n\n"
                                
                            elif sse.event == "token_usage":
                                usage_info = {
                                    "input_tokens": sse_data["prompt_tokens"],
                                    "output_tokens": sse_data["completion_tokens"]
                                }
                            elif sse.event == "done":
                                # Send content block stop
                                content_stop_chunk = MessageStreamChunk(type="content_block_stop")
                                yield f"event: content_block_stop\ndata: {content_stop_chunk.model_dump_json()}\n\n"
                                
                                # Send message delta with usage
                                message_delta_chunk = MessageStreamChunk(
                                    type="message_delta",
                                    delta={"stop_reason": "end_turn", "stop_sequence": None},
                                    usage=usage_info
                                )
                                yield f"event: message_delta\ndata: {message_delta_chunk.model_dump_json()}\n\n"
                                
                                # Send message stop
                                message_stop_chunk = MessageStreamChunk(type="message_stop")
                                yield f"event: message_stop\ndata: {message_stop_chunk.model_dump_json()}\n\n"
                                break
                            elif sse.event == "error":
                                # Send error event
                                error_chunk = MessageStreamChunk(
                                    type="error",
                                    delta={"type": "error", "error": {"type": "api_error", "message": sse.data}}
                                )
                                yield f"event: error\ndata: {error_chunk.model_dump_json()}\n\n"
                                break
                except Exception as e:
                    # If SSE connection fails, send error event
                    error_chunk = MessageStreamChunk(
                        type="error",
                        delta={"type": "error", "error": {"type": "api_error", "message": f"Failed to connect to Trae API: {str(e)}"}}
                    )
                    yield f"event: error\ndata: {error_chunk.model_dump_json()}\n\n"

        return StreamingResponse(stream_response(), media_type="text/event-stream")
    else:
        # Non-streaming response
        async with AsyncClient() as client:
            try:
                async with httpx_sse.aconnect_sse(
                    client,
                    "POST",
                    "https://trae-api-sg.mchost.guru/api/ide/v1/chat",
                    headers={
                        "x-app-id": TRAE_APP_ID,
                        "x-device-brand": TRAE_DEVICE_BRAND,
                        "x-device-cpu": TRAE_DEVICE_CPU,
                        "x-device-id": TRAE_DEVICE_ID,
                        "x-device-type": TRAE_DEVICE_TYPE,
                        "x-ide-token": ide_token,
                        "x-ide-version": TRAE_IDE_VERSION,
                        "x-ide-version-code": TRAE_IDE_VERSION_CODE,
                        "x-ide-version-type": TRAE_IDE_VERSION_TYPE,
                        "x-machine-id": TRAE_MACHINE_ID,
                        "x-os-version": TRAE_OS_VERSION,
                    },
                    json={
                        "chat_history": [
                            {
                                **msg.model_dump(),
                                "status": "success",
                                "locale": "zh-cn",
                            }
                            for msg in request.messages[:-1]
                        ],
                        "context_resolvers": [],
                        "conversation_id": str(uuid.uuid4()),
                        "current_turn": current_turn,
                        "generate_suggested_questions": False,
                        "intent_name": "general_qa_intent",
                        "is_preset": True,
                        "last_llm_response_info": (
                            {"turn": current_turn - 1, "is_error": False, "response": last_assistant_message.content}
                            if last_assistant_message
                            else {}
                        ),
                        "model_name": request.model,
                        "multi_media": [],
                        "provider": "",
                        "session_id": str(uuid.uuid4()),
                        "user_input": request.messages[-1].content,
                        "valid_turns": list(range(current_turn)),
                        "variables": json.dumps(
                            {"locale": "zh-cn", "current_time": datetime.now().strftime("%Y%m%d %H:%M:%S %A")}
                        ),
                    },
                ) as response:
                    content = ""
                    usage_info = {"input_tokens": 0, "output_tokens": 0}
                    
                    async for sse in response.aiter_sse():
                        sse_data = sse.json()
                        if sse.event == "output":
                            content = sse_data["response"]
                        elif sse.event == "token_usage":
                            usage_info = {
                                "input_tokens": sse_data["prompt_tokens"],
                                "output_tokens": sse_data["completion_tokens"]
                            }
                        elif sse.event == "done":
                            break
                        elif sse.event == "error":
                            return JSONResponse(
                                status_code=500,
                                content={"error": {"type": "api_error", "message": sse.data}}
                            )
            except Exception as e:
                # If SSE connection fails, return an error response
                return JSONResponse(
                    status_code=500,
                    content={"error": {"type": "api_error", "message": f"Failed to connect to Trae API: {str(e)}"}}
                )
                
                message_response = MessageResponse(
                    id=str(uuid.uuid4()),
                    created=int(time.time()),
                    model=request.model,
                    content=[{"type": "text", "text": content}],
                    stop_reason="end_turn",
                    usage=usage_info
                )
                
                return JSONResponse(message_response.model_dump())


@app.post("/v1/messages/count_tokens")
async def count_tokens(
    request: CountTokensRequest, ide_token: str = Header(TRAE_IDE_TOKEN, alias="Authorization")
) -> JSONResponse:
    ide_token = ide_token.removeprefix("Bearer ")
    
    # 构建用于计算token的文本内容
    text_content = ""
    for message in request.messages:
        if isinstance(message.content, str):
            text_content += f"{message.role}: {message.content}\n"
        elif isinstance(message.content, list):
            # 处理多模态内容，只提取文本部分
            for content_item in message.content:
                if isinstance(content_item, dict) and content_item.get("type") == "text":
                    text_content += f"{message.role}: {content_item.get('text', '')}\n"
    
    async with AsyncClient() as client:
        try:
            # 使用Trae API来计算token数量
            # 这里我们可以通过发送一个特殊的请求来获取token计数
            # 或者使用一个简单的估算方法
            
            # 简单的token估算：大约每4个字符为1个token（对于英文）
            # 对于中文，大约每个字符为1个token
            estimated_tokens = 0
            
            # 检测是否包含中文字符
            import re
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text_content))
            english_chars = len(text_content) - chinese_chars
            
            # 中文字符按1:1计算，英文字符按4:1计算
            estimated_tokens = chinese_chars + (english_chars // 4)
            
            # 添加一些基础的token开销（系统提示、格式化等）
            estimated_tokens += len(request.messages) * 10  # 每条消息的格式化开销
            
            response = CountTokensResponse(input_tokens=max(1, estimated_tokens))
            return JSONResponse(response.model_dump())
            
        except Exception as e:
            # 如果计算失败，返回一个保守的估算
            fallback_tokens = max(1, len(text_content) // 3)
            response = CountTokensResponse(input_tokens=fallback_tokens)
            return JSONResponse(response.model_dump())
