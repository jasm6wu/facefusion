import asyncio
import json
from typing import Any, List

import cv2
import uvicorn
from litestar import Litestar, WebSocket, get as read, websocket as stream, websocket_listener
from litestar.static_files import create_static_files_router

from facefusion import _preview, choices, execution, state_manager, vision
from facefusion.processors import choices as processors_choices
from facefusion.state_manager import get_state
from facefusion.typing import ExecutionDevice


@read('/choices')
async def read_choices() -> Any:
	__choices__ = {}

	for key in dir(choices):
		if not key.startswith('__'):
			value = getattr(choices, key)

			if isinstance(value, (dict, list)):
				__choices__[key] = value

	return __choices__


@read('/processors/choices')
async def read_processors_choices() -> Any:
	__processors_choices__ = {}

	for key in dir(processors_choices):
		if not key.startswith('__'):
			value = getattr(processors_choices, key)

			if isinstance(value, (dict, list)):
				__processors_choices__[key] = value

	return __processors_choices__


@read('/execution/providers')
async def read_execution_providers() -> Any:
	return execution.get_execution_provider_set()


@stream('/execution/devices')
async def stream_execution_devices(socket : WebSocket[Any, Any, Any]) -> None:
	await socket.accept()

	while True:
		await socket.send_json(execution.detect_execution_devices())
		await asyncio.sleep(0.5)


@read('/execution/devices')
async def read_execution_devices() -> List[ExecutionDevice]:
	return execution.detect_execution_devices()


@read('/static_execution/devices')
async def read_static_execution_devices() -> List[ExecutionDevice]:
	return execution.detect_static_execution_devices()


@stream('/state')
async def stream_state(socket : WebSocket[Any, Any, Any]) -> None:
	await socket.accept()

	while True:
		await socket.send_json(get_state())
		await asyncio.sleep(0.5)


@read('/preview', media_type = 'image/png', mode = "binary")
async def read_preview(frame_number : int) -> bytes:
	_, preview_vision_frame = cv2.imencode('.png', _preview.process_frame(frame_number)) #type:ignore
	return preview_vision_frame.tobytes()


@websocket_listener("/preview", send_mode = "binary")
async def stream_preview(data : str) -> bytes:
	frame_number = int(json.loads(data).get('frame_number'))
	_, preview_vision_frame = cv2.imencode('.png', _preview.process_frame(frame_number)) #type:ignore
	return preview_vision_frame.tobytes()


@read('/ui/preview_slider')
async def read_ui_preview_slider() -> Any:
	target_path = state_manager.get_item('target_path')
	video_frame_total = vision.count_video_frame_total(target_path)

	return\
	{
		'video_frame_total': video_frame_total
	}


api = Litestar(
[
	read_choices,
	read_processors_choices,
	stream_execution_devices,
	read_execution_devices,
	read_static_execution_devices,
	stream_state,
	read_preview,
	read_ui_preview_slider,
	stream_preview,
	create_static_files_router(
		path = '/frontend',
		directories = [ 'facefusion/static' ],
		html_mode = True,
    )
])


def run() -> None:
	uvicorn.run(api)
