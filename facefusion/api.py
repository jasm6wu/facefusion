import asyncio
import time
from io import BytesIO
from typing import Any, List

import cv2
import uvicorn
from litestar import Litestar, WebSocket, get as read, websocket as stream

from facefusion import choices, execution, _preview
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
async def stream_execution_devices(socket : WebSocket) -> None:
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
async def stream_state(socket : WebSocket) -> None:
	await socket.accept()

	while True:
		await socket.send_json(get_state())
		await asyncio.sleep(0.5)


@read('/preview', media_type = 'image/png')
async def read_preview() -> None:
	_, preview_vision_frame = cv2.imencode('.png', _preview.process_frame())
	return preview_vision_frame.tobytes()


api = Litestar(
[
	read_choices,
	read_processors_choices,
	stream_execution_devices,
	read_execution_devices,
	read_static_execution_devices,
	stream_state,
	read_preview
])


def run() -> None:
	uvicorn.run(api)
