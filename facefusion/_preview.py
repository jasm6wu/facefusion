from typing import Optional

import cv2
import numpy

from facefusion import core, state_manager
from facefusion.audio import create_empty_audio_frame, get_audio_frame
from facefusion.common_helper import get_first
from facefusion.content_analyser import analyse_frame
from facefusion.face_analyser import get_average_face, get_many_faces
from facefusion.face_selector import sort_faces_by_order
from facefusion.face_store import get_reference_faces
from facefusion.filesystem import filter_audio_paths, is_image, is_video
from facefusion.processors.core import get_processors_modules
from facefusion.typing import AudioFrame, Face, FaceSet, VisionFrame
from facefusion.vision import get_video_frame, read_static_image, read_static_images, resize_frame_resolution


def process_frame(frame_number : int = 0) -> Optional[VisionFrame]:
	core.conditional_append_reference_faces()
	reference_faces = get_reference_faces() if 'reference' in state_manager.get_item('face_selector_mode') else None
	source_frames = read_static_images(state_manager.get_item('source_paths'))
	source_faces = []

	for source_frame in source_frames:
		temp_faces = get_many_faces([ source_frame ])
		temp_faces = sort_faces_by_order(temp_faces, 'large-small')
		if temp_faces:
			source_faces.append(get_first(temp_faces))
	source_face = get_average_face(source_faces)
	source_audio_path = get_first(filter_audio_paths(state_manager.get_item('source_paths')))
	source_audio_frame = create_empty_audio_frame()

	if source_audio_path and state_manager.get_item('output_video_fps') and state_manager.get_item('reference_frame_number'):
		reference_audio_frame_number = state_manager.get_item('reference_frame_number')
		if state_manager.get_item('trim_frame_start'):
			reference_audio_frame_number -= state_manager.get_item('trim_frame_start')
		temp_audio_frame = get_audio_frame(source_audio_path, state_manager.get_item('output_video_fps'), reference_audio_frame_number)
		if numpy.any(temp_audio_frame):
			source_audio_frame = temp_audio_frame

	if is_image(state_manager.get_item('target_path')):
		target_vision_frame = read_static_image(state_manager.get_item('target_path'))
		preview_vision_frame = process_preview_frame(reference_faces, source_face, source_audio_frame, target_vision_frame)
		return preview_vision_frame

	if is_video(state_manager.get_item('target_path')):
		temp_vision_frame = get_video_frame(state_manager.get_item('target_path'), frame_number)
		preview_vision_frame = process_preview_frame(reference_faces, source_face, source_audio_frame, temp_vision_frame)
		return preview_vision_frame

	return None


def process_preview_frame(reference_faces : FaceSet, source_face : Face, source_audio_frame : AudioFrame, target_vision_frame : VisionFrame) -> VisionFrame:
	target_vision_frame = resize_frame_resolution(target_vision_frame, (1024, 1024))
	source_vision_frame = target_vision_frame.copy()

	if analyse_frame(target_vision_frame):
		return cv2.GaussianBlur(target_vision_frame, (99, 99), 0)

	for processor_module in get_processors_modules(state_manager.get_item('processors')):
		if processor_module.pre_process('preview'):
			target_vision_frame = processor_module.process_frame(
			{
				'reference_faces': reference_faces,
				'source_face': source_face,
				'source_audio_frame': source_audio_frame,
				'source_vision_frame': source_vision_frame,
				'target_vision_frame': target_vision_frame
			})
	return target_vision_frame
