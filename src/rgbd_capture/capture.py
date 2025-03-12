from datetime import datetime, timezone
import os
from typing_extensions import Annotated

import cv2
import numpy as np
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer
import pyrealsense2 as rs

from rgbd_capture import serde

# TODO: Make sure to use path objects wherever possible.
# TODO: Save intrinsics and depth scale. Test to make sure there's
#  nothing else that's necessary.


def intrinsics_validator(intrinsics_list: list[list[float]]) -> np.ndarray:
    intrinsics_array = np.array(intrinsics_list, dtype=np.float64)

    assert intrinsics_array.shape == (3, 3)

    return intrinsics_array


def intrinsics_serializer(intrinsics: np.ndarray) -> list[list[float]]:
    return intrinsics.tolist()


Intrinsics = Annotated[
    np.ndarray,
    BeforeValidator(intrinsics_validator),
    PlainSerializer(intrinsics_serializer, return_type=list[list[float]]),
]


class CameraInfo(BaseModel):
    serial_number: str
    name: str
    frame_dimensions: tuple[int, int]
    intrinsics: Intrinsics
    depth_scale: float
    distortion_coeffs: list[float] = Field(default_factory=lambda: [0, 0, 0, 0, 0])

    model_config = ConfigDict(arbitrary_types_allowed=True)


def reverse_pixel_order(a: np.ndarray) -> np.ndarray:
    """
    Converts RGB to BGR and vice versa.
    """
    return np.flip(a, axis=-1)


def truncate_depth(depth: np.ndarray, max_depth: int = 5_000) -> np.ndarray:
    """
    Zero all grid positions which have a depth greater than
    max_depth. Then, send all of the zeros to max_depth.
    """
    truncated_depth = depth.copy()

    truncated_depth[truncated_depth > max_depth] = 0
    truncated_depth[truncated_depth == 0] = max_depth

    return truncated_depth


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    """
    Assumes depth is an array of uint16s. Converts it to
    a matrix of uint8s with the same dimensions. This is
    mostly useful for colorizing and displaying depth data.

    Could eventually use iinfo (https://numpy.org/doc/stable/reference/generated/numpy.iinfo.html#numpy-iinfo) if there's a need.
    """
    scaling_factor = 65_535 / depth.max()

    normalized_depth = (scaling_factor * depth / 256).astype(np.uint8)

    return normalized_depth


def view(
    frame_dimensions: tuple[int, int] = (1280, 720),
    fps: int = 15,
    max_depth: int = 5_000,
):
    width = frame_dimensions[0]
    height = frame_dimensions[1]
    fps = 15
    max_depth = 5_000

    print("Setting up camera...")
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    pipeline = rs.pipeline()
    pipeline.start(config)
    print("Press ESC to exit.")

    try:
        while True:
            frameset = pipeline.wait_for_frames()
            color_frame = frameset.get_color_frame()
            depth_frame = frameset.get_depth_frame()

            color_array = np.asarray(color_frame.get_data())
            depth_array = np.asarray(depth_frame.get_data())
            bgr_color_array = np.flip(color_array, axis=-1)

            truncated_depth = truncate_depth(depth_array, max_depth)
            normalized_depth = normalize_depth(truncated_depth)
            colorized_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_BONE)

            stacked_frames = np.vstack((bgr_color_array, colorized_depth))

            cv2.imshow("Color and Depth", stacked_frames)
            key = cv2.waitKey(1)
            if key == 27:
                break
    finally:
        pipeline.stop()


def snap(
    color_filename: str | None = None,
    depth_filename: str | None = None,
    frame_dimensions: tuple[int, int] = (1280, 720),
    fps: int = 15,
    frames_to_skip: int = 10,
    preview: bool = True,
):
    width = frame_dimensions[0]
    height = frame_dimensions[1]
    local_timezone = datetime.now().astimezone().tzinfo

    # Check for a connected device
    # Initialize camera
    print("Setting up camera...")
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, 15)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 15)

    pipeline = rs.pipeline()
    pipeline.start(config)

    try:
        # Skip frames
        print("Waiting for auto-exposure to settle...")
        for _ in range(frames_to_skip):
            pipeline.wait_for_frames()

        # Capture frame
        print("Taking a snapshot!")
        frameset = pipeline.wait_for_frames()
        capture_timestamp = datetime.now(local_timezone)
        color_frame = frameset.get_color_frame()
        depth_frame = frameset.get_depth_frame()

        color_array = np.asarray(color_frame.get_data())
        depth_array = np.asarray(depth_frame.get_data())
        bgr_color_array = np.flip(color_array, axis=-1)

        if preview:
            max_depth = 5_000
            truncated_depth = truncate_depth(depth_array, max_depth)
            normalized_depth = normalize_depth(truncated_depth)
            colorized_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_BONE)

            stacked_frames = np.vstack((bgr_color_array, colorized_depth))

            cv2.imshow("Color and Depth", stacked_frames)
            print('Press "y" to save the image.')
            key = cv2.waitKey(0)

            if not (key == ord("Y") or key == ord("y")):
                return

        print("Writing images...")

        # TODO: Revisit if there's a better time format.
        if color_filename is None:
            color_filename = capture_timestamp.strftime("%Y%m%dT%H%M%S%Z.jpg")

        if depth_filename is None:
            depth_filename = capture_timestamp.strftime("%Y%m%dT%H%M%S%Z.png")

        cv2.imwrite(color_filename, bgr_color_array)
        print(f"Color saved to {color_filename}")
        cv2.imwrite(depth_filename, depth_array)
        print(f"Depth saved to {depth_filename}")
    finally:
        pipeline.stop()

    # Save frames and conditionally intrinsics.


def burst(
    burst_dir: str = ".",
    frame_dimensions: tuple[int, int] = (1280, 720),
    fps: int = 15,
    frames_to_skip: int = 10,
    frames_to_capture: int = 10,
):
    width = frame_dimensions[0]
    height = frame_dimensions[1]
    local_timezone = datetime.now().astimezone().tzinfo

    # Check for a connected device
    # Initialize camera
    print("Setting up camera...")
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, 15)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 15)

    pipeline = rs.pipeline()
    pipeline.start(config)

    try:
        # Skip frames
        print("Waiting for auto-exposure to settle...")
        for _ in range(frames_to_skip):
            pipeline.wait_for_frames()

        framesets = []
        timestamps = []

        print(f"Taking burst of {frames_to_capture} snapshots")

        for _ in range(frames_to_capture):
            frameset = pipeline.wait_for_frames()
            capture_timestamp = datetime.now(local_timezone)
            color_frame = frameset.get_color_frame()
            depth_frame = frameset.get_depth_frame()

            color_array = np.asarray(color_frame.get_data())
            depth_array = np.asarray(depth_frame.get_data())

            framesets.append((color_array.copy(), depth_array.copy()))
            timestamps.append(capture_timestamp)

        print(f"Writing images to {burst_dir}")

        for timestamp, frameset in zip(timestamps, framesets):
            color_array, depth_array = frameset
            bgr_color_array = reverse_pixel_order(color_array)

            color_filename = timestamp.strftime("%Y%m%dT%H%M%S%f%Z.jpg")
            depth_filename = timestamp.strftime("%Y%m%dT%H%M%S%f%Z.png")

            cv2.imwrite(os.path.join(burst_dir, color_filename), bgr_color_array)
            cv2.imwrite(os.path.join(burst_dir, depth_filename), depth_array)
    finally:
        pipeline.stop()


def record():
    # Initialize camera
    # Skip frames?
    # Capture frames
    # Wait for time limit or user signal.
    # Encode
    # Save frames and conditionally intrinsics.
    print("Coming soon...")
    pass


def info(frame_dimensions: tuple[int, int], filename: str | None = None):
    width = frame_dimensions[0]
    height = frame_dimensions[1]

    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, 15)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 15)

    pipeline = rs.pipeline()
    pipeline.start(config)

    try:
        profile = pipeline.get_active_profile()
        intrinsics = (
            profile.get_stream(rs.stream.depth)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        # May or may not work generally across camera types.
        device = profile.get_device()
        serial_number = device.get_info(rs.camera_info.serial_number)
        device_type = device.get_info(rs.camera_info.name)
        depth_scale = device.first_depth_sensor().get_depth_scale()

        intrinsics_matrix = np.array(
            [
                [intrinsics.fx, 0, intrinsics.ppx],
                [0, intrinsics.fy, intrinsics.ppy],
                [0, 0, 1],
            ]
        )

        print(f"Device Type: {device_type}")
        print(f"Serial Number: {serial_number}")
        print(f"Frame Dimensions: {frame_dimensions}")
        print(f"Intrinsics Matrix:\n{intrinsics_matrix}")
        print(f"Depth Scale: {depth_scale}")
        print(f"Distortion Coeffs: {list(intrinsics.coeffs)}")

        if filename is not None:
            print(f"Writing to {filename}...")
            info = CameraInfo(
                name=device_type,
                serial_number=serial_number,
                frame_dimensions=(width, height),
                intrinsics=intrinsics_matrix,
                depth_scale=depth_scale,
                distortion_coeffs=list(intrinsics.coeffs),
            )

            serde.save_model(filename, info)
    finally:
        pipeline.stop()
