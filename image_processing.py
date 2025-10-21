# image_processing.py
import os
import io
import concurrent.futures
from datetime import date, datetime
from dataclasses import dataclass
from PIL import Image, ImageOps
import numpy as np
import cv2
from tqdm import tqdm
import logging
from typing import Tuple
from immich_api import get_assets_with_person, download_asset

import insightface
from insightface.model_zoo import get_model


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
tqdm_handler = TqdmLoggingHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
tqdm_handler.setFormatter(formatter)
logger.addHandler(tqdm_handler)


@dataclass
class AppConfig:
    """Configuration for the application."""
    api_key: str
    base_url: str
    person_id: str
    output_folder: str
    resize_size: int
    face_resolution_threshold: int
    pose_threshold: float
    left_eye_pos: Tuple[float, float]
    date_from: str
    date_to: str
    date_format: str | None


def initialize_worker() -> None:
    """Initialize worker process with face predictor.
    """
    global landmark_model
    landmark_model = get_model('buffalo_l/2d106det.onnx', download=True, download_zip=True)
    global landmark_model_3d
    landmark_model_3d = get_model('buffalo_l/1k3d68.onnx', download=True, download_zip=True)


def detect_landmarks(img_np, face_data):
    """
    Detects facial landmarks in the image, resizing if necessary for better detection.

    Args:
        img_np (numpy.ndarray): The input image as a numpy array.
        face_data (numpy.ndarray): The bounding box of the face [x1, y1, x2, y2].

    Returns:
        dict or None: Dictionary containing facial landmarks in numpy arrays if successful,
                     None if face resolution is too low.
    """
    
    # Detect facial landmarks
    face = insightface.app.common.Face()
    face.bbox = face_data
    landmarks = landmark_model.get(img_np, face)
    
    # Convert to numpy arrays for specific facial features. Uses same indices as dlib's 68-point model.
    left_eye = np.array([landmarks[i] for i in [35, 41, 42, 39, 37, 36]])
    right_eye = np.array([landmarks[i] for i in [89, 95, 96, 93, 91, 90]])

    return {
        'left_eye': left_eye,
        'right_eye': right_eye
    }


def check_eye_visibility(left_eye, right_eye, ear_threshold=0.2) -> bool:
    """
    Checks if both eyes are visible by calculating the Eye Aspect Ratio (EAR).

    Args:
        left_eye (numpy.ndarray): Array of left eye landmarks.
        right_eye (numpy.ndarray): Array of right eye landmarks.
        ear_threshold (float): Threshold for eye visibility.

    Returns:
        bool: True if both eyes are open enough, False otherwise.
    """
    def calculate_ear(eye):
        v1 = np.linalg.norm(eye[1] - eye[5])
        v2 = np.linalg.norm(eye[2] - eye[4])
        h = np.linalg.norm(eye[0] - eye[3])
        ear = (v1 + v2) / (2.0 * h)
        return ear

    left_ear = calculate_ear(left_eye)
    right_ear = calculate_ear(right_eye)

    if left_ear < ear_threshold or right_ear < ear_threshold:
        return False

    return True


def get_head_pose(img_np, face_data):
    """
    Estimates the head pose (pitch, yaw, roll) using facial landmarks of insightface's 3D landmarks model.

    Args:
        image (numpy.ndarray): The input image as a numpy array.
        face (insightface.app.common.Face): The face object containing bounding box.

    Returns:
        tuple or None: (pitch, yaw, roll) in degrees if successful; otherwise None.
    """
    face = insightface.app.common.Face()
    face.bbox = face_data
    landmark_model_3d.get(img_np, face)
    
    return {
        'pitch': face.pose[0],
        'yaw': face.pose[1],
        'roll': face.pose[2]
    }


def calculate_eye_alignment_transform(
    left_eye_center: np.ndarray,
    right_eye_center: np.ndarray,
    output_size: int,
    desired_left_eye_pos: Tuple[float, float]
) -> np.ndarray:
    """Calculate the transformation matrix to align eyes at desired positions.
    
    Args:
        left_eye_center: Center coordinates of the left eye
        right_eye_center: Center coordinates of the right eye
        output_size: Size of the output image (width and height)
        desired_left_eye_pos: Desired position of left eye as percentages (x, y)
        
    Returns:
        np.ndarray: 2x3 transformation matrix for cv2.warpAffine
    """
    # Calculate the desired eye positions in the output image
    left_eye_target = np.array([
        output_size * desired_left_eye_pos[0],
        output_size * desired_left_eye_pos[1]
    ])
    right_eye_target = np.array([
        output_size * (1.0 - desired_left_eye_pos[0]),
        output_size * desired_left_eye_pos[1]
    ])

    # Calculate the angle between the current eye line and the target eye line
    current_angle = np.degrees(np.arctan2(
        right_eye_center[1] - left_eye_center[1],
        right_eye_center[0] - left_eye_center[0]
    ))
    target_angle = np.degrees(np.arctan2(
        right_eye_target[1] - left_eye_target[1],
        right_eye_target[0] - left_eye_target[0]
    ))
    rotation_angle = target_angle - current_angle

    # Calculate the scale factor to match the desired eye distance
    current_eye_distance = np.linalg.norm(right_eye_center - left_eye_center)
    target_eye_distance = np.linalg.norm(right_eye_target - left_eye_target)
    scale = target_eye_distance / current_eye_distance

    # Create the transformation matrix
    # First, translate to origin (center of eyes)
    center = np.array([
        (left_eye_center[0] + right_eye_center[0]) / 2,
        (left_eye_center[1] + right_eye_center[1]) / 2
    ])
    M1 = np.array([
        [1, 0, -center[0]],
        [0, 1, -center[1]],
        [0, 0, 1]
    ])

    # Then rotate
    angle_rad = np.radians(rotation_angle)
    M2 = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])

    # Then scale
    M3 = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1]
    ])

    # Finally, translate to target position
    target_center = np.array([
        (left_eye_target[0] + right_eye_target[0]) / 2,
        (left_eye_target[1] + right_eye_target[1]) / 2
    ])
    M4 = np.array([
        [1, 0, target_center[0]],
        [0, 1, target_center[1]],
        [0, 0, 1]
    ])

    # Combine all transformations
    M = M4 @ M3 @ M2 @ M1

    # Convert to 2x3 matrix for OpenCV
    return M[:2, :]

def crop_and_align_face(img_np: np.ndarray, face_data, output_size: int, pose_threshold: float, left_eye_pos: tuple[float, float]):
    """
    Aligns a face in an image by positioning the eyes at specified locations.

    Args:
        image (np.ndarray): The input image.
        face_data (np.ndarray): The bounding box of the face [x1, y1, x2, y2].
        output_size (int): Size to resize the output image to.
        face_resolution_threshold (int): Minimum face resolution threshold.
        pose_threshold (float): Maximum allowed head pose deviation.
        left_eye_pos (tuple): Desired position of the left eye in the output as percentages (x, y).

    Returns:
        PIL.Image or None: The aligned face image if successful, None otherwise.
    """
    try:
        # TODO check face resolution
        #if not landmarks:
        #    logger.info("Face resolution is too low")
        #    return None
        
        # Detect landmarks in the face region
        landmarks = detect_landmarks(img_np, face_data)

        # Check if both eyes are visible
        if not check_eye_visibility(landmarks['left_eye'], landmarks['right_eye']):
            logger.info("Eyes are too closed")
            return None

        # Get head pose
        pose = get_head_pose(img_np, face_data)
        if not pose:
            logger.info("Could not estimate head pose")
            return None

        # Check if head pose is within acceptable range
        if abs(pose['yaw']) > pose_threshold or abs(pose['pitch']) > pose_threshold or abs(pose['roll']) > pose_threshold:
            logger.info(f"Head pose exceeds threshold: pitch={pose['pitch']:.1f}°, yaw={pose['yaw']:.1f}°, roll={pose['roll']:.1f}°")
            return None

        # Get eye positions
        left_eye_center = np.mean(landmarks['left_eye'], axis=0)
        right_eye_center = np.mean(landmarks['right_eye'], axis=0)

        # Calculate transformation matrix
        rotation_matrix = calculate_eye_alignment_transform(
            left_eye_center,
            right_eye_center,
            output_size,
            left_eye_pos
        )

        # Apply transformation
        aligned_face = cv2.warpAffine(
            img_np,
            rotation_matrix,
            (output_size, output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        return aligned_face

    except Exception as e:
        logger.exception(f"Error during face alignment: {e}")
        return None

def add_bottom_center_text(image, text):
    """
    Add text at the bottom center of an image using OpenCV.
    
    Args:
        image: Input image (numpy array)
        text: Text to write
        font_scale: Size of the font
        color: Text color as BGR tuple (default: white)
        thickness: Text thickness
    
    Returns:
        Image with text added
    """
    # Get image dimensions
    height, width = image.shape[:2]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    # Get text size to center it properly
    (text_width, _), _ = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)

    # Calculate position for bottom center
    x = (width - text_width) // 2  # Center horizontally
    y = height - 20  # 20 pixels from bottom
    
    # Add text
    return cv2.putText(image, text, (x, y), font, fontScale=font_scale, color=(255,255,255), thickness=thickness, lineType=cv2.LINE_AA)


def process_asset_worker(asset, config: AppConfig):
    """
    Worker function to process a single asset.

    This function downloads the asset, crops the face based on metadata,
    verifies resolution, aligns the face, and then saves the aligned face.

    Args:
        asset (dict): The asset metadata.
        config (AppConfig): Configuration parameters.

    Returns:
        str or None: The file path of the saved image if processing is successful; otherwise None.
    """
    try:
        asset_id = asset['id']
        image_bytes = download_asset(config.api_key, config.base_url, asset_id)
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
    except Exception as e:
        logger.exception(f"Error processing asset {asset.get('id')}: {e}")
        return None

    matching_person = next((p for p in asset.get('people', []) if p.get('id') == config.person_id), None)
    face_data = matching_person.get('faces', [])[0]

    # Convert image to numpy array for OpenCV processing
    img_np = np.array(image)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    aligned_face = crop_and_align_face(
        img_np,
        scale_detected_face(image, face_data),
        output_size=config.resize_size,
        pose_threshold=config.pose_threshold,
        left_eye_pos=config.left_eye_pos
    )
    
    if aligned_face is None:
        return None

    dt = datetime.fromisoformat(asset['fileCreatedAt'].replace("Z", "+00:00"))

    if config.date_format:
        aligned_face = write_date_text(aligned_face, dt.date(), config.date_format)

    # Convert back to PIL Image
    aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
    aligned_face = Image.fromarray(aligned_face)


    timestamp = dt.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(config.output_folder, f"{timestamp}.jpg")
    aligned_face.save(filename)
    return filename

def write_date_text(image: np.ndarray, timestamp: date, date_format: str) -> np.ndarray:
    """
    Adds date text to the bottom center of the image.

    Args:
        image (np.ndarray): The input image.
        timestamp (date): The date to display.
        date_format (str): The format string for the date.
    """
    text = timestamp.strftime(date_format)

    return add_bottom_center_text(image, text)

def scale_detected_face(raw_image: Image.Image, immich_face_data: dict[str, any]):
    """
    Converts Immich face metadata to an InsightFace Face object.

    Args:
        raw_image (PIL.Image): The original image.
        immich_face_data (dict): Face metadata from Immich API.

    Returns:
        insightface.app.common.Face: Converted Face object.
    """
    face_img_width = int(immich_face_data.get("imageWidth"))
    face_img_height = int(immich_face_data.get("imageHeight"))
    img_width, img_height = raw_image.size
    scale_x = img_width / face_img_width
    scale_y = img_height / face_img_height
    x1 = int(immich_face_data.get("boundingBoxX1") * scale_x)
    y1 = int(immich_face_data.get("boundingBoxY1") * scale_y)
    x2 = int(immich_face_data.get("boundingBoxX2") * scale_x)
    y2 = int(immich_face_data.get("boundingBoxY2") * scale_y)
    
    return np.array([x1, y1, x2, y2], dtype=np.int32)

def process_faces(config: AppConfig, max_workers=1, progress_callback=None, cancel_flag=None):
    """
    Processes assets containing the person and saves aligned face images.

    This function retrieves assets from the API, then uses a process pool to
    concurrently download, crop, and align faces.

    Args:
        config (AppConfig): Configuration parameters.
        max_workers (int): Number of worker processes.
        progress_callback (callable, optional): A callback function for progress updates.
        cancel_flag (callable, optional): A function that returns True if processing should be cancelled.

    Returns:
        list: A list of file paths of the saved images.
    """
    if cancel_flag and cancel_flag():
        logger.info("Processing was cancelled.")
        return []

    assets = get_assets_with_person(config.api_key, config.base_url, config.person_id, config.date_from, config.date_to)
    logger.info(f"Found {len(assets)} assets containing the person.")

    total_assets = len(assets)
    if progress_callback:
        progress_callback(0, total_assets)
    processed_files = []
    completed_count = 0

    with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=initialize_worker) as executor:
        future_to_asset = {executor.submit(process_asset_worker, asset, config): asset
                           for asset in assets}
        for future in tqdm(concurrent.futures.as_completed(future_to_asset), total=total_assets):

            if cancel_flag and cancel_flag():
                logger.info("Processing was cancelled.")
                for f in future_to_asset:
                    f.cancel()
                executor.shutdown(wait=False)
                return processed_files

            try:
                result = future.result()
                if result is not None:
                    processed_files.append(result)
            except Exception as e:
                logger.exception(f"Asset processing failed: {e}")

            completed_count += 1
            if progress_callback:
                progress_callback(completed_count, total_assets)

    logger.info(f"Finished processing. {len(processed_files)} images saved out of {total_assets} assets.")
    return processed_files