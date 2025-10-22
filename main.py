import logging
import multiprocessing
import os
import threading
from dataclasses import dataclass
from typing import Callable, List, Tuple

from flask import Flask, jsonify, render_template, request, Response
from image_processing import ProcessConfig, process_faces
from immich_api import validate_immich_connection
from compile_timelapse import compile_timelapse


# Filter out progress route logs
class ProgressRouteFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/progress" not in record.getMessage()

log = logging.getLogger('werkzeug')
log.addFilter(ProgressRouteFilter())

@dataclass
class GlobalConfig:
    """Configuration for the application."""
    api_key: str
    listen_address: str
    listen_port: int
    base_url: str
    output_folder: str
    left_eye_pos: Tuple[float, float]

# Initialize Flask app
app = Flask(__name__)

def required_env_var(name: str) -> str:
    """Get a required environment variable or raise an error."""
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} environment variable is not set.")
    return value

# Global state
AVAILABLE_CORES = multiprocessing.cpu_count()
progress_info = {"completed": 0, "total": 0, "status": "idle"}
processing_thread: threading.Thread
cancel_requested: bool = False
global_config = GlobalConfig(
    api_key=required_env_var("IMMICH_API_KEY"),
    base_url=required_env_var("IMMICH_BASE_URL"),
    output_folder=os.environ.get("OUTPUT_FOLDER", "output"),
    listen_address=os.environ.get("LISTEN_ADDRESS", "0.0.0.0"),
    listen_port=int(os.environ.get("LISTEN_PORT", 5000)),
    left_eye_pos=(0.35, 0.4),
)

def update_progress(current: int, total: int) -> None:
    """Update the global progress information.
    
    Args:
        current: Number of completed tasks
        total: Total number of tasks
    """
    progress_info["completed"] = current
    progress_info["total"] = total
    progress_info["status"] = "running" if current < total else "done"

def check_output_folder() -> Tuple[bool, int]:
    """Check if the output folder is empty.
    
    Returns:
        Tuple containing (is_empty, file_count)
    """
    if not os.path.exists(global_config.output_folder):
        os.makedirs(global_config.output_folder, exist_ok=True)
        return True, 0

    files = [f for f in os.listdir(global_config.output_folder) 
             if os.path.isfile(os.path.join(global_config.output_folder, f))]
    return len(files) == 0, len(files)

def background_process(
    config: ProcessConfig,
    max_workers: int,
    progress_callback: Callable[[int, int], None],
    cancel_flag: Callable[[], bool],
    do_not_compile_video: bool,
    framerate: int
) -> List[str]:
    """Process faces in the background and optionally compile a timelapse video.
    
    Args:
        max_workers: Number of worker processes for face processing
        progress_callback: Optional callback for progress updates
        cancel_flag: Optional function to check for cancellation
        do_not_compile_video: Whether to not compile a timelapse video after processing
        framerate: Frames per second for the output video
    """
    # Process faces
    process_faces(
        config=config,
        max_workers=max_workers,
        progress_callback=progress_callback,
        cancel_flag=cancel_flag
    )

    progress_callback(1, 1)
    
    if not do_not_compile_video and not cancel_flag():
        progress_info["status"] = "compiling_video"
        video_output_path = os.path.join(global_config.output_folder, "timelapse.mp4")
        success = compile_timelapse(
            image_folder=global_config.output_folder,
            output_path=video_output_path,
            framerate=framerate,
            update_progress=progress_callback
        )
        progress_info["status"] = "video_done" if success else "error:Video compilation failed"

@app.route("/progress")
def progress() -> Response:
    """Get current progress information."""
    return jsonify(progress_info)

@app.route("/check-connection")
def check_connection() -> Response:
    """Check connection to Immich server."""
    is_valid, message = validate_immich_connection(global_config.api_key, global_config.base_url)
    return jsonify({"valid": is_valid, "message": message})

@app.route("/cancel", methods=["POST"])
def cancel() -> Response:
    """Cancel the current processing job."""
    global processing_thread, cancel_requested
    cancel_requested = True
    if processing_thread and processing_thread.is_alive():
        progress_info["status"] = "cancelled"
        return jsonify({"success": True, "message": "Processing cancelled."})
    cancel_requested = False
    return jsonify({"success": False, "message": "No active processing to cancel."})

@app.route("/", methods=["GET", "POST"])
def index() -> str:
    """Handle the main page and processing requests."""
    global processing_thread, cancel_requested

    message = None
    error = None
    warning = None

    # Check output folder status
    is_empty, file_count = check_output_folder()
    if not is_empty:
        warning = f"Output folder is not empty. Contains {file_count} files. New images will be added to this folder."

    # Validate connection on POST
    if request.method == "POST":
        is_valid, message = validate_immich_connection(global_config.api_key, global_config.base_url)
        if not is_valid:
            error = f"Immich server connection error: {message}"
            return render_template("index.html", error=error, warning=warning,
                                max_workers_options=list(range(1, AVAILABLE_CORES + 1)))

        try:
            cancel_requested = False

            # Get form data
            config = ProcessConfig(
                api_key=global_config.api_key,
                base_url=global_config.base_url,
                output_folder=global_config.output_folder,
                left_eye_pos=global_config.left_eye_pos,
                person_id=request.form.get("person_id", type=str, default=""),
                output_image_size=request.form.get("resize_size", type=int, default=512),
                face_resolution_threshold=request.form.get("face_resolution_threshold", type=int, default=80),
                pose_threshold=request.form.get("pose_threshold", type=int, default=25),
                date_from=request.form.get("date_from"),
                date_to=request.form.get("date_to"),
                date_format=request.form.get("date_format") or None,
            )
            if not config.person_id:
                raise ValueError("Person ID is required.")

            max_workers = request.form.get("max_workers", type=int, default=1)
            do_not_compile_video = request.form.get("do_not_compile_video") == "on"
            framerate = request.form.get("framerate", type=int, default=15)

            # Reset progress info
            progress_info.update({
                "completed": 0,
                "total": 0,
                "status": "idle"
            })

            # Start processing
            processing_thread = threading.Thread(
                target=lambda: background_process(
                    config=config,
                    max_workers=max_workers,
                    progress_callback=update_progress,
                    cancel_flag=lambda: cancel_requested,
                    do_not_compile_video=do_not_compile_video,
                    framerate=framerate
                )
            )
            processing_thread.start()
            
            message = "Processing started. Please wait and watch the progress bar below."

        except Exception as e:
            error = f"Error processing request: {e}"

    return render_template("index.html", 
                         message=message, 
                         error=error, 
                         warning=warning,
                         max_workers_options=list(range(1, AVAILABLE_CORES + 1)))

if __name__ == "__main__":
    app.run(host=global_config.listen_address, port=global_config.listen_port, debug=False)
