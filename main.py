import os
import cv2
import numpy as np
import subprocess
from pathlib import Path
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # pip install tqdm

# --------- CONFIGURATION ---------
input_video = "birds10.mp4"              # Input video file
background_image = "background.webp"    # Background image
temp_frames_dir = "./temp_frames"
processed_frames_dir = "./processed_frames"
output_video = "output_1.mp4"            # Final video file (mp4 with audio)

# Use CUDA (GPU) assistance if available for part of the pipeline
HAS_CUDA = False
if hasattr(cv2, "cuda"):
    try:
        HAS_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        HAS_CUDA = False

print(f"CUDA available: {HAS_CUDA}")


# --------- GREEN SCREEN REMOVAL (CPU) ---------
def remove_green_screen_cpu(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1]

    th = cv2.threshold(a_channel, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    masked = cv2.bitwise_and(image, image, mask=th)

    mlab = cv2.cvtColor(masked, cv2.COLOR_BGR2LAB)
    dst = cv2.normalize(
        mlab[:, :, 1], dst=None, alpha=0, beta=255,
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    threshold_value = 200
    dst_th = cv2.threshold(dst, threshold_value, 255, cv2.THRESH_BINARY_INV)[1]
    mlab[:, :, 1][dst_th == 255] = 127

    img2 = cv2.cvtColor(mlab, cv2.COLOR_LAB2BGR)
    img2_bgra = cv2.cvtColor(img2, cv2.COLOR_BGR2BGRA)
    img2_bgra[th == 0] = (0, 0, 0, 0)

    return img2_bgra


# --------- OPTIONAL CUDA-ASSISTED VERSION ---------
def remove_green_screen(image: np.ndarray) -> np.ndarray:
    """Wrapper: try a light CUDA path, fall back to pure CPU."""
    global HAS_CUDA

    if not HAS_CUDA:
        return remove_green_screen_cpu(image)

    try:
        # Use GPU just for color conversion as a demo of CUDA usage.
        gpu = cv2.cuda_GpuMat()
        gpu.upload(image)
        lab_gpu = cv2.cuda.cvtColor(gpu, cv2.COLOR_BGR2LAB)
        lab = lab_gpu.download()

        a_channel = lab[:, :, 1]

        th = cv2.threshold(a_channel, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        masked = cv2.bitwise_and(image, image, mask=th)

        mlab = cv2.cvtColor(masked, cv2.COLOR_BGR2LAB)
        dst = cv2.normalize(
            mlab[:, :, 1], dst=None, alpha=0, beta=255,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        threshold_value = 200
        dst_th = cv2.threshold(dst, threshold_value, 255, cv2.THRESH_BINARY_INV)[1]
        mlab[:, :, 1][dst_th == 255] = 127

        img2 = cv2.cvtColor(mlab, cv2.COLOR_LAB2BGR)
        img2_bgra = cv2.cvtColor(img2, cv2.COLOR_BGR2BGRA)
        img2_bgra[th == 0] = (0, 0, 0, 0)

        return img2_bgra

    except Exception as e:
        print("⚠ CUDA path failed, falling back to CPU:", e)
        HAS_CUDA = False
        return remove_green_screen_cpu(image)


# --------- PER-FRAME PROCESSING (for multiprocessing) ---------
def process_single_frame(args):
    """
    args: (filename, temp_frames_dir, processed_frames_dir, background_image_path, width, height)
    """
    filename, temp_dir, out_dir, bg_path, width, height = args

    frame_path = Path(temp_dir) / filename
    if not frame_path.is_file():
        return

    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        print(f"Skipping unreadable frame: {filename}")
        return

    # Load and resize background per process (simpler, still ok performance-wise)
    background = cv2.imread(bg_path, cv2.IMREAD_COLOR)
    if background is None:
        print(f"❌ Background image missing or unreadable in worker: {bg_path}")
        return
    background = cv2.resize(background, (width, height))

    rgba = remove_green_screen(frame)

    # Split channels
    b, g, r, a = cv2.split(rgba)
    subject = cv2.merge((b, g, r))
    alpha = a.astype(float) / 255.0
    alpha = alpha[..., None]

    # Composite: subject over background
    blended = (alpha * subject + (1 - alpha) * background).astype(np.uint8)

    out_path = Path(out_dir) / filename
    cv2.imwrite(str(out_path), blended)


def main():
    # Create directories
    Path(temp_frames_dir).mkdir(parents=True, exist_ok=True)
    Path(processed_frames_dir).mkdir(parents=True, exist_ok=True)

    # ------------- STEP 1: Read video & extract frames -------------
    video = cv2.VideoCapture(input_video)
    if not video.isOpened():
        print("❌ Could not open input video:", input_video)
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        print("⚠ FPS not detected — defaulting to 30 FPS")
        fps = 30.0

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {input_video}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")

    # Validate background early (main process)
    bg = cv2.imread(background_image, cv2.IMREAD_COLOR)
    if bg is None:
        print("❌ Background image missing or unreadable:", background_image)
        return

    # Extract frames with progress bar
    print("🎞 Extracting frames...")
    frame_idx = 0
    pbar_total = total_frames if total_frames > 0 else None
    with tqdm(total=pbar_total, desc="Extracting", unit="frame") as pbar:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            cv2.imwrite(f"{temp_frames_dir}/{frame_idx:06d}.png", frame)
            frame_idx += 1
            pbar.update(1)

    video.release()
    print(f"✔ Frames extracted: {frame_idx}")

    if frame_idx == 0:
        print("❌ No frames extracted. Exiting.")
        return

    # ------------- STEP 2: Process frames with multiprocessing -------------
    print("🧪 Removing green screen & compositing background (multiprocessing)...")
    frame_files = sorted(os.listdir(temp_frames_dir))
    tasks = [
        (
            fname,
            temp_frames_dir,
            processed_frames_dir,
            background_image,
            width,
            height,
        )
        for fname in frame_files
    ]

    # Use all CPU cores
    workers = max(1, cpu_count() - 1)  # leave 1 core free
    print(f"Using {workers} workers")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        list(
            tqdm(
                executor.map(process_single_frame, tasks),
                total=len(tasks),
                desc="Processing",
                unit="frame",
            )
        )

    print("✔ All frames processed")

    # ------------- STEP 3: Stitch back to video with audio -------------
    print("🎬 Creating final video with audio...")

    # ffmpeg command:
    # - first input: processed frames (video)
    # - second input: original video (for audio)
    # -c:a copy = copy original audio
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", f"{processed_frames_dir}/%06d.png",
        "-i", input_video,
        "-c:v", "mpeg4",          # if you get errors, this is safe on most builds
        "-pix_fmt", "yuv420p",
        "-q:v", "2",              # quality (1=best, 31=worst)
        "-c:a", "copy",
        "-shortest",              # stop when shortest stream ends
        output_video,
    ]

    print("Running FFmpeg:")
    print(" ".join(cmd))

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("🎯 Final video generated successfully →", output_video)
    else:
        print("❌ FFmpeg failed — scroll up to see the error output.")


if __name__ == "__main__":
    main()
