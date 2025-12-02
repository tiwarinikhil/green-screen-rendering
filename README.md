# 🎬 Automatic Green Screen Removal + Background Replacement + Video Rendering (Python)

This project takes an input video, removes its **green screen frame-by-frame**, overlays the subject on a **custom background image**, and exports a **final video with preserved audio**.

The entire pipeline is **GPU-accelerated (CUDA if available)** and optimized with **multiprocessing + progress bars** for large videos.

---

https://github.com/tiwarinikhil/green-screen-rendering/assets/birds

https://github.com/tiwarinikhil/green-screen-rendering/assets/output

## 🚀 Features

| Feature                              | Status |
| ------------------------------------ | ------ |
| Green screen removal (chroma key)    | ✅     |
| Background image replacement         | ✅     |
| Transparent PNG compositing          | ✅     |
| Multiprocessing (fast)               | ✅     |
| CUDA GPU acceleration (if available) | ⚡     |
| Progress Bars (tqdm)                 | 📊     |
| Original video audio preserved       | 🔊     |
| MP4 export                           | 🎥     |

---

## 🏗 How it works (Pipeline)

## Installation Guide

Input Video (.mp4)
⬇
Extract Frames
⬇
Remove Green Screen (PNG + Alpha)
⬇
Overlay Subject on Background Image
⬇
Re-assemble Processed Frames + Original Audio
⬇
Final Output Video

## 📦 Requirements

| Dependency | Version                  |
| ---------- | ------------------------ |
| Python     | 3.8+                     |
| OpenCV     | 4.x                      |
| NumPy      | latest                   |
| tqdm       | latest                   |
| Pillow     | latest                   |
| FFmpeg     | installed on system PATH |

## 🔧 Installation

1. Clone the Repository
   - First, clone the repository or download the script to your local machine.
   ```bash
   git clone https://github.com/your-repo/greenscreen-remover.git
   ```
2. Install Python
   - Ensure you have Python installed. You can download it from the official Python website.
3. Set up a Virtual Environment (Optional, but recommended)
   - Navigate to the directory where you downloaded the script.
   - Create a virtual environment:
     ```shell
     python -m venv env
     ```
   - Activate the virtual environment:
     - On Windows:
       ```shell
       .\env\Scripts\activate
       ```
     - On macOS/Linux:
       ```shell
       source env/bin/activate
       ```
4. Install dependencies
   - ```shell
     cd green-screen-rendering
     pip install -r requirements.txt
     ```

## Usage Guide

1. Place in the project directory:
   - input.mp4 → the video containing green screen
   - background.jpg → the new background
2. Run the script
   - ```shell
     python main.py
     ```

## Adjustable Settings

```shell
input_video = "input.mp4"
background_image = "background.jpg"
output_video = "output.mp4"
```

To export in .mov ProRes format instead of MP4:

```shell
output_video = "output.mov"
```
