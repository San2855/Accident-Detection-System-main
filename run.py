import threading
import subprocess

def run_camera():
    subprocess.run(["python", "camera.py"])

def run_detection():
    subprocess.run(["python", "detection.py"])

if __name__ == "__main__":
    camera_thread = threading.Thread(target=run_camera)
    detection_thread = threading.Thread(target=run_detection)

    camera_thread.start()
    detection_thread.start()

    camera_thread.join()
    detection_thread.join()
