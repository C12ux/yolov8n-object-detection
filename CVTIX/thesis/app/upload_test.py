import requests

url = "http://127.0.0.1:8000/upload-video/"
video_path = r"E:\CVTIX\thesis\videos\CCTV_VIDEO.mp4"  # Use raw string

with open(video_path, "rb") as f:
    files = {"file": (video_path, f, "video/mp4")}
    response = requests.post(url, files=files)

print(response.json())