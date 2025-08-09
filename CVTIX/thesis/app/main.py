from fastapi import FastAPI, UploadFile, File
import shutil, os, cv2
from ultralytics import YOLO
from .tracker import get_tracker       
from .utils import save_snapshot      

app = FastAPI()

MODEL_PATH = "app/yolov8_model.pt"
model = YOLO(MODEL_PATH) if os.path.exists(MODEL_PATH) else YOLO("yolov8n.pt")

tracker = get_tracker()

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(temp_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if width == 0 or height == 0 or fps == 0:
        cap.release()
        os.remove(temp_path)
        return {"error": "Invalid video file or unsupported format."}

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = f"processed_{file.filename}"
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    saved_ids = set()
    results_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = model(frame)[0]
        boxes = []
        for det in detections.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            conf = float(det.conf[0])
            cls = int(det.cls[0])
            boxes.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        tracks = tracker.update_tracks(boxes, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_ltrb()
            results_data.append({"id": track_id, "bbox": bbox})

            # Save only one snapshot per track
            if track_id not in saved_ids:
                save_snapshot(frame, bbox, track_id)
                saved_ids.add(track_id)

        out.write(frame)

    cap.release()
    out.release()
    os.remove(temp_path)
    return {
        "results": results_data,
        "processed_video": out_path,
        "message": "Video processed successfully with bounding boxes."
    }