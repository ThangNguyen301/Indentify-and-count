from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train2/weights/best.pt")
# ===== CHỌN NGUỒN =====
source = r"E:\Identify and count\images\Untitled-1-1-533x400.jpg"
#source = "E:\\Identify and count\\videos\\18394353-hd_1080_1920_30fps.mp4"
#source = 0 # webcam

# ===== ẢNH =====
if isinstance(source, str) and source.lower().endswith((".jpg", ".png", ".jpeg", ".webp")):

    img = cv2.imread(source)

    if img is None:
        print("❌ Không đọc được ảnh")
        exit()

    results = model(img, conf=0.25)

    count = 0
    for r in results:
        boxes = r.boxes
        count = len(r.boxes)

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            

    cv2.putText(img, f"Chicken: {count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ===== VIDEO / WEBCAM =====
else:
    cap = cv2.VideoCapture(source)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 640))

    results = model(frame, conf=0.5, iou=0.5)

    count = 0
    for r in results:
        count = len(r.boxes)

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(frame, f"Chicken: {count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Webcam/Video", frame)

    if cv2.waitKey(1) == 27:
        break