import cv2
import detect

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    model = detect.detectapi(weights='weights/yolov5s.pt')
    while True:

        rec, img = cap.read()

        result, names = model.detect([img])
        # img = result[0][0]  # 图片的处理结果图片

        for cls, (x1, y1, x2, y2), conf in result[0][1]:  # 图片的处理结果标签。
            # print(cls, x1, y1, x2, y2, conf)
            if cls == 67:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255))
                cv2.putText(img, names[cls], (x1, y1 - 20), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 255))
                break

        cv2.imshow("vedio", img)

        if cv2.waitKey(1) == ord('q'):
            break
