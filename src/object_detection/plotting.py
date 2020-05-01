import cv2


def add_rectangle_to_image(image, bbox):
    (x, y, w, h) = bbox
    start_point = (x, y)
    end_point = (x + w, y + h)
    cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)


def add_text_to_image(image, text, x, y):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), int(1.5))


def add_detection_boxes_to_image(image_array, detections):
    for detection in detections:
        add_rectangle_to_image(image_array, detection["bbox"])
        add_text_to_image(image_array, detection["class"], detection["bbox"][0], detection["bbox"][1] - 5)
    return image_array
