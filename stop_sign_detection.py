import cv2

def distance_to_camera(known_width, focal_length, pixel_width):
    """Calculates the distance from the marker to the camera."""
    return (known_width * focal_length) / pixel_width

# Stop Sign Cascade Classifier xml
stop_sign = cv2.CascadeClassifier('cascade_stop_sign.xml')

img = cv2.imread('stop_sign_dataset\photo-1727156275339-aad186798856.jpg')

known_width = 76.0  # cm
focal_length = 800  # Example focal length in pixels

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
stop_sign_scaled = stop_sign.detectMultiScale(gray, 1.3, 5)

# Detect the stop sign, x,y = origin points, w = width, h = height
for (x, y, w, h) in stop_sign_scaled:

    distance = distance_to_camera(known_width, focal_length, w)
    # Draw rectangle around the stop sign
    stop_sign_rectangle = cv2.rectangle(img, (x,y),
                                        (x+w, y+h),
                                        (0, 255, 0), 3)
    # Write "Distance" on the bottom of the rectangle
    stop_sign_text = cv2.putText(img=stop_sign_rectangle,
                                 text=f"Distance: {distance:.2f} cm",
                                 org=(x, y+h+30),
                                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                 fontScale=1, color=(0, 0, 255),
                                 thickness=2, lineType=cv2.LINE_4)
cv2.imshow("img", img)
key = cv2.waitKey(-1)
if key == ord('q'):
    cv2.destroyAllWindows()
