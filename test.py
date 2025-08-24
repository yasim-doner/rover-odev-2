import cv2 as cv

img = cv.imread('stop_sign_dataset/photo-1558626219-fa0c107b5613.jpg')

b, g, r = cv.split(img)

ret, thresh_red = cv.threshold(r, 100, 255, cv.THRESH_BINARY, r)
ret, thresh_blue = cv.threshold(b, 40, 255, cv.THRESH_BINARY_INV, b)
ret, thresh_green = cv.threshold(g, 40, 255, cv.THRESH_BINARY_INV, g)

red_filter = cv.bitwise_and(thresh_blue, cv.bitwise_and(thresh_red, thresh_green))

cv.imshow('Red Channel Threshold', red_filter)

cv.waitKey(0)
cv.destroyAllWindows()


'''
capture = cv.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if not ret:
        break

    cv.imshow('Camera Feed', frame)

    if cv.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
'''