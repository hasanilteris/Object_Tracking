import cv2

cap = cv2.VideoCapture('film.mp4')
fgbg=cv2.createBackgroundSubtractorMOG2(detectShadows=False)

while(1):
    ret,frame=cap.read()
    fgmask = fgbg.apply(frame)
    median = cv2.medianBlur(fgmask,3)

    (contours,hierarchy)=cv2.findContours(median.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) < 500:
            continue
        (x,y,w,h)=cv2.boundingRect(c)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)

    background = cv2.resize(median,(600,360))
    frame1 = cv2.resize(frame,(600,360))

    cv2.imshow('background',background)
    cv2.imshow('frame',frame1)

    k = cv2.waitKey(1) & 0xff
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()