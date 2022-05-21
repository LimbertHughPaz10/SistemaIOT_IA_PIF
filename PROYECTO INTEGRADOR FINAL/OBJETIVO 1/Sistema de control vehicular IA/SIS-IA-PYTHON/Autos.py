import cv2

from Rastreador import *

seguimiento = Rastreador()
cap = cv2.VideoCapture("Prueba2.mp4")

deteccion = cv2.createBackgroundSubtractorMOG2(history=100000, varThreshold=13)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame,(1280,720))
    zona = frame[530:720, 300:850]

    mascara = deteccion.apply(zona)
    _, mascara = cv2.threshold(mascara, 254, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detecciones = []

    for cont in contornos:
        area = cv2.contourArea(cont)
        if area > 800:
            x, y, ancho, alto = cv2.boundingRect(cont)
            cv2.rectangle(zona,(x,y), (x + ancho, y + alto), (255,255,0), 3)
            detecciones.append([x, y, ancho, alto])
    
    info_id = seguimiento.rastreo(detecciones)
    for inf in info_id:
        x, y, ancho, alto, id = inf
        cv2.putText(zona, str(id), (x, y-15), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255),2)
        cv2.rectangle(zona, (x, y), (x + ancho, y + alto), (255,255,0), 3)

    print(info_id)
    cv2.imshow("Zona de Interes", zona)
    cv2.imshow("Camara 01", frame)
    cv2.imshow("Mascara", mascara)

    key = cv2.waitKey(5)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
