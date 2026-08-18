# Un singolo box rilevato da YOLOE, in coordinate pixel del frame passato
# nella richiesta (convenzione xyxy: x1,y1 = angolo alto-sinistra, x2,y2 =
# angolo basso-destra), stesso formato di box.xyxy() di ultralytics.

float32 x1
float32 y1
float32 x2
float32 y2
float32 score
string class_name
