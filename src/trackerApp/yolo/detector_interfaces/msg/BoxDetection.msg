# Un singolo box rilevato da YOLOE, in coordinate pixel del frame passato
# nella richiesta (convenzione xyxy: x1,y1 = angolo alto-sinistra, x2,y2 =
# angolo basso-destra), stesso formato di box.xyxy() di ultralytics.

float32 x1
float32 y1
float32 x2
float32 y2
float32 score
string class_name
int32 track_id   # ID assegnato da BoT-SORT (persistente tra chiamate consecutive
                  # con persist=True). -1 se la detection non ha un track associato
                  # (capita: non tutte le detection vengono per forza agganciate a
                  # un track in un dato frame) o se il tracciamento non e' attivo.

float32[] embedding  # Vettore di feature d'aspetto (da un modello di classificazione
                      # separato, es. yolo11n-cls.pt, penultimo layer) — usato dal
                      # NOSTRO ReID in demo_package (confronto coseno), indipendente
                      # da BoT-SORT. Array VUOTO se embedding_model_path non e'
                      # configurato lato DetectorNode — il chiamante deve gestire
                      # questo caso, non assumere che sia sempre popolato.
