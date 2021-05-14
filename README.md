# DS_Challenge

1. Instalar los paquetes que están en el archivo requirements.txt
   pip install -r requeriments.txt
2. Las siguientes librerias no se instalan (por alguna razón) con el anterior paso hay que correr 
   pip install category_encoders
   pip install tabulate 
   pip install imblearn
   pip install imbalanced-learn
   
3.Abrir el archivo ds_challenge.pdf, allí está el análisis de la data y las respuestas a las preguntas del reto. 
4.Al correr CustumerClassification.py se imprimen dos clasificasiones de los clientes correspondietes a KMeans y a Clusstering Jerárquico
5.Al correr FraudstersDetector.py aparecen imperesas caracteísticas de los modelos que comparé para elegir el final, luego se imprimen resultados del modelo elegido.
6.FinalM.py es  el modelo para producción, al correrlo se ejecuta sobre un conjunto de prueba, generado por scikitlearn.
   
