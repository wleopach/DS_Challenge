import pandas as pd
import category_encoders as ce
from DataReader import  data
#Se crea el objeto target encoding
cat = ['genero', 'fecha', 'dispositivo', 'establecimiento','ciudad', 'tipo_tc', 'status_txn', 'is_prime']
encoder = ce.TargetEncoder(cols=cat)
numD = encoder.fit_transform(data[cat], data['fraude'])
