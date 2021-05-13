import pandas as pd
import pandas_profiling as pp
import matplotlib.pyplot as plt
import category_encoders as ce
# Lectura del csv
data = pd.read_csv('ds_challenge_2021.csv', sep=',')

# Pie diagram para dia de la semana
# labels = ['Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo', 'Lunes', 'Martes']
# size = [data['fecha'].value_counts()[0], data['fecha'].value_counts()[1], data['fecha'].value_counts()[2],
#         data['fecha'].value_counts()[3], data['fecha'].value_counts()[4], data['fecha'].value_counts()[5],
#         data['fecha'].value_counts()[6]]
# colors = ['grey', 'yellow', 'red', 'blue', 'green', 'orange', 'brown']
# explode = (0, 0, 0.1, 0, 0, 0, 0)
# plt.pie(size, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
# plt.axis('equal')
# plt.show()
# # Pie diagram para genero
# labels = ['M', 'F', 'No definido']
# size = [data['genero'].value_counts()[0], data['genero'].value_counts()[1], data['genero'].value_counts()[2]]
# colors = ['grey', 'yellow', 'red']
# explode = (0.1, 0, 0)
# plt.pie(size, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
# plt.axis('equal')
# plt.show()
# # Pie diagram para fraude
# labels = ['Fraude', 'Legal']
# size = [data['fraude'].value_counts()[1], data['fraude'].value_counts()[0]]
# colors = ['grey', 'yellow']
# explode = (0.1, 0)
# plt.pie(size, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
# plt.axis('equal')
# plt.show()
# # Pie diagram para establecimiento
# labels = ['Restaurante', 'Abarrotes', 'Super', 'MPago', 'Farmacia']
# size = [data['establecimiento'].value_counts()[0], data['establecimiento'].value_counts()[1],
#         data['establecimiento'].value_counts()[2], data['establecimiento'].value_counts()[3],
#         data['establecimiento'].value_counts()[4]]
# colors = ['grey', 'yellow', 'blue', 'green', 'white']
# explode = (0.1, 0, 0, 0, 0)
# plt.pie(size, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
# plt.axis('equal')
# plt.show()
# # Pie diagram para ciudad
# labels = ['Toluca', 'Guadalajara', 'Merida', 'Monterrey']
# size = [data['ciudad'].value_counts()['Toluca'], data['ciudad'].value_counts()['Guadalajara'],
#         data['ciudad'].value_counts()['Merida'], data['ciudad'].value_counts()['Monterrey']]
# colors = ['grey', 'yellow', 'blue', 'green']
# explode = (0.1, 0, 0, 0)
# plt.pie(size, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
# plt.axis('equal')
# plt.show()
# # Pie diagram para status
# labels = ['Aceptada', 'En Proceso', 'Rechazada']
# size = [data['status_txn'].value_counts()['Aceptada'], data['status_txn'].value_counts()['En proceso'],
#         data['status_txn'].value_counts()['Rechazada']]
# colors = ['grey', 'yellow', 'blue']
# explode = (0.1, 0, 0)
# plt.pie(size, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
# plt.axis('equal')
# plt.show()
# # Pie diagram para tipo
# labels = ['Física', 'Virtual']
# size = [data['tipo_tc'].value_counts()[1], data['tipo_tc'].value_counts()[0]]
# colors = ['grey', 'yellow']
# explode = (0.1, 0)
# plt.pie(size, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
# plt.axis('equal')
# plt.show()
# # Pie diagram para prime
# labels = ['Prime', 'No Prime']
# size = [data['is_prime'].value_counts()[1], data['is_prime'].value_counts()[0]]
# colors = ['grey', 'yellow']
# explode = (0.1, 0)
# plt.pie(size, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
# plt.axis('equal')
# plt.show()
