import numpy as numpy
from matplotlib import pyplot as plt
from JST import *
import pandas as pd 
from sklearn.model_selection import KFold
import pickle

file_V = 'dataPickle/dataV.pkl'
file_W = 'dataPickle/dataW.pkl'

fileV = open(file_V, 'rb')
fileW = open(file_W, 'rb')

V = pickle.load(fileV)
W = pickle.load(fileW)

fileV.close
fileW.close

jst = JST();

n_input = 5
n_hidden = 4
n_output = 1
n_datauji = 8

# membaca data
namafile = "E:\\akademik\\python\\Skripsi\\peramalanJST_Bismillah\\datasetPengujian2020.xlsx"
data = pd.read_excel(namafile, sheet_name="Sheet1")
data = data.to_numpy()

data_normalisasi = jst.normalisasi(data)
print("data Normalisasi :", data_normalisasi)
# print(data_normalisasi)

data_uji = data_normalisasi[0:n_datauji, 0:5]
output_sebenarnya = data_normalisasi[0:n_datauji, 5]
hasil_prediksi = np.zeros((n_datauji, 1))

# proses pengujian
for j in range(n_datauji):
    [Z, Y] = jst.perambatanMaju(data_uji[j,:], V, W, n_hidden, n_output)
    hasil_prediksi[j, 0] = Y[0, 0]
    

# denormalisasi hasil prediksi dan data sebenarnya
mindata = min(data[:,5])
maksdata = max(data[:,5])

# membuat matriks
hasil_prediksi_denormalisasi = np.zeros((n_datauji, 1))
hasil_prediksi_normalisasi = np.zeros((n_datauji, 1))
output_sebenarnya_denormalisasi = np.zeros((n_datauji, 1))
output_sebenarnya_normalisasi = np.zeros((n_datauji, 1))

# fungsi memasukkan nilai prediksi pada matriks yang telah dibuat
for i in range(n_datauji):
    hasil_prediksi_denormalisasi[i, 0] = jst.denormalisasi(hasil_prediksi[i, 0], mindata, maksdata)
    hasil_prediksi_normalisasi[i, 0] = hasil_prediksi[i, 0]
    output_sebenarnya_denormalisasi[i, 0] = jst.denormalisasi(output_sebenarnya[i], mindata, maksdata)
    output_sebenarnya_normalisasi[i, 0] = output_sebenarnya[i]

# print('hpd', hasil_prediksi_denormalisasi) # bentuknya list
# print('osd', output_sebenarnya_denormalisasi)

nilaiMSE = []
# menampilkan hasil prediksi
print("Data ke- \t X1 \t X2 \t X3 \t Output \t Output_JST \t Eror \t MSE \t\t RMSE \t\t Output_Den \t Output_JST_Den")
for i in range(n_datauji):
    hasilDenJST = hasil_prediksi_denormalisasi[i, 0]
    hasilNorJST = hasil_prediksi_normalisasi[i, 0]
    dataDenSebenarnya = output_sebenarnya_denormalisasi[i, 0]
    dataNorSebenarnya = output_sebenarnya_normalisasi[i, 0]
    errorHasil = np.round(abs((dataNorSebenarnya - hasilNorJST)), 5)
    errorHasil2 = errorHasil**2
    mse = np.round((errorHasil2 / n_datauji), 5)
    mse2 = np.mean
    nilaiMSE.append(mse)
    rmse = np.round(np.sqrt(mse), 3)
    print((i + 1), "\t\t", data_uji[i, 0], "\t", data_uji[i, 1], "\t", data_uji[i, 2], "\t", dataNorSebenarnya, "\t\t", hasilNorJST, "\t\t", errorHasil, "\t", mse, "\t\t", rmse, "\t\t", dataDenSebenarnya, "\t\t", hasilDenJST)

MSE = np.mean(nilaiMSE)
print("MSE", MSE)


# menampilkan grafik proses pelatihan
y1 = hasil_prediksi_denormalisasi
y2 = output_sebenarnya_denormalisasi
x_tmp = list(range(1, n_datauji+ 1))
x = np.array([x_tmp]).transpose()

plt.figure()
plt.plot(x, y1, 'r', x, y2, 'g')
plt.xlabel('Data uji ke-i, (0 < i <' + str(n_datauji) + ')')
plt.ylabel('Prediksi Curah Hujan')
plt.title('Grafik Perbandingan Hasil Prediksi JST dan Data Sebenarnya')
plt.legend(('Hasil Prediksi JST', 'Data Sebenarnya'), loc = 'upper right')
plt.show()
