import numpy as numpy
from matplotlib import pyplot as plt
from JST import *
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pickle

datanya = []
dataInputPickle = []
dataHiddenPickle = []
dataOutputPickle = []
dataAlphaPickle = []
dataToleransiErrorPickle = []
dataIterasiPickle = []
dataKfold = []
dataMSEPelatihan = []
dataErrorPengujianTerkecil = []
dataError2 = []
DataAwalVW = []
dataVW = []
MSEPelatihan = []
len_train = []
len_test = []

# menciptakan objek dari kelas JST
jst = JST();

# inisialisasi parameter-paramter JST
n_input = 5
n_hidden = 4
n_output = 1
alpha = 0.4
toleransi_error = 0.001
iterasi = 100

dataInputPickle.append(n_input)
dataHiddenPickle.append(n_hidden)
dataOutputPickle.append(n_output)
dataAlphaPickle.append(alpha)
dataToleransiErrorPickle.append(toleransi_error)
dataIterasiPickle.append(iterasi)


# membaca data
namafile = "E:\\akademik\\python\\Skripsi\\peramalanJST_Bismillah\\dataset.xlsx"
data = pd.read_excel(namafile, sheet_name="Sheet1")
data = data.to_numpy()
data_normalisasi = jst.normalisasi(data)
datanya.append(data)

# proses pelatihan menggunakan kfold
print(' PROSES PELATIHAN ')
kf = KFold(n_splits=10, shuffle=False) # split kfold
h = 1

# perulangan kfold
for train_index, test_index in kf.split(data_normalisasi):
	print("fold -", h)
	dataKfold.append(h)
	# datatest.append(test_index)
	print('train index', train_index)
	print('test index', test_index)
	print('test index', len(train_index))
	print('test index', len(test_index))
	print('test index', len(data))

	len_train1 = len(train_index)
	len_test1 = len(test_index)

	len_train.append(len_train1)
	len_test.append(len_test1)

	# print('train data', data[train_index])
	# print('test data', data[test_index])

	n_datalatih = len(data_normalisasi[train_index]) # mengetahui panjang data latih dan data uji
	n_datauji = len(data_normalisasi[test_index])

	data_latih1 = data_normalisasi[train_index] # train index pada kfold dijadikan data latih
	data_uji1 = data_normalisasi[test_index]

	#menentukan data latih dan target output
	data_latih = data_latih1[0:n_datalatih,0:5] # data latih pada baris 1, 2, 3, 4
	# print('data latih', data_latih, 'panjangnya', len(data_latih))
	t_output = data_latih1[0:n_datalatih,5] # data target pada baris ke 5
	# print('t output', t_output, 'panjangnya', len(t_output))

	# inisialisasi bobot V dan bobot W
	[V, W] = jst.acakBobot(n_input, n_hidden, n_output)
	DataAwalVW.append([V, W])

	# print('-----'*100)
	eror = np.zeros((n_datalatih, 1))
	mse = np.zeros((iterasi, 1))
	jumlah_iterasi = 0

	erornya = []
	MSESimpan = []
	# perulangan pada training untuk update bobot
	for i in range(iterasi):
		# print('Iterasi ke-', (i + 1))
		for j in range(n_datalatih):
			[Z, Y] = jst.perambatanMaju(data_latih[j,:], V, W, n_hidden, n_output)
			[W, V] = jst.perambatanMundur(t_output[j], Y, data_latih[j,:], alpha, Z, W, V)

			eror[j, 0] = abs(t_output[j] - Y[0, 0])**2
			erornya.append(eror[j, 0])
			# print(Y[0,0])

		mse[i, 0] = np.round(np.mean(erornya), 5)
		MSESimpan.append(mse[i, 0])
		# print(MSESimpan)
		print("MSE pelatihan", mse[i, 0], "iterasi ke-", (i+1))
		# print("errornya2", mse[i,0])

		if mse[i, 0] <= toleransi_error: # jika toleransi error dibawah parameter perhitungan akan berhenti
			jumlah_iterasi = i + 1
			break

		jumlah_iterasi = i + 1

	# print('MSE Pelatihan:', mse[i, 0], 'pada iterasi:', jumlah_iterasi)
	dataMSE = mse[i, 0]
	dataMSEPelatihan.append(dataMSE)
	h += 1
	dataVW.append([V, W])

	MSEPelatihan.append(MSESimpan)
	# print('MSEPELATIHAN', MSEPelatihan)
	# print(len(MSEPelatihan))
	
	# pengujian dengan test index pada kfold
	print(' PROSES PENGUJIAN ')
	data_uji = data_uji1[0 : n_datauji, 0:5]
	print('data uji', data_uji, 'panjangnya', len(data_uji))
	output_sebenarnya = data_uji1[0 : n_datauji, 5]
	print('data output', output_sebenarnya, 'panjangnya', len(output_sebenarnya))
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
	hasil_prediksi_normal = np.zeros((n_datauji, 1))
	output_sebenarnya_denormalisasi = np.zeros((n_datauji, 1))
	output_sebenarnya_normal = np.zeros((n_datauji, 1))

	for i in range(n_datauji):
		hasil_prediksi_denormalisasi[i, 0] = jst.denormalisasi(hasil_prediksi[i, 0], mindata, maksdata)
		hasil_prediksi_normal[i, 0] = hasil_prediksi[i, 0]
		output_sebenarnya_denormalisasi[i, 0] = jst.denormalisasi(output_sebenarnya[i], mindata, maksdata)
		output_sebenarnya_normal[i, 0] = output_sebenarnya[i]

	mse = mean_squared_error(output_sebenarnya_normal, hasil_prediksi_normal)
	print('mse', mse)
	
	dataError = []
	# dataError2.append(mse)
	for i in range(n_datauji):
		hasilJST2 = hasil_prediksi_normal[i, 0]
		# print('hasil JST:', hasilJST2)
		dataSebenarnya2 = output_sebenarnya_normal[i, 0]
		# print('hasil output:', dataSebenarnya2)

		errorHasil = np.round(abs(dataSebenarnya2 - hasilJST2)**2, 5)

		dataError.append(errorHasil)

	rata3 = np.round(np.mean(dataError), 5)
	print("MSE Pengujian", rata3)
	dataError2.append(rata3)

print("list error pada 10 folds :", dataError2)

print("len train :", len_train)
print("len test :", len_test)

ErrTerkecil = min(dataError2)
print("Error terkecil pada list :", ErrTerkecil)
dataErrorPengujianTerkecil.append(ErrTerkecil)

PosisiErrTerkecil = dataError2.index(min(dataError2))
print("Posisi Error terkecil pada list :", PosisiErrTerkecil)

dataV = dataVW[PosisiErrTerkecil][0]
dataW = dataVW[PosisiErrTerkecil][1]

V = DataAwalVW[PosisiErrTerkecil][0]
W = DataAwalVW[PosisiErrTerkecil][1]
print('V', V)
print('W', W)
print('VW', DataAwalVW)

print('V', dataV)
print('W', dataW)
print('VW', dataVW)
# print("V", dataV)
# print("W", dataW)
# print("V", V)
# print("W", W)
# print("data V W :", dataVW)
# print("data awal V W :", DataAwalVW)

# print('MSE Pelatihan', MSEPelatihan)
MSEPelatihanTerkecil = MSEPelatihan[PosisiErrTerkecil]
# print('MSE Pelatihan Terkecil', MSEPelatihann)

# simpan bobot V awal
pickle.dump(V, open('dataPickle/V.pkl', 'wb'))
V = pickle.load(open('dataPickle/V.pkl', 'rb'))

# simpan bobot W awal
pickle.dump(W, open('dataPickle/W.pkl', 'wb'))
W = pickle.load(open('dataPickle/W.pkl', 'rb'))

# simpan bobot V
pickle.dump(dataV, open('dataPickle/dataV.pkl', 'wb'))
dataV = pickle.load(open('dataPickle/dataV.pkl', 'rb'))

# simpan bobot W
pickle.dump(dataW, open('dataPickle/dataW.pkl', 'wb'))
dataW = pickle.load(open('dataPickle/dataW.pkl', 'rb'))

# simpan MSE Pelatihan
pickle.dump(PosisiErrTerkecil, open('dataPickle/PosisiErrTerkecil.pkl', 'wb'))
PosisiErrTerkecil = pickle.load(open('dataPickle/PosisiErrTerkecil.pkl', 'rb'))

# simpan MSE Pelatihan
pickle.dump(MSEPelatihanTerkecil, open('dataPickle/MSEPelatihanTerkecil.pkl', 'wb'))
MSEPelatihanTerkecil = pickle.load(open('dataPickle/MSEPelatihanTerkecil.pkl', 'rb'))

# simpan len train
pickle.dump(len_train, open('dataPickle/len_train.pkl', 'wb'))
len_train = pickle.load(open('dataPickle/len_train.pkl', 'rb'))

# simpan len test
pickle.dump(len_test, open('dataPickle/len_test.pkl', 'wb'))
len_test = pickle.load(open('dataPickle/len_test.pkl', 'rb'))

# simpan n_input
pickle.dump(dataInputPickle, open('dataPickle/dataInputPickle.pkl', 'wb'))
dataInputPickle = pickle.load(open('dataPickle/dataInputPickle.pkl', 'rb'))

# simpan n_hidden
pickle.dump(dataHiddenPickle, open('dataPickle/dataHiddenPickle.pkl', 'wb'))
dataHiddenPickle = pickle.load(open('dataPickle/dataHiddenPickle.pkl', 'rb'))

# simpan n_output
pickle.dump(dataOutputPickle, open('dataPickle/dataOutputPickle.pkl', 'wb'))
dataOutputPickle = pickle.load(open('dataPickle/dataOutputPickle.pkl', 'rb'))

# simpan learning rate
pickle.dump(dataAlphaPickle, open('dataPickle/dataAlphaPickle.pkl', 'wb'))
dataAlphaPickle = pickle.load(open('dataPickle/dataAlphaPickle.pkl', 'rb'))

# simpan toleransi error
pickle.dump(dataToleransiErrorPickle, open('dataPickle/dataToleransiErrorPickle.pkl', 'wb'))
dataToleransiErrorPickle = pickle.load(open('dataPickle/dataToleransiErrorPickle.pkl', 'rb'))

# simpan iterasi
pickle.dump(dataIterasiPickle, open('dataPickle/dataIterasiPickle.pkl', 'wb'))
dataIterasiPickle = pickle.load(open('dataPickle/dataIterasiPickle.pkl', 'rb'))

# simpan iterasi kfold
pickle.dump(dataKfold, open('dataPickle/dataKfold.pkl', 'wb'))
dataKfold = pickle.load(open('dataPickle/dataKfold.pkl', 'rb'))

# simpan MSE pelatihan
pickle.dump(dataMSEPelatihan, open('dataPickle/dataMSEPelatihan.pkl', 'wb'))
dataMSEPelatihan = pickle.load(open('dataPickle/dataMSEPelatihan.pkl', 'rb'))

# simpan Error Pengujian
pickle.dump(dataError2, open('dataPickle/dataError2.pkl', 'wb'))
dataError2 = pickle.load(open('dataPickle/dataError2.pkl', 'rb'))

# simpan Error Pengujian Terkecil
pickle.dump(dataErrorPengujianTerkecil, open('dataPickle/dataErrorPengujianTerkecil.pkl', 'wb'))
dataErrorPengujianTerkecil = pickle.load(open('dataPickle/dataErrorPengujianTerkecil.pkl', 'rb'))

# simpan Error Pengujian Terkecil
pickle.dump(datanya, open('dataPickle/datanya.pkl', 'wb'))
datanya = pickle.load(open('dataPickle/datanya.pkl', 'rb'))
