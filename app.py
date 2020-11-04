import numpy as np 
import pickle
from JST import *
import pandas as pd 
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from flask import Flask, render_template, request

app = Flask(__name__, static_url_path='',  static_folder='templates')

@app.route("/")
def index():
    # import data dari file pickle
    fileInput = "dataPickle/dataInputPickle.pkl"
    fileHidden = "dataPickle/dataHiddenPickle.pkl"
    fileOutput = "dataPickle/dataOutputPickle.pkl"
    fileAlpha = "dataPickle/dataAlphaPickle.pkl"
    fileTE = "dataPickle/dataToleransiErrorPickle.pkl"
    fileIterasi = "dataPickle/dataIterasiPickle.pkl"
    fileKfold = "dataPickle/dataKfold.pkl"
    fileMSEPelatihan = "dataPickle/dataMSEPelatihan.pkl"
    fileErrorPengujian = "dataPickle/dataError2.pkl"
    fileErrorTerkecil = "dataPickle/dataErrorPengujianTerkecil.pkl"
    fileV = 'dataPickle/dataV.pkl'
    fileW = 'dataPickle/dataW.pkl'
    fileAwalV = 'dataPickle/V.pkl'
    fileAwalW = 'dataPickle/W.pkl'
    fileMSE = 'dataPickle/MSEPelatihanTerkecil.pkl'
    filePosisiTerkecil = 'dataPickle/PosisiErrTerkecil.pkl'
    fileLenTrain = 'dataPickle/len_train.pkl'
    fileLenTest = 'dataPickle/len_test.pkl'

    fileInput = open(fileInput, 'rb')
    fileHidden = open(fileHidden, 'rb')
    fileOutput = open(fileOutput, 'rb')
    fileAlpha = open(fileAlpha, 'rb')
    fileTE = open(fileTE, 'rb')
    fileIterasi = open(fileIterasi, 'rb')
    fileKfold = open(fileKfold, 'rb')
    fileMSEPelatihan = open(fileMSEPelatihan, 'rb')
    fileErrorPengujian = open(fileErrorPengujian, 'rb')
    fileErrorTerkecil = open(fileErrorTerkecil, 'rb')
    fileV = open(fileV, 'rb')
    fileW = open(fileW, 'rb')
    fileAwalV = open(fileAwalV, 'rb')
    fileAwalW = open(fileAwalW, 'rb')
    fileMSE = open(fileMSE, 'rb')
    filePosisiTerkecil = open(filePosisiTerkecil, 'rb')
    fileLenTrain = open(fileLenTrain, 'rb')
    fileLenTest = open(fileLenTest, 'rb')

    dataInputPickle = pickle.load(fileInput)
    dataHiddenPickle = pickle.load(fileHidden)
    dataOutputPickle = pickle.load(fileOutput)
    dataAlphaPickle = pickle.load(fileAlpha)
    dataToleransiErrorPickle = pickle.load(fileTE)
    dataIterasiPickle = pickle.load(fileIterasi)
    dataKfoldPickle = pickle.load(fileKfold)
    dataMSEPelatihan = pickle.load(fileMSEPelatihan)
    dataErrorPengujian = pickle.load(fileErrorPengujian)
    dataErrorPengujianTerkecil = pickle.load(fileErrorTerkecil)
    dataV = pickle.load(fileV)
    dataW = pickle.load(fileW)
    dataAwalV = pickle.load(fileAwalV)
    dataAwalW = pickle.load(fileAwalW)
    MSEPelatihan = pickle.load(fileMSE)
    PosisiTerkecil = pickle.load(filePosisiTerkecil)
    len_train = pickle.load(fileLenTrain)
    len_test = pickle.load(fileLenTest)

    lenMSEPelatihan = len(MSEPelatihan)

    fileInput.close
    fileHidden.close
    fileOutput.close
    fileAlpha.close
    fileTE.close
    fileIterasi.close
    fileKfold.close
    fileMSEPelatihan.close
    fileErrorPengujian.close
    fileErrorTerkecil.close
    fileV.close
    fileW.close
    fileAwalV.close
    fileAwalW.close
    fileMSE.close
    filePosisiTerkecil.close
    fileLenTrain.close
    fileLenTest.close

    banyakDataV = len(dataV)
    banyakDataW = len(dataW)
    banyakDataAwalV = len(dataAwalV)
    banyakDataAwalW = len(dataAwalW)

    banyakIterasi = 0
    dataIterasinya = []
    for i in range(lenMSEPelatihan):
        banyakIterasi = i + 1
        dataIterasinya.append(banyakIterasi)
    
    namaV = [['v11', 'v12', 'v13', 'v14'], 
            ['v21', 'v22', 'v23', 'v24'],
            ['v31', 'v32', 'v33', 'v34'],
            ['v41', 'v42', 'v43', 'v44'],
            ['v51', 'v52', 'v53', 'v54'],
            ['v01', 'v02', 'v03', 'v04']]

    namaW = [['w1'], ['w2'], ['w3'], ['w4'], ['w0']]

    return render_template("home.html", 
                        dataInputPickle = dataInputPickle,
                        dataHiddenPickle = dataHiddenPickle,
                        dataOutputPickle = dataOutputPickle,
                        dataAlphaPickle = dataAlphaPickle,
                        dataToleransiErrorPickle = dataToleransiErrorPickle,
                        dataIterasiPickle = dataIterasiPickle,
                        dataKfoldPickle = dataKfoldPickle,
                        dataErrorPengujian = dataErrorPengujian,
                        dataErrorPengujianTerkecil = dataErrorPengujianTerkecil,
                        dataV = dataV,
                        dataW = dataW,
                        dataAwalV = dataAwalV,
                        dataAwalW = dataAwalW,
                        banyakDataV = banyakDataV,
                        banyakDataW = banyakDataW,
                        banyakDataAwalV = banyakDataAwalV,
                        banyakDataAwalW = banyakDataAwalW,
                        MSEPelatihan = MSEPelatihan,
                        dataIterasinya = dataIterasinya,
                        namaV = namaV,
                        namaW = namaW,
                        PosisiTerkecil = PosisiTerkecil,
                        len_test = len_test,
                        len_train = len_train,
                        len = len(dataKfoldPickle))

@app.route("/train", methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
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
        n_input = request.form['n_input']
        n_input = int(n_input)
        n_hidden = request.form['n_hidden']
        n_hidden = int(n_hidden)
        n_output = request.form['n_output']
        n_output = int(n_output)
        alpha = request.form['alpha']
        alpha = float(alpha)
        toleransi_error = request.form['toleransi_error']
        toleransi_error = float(toleransi_error)
        iterasi = request.form['iterasi']
        iterasi = int(iterasi)

        dataInputPickle.append(n_input)
        dataHiddenPickle.append(n_hidden)
        dataOutputPickle.append(n_output)
        dataAlphaPickle.append(alpha)
        dataToleransiErrorPickle.append(toleransi_error)
        dataIterasiPickle.append(iterasi)

        # membaca data
        f = request.files['file']
        data_xls = pd.read_excel(f)
        data = data_xls.to_numpy()
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
            # print('train data', data[train_index])
            # print('test data', data[test_index])

            len_train1 = len(train_index)
            len_test1 = len(test_index)

            len_train.append(len_train1)
            len_test.append(len_test1)

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
            
            # pengujian dengan test index pada kfold
            print(' PROSES PENGUJIAN ')
            data_uji = data_uji1[0 : n_datauji, 0:5]
            # print('data uji', data_uji, 'panjangnya', len(data_uji))
            output_sebenarnya = data_uji1[0 : n_datauji, 5]
            # print('data output', output_sebenarnya, 'panjangnya', len(output_sebenarnya))
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
            # print('mse', mse)
            
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
            # print("MSE Pengujian", rata3)
            dataError2.append(rata3)

        # print("list error pada 10 folds :", dataError2)

        ErrTerkecil = min(dataError2)
        # print("Error terkecil pada list :", ErrTerkecil)
        dataErrorPengujianTerkecil.append(ErrTerkecil)

        PosisiErrTerkecil = dataError2.index(min(dataError2))
        # print("Posisi Error terkecil pada list :", PosisiErrTerkecil)

        dataV = dataVW[PosisiErrTerkecil][0]
        dataW = dataVW[PosisiErrTerkecil][1]
        
        V = DataAwalVW[PosisiErrTerkecil][0]
        W = DataAwalVW[PosisiErrTerkecil][1]

        # print("V", dataV)
        # print("W", dataW)
        # print("V", V)
        # print("W", W)
        # print("data V W :", dataVW)
        # print("data awal V W :", DataAwalVW)

        # print('MSE Pelatihan', MSEPelatihan)
        MSEPelatihanTerkecil = MSEPelatihan[PosisiErrTerkecil]
        # print('MSE Pelatihan Terkecil', MSEPelatihann)
        # print('MSE Pelatihan ', MSEPelatihann)

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

        # import data dari file pickle
        fileInput = "dataPickle/dataInputPickle.pkl"
        fileHidden = "dataPickle/dataHiddenPickle.pkl"
        fileOutput = "dataPickle/dataOutputPickle.pkl"
        fileAlpha = "dataPickle/dataAlphaPickle.pkl"
        fileTE = "dataPickle/dataToleransiErrorPickle.pkl"
        fileIterasi = "dataPickle/dataIterasiPickle.pkl"
        fileKfold = "dataPickle/dataKfold.pkl"
        fileMSEPelatihan = "dataPickle/dataMSEPelatihan.pkl"
        fileErrorPengujian = "dataPickle/dataError2.pkl"
        fileErrorTerkecil = "dataPickle/dataErrorPengujianTerkecil.pkl"
        fileV = 'dataPickle/dataV.pkl'
        fileW = 'dataPickle/dataW.pkl'
        fileAwalV = 'dataPickle/V.pkl'
        fileAwalW = 'dataPickle/W.pkl'
        fileMSE = 'dataPickle/MSEPelatihanTerkecil.pkl'
        filePosisiTerkecil = 'dataPickle/PosisiErrTerkecil.pkl'
        fileLenTrain = 'dataPickle/len_train.pkl'
        fileLenTest = 'dataPickle/len_test.pkl'

        fileInput = open(fileInput, 'rb')
        fileHidden = open(fileHidden, 'rb')
        fileOutput = open(fileOutput, 'rb')
        fileAlpha = open(fileAlpha, 'rb')
        fileTE = open(fileTE, 'rb')
        fileIterasi = open(fileIterasi, 'rb')
        fileKfold = open(fileKfold, 'rb')
        fileMSEPelatihan = open(fileMSEPelatihan, 'rb')
        fileErrorPengujian = open(fileErrorPengujian, 'rb')
        fileErrorTerkecil = open(fileErrorTerkecil, 'rb')
        fileV = open(fileV, 'rb')
        fileW = open(fileW, 'rb')
        fileAwalV = open(fileAwalV, 'rb')
        fileAwalW = open(fileAwalW, 'rb')
        fileMSE = open(fileMSE, 'rb')
        filePosisiTerkecil = open(filePosisiTerkecil, 'rb')
        fileLenTrain = open(fileLenTrain, 'rb')
        fileLenTest = open(fileLenTest, 'rb')

        dataInputPickle = pickle.load(fileInput)
        dataHiddenPickle = pickle.load(fileHidden)
        dataOutputPickle = pickle.load(fileOutput)
        dataAlphaPickle = pickle.load(fileAlpha)
        dataToleransiErrorPickle = pickle.load(fileTE)
        dataIterasiPickle = pickle.load(fileIterasi)
        dataKfoldPickle = pickle.load(fileKfold)
        dataMSEPelatihan = pickle.load(fileMSEPelatihan)
        dataErrorPengujian = pickle.load(fileErrorPengujian)
        dataErrorPengujianTerkecil = pickle.load(fileErrorTerkecil)
        dataV = pickle.load(fileV)
        dataW = pickle.load(fileW)
        dataAwalV = pickle.load(fileAwalV)
        dataAwalW = pickle.load(fileAwalW)
        MSEPelatihan = pickle.load(fileMSE)
        PosisiTerkecil = pickle.load(filePosisiTerkecil)
        len_train = pickle.load(fileLenTrain)
        len_test = pickle.load(fileLenTest)

        lenMSEPelatihan = len(MSEPelatihan)

        fileInput.close
        fileHidden.close
        fileOutput.close
        fileAlpha.close
        fileTE.close
        fileIterasi.close
        fileKfold.close
        fileMSEPelatihan.close
        fileErrorPengujian.close
        fileErrorTerkecil.close
        fileV.close
        fileW.close
        fileAwalV.close
        fileAwalW.close
        fileMSE.close
        filePosisiTerkecil.close
        fileLenTrain.close
        fileLenTest.close

        banyakDataV = len(dataV)
        banyakDataW = len(dataW)
        banyakDataAwalV = len(dataAwalV)
        banyakDataAwalW = len(dataAwalW)

        banyakIterasi = 0
        dataIterasinya = []
        for i in range(lenMSEPelatihan):
            banyakIterasi = i + 1
            dataIterasinya.append(banyakIterasi)

        namaV = [['v11', 'v12', 'v13', 'v14'], 
            ['v21', 'v22', 'v23', 'v24'],
            ['v31', 'v32', 'v33', 'v34'],
            ['v41', 'v42', 'v43', 'v44'],
            ['v51', 'v52', 'v53', 'v54'],
            ['v01', 'v02', 'v03', 'v04']]

        namaW = [['w1'], ['w2'], ['w3'], ['w4'], ['w0']]

        return render_template('home.html', 
                        dataInputPickle = dataInputPickle,
                        dataHiddenPickle = dataHiddenPickle,
                        dataOutputPickle = dataOutputPickle,
                        dataAlphaPickle = dataAlphaPickle,
                        dataToleransiErrorPickle = dataToleransiErrorPickle,
                        dataIterasiPickle = dataIterasiPickle,
                        dataKfoldPickle = dataKfoldPickle,
                        dataMSEPelatihan = dataMSEPelatihan,
                        dataErrorPengujian = dataErrorPengujian,
                        dataErrorPengujianTerkecil = dataErrorPengujianTerkecil,
                        dataV = dataV,
                        dataW = dataW,
                        dataAwalV = dataAwalV,
                        dataAwalW = dataAwalW,
                        banyakDataV = banyakDataV,
                        banyakDataW = banyakDataW,
                        banyakDataAwalV = banyakDataAwalV,
                        banyakDataAwalW = banyakDataAwalW,
                        MSEPelatihan = MSEPelatihan,
                        dataIterasinya = dataIterasinya,
                        namaV = namaV,
                        namaW = namaW,
                        PosisiTerkecil = PosisiTerkecil,
                        len_test = len_test,
                        len_train = len_train,
                        len = len(dataKfoldPickle))
    return render_template('train.html')

@app.route("/test", methods=['GET', 'POST'])
def test():
    if request.method == 'POST':
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

        r1 = request.form['r1']
        r1 = float(r1)
        r2 = request.form['r2']
        r2 = float(r2)
        r3 = request.form['r3']
        r3 = float(r3)
        r4 = request.form['r4']
        r4 = float(r4)
        r5 = request.form['r5']
        r5 = float(r5)
        r6 = request.form['r6']
        r6 = float(r6)
        r7 = request.form['r7']
        r7 = float(r7)
        r8 = request.form['r8']
        r8 = float(r8)

        ra = (r1, r2, r3, r4, r5, r6, r7, r8)

        rataHujan = (ra)
        rataHujan = np.vstack(rataHujan)

        a1 = request.form['xt_1_1']
        xt_1_1 = float(a1)
        a2 = request.form['xt_11_1']
        xt_11_1 = float(a2)
        a3 = request.form['xt_12_1']
        xt_12_1 = float(a3)
        a4 = request.form['xt_13_1']
        xt_13_1 = float(a4)
        a5 = request.form['xt_24_1']
        xt_24_1 = float(a5)
        a6 = request.form['y_1']
        y_1 = float(a6)

        b1 = request.form['xt_1_2']
        xt_1_2 = float(b1)
        b2 = request.form['xt_11_2']
        xt_11_2 = float(b2)
        b3 = request.form['xt_12_2']
        xt_12_2 = float(b3)
        b4 = request.form['xt_13_2']
        xt_13_2 = float(b4)
        b5 = request.form['xt_24_2']
        xt_24_2 = float(b5)
        b6 = request.form['y_2']
        y_2 = float(b6)

        c1 = request.form['xt_1_3']
        xt_1_3 = float(c1)
        c2 = request.form['xt_11_3']
        xt_11_3 = float(c2)
        c3 = request.form['xt_12_3']
        xt_12_3 = float(c3)
        c4 = request.form['xt_13_3']
        xt_13_3 = float(c4)
        c5 = request.form['xt_24_3']
        xt_24_3 = float(c5)
        c6 = request.form['y_3']
        y_3 = float(c6)

        c1 = request.form['xt_1_4']
        xt_1_4 = float(c1)
        c2 = request.form['xt_11_4']
        xt_11_4 = float(c2)
        c3 = request.form['xt_12_4']
        xt_12_4 = float(c3)
        c4 = request.form['xt_13_4']
        xt_13_4 = float(c4)
        c5 = request.form['xt_24_4']
        xt_24_4 = float(c5)
        c6 = request.form['y_4']
        y_4 = float(c6)

        c1 = request.form['xt_1_5']
        xt_1_5 = float(c1)
        c2 = request.form['xt_11_5']
        xt_11_5 = float(c2)
        c3 = request.form['xt_12_5']
        xt_12_5 = float(c3)
        c4 = request.form['xt_13_5']
        xt_13_5 = float(c4)
        c5 = request.form['xt_24_5']
        xt_24_5 = float(c5)
        c6 = request.form['y_5']
        y_5 = float(c6)

        c1 = request.form['xt_1_6']
        xt_1_6 = float(c1)
        c2 = request.form['xt_11_6']
        xt_11_6 = float(c2)
        c3 = request.form['xt_12_6']
        xt_12_6 = float(c3)
        c4 = request.form['xt_13_6']
        xt_13_6 = float(c4)
        c5 = request.form['xt_24_6']
        xt_24_6 = float(c5)
        c6 = request.form['y_6']
        y_6 = float(c6)

        c1 = request.form['xt_1_7']
        xt_1_7 = float(c1)
        c2 = request.form['xt_11_7']
        xt_11_7 = float(c2)
        c3 = request.form['xt_12_7']
        xt_12_7 = float(c3)
        c4 = request.form['xt_13_7']
        xt_13_7 = float(c4)
        c5 = request.form['xt_24_7']
        xt_24_7 = float(c5)
        c6 = request.form['y_7']
        y_7 = float(c6)

        c1 = request.form['xt_1_8']
        xt_1_8 = float(c1)
        c2 = request.form['xt_11_8']
        xt_11_8 = float(c2)
        c3 = request.form['xt_12_8']
        xt_12_8 = float(c3)
        c4 = request.form['xt_13_8']
        xt_13_8 = float(c4)
        c5 = request.form['xt_24_8']
        xt_24_8 = float(c5)
        c6 = request.form['y_8']
        y_8 = float(c6)

        b1 = (xt_1_1, xt_11_1, xt_12_1, xt_13_1, xt_24_1, y_1)
        b2 = (xt_1_2, xt_11_2, xt_12_2, xt_13_2, xt_24_2, y_2)
        b3 = (xt_1_3, xt_11_3, xt_12_3, xt_13_3, xt_24_3, y_3)
        b4 = (xt_1_4, xt_11_4, xt_12_4, xt_13_4, xt_24_4, y_4)
        b5 = (xt_1_5, xt_11_5, xt_12_5, xt_13_5, xt_24_5, y_5)
        b6 = (xt_1_6, xt_11_6, xt_12_6, xt_13_6, xt_24_6, y_6)
        b7 = (xt_1_7, xt_11_7, xt_12_7, xt_13_7, xt_24_7, y_7)
        b8 = (xt_1_8, xt_11_8, xt_12_8, xt_13_8, xt_24_8, y_8)

        datasetnya = (b1, b2, b3, b4, b5, b6, b7, b8)
        data = np.vstack(datasetnya)

        data_normalisasi = jst.normalisasi(data)
        print(data_normalisasi)

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

        y1 = hasil_prediksi_denormalisasi
        y2 = output_sebenarnya_denormalisasi 
        y3 = hasil_prediksi_normalisasi        
        y4 = output_sebenarnya_normalisasi

        sifatHujanOutput = np.round((output_sebenarnya_denormalisasi * 100 / rataHujan), 1)
        sifatHujanPrediksi = np.round((hasil_prediksi_denormalisasi * 100 / rataHujan), 1)

        errorPerBulan = []
        errorPerBulan2= []
        for i in range(n_datauji):
            hasilNorJST = hasil_prediksi_normalisasi[i, 0]
            dataNorSebenarnya = output_sebenarnya_normalisasi[i, 0]
            errorHasil = np.round(abs(dataNorSebenarnya - hasilNorJST)**2, 5)
            errorPerBulan.append(errorHasil)
            
            nilaiEror = abs(dataNorSebenarnya - hasilNorJST)
            persentaseEror = ((nilaiEror*100)/max(dataNorSebenarnya, hasilNorJST))
            persentaseAkurasi = 100 - persentaseEror
            errorPerBulan2.append(persentaseAkurasi)
            
        mse = np.round(np.mean(errorPerBulan), 5)
        rmse = sqrt(mse)
        rmse = np.round((rmse), 5)
        nilaiMSE = mse
        akurasi = round(np.mean(errorPerBulan2), 2)

        sifatHujanO = []
        sifatHujanP = []
        konv = []
        for i in range(n_datauji):     
            sft1 = sifatHujanOutput[i]
            sft2 = sifatHujanPrediksi[i]

            if sft1 >= 115:
                sifatHujannya1 = "AN"
            elif sft1 <= 115 and sft1 >= 85:
                sifatHujannya1 = "N"
            else:
                sifatHujannya1 = "BN"
            sifatHujanO.append(sifatHujannya1)

            if sft2 >= 115:
                sifatHujannya2 = "AN"
            elif sft2 <= 115 and sft2 >= 85:
                sifatHujannya2 = "N"
            else:
                sifatHujannya2 = "BN"
            sifatHujanP.append(sifatHujannya2)

            if sifatHujannya1 == sifatHujannya2:
                sifat = '1'
            elif sifatHujannya1 != sifatHujannya2:
                sifat = '0'
            konv.append(sifat)

        sifatHujanO = sifatHujanO
        sifatHujanP = sifatHujanP
        konv = konv
        banyakKonv = konv.count('1')
        nilaiKonv = ((banyakKonv*100)/8)

        Bulan = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus']
        return render_template("test.html", y1=y1, y2=y2, y3=y3, y4=y4, Bulan=Bulan, Banyak=len(Bulan), max=500, title='grafik', 
                                rataHujan=rataHujan,  sifatHujanOutput=sifatHujanOutput, sifatHujanPrediksi=sifatHujanPrediksi, 
                                sifatHujanO=sifatHujanO, sifatHujanP=sifatHujanP, konv=konv, banyakKonv=banyakKonv,
                                nilaiKonv=nilaiKonv, nilaiMSE=nilaiMSE, errorPerBulan=errorPerBulan, rmse=rmse, akurasi=akurasi)
    return render_template("test.html")
    

if __name__ == "__main__":
    app.run(debug=True)