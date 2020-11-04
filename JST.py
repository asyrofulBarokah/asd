import numpy as np 
import sys
from math import *

class JST:

    def normalisasi(self,data):
        try:
            data_normalisasi= np.round((data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0)), 3)

            return data_normalisasi
        except:
            print('Terjadi kesalahan pada proses Normalisasi Data',sys.exc_info()[0])

    def denormalisasi(self,data,mindata,maksdata):
        try:
            data_denormalisasi = round((data * maksdata - data * mindata) + mindata)

            return data_denormalisasi
        except:
            print('Terjadi kesalahan pada proses Denormalisasi Data',sys.exc_info()[0])

    # fungsi untuk random bobot awal
    def acakBobot(self, n_input, n_hidden, n_output):
        try:
            V = np.random.rand(n_input + 1, n_hidden) # bobot random pada input, bias, dan hidden
            W = np.random.rand(n_hidden + 1, n_output)
            V = np.round((V), 2)
            W = np.round((W), 2)
            # print("bobot V:", bobot_V)
            # print("bobot W:", bobot_W)

            return [V, W]
        except:
            print('terjadi kesalahan pada proses random bobot')

    # fungsi perhitungan nilai neuron pada hidden layer
    def input2Hidden(self, data, n_hidden, V):
        try:
            n_data = data.shape[0] # menetapkan n_data untuk kolom pada matriks
            Z = np.zeros((1, n_hidden)) # membuat matriks untuk Z

            for j in range(n_hidden): # looping sebanyak neuron hidden
                Z_in = 0
                for i in range(n_data): # looping sebanyak baris pada matriks
                    Z_in = Z_in + V[i + 1, j] * data[i] # feedforward bobot dan input

                Z_in = V[0, j] + Z_in # menambahkan bias
                Z[0, j] = round( 1 / ( 1 + exp(-Z_in)), 3) # fungsi aktivasi, round membulatkan dengan 3 dibelakang koma

            return Z
        except:
            print('terjadi kesalahan pada proses perhitungan z (input - hidden)')

    # fungsi melakukan perhitungan nilai neuron hidden layer
    def hidden2Output(self, Z, n_output, W): # Z punya atasnya
        try:
            baris, kolom = Z.shape # menentukan kolom matriks
            Y = np.zeros((1, n_output)) # membuat matriks untuk Y

            for k in range(n_output): # looping sebanyak neuron output
                Y_in = 0
                for j in range(kolom): # looping sebanyak
                    Y_in = Y_in + W[j + 1, k] * Z[k, j] # feedforward

                Y_in = W[0, k] + Y_in # menambahkan bias
                Y[0, k] = round(1 / (1 + exp(-Y_in)), 3) # fungsi aktivasi
            return Y
        except:
            print('terjadi kesalahan pada proses perhitungan y')

    # fungsi perambatan maju
    def perambatanMaju(self, data, V, W, n_hidden, n_output):
        try:
            Z = self.input2Hidden(data, n_hidden, V)
            Y = self.hidden2Output(Z, n_output, W)

            return Z, Y
        except:
            print("terjadi kesalahan pada proses perambatan maju")

    
    # fungsi pembaruan bobot W
    def outputHidden(self, t_output, output, Z, alpha, W):
        try:
            baris, kolom = output.shape
            lossK = np.zeros((baris, kolom)) # membuat matriks nilai error

            for i in range(baris):
                for j in range(kolom):
                    lossK[i, j] = (t_output - output[i, j]) * output[i, j]*(1 - output[i, j]) # mencari nilai error unit output

            baris, kolom = lossK.shape
            baris1, kolom1 = Z.shape
            deltaW = np.zeros((kolom1 + 1, kolom)) # membuat matriks untuk koreksi bobot W

            for i in range(kolom): 
                for j in range(kolom1):
                    deltaW[j + 1, i] = round(alpha * lossK[0, i] * Z[i, j], 3) # koreksi bobot W 

                deltaW[0, i] = round(alpha * lossK[0, i], 3) # koreksi bobot bias

            W_baru = W + deltaW # bobot baru

            return W_baru
        except:
            print("terjadi kesalahan pada proses perambatan mundur")

    
    # fungsi pembaruan bobot V
    def hiddenInput(self, t_output, output, data, alpha, Z, W, V):
        try:
            baris, kolom = output.shape # 
            loss = np.zeros((baris, kolom))

            for i in range(baris):
                for j in range(kolom):
                    loss[i, j] = (t_output - output[i, j]) * output[i, j] * (1 - output[i, j]) # mencari nilai error output

            baris1, kolom1 = W.shape
            baris2, kolom2 = Z.shape
            lossJ = np.zeros((baris2, kolom2))

            for i in range(kolom2):
                lossHidden = 0
                for j in range(kolom):
                    lossHidden = round(loss[0, j] * W[i + 1, j], 3) # mentransfer nilai nilai dari kanan hidden layer (output layer)

                lossJ[0, i] = round(lossHidden * Z[0, i] * (1 - Z[0, i]), 3) #  menghitung nilai eror unit hidden

            baris, kolom = lossJ.shape
            n_data = data.shape[0]
            m, n = V.shape
            deltaV = np.zeros((m, n))

            for j in range(kolom):
                lossHidden = 0
                for i in range(n_data):
                    deltaV[i + 1, j] = round(alpha * lossJ[0, j] * data[i], 3) # koreksi bobot V

                deltaV[0, j] = round(alpha * lossJ[0, j], 3) # koreksi bias V

            Vbaru = V + deltaV # update bobot V

            return Vbaru
        except:
            print('terjadi kesalahan pada proses perambatan mundur V')

    # fungsi melakukan perambatan mundur
    def perambatanMundur(self, t_output, output, data, alpha, Z, W, V):
        try:
            W_baru = self.outputHidden(t_output, output, Z, alpha, W)
            V_baru = self.hiddenInput(t_output, output, data, alpha, Z, W, V)

            return W_baru, V_baru
        except:
            print('terjadi kesalan pada proses perambatan maju W')

