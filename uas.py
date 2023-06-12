#Librari yang dibutuh kan
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Melakukan Pembacaan Judul (Introduction)
st.title("APK PREDIKSI DATA HUMAN STRESS DETECTION")
st.write("##### NAMA : JHON FILIUS GULTOM")
st.write("##### NIM : 210411100019 ")
st.write("##### KELAS : PENAMBANGAN DATA B ")


# Tampilan Aturan Navbarnya 
penjelasan, preprocessing, modeling, implementasi = st.tabs(["Penjelasan Data", "Preprocessing", "Modeling", "Implementasi"])

#Data Yang digunakan
df = pd.read_csv('https://raw.githubusercontent.com/Jhonfiliusgultom3/Data-Project/main/SaYoPillow_Data%20Uas.csv')


#Penjelasan Data
with penjelasan:
    st.write("###### Data Set Ini merupakan : Human Stress Detection in and through Sleep ")
    st.write("###### Sumber Data Set dari Kaggle : https://www.kaggle.com/datasets/laavanya/human-stress-detection-in-and-through-sleep?select=SaYoPillow.csv ")
    st.write("###### Link Dataset : https://raw.githubusercontent.com/Jhonfiliusgultom3/Data-Project/main/SaYoPillow_Data%20Uas.csv ")
    st.write("""###### Jumlah Data  : 630 Data """)
    st.write("""###### Jumlah Kolom: 9 Kolom """)
    st.write("""###### Penjelasan setiap kolom : """)

    st.write("""1. Snoring Rate (sr) :

   Snoring Rate (SR) adalah salah satu fitur yang digunakan dalam model prediksi pada implementasi Anda. Fitur ini menggambarkan tingkat kebisingan atau suara mendengkur pada subjek.
    """)
    st.write("""2. Respiration Rate (rr) :

   Respiration Rate (RR) adalah fitur yang digunakan dalam model prediksi pada implementasi Anda. Fitur ini menggambarkan tingkat laju pernapasan pada subjek.
    """)
    st.write("""3. Body Temeperature (t) :

    Body Temperature (t) adalah fitur yang digunakan dalam model prediksi pada implementasi Anda. Fitur ini menggambarkan suhu tubuh.
    """)
    st.write("""4. Limb Movement (lm) :

    Limb Movement (lm) adalah fitur yang digunakan dalam model prediksi pada implementasi Anda. Fitur ini menggambarkan gerakan anggota tubuh subjek.
    """)
    st.write("""5. Blood Oxygen (bo) :

    Menjelaskan Tentang berapa oksigen darah pasien
    """)
    st.write("""6. Eye Movement (rem) :

    Eye Movement (rem) adalah singkatan dari Rapid Eye Movement. REM adalah fase tidur yang ditandai dengan gerakan mata yang cepat dan tidak teratur. 
    """)
    st.write("""7. Sleeping Hours (sh) :

    Menjelaskan Tentang Berapa banyak waktu jam tidur yang dipakai
    """)
    st.write("""8. Heart Rate (hr) :

    Heart Rate (hr) adalah ukuran jumlah denyut jantung per menit. Denyut jantung merupakan jumlah kali kontraksi dan relaksasi yang dilakukan oleh jantung dalam satu menit. Heart Rate dapat memberikan informasi mengenai aktivitas jantung seseorang dan kondisi kardiovaskularnya.
    """)
    st.write("""9. Stres Level (lm) :

    Menjelaskan Tentang Level Stres Pasien (Outputnya ) Jika 0 = Normal, Jika 1 = Medium Low (Sedang Rendah), Jika 2 = Medium (Sedang), Jika 3 = Medium High (Sedang Tinggi), Jika 4 Hingh (Tinggi)
    """)


# Preprocessing
with preprocessing:
    st.subheader("Normalisasi Data")
    st.write("Normalisasi adalah proses mengubah data dalam skala yang sama atau rentang yang serupa. Tujuan normalisasi adalah untuk memastikan bahwa semua variabel memiliki pengaruh yang sebanding pada analisis atau model yang akan digunakan Dalam konteks data numerik, normalisasi sering dilakukan dengan mengubah nilai-nilai data ke dalam rentang tertentu, seperti 0 hingga 1 atau -1 hingga 1. Normalisasi dapat dilakukan menggunakan berbagai metode, termasuk Min-Max Scaling, Z-Score Scaling, dan lainnya")
    st.write("Rumus Normalisasi Data:")
    st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    st.markdown(
        """
        Dimana:
        - X = Merupakan suatu data yang akan dinormalisasi atau data asli (Dimana Disini Datanya adalah Data Humans Stress)
        - min = nilai minimum semua data asli (Nilai Paling Kecil)
        - max = nilai maksimum semua data asli (Nilai Paling Besar)
        """
    )
    #data yang tidak digunakan dalam normalisasi yaitu kolom "sl"
    X = df.drop(columns=["sl"])
    y = df["sl"].values
    df_min = X.min()
    df_max = X.max()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    feature_names = X.columns.copy()
    scaled_features = pd.DataFrame(scaled, columns=feature_names)

    st.subheader("Hasil Normalisasi Data dari Data Human Stress")
    st.write(scaled_features)


# Modeling
with modeling:

    training, test, training_label, test_label = train_test_split(scaled_features, y, test_size=0.2, random_state=42)
    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilih Model untuk menghitung akurasi:")
        ann = st.checkbox('ANN')
        naive_bayes = st.checkbox('Naive Bayes')
        knn = st.checkbox('K-Nearest Neighbors')
        dt = st.checkbox('Decision Tree')
        submitted = st.form_submit_button("Submit")

        # ANN
        mlp = MLPClassifier(hidden_layer_sizes=(4,), max_iter=443, random_state=42)
        mlp.fit(training, training_label)
        mlp_pred = mlp.predict(test)
        mlp_accuracy = round(100 * accuracy_score(test_label, mlp_pred))

        # Naive Bayes
        nb = GaussianNB()
        nb.fit(training, training_label)
        nb_pred = nb.predict(test)
        nb_accuracy = round(100 * accuracy_score(test_label, nb_pred))

        # KNN
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(training, training_label)
        knn_pred = knn.predict(test)
        knn_accuracy = round(100 * accuracy_score(test_label, knn_pred))

        # Decision Tree
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(training, training_label)
        dt_pred = dt.predict(test)
        dt_accuracy = round(100 * accuracy_score(test_label, dt_pred))

        if submitted:
            if ann:
                st.write("Model ANN accuracy score: {0:0.2f}".format(mlp_accuracy))
            if naive_bayes:
                st.write("Model Naive Bayes accuracy score: {0:0.2f}".format(nb_accuracy))
            if knn:
                st.write("Model K-Nearest Neighbors accuracy score: {0:0.2f}".format(knn_accuracy))
            if dt:
                st.write("Model Decision Tree accuracy score: {0:0.2f}".format(dt_accuracy))

 

# Implementasi
with implementasi :
 with st.form("jhon_form"):
    st.subheader('IMPLEMENTASI (PREDIKSI DATA)')
    st.write("Masukkan nilai-nilai yang akan diprediksi:")
    sr  = st.number_input('Masukkan Hasil Snoring Rate (sr)')
    rr  = st.number_input('Masukkan Hasil Respiration Rate (rr)')
    t  = st.number_input('Masukkan Hasil Body Temperature (t)')
    lm  = st.number_input('Masukkan Hasil Limb Movement (lm)')
    bo  = st.number_input('Masukkan Hasil Blood Oxygen (bo)')
    rem  = st.number_input('Masukkan Hasil Eye Movement (rem)')
    sh  = st.number_input('Masukkan Hasil Sleeping Hours (sh)')
    hr  = st.number_input('Masukkan Hasil Heart Rate (hr)')


    model = st.selectbox('Pilih model untuk melakukan prediksi:',
                         ('ANN', 'Naive Bayes', 'KNN', 'Decision Tree'))

    prediksi = st.form_submit_button("Submit")
    if prediksi:
        inputs = np.array([sr,rr,t,lm,bo,rem,sh,hr])
        input_norm = (inputs - df_min) / (df_max - df_min)
        input_norm = np.array(input_norm).reshape(1, -1)

        mod = None
        if model == 'ANN':
            mod = mlp
        elif model == 'Naive Bayes':
            mod = nb
        elif model == 'KNN':
            mod = knn
        elif model == 'Decision Tree':
             mod = dt

        if mod is not None:
            input_pred = mod.predict(input_norm)

            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Pemodelan:', model)
            st.write('Pasien Mendapatkan Level Strees :', input_pred)
        else:
            st.write ('Model belum dipilih')