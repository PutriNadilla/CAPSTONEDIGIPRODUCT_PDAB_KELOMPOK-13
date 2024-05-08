import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier


data = pd.read_csv("cwurData.csv")
# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("cwurData.csv")  
    return data

# Main function
def main():
    st.sidebar.title("Menu")
    menu_options = ["Dashboard", "Klasifikasi"]
    choice = st.sidebar.selectbox("Pilih Halaman", menu_options)

    if choice == "Dashboard":
        show_dashboard()
    elif choice == "Klasifikasi":
        show_classification()

def show_dashboard():
    st.title("Dashboard")

    # Load data
    data = load_data()

    # Visualisasi jumlah institusi di setiap negara menggunakan bar plot Matplotlib
    st.subheader("Jumlah Institusi di Setiap Negara")
    country_counts = data['country'].value_counts()
    plt.figure(figsize=(10, 6))
    plt.bar(country_counts.index, country_counts.values)
    plt.xticks(rotation=90)
    plt.xlabel('Negara')
    plt.ylabel('Jumlah Institusi')
    plt.title('Jumlah Institusi di Setiap Negara')
    st.pyplot(plt)
    st.write("""
    Seperti yang terlihat pada bar plot, dataset ini didominasi oleh institusi-institusi yang berasal dari negara-negara maju seperti Amerika Serikat, Tiongkok, Jepang, Inggris Raya, Jerman, dan Perancis. 
    Hal ini dapat dilihat dari jumlah institusi yang signifikan dari negara-negara tersebut dalam dataset.
    """)

    # Visualisasi persebaran institusi
    st.subheader("Persebaran Asal Institusi")
    fig = px.scatter_geo(data, 
                            locations="country", 
                            locationmode="country names", 
                            color="country", 
                            hover_name="institution",
                            size_max=100, 
                            projection="natural earth",
                            title="Persebaran Asal Institusi")
    st.plotly_chart(fig)
    st.write("""
    Dari persebaran data dapat dilihat bahwa dataset yang digunakan sudah mencakup seluruh institusi perguruan tinggi yang ada di seluruh dunia, hanya saja kebanyakan didominasi oleh negara-negara maju.
    """)

    # Tampilkan tabel 50 universitas teratas
    top_50_data = data.head(50)
    st.subheader("50 Universitas Teratas Berdasarkan Peringkat Dunia")
    st.write(top_50_data)

    # Buat grafik berdasarkan negara asal
    st.subheader("Grafik Negara Asal 50 Universitas Teratas")
    fig = px.histogram(top_50_data, x="country", title="Negara Asal 50 Universitas Teratas")
    fig.update_layout(xaxis_title="Negara", yaxis_title="Jumlah Universitas", xaxis_tickangle=-45)
    st.plotly_chart(fig)
    st.write("""
    Bahkan pada 50 besar institusi terbaik tetap didominasi oleh Amerika Serikat karena memang tidak dipungkiri perguruan tinggi di Amerika Serikat memang perguruan tinggi yang berkualitas dan terbaik di dunia. Sisanya diikuti oleh Inggris Raya, Jepang, Swiss, Israel, Canada, dan Perancis yang merupakan negara-negara maju.
    """)

    # Tampilkan distribusi peringkat nasional
    st.subheader("Distribusi Peringkat Nasional")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='national_rank', bins=20, kde=True)
    plt.xlabel('Peringkat Nasional')
    plt.ylabel('Frekuensi')
    plt.title('Distribusi Peringkat Nasional')
    st.pyplot(plt)
    st.write("""
    Dari persebaran peringkat nasional bisa dilihat bahwa dataset sudah berisi institusi-institusi perguruan tinggi yang memang sudah menjadi institusi unggulan di negaranya masing-masing, dimana dari grafik terlihat bahwa 900 kampus lebih memiliki peringkat nasional yang tinggi dan memang unggulan di negaranya.
    """)


def show_classification():
    # Load data
    df = pd.read_csv("university.csv")

    # Menghapus kolom yang tidak diperlukan
    columns_to_drop = ['world_rank', 'institution', 'country', 'year', 'national_rank', 'AkreditasiRank']
    features = df.drop(columns=columns_to_drop)

    # Encoding kolom target 'AkreditasiRank'
    label_encoder = LabelEncoder()
    df['AkreditasiRank_encoded'] = label_encoder.fit_transform(df['AkreditasiRank'])

    # Normalisasi data
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    # Melatih model KNN dengan parameter yang ditentukan
    knn_classifier = KNeighborsClassifier(n_neighbors=2)
    knn_classifier.fit(scaled_features, df['AkreditasiRank_encoded'])

    # Menampilkan judul dan deskripsi aplikasi
    st.title("Klasifikasi Akreditasi Perguruan Tinggi")
    st.write("Aplikasi ini melakukan klasifikasi akreditasi perguruan tinggi berdasarkan fitur-fitur yang diberikan.")

    # Menampilkan input dari pengguna
    st.write("Silakan masukkan nilai fitur untuk melakukan prediksi akreditasi:")
    input_features = {}
    for feature in features.columns:
        input_features[feature] = st.number_input(feature)

    # Prediksi akreditasi
    input_data = pd.DataFrame([input_features])
    scaled_input = scaler.transform(input_data)
    prediction = knn_classifier.predict(scaled_input)

    # Mengembalikan label asli dari hasil prediksi
    decoded_prediction = label_encoder.inverse_transform(prediction)

    # Menampilkan hasil prediksi
    st.write("Hasil Prediksi Akreditasi:")
    st.write(decoded_prediction[0])

if __name__ == '__main__':
    main()
