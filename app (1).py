# --- 1. Impor Pustaka ---
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pickle

# --- 2. Konfigurasi Halaman Aplikasi ---
st.set_page_config(
    page_title="Deteksi Penyakit Kulit",
    page_icon="⚕️",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- 3. Fungsi untuk Memuat Model (Dengan Caching) ---
@st.cache_resource
def load_model():
    """
    Memuat model ekstraksi fitur (MobileNetV2) dan model klasifikasi (dari file pickle).
    Menggunakan caching agar model tidak dimuat ulang setiap kali ada interaksi.
    """
    # Memuat model ekstraksi fitur MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(128, 128, 3),
        pooling='avg'
    )
    
    # Path ke file model klasifikasi Anda
    model_path = 'model_kulit_gnb.pkl'
    classifier = None # Inisialisasi classifier sebagai None
    
    try:
        # PENTING: Buka file dalam mode 'read-binary' ('rb') untuk menghindari error
        with open(model_path, 'rb') as file:
            classifier = pickle.load(file)
            
    except FileNotFoundError:
        st.error(
            f"File model '{model_path}' tidak ditemukan. "
            "Pastikan file model berada di direktori yang sama dengan app.py."
        )
    except Exception as e:
        st.error(f"Terjadi error saat memuat model: {e}")
        
    return base_model, classifier

# --- 4. Fungsi untuk Prediksi ---
def predict(image, base_model, classifier):
    """
    Memproses gambar yang diunggah dan mengembalikan prediksi nama penyakit.
    """
    # Pastikan gambar dalam mode RGB dan ubah ukurannya
    img = image.convert('RGB')
    img = img.resize((128, 128))
    
    # Ubah gambar menjadi array dan lakukan pra-pemrosesan
    img_array = np.array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(img_array_expanded)

    # Ekstrak fitur dan lakukan prediksi
    features = base_model.predict(preprocessed_img)
    prediction = classifier.predict(features)
    
    # Ambil nama kelas dari hasil prediksi
    predicted_class_index = int(prediction[0])
    predicted_class_name = CLASS_NAMES[predicted_class_index]

    return predicted_class_name

# --- 5. Tampilan Utama Aplikasi ---

# Judul dan Deskripsi
st.title("⚕️ Deteksi Penyakit Kulit")
st.write(
    "Unggah gambar untuk mendeteksi salah satu dari lima kondisi kulit: "
    "Jerawat, Eksim, Herpes, Panu, atau Rosacea."
)

# Muat model
base_model, classifier = load_model()

# Daftar nama kelas (sesuaikan dengan urutan saat pelatihan)
CLASS_NAMES = ['Jerawat', 'Eksim', 'Herpes', 'Panu', 'Rosacea']

# Komponen untuk unggah file
uploaded_file = st.file_uploader(
    "Pilih sebuah gambar...",
    type=["jpg", "jpeg", "png"]
)

# Logika setelah gambar diunggah
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # PERBAIKAN: Menggunakan use_container_width yang direkomendasikan
    st.image(image, caption='Gambar yang Diunggah', use_container_width=True)

    # Tombol untuk memulai deteksi
    if st.button('Deteksi Penyakit Kulit'):
        # Hanya jalankan prediksi jika model berhasil dimuat
        if base_model is not None and classifier is not None:
            with st.spinner('Sedang menganalisis...'):
                prediction_result = predict(image, base_model, classifier)
            st.success(f"**Hasil Deteksi:** {prediction_result}")
        else:
            st.warning("Prediksi tidak dapat dilakukan karena model gagal dimuat.")

# --- 6. Informasi Tambahan di Sidebar ---
st.sidebar.header("Tentang Aplikasi")
st.sidebar.info(
    "Aplikasi ini menggunakan MobileNetV2 untuk ekstraksi fitur dan "
    "sebuah model klasifikasi untuk memprediksi kondisi kulit berdasarkan gambar."
)
