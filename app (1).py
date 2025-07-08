import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pickle # Menggunakan pickle sesuai dengan error Anda

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Deteksi Penyakit Kulit",
    page_icon="⚕️", # Menggunakan ikon yang lebih relevan
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Judul dan Deskripsi ---
st.title("⚕️ Deteksi Penyakit Kulit")
st.write(
    "Unggah gambar untuk mendeteksi salah satu dari lima kondisi kulit: "
    "Jerawat, Eksim, Herpes, Panu, atau Rosacea."
)

# --- Memuat Model (BAGIAN YANG DIPERBAIKI) ---
@st.cache_resource
def load_model():
    """Memuat model yang telah dilatih."""
    # Memuat model ekstraksi fitur (MobileNetV2)
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(128, 128, 3),
        pooling='avg'
    )
    
    # Ganti 'modelGNB_kondisi_kulit (2).pkl' dengan nama file model Anda
    model_path = 'model_kondisi_kulit.pkl' 
    
    try:
        # Buka file dalam mode 'read-binary' ('rb')
        with open(model_path, 'rb') as file:
            # BARIS INI YANG DIPERBAIKI:
            # Sebelumnya: classifier = model_kondisi_kulit.pklickle.load(file)
            # Seharusnya:
            classifier = pickle.load(file)
            
    except FileNotFoundError:
        st.error(
            f"File model '{model_path}' tidak ditemukan. "
            "Pastikan Anda telah melatih dan menyimpan model dari notebook Anda, "
            "dan letakkan file tersebut di direktori yang sama dengan aplikasi ini."
        )
        classifier = None
    except Exception as e:
        st.error(f"Terjadi error saat memuat model: {e}")
        classifier = None
        
    return base_model, classifier

base_model, classifier = load_model()

# --- Daftar Kelas ---
class_names = ['Jerawat', 'Eksim', 'Herpes', 'Panu', 'Rosacea']

# --- Fungsi untuk Pra-pemrosesan dan Prediksi ---
def predict(image, base_model, classifier):
    """Fungsi untuk memproses gambar dan melakukan prediksi."""
    img = image.convert('RGB') # Pastikan gambar dalam mode RGB
    img = img.resize((128, 128))
    img_array = np.array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(img_array_expanded)

    features = base_model.predict(preprocessed_img)
    prediction = classifier.predict(features)
    predicted_class_index = int(prediction[0])
    predicted_class_name = class_names[predicted_class_index]

    return predicted_class_name

# --- Komponen Unggah File ---
uploaded_file = st.file_uploader(
    "Pilih gambar...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang Diunggah', use_container_width=True)

    if st.button('Deteksi Penyakit Kulit'):
        if base_model is not None and classifier is not None:
            with st.spinner('Sedang menganalisis...'):
                prediction = predict(image, base_model, classifier)
                st.success(f"**Hasil Deteksi:** {prediction}")
        else:
            st.warning("Model tidak dapat dimuat, prediksi tidak dapat dilakukan.")

# --- Informasi Tambahan ---
st.sidebar.header("Tentang Aplikasi")
st.sidebar.info(
    "Aplikasi ini dibuat berdasarkan notebook Jupyter untuk klasifikasi "
    "penyakit kulit. Aplikasi ini menggunakan MobileNetV2 untuk ekstraksi fitur dan "
    "sebuah model klasifikasi untuk memprediksi kondisi kulit."
)
