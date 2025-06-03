import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import gdown
import os
import matplotlib.pyplot as plt
import random
    
# Konfigurasi
st.set_page_config(page_title="Klasifikasi Daun Tomat", page_icon="🍅", layout="wide")

# --- Unduh model ---
model_path = "tomato_leaf_model.h5"
gdrive_id = "1BavaDWnAalzugPrMtfNcc7NL_UWLWMA2"

if not os.path.exists(model_path):
    with st.spinner("Mengunduh model dari Google Drive..."):
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        gdown.download(url, model_path, quiet=False)

# Load model
model = tf.keras.models.load_model(model_path)

# Daftar label kelas
class_labels = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Tabs UI
tab1, tab2, tab3 = st.tabs(["📷 Prediksi Gambar", "📊 Penjelasan", "🧪 Contoh Prediksi"])

with tab1:
    st.title("🍅 Klasifikasi Penyakit Daun Tomat")
    uploaded_file = st.file_uploader("Upload gambar daun tomat...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Gambar yang diupload", use_column_width=True)

        img = img.resize((224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)
        pred_class = np.argmax(pred)
        confidence = np.max(pred) * 100

        st.markdown(f"### ✅ Prediksi: `{class_labels[pred_class]}`")
        st.markdown(f"**Tingkat Keyakinan:** {confidence:.2f}%")

with tab2:
    st.header("📑 Penjelasan Sistem")
    
    st.subheader("📂 Dataset")
    st.markdown("""
    Dataset yang digunakan merupakan kumpulan citra daun tomat yang terdiri dari 10 kelas, 
    baik daun sehat maupun yang terinfeksi penyakit seperti *Bacterial Spot*, *Early Blight*, 
    *Leaf Mold*, *Tomato Mosaic Virus*, dan lainnya.

    Dataset ini dibagi menjadi dua folder utama:
    - **train/**: digunakan untuk pelatihan model
    - **val/**: digunakan untuk validasi model
    """)

    st.subheader("⚙️ Preprocessing")
    st.markdown("""
    Setiap gambar di-*resize* menjadi ukuran **224x224 piksel**, lalu di-*normalize* ke rentang [0, 1]. 
    Untuk dataset training, dilakukan juga augmentasi berupa:
    - Rotasi acak
    - Flip horizontal
    - Zoom dan geser
    Agar model tidak overfitting dan mampu generalisasi lebih baik.
    """)

    st.subheader("🧠 Model yang Digunakan")
    st.markdown("""
    Model utama yang digunakan adalah **ResNet50** pretrained dari ImageNet.
    Tahapan pelatihan:
    1. **Feature Extraction**: hanya layer akhir (Dense) yang dilatih dulu
    2. **Fine-Tuning**: lapisan akhir ResNet dibuka untuk penyesuaian dengan dataset tomat

    Optimizer: **Adam**  
    Loss Function: **Categorical Crossentropy**
    """)

with tab3:
    st.header("🧪 Contoh Prediksi dari Dataset")
    st.markdown("Klik tombol di bawah untuk menampilkan 5 gambar acak dari dataset validasi beserta hasil prediksinya.")

    if st.button("Tampilkan Contoh Prediksi"):
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Hanya load val data untuk testing
        val_datagen = ImageDataGenerator(rescale=1./255)
        val_data = val_datagen.flow_from_directory(
            "tomato/val",  # Pastikan ini path folder validasi kamu di lokal/Colab
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )

        images, labels = next(val_data)
        preds = model.predict(images)
        pred_classes = np.argmax(preds, axis=1)
        true_classes = np.argmax(labels, axis=1)

        # Ambil 5 gambar acak dari batch
        indices = random.sample(range(len(images)), 5)
        fig, axs = plt.subplots(1, 5, figsize=(20, 5))

        for i, idx in enumerate(indices):
            axs[i].imshow(images[idx])
            axs[i].axis('off')
            true_label = class_labels[true_classes[idx]]
            pred_label = class_labels[pred_classes[idx]]
            color = 'green' if true_label == pred_label else 'red'
            axs[i].set_title(f"Pred: {pred_label}\nTrue: {true_label}", color=color, fontsize=10)

        st.pyplot(fig)
