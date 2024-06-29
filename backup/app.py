import streamlit as st
import os
import tempfile
from OCR import run
from ottoman import prepare_dataset
from glob import glob
from translate import Translator
import urllib.parse
import tqdm

# chars klasöründeki her harfin içinde kaç dosya olduğunu hesapla
char_counts = {}
total_char_count = 0

chars_folder_path = os.path.join(os.getcwd(), 'Dataset', 'chars')
if os.path.exists(chars_folder_path):
    for char_folder in os.listdir(chars_folder_path):
        char_path = os.path.join(chars_folder_path, char_folder)
        if os.path.isdir(char_path):
            char_count = len(os.listdir(char_path))
            char_counts[char_folder] = char_count
            total_char_count += char_count


st.title("Arapça OCR Uygulaması")
menu = st.sidebar.radio(
    "Menü", ["Ana Sayfa", "OCR Yap", "Dataset Oluştur", "Model Eğit"])

# Dataset dosya yollarını tanımla
img_paths = glob(os.path.join(os.getcwd(), 'Dataset', 'scanned', '*.png'))
txt_paths = glob(os.path.join(os.getcwd(), 'Dataset', 'text', '*.txt'))

if menu == "Ana Sayfa":
    st.header("Hoş geldiniz!")
    st.write("Lütfen soldaki menüden bir seçenek seçin.")

elif menu == "OCR Yap":
    uploaded_file = st.file_uploader(
        "Resim yükleyin", type=["jpg", "png", "bmp"])

    if uploaded_file is not None:
        file_name = uploaded_file.name
        st.write(f"Yüklenen dosya adı: {file_name}")

        temp_file = os.path.join('test', file_name)
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.read())

        if st.button("OCR İşlemi Başlat"):
            st.spinner("OCR işlemi devam ediyor...")

            img_name, character_count, word_count, ocr_result_arabic = run(
                temp_file)
            st.success(
                f"OCR işlemi tamamlandı! Sonuçlar: \nDosya Adı: {img_name}\nToplam Karakter Sayısı: {character_count}\nToplam Kelime Sayısı: {word_count}")

            st.subheader("OCR Sonuçları (Arapça):")
            st.text_area("OCR Çıktısı (Arapça):", ocr_result_arabic,
                         height=200, key="ocr_result_arabic")

            if st.button("Çevir"):
                st.spinner("Çeviri işlemi devam ediyor...")

                translator = Translator(to_lang="tr")
                translated_text = translator.translate(ocr_result_arabic)
                encoded_text = urllib.parse.quote(translated_text)

                st.success("Çeviri işlemi tamamlandı!")

                st.subheader("Çevrilen Metin (Türkçe):")
                st.text_area("Türkçe Çeviri:", translated_text,
                             height=200, key="translated_text")

                google_translate_url = f"https://translate.google.com/?sl=ar&tl=tr&text={encoded_text}"
                st.write(
                    f"[Google Translate'de Göster]({google_translate_url})")

        os.unlink(temp_file)

elif menu == "Dataset Oluştur":
    st.header("Dataset Oluştur")
    st.write("PNG ve TXT formatındaki dosyaları yükleyin.")

    # PNG dosyası yükleme
    png_file = st.file_uploader("PNG Dosyası Yükleyin", type=["png"])
    txt_file = st.file_uploader("TXT Dosyası Yükleyin", type=["txt"])

    if png_file is not None and txt_file is not None:
        # Geçici klasörü oluştur
        temp_dir = tempfile.mkdtemp()

        # Dosyaları geçici klasöre kaydet
        scanned_path = os.path.join(temp_dir, png_file.name)
        text_path = os.path.join(temp_dir, txt_file.name)

        with open(scanned_path, "wb") as f:
            f.write(png_file.read())

        with open(text_path, "wb") as f:
            f.write(txt_file.read())

        if st.button("Dataset Oluştur"):
            progress_bar = st.progress(0)  # İlerleme çubuğunu oluştur

            # Geçici dosyaların yollarını prepare_dataset() fonksiyonuna ileterek işlem yap
            prepare_dataset(img_paths=[scanned_path], txt_paths=[text_path])

            # İlerleme çubuğunu tamamlanmış olarak güncelle
            progress_bar.progress(1.0)

            st.success(
                "Dataset oluşturma işlemi tamamlandı! Modeli eğitme menüsünden modeli eğitmeye geçebilirsiniz.")
            
            # Her harfin ve dosya sayısının bilgisini yan yana yazdır
            char_info = "\n".join([f"{char}: {count}" for char, count in char_counts.items()])
            st.write(f"Toplam Görüntü Sayısı: {total_char_count}")
            st.write("Her Harfin İçindeki Dosya Sayıları:")
            st.write(char_info)

    else:
        st.warning("Lütfen PNG ve TXT dosyalarını yükleyin. Her iki dosyada mutlaka aynı isimde olmalıdır.")


elif menu == "Model Eğit":
    st.header("Model Eğit")
    # Model eğitme kodu buraya gelebilir (yukarıdaki kod örneğinden devralabilir)
