import streamlit as st
import os
import tempfile
from ocr import run
from ottomanDataset import prepare_dataset
from glob import glob
from translate import Translator
import train  # train.py dosyasından train fonksiyonunu içe aktar
from train import set_selected_models
import urllib.parse
import requests
import pandas as pd
import subprocess
import json
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


# st.title("Osmanlı Türkçesi OCR Uygulaması")
menu_group = st.sidebar.radio(
    "OCR Model", ["Ana Sayfa", "Dataset Oluştur", "Model Eğit", "OCR Yap", "Neural Machine Translation"])

# Dataset dosya yollarını tanımla
img_paths = glob(os.path.join(os.getcwd(), 'Dataset', 'scanned', '*.png'))
txt_paths = glob(os.path.join(os.getcwd(), 'Dataset', 'text', '*.txt'))

if menu_group == "Ana Sayfa":
    st.write(
        "Bu uygulama, OCR (Optical Character Recognition) yani Optik Karakter Tanıma işlemi yapmanıza olanak tanır. "
        "Menülerden uygun olanı seçerek işlemleri gerçekleştirebilirsiniz."
    )

    st.subheader("Dataset Oluşturma")
    st.write(
        "Bu bölümde, OCR modeli için kullanılacak olan veri kümesini oluşturmalısınız. Veri kümesi, "
        "tahminlerinizi değerlendirmek ve modelinizi eğitmek için kullanılacaktır. "
        "Lütfen çeşitli karakterleri içeren doğru ve temiz görüntüler ekleyin. "
        "Her bir karakter için birkaç farklı örnek eklemek, modelin doğruluğunu artırabilir."
    )

    st.subheader("Modeli Eğitme")
    st.write(
        "Bu bölümde, oluşturduğunuz veri kümesi üzerinde bir veya daha fazla modeli eğitebilirsiniz. "
        "Eğitim, seçtiğiniz sınıflandırıcı algoritmaları kullanarak veri kümesini öğrenmeyi içerir. "
        "Daha sonra eğitilen modeller, OCR işlemi için kullanılacaktır. "
        "Modeli eğitirken dikkate almanız gereken bazı parametreler ve dikkat edilmesi gereken noktalar şunlar olabilir:"
        "\n\n1. **Veri Bölme**: Veri kümesi genellikle eğitim ve test verileri olarak ikiye ayrılır. "
        "Eğitim verileriyle model öğrenilirken, test verileriyle modelin performansı değerlendirilir. "
        "Bu iki veri setini uygun oranlarda bölmek önemlidir."
        "\n\n2. **Sınıflandırıcı Seçimi**: Hangi sınıflandırıcı algoritmayı kullanacağınızı seçmelisiniz. "
        "Destek vektör makineleri (SVM), yapay sinir ağları, Naive Bayes vb. gibi farklı algoritmalar farklı veri tiplerine uygun olabilir."
        "\n\n3. **Özellik Çıkarımı**: Görüntü verilerinden anlam çıkarabilmek için özellik çıkarımı yapmanız gerekebilir. "
        "Bu, görüntüleri düzleştirme, histogram eşitleme, kenar tespiti gibi işlemleri içerebilir."
        "\n\n4. **Hiperparametre Ayarı**: Kullandığınız algoritmanın hiperparametrelerini ayarlamak, modelin performansını etkileyebilir."
        "\n\n5. **Eğitim Süresi**: Büyük veri setleriyle çalışıyorsanız, model eğitimi uzun sürebilir. "
        "Bu süreyi azaltmak için daha fazla işlem gücüne sahip bir bilgisayar veya bulut tabanlı çözümler kullanabilirsiniz."
    )

    st.subheader("OCR İşlemi")
    st.write(
        "Bu bölümde, eğittiğiniz modeli kullanarak metin tanıma işlemi gerçekleştirebilirsiniz. "
        "OCR (Optical Character Recognition) işlemi, görüntülerdeki yazıları algılayıp metin olarak dönüştürme sürecidir. "
        "OCR işlemi yaparken dikkate almanız gereken bazı noktalar şunlar olabilir:"
        "\n\n1. **Görüntü Hazırlama**: OCR işleminden önce görüntülerin doğru bir şekilde işlenmiş ve ölçeklendirilmiş olması önemlidir."
        "\n\n2. **Tahmin ve Sonuç Analizi**: Model tarafından yapılan tahminlerin doğruluğunu değerlendirmek için sonuçları analiz etmelisiniz. "
        "Yanlış tahminlerin nedenlerini anlamak, modelinizi geliştirmek için önemlidir."
        "\n\n3. **Performans Optimizasyonu**: Modelin doğruluğunu artırmak için görüntü işleme tekniklerini veya farklı özellik çıkarımı yöntemlerini deneyebilirsiniz."
        "\n\n4. **Gerçek Zamanlı Uygulamalar**: OCR işlemi gerçek zamanlı bir uygulamada kullanılacaksa, modelin hızı da önemli bir faktördür. "
        "Düşük gecikme süreleri gerektiren uygulamalar için hızlı modeller tercih edilmelidir."
    )
    st.write(
        "Sol taraftaki menüden istediğiniz işlemi seçerek başlayabilirsiniz. İyi çalışmalar!")


if menu_group == "OCR Yap":
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

if menu_group == "Dataset Oluştur":
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
            char_info = "\n".join(
                [f"{char}: {count}" for char, count in char_counts.items()])
            st.write(f"Toplam Görüntü Sayısı: {total_char_count}")
            st.write("Her Harfin İçindeki Dosya Sayıları:")
            st.write(char_info)

    else:
        st.warning(
            "Lütfen PNG ve TXT dosyalarını yükleyin. Her iki dosyada mutlaka aynı isimde olmalıdır.")


if menu_group == "Model Eğit":
    st.header("Model Eğit")

    # Her harfin ve dosya sayısının bilgisini yan yana yazdır
    char_info = "\n".join(
        [f"{char}: {count}" for char, count in char_counts.items()])
    st.write(f"Model eğitiminde seçeneklerden istediğiniz modeli seçebilirsiniz. Model eğitimi tamamlanınca oluşturduğunuz her model için bir skor verecektir. Skor 0-1 arasındadır. 0.98 skor modelin % 98 doğrulukla doğru tahminde bulunduğunu ifade eder.")
    st.write(f"Model eğitimi için toplam görüntü sayısı: {total_char_count}")
    st.write("Her Harfin İçindeki Dosya Sayıları:")
    st.write(char_info)

    # Kullanıcıdan hangi modellerin oluşturulacağını seçmesini iste
    selected_models = st.multiselect(
        "Hangi Modelleri Oluşturmak İstiyorsunuz?", train.names)

    # Seçilen modelleri train.py dosyasına iletmek için train.py içindeki bir fonksiyonu kullan
    set_selected_models(selected_models)

    # Model eğitimine başlamak için düğmeye basıldığında
    if st.button("Model Eğitimine Başla"):
        st.spinner("Model eğitimi devam ediyor...")
        train.train()  # train.py dosyasındaki train fonksiyonunu çağırarak model eğitimini başlat

        st.success("Model eğitimi tamamlandı!")

        # Eğitim tamamlandıktan sonra oluşturulan modellerin isimlerini, skorlarını ve indirme bağlantılarını göster
        st.subheader("Oluşturulan Modeller ve Skorlar:")
        for model_name, score in zip(train.names, train.scores):
            st.write(f"- **{model_name}** - Skor: {score:.2f}")
            # Modelin indirme bağlantısını oluştur ve kullanıcıya göster
            model_link = f"[İndir: {model_name} Model](models/{model_name}.sav)"
            st.markdown(model_link, unsafe_allow_html=True)

elif menu_group == "Neural Machine Translation":
    nmt_menu = st.sidebar.radio("Neural Machine Translation", [
                                "Veri Girişi", "NMT Data", "Vocab Oluştur", "NMT Model Oluştur", "NMT Çeviri"])

    if nmt_menu == "Veri Girişi":
        st.header("NMT Veri Girişi")

        # Ottoman ve Turkish metinleri tutacak sözlük
        translations = {
            "ottoman": [],
            "turkish": []
        }

        # Metin giriş widget'ları
        ottoman_text = st.text_area("Ottoman Text")
        turkish_text = st.text_area("Turkish Text")

        # Ekleme düğmesi
        if st.button("Submit"):
            if ottoman_text and turkish_text:
                # Veriyi gönder
                payload = {"ottoman": ottoman_text, "turkish": turkish_text}
                response = requests.post(
                    "http://localhost/ottoman/api/api.php", data=payload)
                if response.text == "success":
                    st.success(
                        "Success! Data has been submitted successfully.")
                else:
                    st.error(
                        "Error! There was an error while submitting the data.")
            else:
                st.error("Please fill out both Ottoman and Turkish texts.")

    elif nmt_menu == "NMT Data":
        st.header("NMT Data")
        # API endpoint URL
        api_url = "http://localhost/ottoman/api/api.php?action=get_data"

        # API'den veri çekme
        response = requests.get(api_url)

        # API'den gelen JSON verisini Python sözlüğüne çevirme
        data = response.json()
        # Toplam kayıt sayısını al
        total_records = data.get("total_records", 0)
        # Toplam kayıt sayısını gösterme
        st.write(f"Toplam kayıt sayısı: {total_records}")
        # Veriyi DataFrame'e dönüştürme
        if data.get("data"):
            df = pd.DataFrame(data["data"][-4:])
            df.columns = ["Ottoman Text", "Turkish Translation"]

            # Tablo olarak gösterme
            st.table(df)
        else:
            st.write("Veri bulunamadı.")

        st.write(
            "Yeterli düzeyde veri girişi yaptıktan sonra aşağıdaki butonları kullanarak Neural Machine Translation modelimizi oluşturacağımız dataları indirmemiz gerekli. Data nmtData klasörüne iner. Model eğitimine geçmeden önce 4 adet dosyanın ilgili klasörde olduğunu kontrol ediniz. SRC Train:Çevrilmek istenen metnin orijinal dilindeki versiyonunu ifade eder. TGT Train:Çevrilmek istenen metnin hedef dildeki versiyonunu ifade eder. SRC Val: Modelin doğruluğunu değerlendirmek için kullanılan orijinal metinleri içerir. TGT Val: Modelin çevirilerini değerlendirmek için kullanılan hedef dildeki metinleri içerir. Önemli bir not; genellikle verinin %80'i eğitim verisi olarak kullanılırken, geriye kalan %20'si doğrulama ve test verisi olarak ayrılmalıdır.  Bu oranlar, kullanılan veri kümesinin büyüklüğüne ve çeşitliliğine göre değişebilir.Train ve Val verilerinde bir ayrım yapılmadı. Projeyi geliştirmek isteyenler database e bir field açarak bu sekilde bir ayrımı gerçekleştirebilir ve train ve val verilerini buna göre indirebilir."
        )

        # SRC Train, TGT Train, SRC Val, TGT Val butonlarını yatay düzende gösterme
        col1, col2, col3, col4 = st.columns(4)

        # Dosya yolu belirleme
        file_path = os.path.join("..", "nmtData", "src-train.txt")

        if col1.button("SRC Train"):
            # API'ye istek gönderme
            response = requests.post(
                api_url, data={"action": "get_data"})  # POST isteği gönder

            if response.status_code == 200:
                data = response.json()  # JSON verisini al
                ottoman_texts = [entry["ottoman_text"]
                                 for entry in data["data"]]  # Ottoman metinlerini al

                # Dosyayı belirtilen yola yazma
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(" ".join(ottoman_texts))  # Verileri dosyaya yaz

                st.success(
                    f"Veriler başarıyla '{file_path}' dosyasına kaydedildi.")
            else:
                st.error("API'den veri alınamadı. Lütfen tekrar deneyin.")

        if col2.button("TGT Train"):
            # Dosya yolu belirleme
            turkish_texts = [entry["turkish_translation"]
                             for entry in data["data"]]  # Türkçe metinlerini al
            file_path = os.path.join("..", "nmtData", "tgt-train.txt")
            response = requests.post(
                api_url, data={"action": "get_data"})  # POST isteği gönder

            if response.status_code == 200:
                data = response.json()  # JSON verisini al
                turkish_texts = [entry["turkish_translation"]
                                 for entry in data["data"]]  # Türkçe metinlerini al

                # Dosyayı belirtilen yola yazma
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(" ".join(turkish_texts))  # Verileri dosyaya yaz

                st.success(
                    f"Veriler başarıyla '{file_path}' dosyasına kaydedildi.")
            else:
                st.error("API'den veri alınamadı. Lütfen tekrar deneyin.")

        if col3.button("SRC Val"):
            # Dosya yolu belirleme
            turkish_texts = [entry["turkish_translation"]
                             for entry in data["data"]]  # Türkçe metinlerini al
            file_path = os.path.join("..", "nmtData", "src-val.txt")
            response = requests.post(
                api_url, data={"action": "get_data"})  # POST isteği gönder

            if response.status_code == 200:
                data = response.json()  # JSON verisini al
                turkish_texts = [entry["turkish_translation"]
                                 for entry in data["data"]]  # Türkçe metinlerini al

                # Dosyayı belirtilen yola yazma
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(" ".join(turkish_texts))  # Verileri dosyaya yaz

                st.success(
                    f"Veriler başarıyla '{file_path}' dosyasına kaydedildi.")
            else:
                st.error("API'den veri alınamadı. Lütfen tekrar deneyin.")

        if col4.button("TGT Val"):
            # Dosya yolu belirleme
            turkish_texts = [entry["turkish_translation"]
            for entry in data["data"]]  # Türkçe metinlerini al
            file_path = os.path.join("..", "nmtData", "tgt-val.txt")
            response = requests.post(
                api_url, data={"action": "get_data"})  # POST isteği gönder

            if response.status_code == 200:
                data = response.json()  # JSON verisini al
                turkish_texts = [entry["turkish_translation"]
                for entry in data["data"]]  # Türkçe metinlerini al

                # Dosyayı belirtilen yola yazma
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(" ".join(turkish_texts))  # Verileri dosyaya yaz

                st.success(
                    f"Veriler başarıyla '{file_path}' dosyasına kaydedildi.")
            else:
                st.error("API'den veri alınamadı. Lütfen tekrar deneyin.")


    elif nmt_menu == "Vocab Oluştur":
        st.header("Vocab Dosyalarını Oluştur")
        nmt_folder = "nmt"  # YAML dosyalarının bulunduğu klasörün adı
        vocab_options = ["10.000 Vocab", "100.000 Vocab", "1.000.000 Vocab", "100.000.000 Vocab"]
        selected_vocab = st.selectbox("Vocab Değerini Seçin", vocab_options, index=0)

        vocab_size = 10000  # Varsayılan değer olarak 10.000 vocab seçiliyor
        if selected_vocab == "10.000 Vocab":
            vocab_size = 10000
        elif selected_vocab == "100.000 Vocab":
            vocab_size = 100000
        elif selected_vocab == "1.000.000 Vocab":
            vocab_size = 1000000
        elif selected_vocab == "100.000.000 Vocab":
            vocab_size = 100000000

        vocab_config_file = os.path.join(nmt_folder, "easy.yaml")  # Vocab oluşturma işlemi için varsayılan olarak easy.yaml dosyasını kullanıyoruz
        run_folder = os.path.join(nmt_folder, "run")
        existing_vocab_path = os.path.join(run_folder, f"vocab-{vocab_size}.txt")

        # nmt/run klasöründe vocab dosyası varsa dosyaları göster
        if os.path.exists(run_folder):
            existing_files = os.listdir(run_folder)
            st.info(f"nmt/run klasöründe bulunan dosyalar: {', '.join(existing_files)}  zaten var. Yine de oluşturmak için Vocab değerini seçerek Vocab Oluştur butonuna tıklayabilirsiniz.")
            

        if st.button("Vocab Oluştur"):
            try:
                # onmt_build_vocab komutunu doğrudan çalıştırarak vocab dosyasını oluşturun
                subprocess.run(["onmt_build_vocab", "-config", vocab_config_file, "-n_sample", str(vocab_size)], check=True, text=True)
                
                st.success("Vocab Dosyası Başarıyla Oluşturuldu!")  # İşlem tamamlandı uyarısı
            except subprocess.CalledProcessError as e:
                st.error(f"Vocab Dosyası Oluşturulurken Hata Oluştu: {e}")  # Hata uyarısı              



    elif nmt_menu == "NMT Model Oluştur":
        st.header("NMT Model Oluştur")
        nmt_folder = "nmt"  # YAML dosyalarının bulunduğu klasörün adı
        model_types = {
            "Basit Ayarlarla Model Oluştur": "easy.yaml",
            "CPU Kullanarak Oluştur": "config-cpu.yaml",
            "Özet Oluştur": "config-rnn-summarization.yaml",
            "1 GPU Kullanarak Oluştur": "config-transformer-base-1GPU.yaml",
            "4 GPU Kullanarak Oluştur": "config-transformer-base-4GPU.yaml"
        }
        selected_model_name = st.selectbox("Model Türünü Seçin", list(model_types.keys()), index=0)  # Model isimlerinin listesi
        selected_model_filename = model_types[selected_model_name]  # Seçilen modelin dosya adı

        if st.button("NMT Model Oluştur"):
            config_file = os.path.join(nmt_folder, selected_model_filename)  # Seçilen modele göre config dosya adı
            
            st.info("NMT Modeli Oluşturma İşlemi Başladı. Lütfen Bekleyin...")  # İşlem başladı uyarısı
            
            try:
                # onmt_train komutunu doğrudan çalıştırarak işlemi başlatın
                subprocess.run(["onmt_train", "-config", config_file], check=True, text=True)

                st.success("NMT Modeli Başarıyla Oluşturuldu!")  # İşlem tamamlandı uyarısı
            except subprocess.CalledProcessError as e:
                st.error(f"NMT Modeli Oluşturulurken Hata Oluştu: {e}")  # Hata uyarısı

    elif nmt_menu == "NMT Çeviri":
        st.header("NMT Çeviri")

        nmt_folder = "nmt/models"  # Model dosyalarının bulunduğu klasörün adı
        model_files = [f for f in os.listdir(nmt_folder) if f.endswith(".pt")]  # NMT model dosyalarının listesi
        selected_model = st.selectbox("Modeli Seçin", model_files, index=0)  # Model dosyalarını seçim listesi

        input_text = st.text_area("Osmanlıca Metin Girin", "")  # Kullanıcıdan Osmanlıca metin girişi
        translate_button = st.button("Çevir")  # Çeviri butonu

        if translate_button:
            output_file = "translated_text.txt"  # Çevrilen metnin kaydedileceği dosya adı
            model_path = os.path.join(nmt_folder, selected_model)  # Seçilen model dosyasının tam yolu

            try:
                # onmt_translate komutunu çalıştırarak çeviri işlemi gerçekleştirin
                process = subprocess.Popen(["onmt_translate", "-model", model_path, "-src", "-", "-output", output_file, "-verbose"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                stdout, stderr = process.communicate(input=input_text)

                # Çevrilen metni dosyadan okuyarak ekranda göster
                with open(output_file, "r") as f:
                    translated_text = f.read()
                    st.text("Çevrilen Metin:\n" + translated_text)  # Çevrilen metni ekranda göster

            except subprocess.CalledProcessError as e:
                st.error(f"Çeviri İşlemi Sırasında Hata Oluştu: {e}")  # Hata mesajını ekranda göster
