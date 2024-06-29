# Kütüphanelerin ve Modüllerin İçe Aktarılması
import cv2 as cv  # Görüntü işleme için OpenCV kütüphanesi
import os  # İşletim sistemi işlemleri için
import time  # Zaman fonksiyonları için
from tqdm import tqdm  # İlerleme çubuğu oluşturmak için
from glob import glob  # Dosya yollarını almak için
# Karakterleri ayırmak için özel modül
from character_segmentation import segment
from segmentation import extract_words  # Kelimeleri ayırmak için özel modül
# Modelin hazırlık işlemleri için özel modül
from train import prepare_char, featurizer
import pickle  # Nesneleri seri hale getirmek ve geri çevirmek için
import multiprocessing as mp  # Çoklu iş parçacığı desteği sağlayan modül

# Önceden eğitilmiş modelin yüklenmesi
model_name = '2L_NN.sav'


def load_model():
    location = 'models'  # Önceden eğitilmiş modeller models klasörü içerisinde
    if os.path.exists(location):
        model = pickle.load(open(f'models/{model_name}', 'rb'))
        return model

# OCR işlemi için her bir kelimenin karakter dizilimini bulan fonksiyon


def run2(obj):
    word, line = obj
    model = load_model()
    # Her bir kelime için karakterleri ayır
    char_imgs = segment(line, word)
    txt_word = ''
    # Her bir karakter için
    for char_img in char_imgs:
        try:
            ready_char = prepare_char(char_img)  # Karakteri hazırla
        except:
            # breakpoint()
            continue
        feature_vector = featurizer(ready_char)   # Özellik vektörünü oluştur
        # Karakterin ne olduğunu tahmin et
        predicted_char = model.predict([feature_vector])[0]
        txt_word += predicted_char  # Tahmin edilen karakteri kelimeye ekle
    return txt_word


# Görüntüdeki metinleri tanıyan fonksiyon
def run(image_path):
    # Verilen görüntüyü oku
    full_image = cv.imread(image_path)
    predicted_text = ''

    # Zaman ölçümü başlat
    before = time.time()
    # Görüntüden kelimeleri ayır, [(kelime, satır), (kelime, satır), ...]
    words = extract_words(full_image)
    pool = mp.Pool(mp.cpu_count())
    # Her kelimenin karakterlerini tahmin et
    predicted_words = pool.map(run2, words)
    pool.close()
    pool.join()
    # Zaman ölçümü durdur
    after = time.time()

    # Toplam metni oluştur
    for word in predicted_words:
        predicted_text += word
        predicted_text += ' '

    exc_time = after-before  # İşlem süresini hesapla
    # Görüntünün adıyla aynı isimde bir dosya oluştur
    img_name = image_path.split('\\')[1].split('.')[0]

    with open(f'output/text/{img_name}.txt', 'w', encoding='utf8') as fo:
        # Tahmin edilen metni dosyaya yaz. output/text klasöründe txt formatında yazdırır.
        fo.writelines(predicted_text)

    # İşlem süresiyle birlikte (dosya_adı, işlem_süresi) çiftini döndür
    return (img_name, exc_time)


if __name__ == "__main__":

    # Eski verileri içeren running_time.txt dosyasını temizle
    if not os.path.exists('output'):
        os.mkdir('output')
    open('output/running_time.txt', 'w').close()

    destination = 'output/text'
    if not os.path.exists(destination):
        os.makedirs(destination)

    types = ['png', 'jpg', 'bmp']
    images_paths = []
    for t in types:
        # Belirtilen uzantılara sahip görüntü dosyalarını bul
        images_paths.extend(glob(f'test/*.{t}'))
    before = time.time()

    # pool = mp.Pool(mp.cpu_count())

    # # Method1
    # for image_path in images_paths:
    #     pool.apply_async(run,[image_path])

    # Method2
    # for _ in tqdm(pool.imap_unordered(run, images_paths), total=len(images_paths)):
    #     pass

    running_time = []

    # Her bir görüntü için işlem süresini hesapla ve running_time.txt dosyasına yaz
    for images_path in tqdm(images_paths, total=len(images_paths)):
        running_time.append(run(images_path))

    running_time.sort()
    with open('output/running_time.txt', 'w') as r:
        for t in running_time:
            # Dosya adı ve işlem süresini running_time.txt dosyasına yaz
            r.writelines(f'image#{t[0]}: {t[1]}\n')

    # pool.close()
    # pool.join()
    after = time.time()
    print(f'total time to finish {len(images_paths)} images:')
    print(after - before)  # Toplam işlem süresini ekrana yazdır
