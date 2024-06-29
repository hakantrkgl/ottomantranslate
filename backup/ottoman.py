import re
from pyarabic.araby import tokenize
import cv2 as cv
import os
from segmentation import extract_words
from character_segmentation import segment
from glob import glob
from tqdm import tqdm


# Diğer değişkenlerin tanımlanması
script_path = os.getcwd()  # Betik dosyasının yolu
directory = {}  # Karakter dizinini tutan sözlük
chars = ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف',
         'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي', 'لا', 'پ', 'چ', 'ژ', '،', 'ی']  # İşlenen karakterlerin listesi

# Karakter dizinini oluşturma
for char in chars:
    directory[char] = 0

# Arapça metni ayırma fonksiyonu
def tokenize_arabic(text):
    tokens = tokenize(text)
    return tokens

# Temizleme fonksiyonu
def clean_text(text):
    # Sadece Arap harflerini ve boşlukları koru
    cleaned_text = re.sub(r'[^ء-ي\s]', '', text)
    return cleaned_text.strip()  # Metin başındaki ve sonundaki boşlukları kaldır

# Veri kümesini hazırlayan fonksiyon
def prepare_dataset(img_paths=None, txt_paths=None, limit=None):
    print("Processing Images")

    # Görüntü ve metin dosyalarını işleyerek veri kümesini oluştur
    for img_path, txt_path in tqdm(zip(img_paths[:limit], txt_paths[:limit]), total=len(img_paths)):

        assert (img_path.split('\\')[-1].split('.')[0]
                == txt_path.split('\\')[-1].split('.')[0])

        # Metin kısmı
        with open(txt_path, 'r', encoding='utf8') as fin:
            lines = fin.readlines()
            line = lines[0].rstrip()
            txt_word = clean_text(line)  # Temizleme işlemi
            txt_chars = tokenize_arabic(txt_word)  # Arapça metni ayırma işlemi

        # Görüntü kısmı
        img = cv.imread(img_path)
        img_words = extract_words(img)

        # Görüntüden kelimeleri çıkar ve karşılaştır
        for img_word, txt_char in tqdm(zip(img_words, txt_chars), total=len(txt_chars)):
            # Metin kısmındaki karakterleri al
            line = img_word[1]  # Görüntüdeki satır
            word = img_word[0]  # Görüntüdeki kelime

            img_chars = segment(line, word)  # Görüntüdeki karakterleri ayır

            if len(txt_char) == len(img_chars):
                for img_char, txt_char in zip(img_chars, txt_char):
                    number = directory[txt_char]  # Karakter sayısını güncelle
                    # Karakterin hedef yolu
                    destination = f'../Dataset/chars/{txt_char}'
                    if not os.path.exists(destination):
                        os.makedirs(destination)  # Klasör yoksa oluştur
                    os.chdir(destination)
                    # Karakter resmini kaydet
                    cv.imwrite(f'{number}.png', img_char)
                    os.chdir(script_path)  # Betik dosyasının yoluna geri dön
                    directory[txt_char] += 1  # Karakter sayısını güncelle

    print('\nDone')

if __name__ == "__main__":
    prepare_dataset(img_paths=img_paths, txt_paths=txt_paths, limit=len(img_paths))  # Veri kümesini hazırla