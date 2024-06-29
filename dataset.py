# Gerekli kütüphanelerin içe aktarılması
import cv2 as cv  # Görüntü işleme için OpenCV kütüphanesi
import os  # İşletim sistemi işlemleri için
from segmentation import extract_words  # Kelimeleri ayırmak için özel modül
# Karakterleri ayırmak için özel modül
from character_segmentation import segment
from glob import glob  # Dosya yollarını almak için
from utilities import projection  # Yardımcı fonksiyonlar için özel modül
from tqdm import tqdm  # İlerleme çubuğu oluşturmak için

# Görüntü ve metin dosyalarının yollarının listelenmesi
img_paths = glob('Dataset\\scanned/*.png')  # Taranmış görüntülerin yolu
txt_paths = glob('Dataset\\text/*.txt')  # Metin dosyalarının yolu


# Diğer değişkenlerin tanımlanması
script_path = os.getcwd()  # Betik dosyasının yolu
width = 25  # Karakter resminin genişliği
height = 25  # Karakter resminin yüksekliği
dim = (width, height)  # Resim boyutları
directory = {}  # Karakter dizinini tutan sözlük
chars = ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف',
         'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي', 'لا', 'پ', 'چ', 'ژ','،','ی']  # İşlenen karakterlerin listesi

# Karakter dizinini oluşturma
for char in chars:
    directory[char] = 0

# 'لام-ألف' kontrolü yapan fonksiyon


def check_lamAlf(word, idx):

    if idx != len(word)-1 and word[idx] == 'ل':
        if word[idx+1] == 'ا':
            return True

    return False

# Kelimede bulunan karakterleri ayıran fonksiyon


def get_word_chars(word):

    i = 0
    chars = []
    while i < len(word):
        if check_lamAlf(word, i):
            chars.append(word[i:i+2])
            i += 2
        else:
            chars.append(word[i])
            i += 1

    return chars

# Veri kümesini hazırlayan fonksiyon

def prepare_dataset(limit=  None):
    print("Processing Images")

    # Görüntü ve metin dosyalarını işleyerek veri kümesini oluştur
    for img_path, txt_path in tqdm(zip(img_paths[:limit], txt_paths[:limit]), total=len(img_paths)):

        assert (img_path.split('\\')[-1].split('.')[0]
                == txt_path.split('\\')[-1].split('.')[0])

        # Metin kısmı
        with open(txt_path, 'r', encoding='utf8') as fin:

            lines = fin.readlines()
            line = lines[0].rstrip()

            txt_words = line.split()

        # Görüntü kısmı
        img = cv.imread(img_path)
        img_words = extract_words(img)

        # breakpoint()

        lst = []
        error = 0

        # Görüntüden kelimeleri çıkar ve karşılaştır
        for img_word, txt_word in tqdm(zip(img_words, txt_words), total=len(txt_words)):

            # Metin kısmındaki karakterleri al
            txt_chars = get_word_chars(txt_word)

            line = img_word[1]  # Görüntüdeki satır
            word = img_word[0]  # Görüntüdeki kelime

            img_chars = segment(line, word)  # Görüntüdeki karakterleri ayır

            if len(txt_chars) == len(img_chars):
                for img_char, txt_char in zip(img_chars, txt_chars):
                    number = directory[txt_char]  # Karakter sayısını güncelle
                    # Karakterin hedef yolu
                    destination = f'Dataset/chars/{txt_char}'
                    if not os.path.exists(destination):
                        os.makedirs(destination)  # Klasör yoksa oluştur
                    os.chdir(destination)
                    # Karakter resmini kaydet
                    cv.imwrite(f'{number}.png', img_char)
                    os.chdir(script_path)  # Betik dosyasının yoluna geri dön
                    directory[txt_char] += 1  # Karakter sayısını güncelle
            else:
                error += 1
        # Hata oranını yazdır
        # print(f'\nAcc: {100-(error*100/len(img_words))}')

    print('\nDone')


# Ana fonksiyon
if __name__ == "__main__":
    prepare_dataset(limit=len(img_paths))  # Veri kümesini hazırla
