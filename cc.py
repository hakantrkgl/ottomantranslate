import os

chars = ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف',
         'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي', 'لا', 'پ', 'چ', 'ژ', '،', 'ی']

base_directory = 'Dataset/chars'

# 'Dataset/chars' klasörü varsa oluşturmaya gerek yok
if not os.path.exists(base_directory):
    os.makedirs(base_directory)

# Her bir karakter için klasör oluştur
for char in chars:
    char_directory = os.path.join(base_directory, char)
    if not os.path.exists(char_directory):
        os.makedirs(char_directory)

print("Klasörler oluşturuldu.")
