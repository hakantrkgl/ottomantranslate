#!/usr/bin/python
# Bu betiğin çalışabilmesi için python yürütücüsünün yolunu belirten shebang satırı

from __future__ import print_function  # Python 2.x uyumluluğu için

import sys, os, editdistance  # Sistem, işletim sistemi ve editdistance modüllerinin içe aktarılması

# Komut satırı argümanlarının doğruluğunun kontrolü
if len(sys.argv) != 3:
    sys.exit('USAGE: edit.py PREDICTED_PATH TRUTH_PATH')  # Doğru kullanım şeklini ekrana yazdır

distances = []  # Edit mesafesi değerlerini tutacak liste
accuracies = []  # Doğruluk oranlarını tutacak liste

# PREDICTED_PATH klasöründeki dosyalar ile TRUTH_PATH klasöründeki dosyaları karşılaştır
for file_name in os.listdir(sys.argv[1]):  # PREDICTED_PATH klasöründeki dosyaları listele
    with open(os.path.join(sys.argv[1], file_name), encoding='utf8') as f:
        predicted = ''.join(f.read().split())  # Dosyadan tahmin edilen metni oku ve boşlukları kaldır
    with open(os.path.join(sys.argv[2], file_name), encoding='utf8') as f:
        truth = ''.join(f.read().split())  # Doğru metni oku ve boşlukları kaldır
    distance = editdistance.eval(predicted, truth)  # Edit mesafesini hesapla
    distances.append(distance)  # Edit mesafesini distances listesine ekle
    accuracies.append(max(0, 1 - distance / len(truth)))  # Doğruluk oranını hesapla ve accuracies listesine ekle
    print(f'{file_name}: {distance}')  # Dosya adı ve edit mesafesini ekrana yazdır

print(f'Total distance = {sum(distances)}')  # Toplam edit mesafesi değerini ekrana yazdır
print('Average Accuracy = %.2f%%' % (sum(accuracies) / len(accuracies) * 100))  # Ortalama doğruluk oranını ekrana yazdır
