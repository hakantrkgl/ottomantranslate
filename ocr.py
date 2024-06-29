# OCR.py

import cv2 as cv
import os
import time
from tqdm import tqdm
from glob import glob
import multiprocessing as mp
from ocr_utils import run2
from segmentation import extract_words

model_name = '2L_NN.sav'


def run(file_path):
    full_image = cv.imread(file_path)
    img_name = os.path.basename(file_path).split('.')[0]
    predicted_text = ''
    
    if full_image is None:
        print(f"Error: Dosya bulunamadı: {file_path}")
        return None, None

    words = extract_words(full_image)

    if not words:
        print("Error: Kelimeleri ayırmada hata oluştu.")
        return None, None

    pool = mp.Pool(mp.cpu_count())
    predicted_words = pool.map(run2, words)
    pool.close()
    pool.join()

    for word in predicted_words:
        predicted_text += word
        predicted_text += ' '

    with open(f'output/text/{img_name}.txt', 'w', encoding='utf8') as fo:
        fo.writelines(predicted_text)

    return img_name, len(predicted_text), len(words), predicted_text



if __name__ == "__main__":
    if not os.path.exists('output'):
        os.mkdir('output')
    open('output/running_time.txt', 'w').close()

    destination = 'output/text'
    if not os.path.exists(destination):
        os.makedirs(destination)

    types = ['png', 'jpg', 'bmp']
    images_paths = []
    for t in types:
        images_paths.extend(glob(f'src/test/*.{t}'))

    before = time.time()
    running_time = []

    for image_path in tqdm(images_paths, total=len(images_paths)):
        running_time.append(run(image_path))

    running_time.sort()
    with open('output/running_time.txt', 'w') as r:
        for t in running_time:
            r.writelines(f'image#{t[0]}: {t[1]} characters in {t[2]} words\n')

    after = time.time()
    print(f'total time to finish {len(images_paths)} images:')
    print(after - before)
