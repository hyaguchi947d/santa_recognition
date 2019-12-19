import numpy as np
import cv2 as cv

from sklearn import svm
from santa_utils import dataset_files, precision_recall

files_train_p, files_train_n, files_test_p, files_test_n = dataset_files('/home/h-yaguchi/Pictures/santa_images')

file_hiro_not_santa = '/home/h-yaguchi/Pictures/santa_images/hiro/dr_hiro_avator_00.png'
file_hiro_santa = '/home/h-yaguchi/Pictures/santa_images/hiro/avatar_santa.png'
file_santa_kamo = '/home/h-yaguchi/Pictures/santa_images/hiro/santa_kamosirenai.png'

def color_hist_from_file(imgfile):
    img = cv.imread(imgfile)
    hist = cv.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_1d = hist.flatten()[1:-1]  # remove bins with (0,0,0) and (255,255,255)
    n_hist_1d = hist_1d / np.linalg.norm(hist_1d)
    return n_hist_1d
    
hist_train_p = map(color_hist_from_file, files_train_p)
hist_train_n = map(color_hist_from_file, files_train_n)

hist_test_p = map(color_hist_from_file, files_test_p)
hist_test_n = map(color_hist_from_file, files_test_n)

trains = hist_train_p + hist_train_n
labels = [1] * len(hist_train_p) + [0] * len(hist_train_n)

clf = svm.SVC()
clf.fit(trains, labels)

result_train_p = clf.predict(hist_train_p).tolist()
result_train_n = clf.predict(hist_train_n).tolist()
train_accuracy, train_precision, train_recall = precision_recall(result_train_p, result_train_n)

print('[train] positive:')
print(result_train_p)
print('[train] negative:')
print(result_train_n)
print("[train] accuracy = %f" % train_accuracy)
print("[train] precision = %f" % train_precision)
print("[train] recall = %f" % train_recall)


result_test_p = clf.predict(hist_test_p).tolist()
result_test_n = clf.predict(hist_test_n).tolist()
test_accuracy, test_precision, test_recall = precision_recall(result_test_p, result_test_n)


print('[test] positive:')
print(result_test_p)
print('[test] negative:')
print(result_test_n)
print("[test] accuracy = %f" % test_accuracy)
print("[test] precision = %f" % test_precision)
print("[test] recall = %f" % test_recall)

print("false positive:")
for i in (i for i, x in enumerate(result_test_p) if x == 0):
    print(files_test_p[i])
print("false negative:")
for i in (i for i, x in enumerate(result_test_n) if x == 1):
    print(files_test_n[i])

# hiro
hist_hiro_santa = color_hist_from_file(file_hiro_santa)
hist_hiro_not_santa = color_hist_from_file(file_hiro_not_santa)
hist_santa_kamo = color_hist_from_file(file_santa_kamo)

result_hiro_santa = clf.predict([hist_hiro_santa])[0]
result_hiro_not_santa = clf.predict([hist_hiro_not_santa])[0]
result_santa_kamo = clf.predict([hist_santa_kamo])[0]

print('hiro_santa = %d' % result_hiro_santa)
print('hiro_not_santa = %d' % result_hiro_not_santa)
print('santa_kamo = %d' % result_santa_kamo)
