import numpy as np
import cv2 as cv

from sklearn import svm
from santa_utils import dataset_files, precision_recall

files_train_p, files_train_n, files_test_p, files_test_n = dataset_files('/home/h-yaguchi/Pictures/santa_images')

file_hiro_not_santa = '/home/h-yaguchi/Pictures/santa_images/hiro/dr_hiro_avator_00.png'
file_hiro_santa = '/home/h-yaguchi/Pictures/santa_images/hiro/avatar_santa.png'
file_santa_kamo = '/home/h-yaguchi/Pictures/santa_images/hiro/santa_kamosirenai.png'

detector = cv.AKAZE_create(2)  # AKAZE_UPRIGHT
matcher = cv.BFMatcher()
bof_train = cv.BOWKMeansTrainer(16)
bof_ext = cv.BOWImgDescriptorExtractor(detector, matcher)

def key_desc_from_file(imgfile):
    img = cv.imread(imgfile, 0)
    keypoints, descriptors = detector.detectAndCompute(img, None)
    return (img, keypoints, descriptors)
    
kd_train_p = list(map(key_desc_from_file, files_train_p))
kd_train_n = list(map(key_desc_from_file, files_train_n))
kd_test_p = list(map(key_desc_from_file, files_test_p))
kd_test_n = list(map(key_desc_from_file, files_test_n))

# create dictionary
print('create dictionary...')
for img, key, desc in kd_train_p:
    bof_train.add(desc.astype(np.float32))
for img, key, desc in kd_train_n:
    bof_train.add(desc.astype(np.float32))
dictionary = bof_train.cluster()

bof_ext.setVocabulary(dictionary)
print('dictionary created')

def extract_bof_desc(ikd):
    return bof_ext.compute(ikd[0], ikd[1])[0]

print('convert training images to descriptors')
bof_desc_p = list(map(extract_bof_desc, kd_train_p))
bof_desc_n = list(map(extract_bof_desc, kd_train_n))

print('create SVM')
trains = bof_desc_p + bof_desc_n
labels = [1] * len(bof_desc_p) + [0] * len(bof_desc_n)

clf = svm.SVC()
clf.fit(trains, labels)
print('training done.')

result_train_p = clf.predict(bof_desc_p).tolist()
result_train_n = clf.predict(bof_desc_n).tolist()
train_accuracy, train_precision, train_recall = precision_recall(result_train_p, result_train_n)

print('[train] positive:')
print(result_train_p)
print('[train] negative:')
print(result_train_n)
print("[train] accuracy = %f" % train_accuracy)
print("[train] precision = %f" % train_precision)
print("[train] recall = %f" % train_recall)

# test
bof_test_p = list(map(extract_bof_desc, kd_test_p))
bof_test_n = list(map(extract_bof_desc, kd_test_n))

result_test_p = clf.predict(bof_test_p).tolist()
result_test_n = clf.predict(bof_test_n).tolist()
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
kd_hiro_santa = key_desc_from_file(file_hiro_santa)
kd_hiro_not_santa = key_desc_from_file(file_hiro_not_santa)
kd_santa_kamo = key_desc_from_file(file_santa_kamo)

bof_hiro_santa = extract_bof_desc(kd_hiro_santa)
bof_hiro_not_santa = extract_bof_desc(kd_hiro_not_santa)
bof_santa_kamo = extract_bof_desc(kd_santa_kamo)

result_hiro_santa = clf.predict([bof_hiro_santa])[0]
result_hiro_not_santa = clf.predict([bof_hiro_not_santa])[0]
result_santa_kamo = clf.predict([bof_santa_kamo])[0]

print('hiro_santa = %d' % result_hiro_santa)
print('hiro_not_santa = %d' % result_hiro_not_santa)
print('santa_kamo = %d' % result_santa_kamo)
