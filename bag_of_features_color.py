import numpy as np
import cv2 as cv

from sklearn import svm
from santa_utils import dataset_files, precision_recall

files_train_p, files_train_n, files_test_p, files_test_n = dataset_files('/home/h-yaguchi/Pictures/santa_images')

file_hiro_not_santa = '/home/h-yaguchi/Pictures/santa_images/hiro/dr_hiro_avator_00.png'
file_hiro_santa = '/home/h-yaguchi/Pictures/santa_images/hiro/avatar_santa.png'
file_santa_kamo = '/home/h-yaguchi/Pictures/santa_images/hiro/santa_kamosirenai.png'


dict_size = 16
detector = cv.AKAZE_create(2)  # AKAZE_UPRIGHT
matcher = cv.BFMatcher()
bof_train_b = cv.BOWKMeansTrainer(dict_size)
bof_ext_b = cv.BOWImgDescriptorExtractor(detector, matcher)
bof_train_g = cv.BOWKMeansTrainer(dict_size)
bof_ext_g = cv.BOWImgDescriptorExtractor(detector, matcher)
bof_train_r = cv.BOWKMeansTrainer(dict_size)
bof_ext_r = cv.BOWImgDescriptorExtractor(detector, matcher)


def key_desc_from_file(imgfile):
    img = cv.imread(imgfile)
    img_b, img_g, img_r = cv.split(img)
    keypoints_b, descriptors_b = detector.detectAndCompute(img_b, None)
    keypoints_g, descriptors_g = detector.detectAndCompute(img_g, None)
    keypoints_r, descriptors_r = detector.detectAndCompute(img_r, None)
    return (img_b, keypoints_b, descriptors_b,
            img_g, keypoints_g, descriptors_g,
            img_r, keypoints_r, descriptors_r)
    
kd_train_p = list(map(key_desc_from_file, files_train_p))
kd_train_n = list(map(key_desc_from_file, files_train_n))
kd_test_p = list(map(key_desc_from_file, files_test_p))
kd_test_n = list(map(key_desc_from_file, files_test_n))

# create dictionary
print('create dictionary...')
for img_b, key_b, desc_b, img_g, key_g, desc_g, img_r, key_r, desc_r in kd_train_p:
    bof_train_b.add(desc_b.astype(np.float32))
    bof_train_g.add(desc_g.astype(np.float32))
    bof_train_r.add(desc_r.astype(np.float32))
for img_b, key_b, desc_b, img_g, key_g, desc_g, img_r, key_r, desc_r in kd_train_n:
    bof_train_b.add(desc_b.astype(np.float32))
    bof_train_g.add(desc_g.astype(np.float32))
    bof_train_r.add(desc_r.astype(np.float32))
    
dictionary_b = bof_train_b.cluster()
dictionary_g = bof_train_g.cluster()
dictionary_r = bof_train_r.cluster()

bof_ext_b.setVocabulary(dictionary_b)
bof_ext_g.setVocabulary(dictionary_g)
bof_ext_r.setVocabulary(dictionary_r)
print('dictionary created')

def extract_bof_desc(ikd):
    result_b = [0] * dict_size
    bof_res_b = bof_ext_b.compute(ikd[0], ikd[1])
    if bof_res_b is not None:
        result_b = bof_res_b[0].tolist()
    result_g = [0] * dict_size
    bof_res_g = bof_ext_g.compute(ikd[3], ikd[4])
    if bof_res_g is not None:
        result_g = bof_res_g[0].tolist()
    result_r = [0] * dict_size
    bof_res_r = bof_ext_r.compute(ikd[6], ikd[7])
    if bof_res_r is not None:
        result_r = bof_res_r[0].tolist()
    return np.array(result_b + result_g + result_r)

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
