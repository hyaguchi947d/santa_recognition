import glob

## @brief enumerate dataset files
##  put your image files as follows:
##  - dataset_path
##    - train
##      - p : positive data
##      - n : negative data
##    - test
##      - p : positive data
##      - n : negative data
## @param[in] dataset_path path of dataset
## @return (train_positive, train_negative, test_positive, test_negative) as list of string
def dataset_files(dataset_path):
    files_train_p = glob.glob(dataset_path + '/train/p/*')
    files_train_n = glob.glob(dataset_path + '/train/n/*')
    files_test_p = glob.glob(dataset_path + '/test/p/*')
    files_test_n = glob.glob(dataset_path + '/test/n/*')
    return (files_train_p, files_train_n, files_test_p, files_test_n)


## @brief calc accuracy, recall, and precision
## @param[in] result_p result of positive dataset, 1 is true positive
## @param[in] result_n result of negative dataset, 0 is true negative
## @return (accuracy, recall, precision)
def precision_recall(result_p, result_n):
    true_positive = result_p.count(1)
    false_negative = len(result_p) - true_positive
    true_negative = result_n.count(0)
    false_positive = len(result_n) - true_negative
    return (
        (float(true_positive + true_negative) / float(true_positive + true_negative + false_positive + false_negative)),
        (float(true_positive) / float(true_positive + false_positive)),
        (float(true_positive) / float(true_positive + false_negative)))
