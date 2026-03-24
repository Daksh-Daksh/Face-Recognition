import os
import numpy as np
from PIL import Image
from keras.utils.np_utils import to_categorical
from keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
    14 is skipped when staff name this image data set
"""
# dictionary -> { key: value }
test_label_dict = {'11': 0, '12': 1, '13': 2, '15': 3, '16': 4, '17': 5, '18': 6, '19': 7, '20': 8, '21': 9,
                   '22': 10, '23': 11, '24': 12, '25': 13, '26': 14, '27': 15, '28': 16, '29': 17, '30': 18, '31': 19,
                   '32': 20, '33': 21, '34': 22, '35': 23, '36': 24, '37': 25, '38': 26, '39': 27}


def return_test_label(label_str):
    return test_label_dict[label_str]


def get_testing_set(out_dim):
    Test_data_list = []
    Test_label_list = []
    for _, img_name in enumerate(os.listdir(test_filepath)):
        if os.path.splitext(img_name)[-1] != '.png':
            print('Not png')
            continue

        label_tmp = (img_name.split('_')[0]).split('yaleB')[-1]
        real_label = return_test_label(label_tmp)

        img_filepath = os.path.join(test_filepath, img_name)  # Yale Face Database/val\yaleB11_P00A+000E+00.png
        img_tmp = Image.open(img_filepath)
        img_num_arr = np.array(img_tmp).reshape(32, 32, 1)
        img_num_arr = img_num_arr.astype('float32')
        img_num_arr /= 255

        Test_data_list.append(img_num_arr)
        Test_label_list.append(to_categorical(real_label, out_dim))

    Test_data_arr = np.array(Test_data_list)
    Test_label_arr = np.array(Test_label_list)
    return Test_data_arr, Test_label_arr


if __name__ == '__main__':
    test_filepath = 'Yale Face Database/val'  # testing set

    out_dim = 28
    x_test, y_test = get_testing_set(out_dim)
    # print(x_test.shape)  # (2128, 32, 32, 1)
    # print(y_test.shape)  # (2128, 28)

    with open('Model_name.txt', 'r') as Model_reader:
        for Model_name in Model_reader:
            predict_model = load_model(Model_name[:-1])  # eliminate the last char ('\n')
            model_predict = predict_model.predict(x_test, verbose=0)

            predict_result = []
            for i in model_predict:
                predict_result.append(i.argmax())

            total_test_num = 0  # Count for how many test data(image)
            correct_predict_num = 0  # Count for how many prediction are correct
            for i in range(len(predict_result)):
                total_test_num += 1

                predict_label = predict_result[i]
                actual_label = y_test[i].argmax()

                if predict_label == actual_label:
                    correct_predict_num += 1

            predict_accuracy = correct_predict_num / total_test_num
            print(Model_name[:-1])
            print("Accuracy - {a:.3f}% [{b} / {c}]".format(a=predict_accuracy * 100, b=correct_predict_num,
                                                           c=total_test_num))
            print()
