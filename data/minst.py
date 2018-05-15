import urllib2
import os
import csv
import numpy as np

basedir = os.path.dirname(os.path.realpath(__file__))
pathfile = os.path.join(basedir, 'mnist_train.csv')


def get_mminst(url='https://raw.githubusercontent.com/sbussmann/kaggle-mnist/master/Data/train.csv'):
    if os.path.isfile(pathfile):
        print('File already existed in %s ' % pathfile)
    else:
        print('Beginning file download with urllib2...')
        filedata = urllib2.urlopen(url=url)
        datatowrite = filedata.read()
        with open(pathfile, 'wb') as f:
            f.write(datatowrite)


def get_data(train_size, test_size):
    count = 0
    train = []
    target_train = []
    test = []
    target_test = []
    with open(pathfile, 'rb') as f:
        reader = csv.reader(f)
        header = next(reader)
        print(header)
        for row in reader:
            count += 1
            if count <= train_size + test_size:
                if count <= train_size:
                    train.append([float(pix) for pix in row[1:]])
                    target_train.append([int(row[0])/255.0])
                else:
                    test.append([float(pix) for pix in row[1:]])
                    target_test.append([int(row[0])/255.0])
            else:
                break
    (train, target_train, test, target_test) = np.asarray(train), np.asarray(target_train), np.asarray(test), np.asarray(
        target_test)
    return (train, target_train, test, target_test, header)
