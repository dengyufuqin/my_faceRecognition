import json
from io import BytesIO
import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify
from keras.preprocessing import image
import random
from PIL import Image
import urllib.request
import requests
import base64
import os
from predict import *
from create_dataset import *
import shutil
import argparse
import glob
import sys
from builtins import str

app = Flask(__name__)

def upload1(image_path):
    url = 'http://192.168.0.5:8080/jeecg-boot/fastdfs/fastDFS/fastDFSUpload'
    with open(image_path, 'rb') as f_abs:
        body = {
            'file': f_abs
        }
        rep = requests.post(url=url,files=body).json  # 发送get请求
        img = rep['message']
        return img

def remove_cache_image(cache_path):
    fileList = os.listdir(cache_path)

    for cache_name in fileList:
        image_path = cache_path+cache_name
        os.remove(image_path)

def merge(file1, file2):
    f1 = open(file1, 'a+', encoding='utf-8')
    with open(file2, 'r', encoding='utf-8') as f2:
        for i in f2:
            f1.write(i)


@app.route('/faceShibe/', methods=('POST', 'GET'))
def face_shibie():
    global dataset_emb, names_list, unknow_dataset_emb, unknow_names_list


    url = request.form['url']
    resp = urllib.request.urlopen(url)
    img = resp.read()
    b64 = base64.b64encode(img)
    b64 = str(b64, 'utf-8')

    resualt = {}
    img = base64.b64decode(b64)
    nparr = np.frombuffer(img, np.uint8)
    img_cv2 = cv2.imdecode(nparr, cv2.COLOR_RGB2BGR)
    image = img_cv2[:, :, ::-1]

    images = []
    flag = False

    pred_names, bboxes = face_recognition_image1(dataset_emb, names_list, face_detect, face_net, image)

    for img_name, bbox in zip(pred_names, bboxes):
        bbox = [int(b) for b in bbox]
        for i in range(len(bbox)):
            if bbox[i] < 0:
                bbox[i] = 0
        if img_name != 'unknow':
            resualt['status'] = '0'
            resualt['id'] = img_name
        if img_name == 'unknow':
            flag = True
            unknow_image = image[bbox[1]: bbox[3], bbox[0]: bbox[2]]
            try:
                unknow_pred_names, bboxes = face_recognition_image1(unknow_dataset_emb, unknow_names_list, face_detect, face_net, unknow_image)
                print(unknow_pred_names)
                for unknow_img_name, bbox in zip(unknow_pred_names, bboxes):
                    bbox = [int(b) for b in bbox]

                    if unknow_img_name != 'unknow':
                        resualt['status'] = '1'
                        resualt['id'] = unknow_img_name
            except:
                continue

        images.append(resualt)
        resualt = {}
    return json.dumps(images)

@app.route('/faceRecognition/', methods=('POST', 'GET'))
def face_rec():
    global dataset_emb, names_list, unknow_dataset_emb, unknow_names_list

    rnd_name = ''.join(random.sample('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', 32))



    url = request.form['url']
    resp = urllib.request.urlopen(url)
    img = resp.read()
    b64 = base64.b64encode(img)
    b64 = str(b64, 'utf-8')

    resualt = {}
    img = base64.b64decode(b64)
    nparr = np.frombuffer(img, np.uint8)
    img_cv2 = cv2.imdecode(nparr, cv2.COLOR_RGB2BGR)
    image = img_cv2[:, :, ::-1]

    images = []
    flag = False
    pred_names, bboxes = face_recognition_image1(dataset_emb, names_list, face_detect, face_net, image)

    cache_dataset = rnd_name+'/emb/faceEmbedding.npy'
    cache_zairu_path = rnd_name+'/images'
    cache_filename = rnd_name+'/emb/name.txt'
    cache_out_emb_path = rnd_name+'/emb/faceEmbedding.npy'
    cache_out_filename = rnd_name+'/emb/name.txt'

    os.mkdir(rnd_name)
    os.mkdir(rnd_name+'/emb')
    os.mkdir(rnd_name+'/images')
    for img_name, bbox in zip(pred_names, bboxes):
        bbox = [int(b) for b in bbox]
        for i in range(len(bbox)):
            if bbox[i] < 0:
                bbox[i] = 0
        if img_name != 'unknow':
            resualt['status'] = '0'
            resualt['id'] = img_name

            str1 = 'rlsb1000'
            rnd = ''.join(random.choice("0123456789") for i in range(8))
            rnd = str1 + rnd

            know_img = img_cv2[bbox[1]: bbox[3], bbox[0]: bbox[2]]

            cv2.imwrite('dataset/images/'+img_name+'/'+rnd+'.jpg', know_img)

            img_path = upload('dataset/images/'+img_name+'/'+rnd+'.jpg')
            resualt['url'] = img_path
            os.remove('dataset/images/'+img_name+'/'+rnd+'.jpg')
        if img_name == 'unknow':


            unknow_image = image[bbox[1]: bbox[3], bbox[0]: bbox[2]]
            try:
                unknow_pred_names, bboxes = face_recognition_image1(unknow_dataset_emb, unknow_names_list, face_detect, face_net, unknow_image)

                for unknow_img_name, bbox in zip(unknow_pred_names, bboxes):

                    bbox = [int(b) for b in bbox]

                    if unknow_img_name != 'unknow':
                        flag = True
                        count = glob.glob('unknow/images/' + unknow_img_name + '/*.jpg')
                        count1 = len(count)
                        resualt['status'] = '1'
                        resualt['id'] = unknow_img_name
                        path_file_number = glob.glob('unknow/images/'+unknow_img_name+'/*.jpg')
                        # if len(path_file_number) > 20:
                        #
                        #     resualt.append(img_name)
                        #     continue
                        str1 = 'rlsb1000'
                        rnd = ''.join(random.choice("0123456789") for i in range(8))
                        rnd = str1 + rnd
                        unknow_img = unknow_image[:, :, ::-1]
                        cv2.imwrite('unknow/images/' + unknow_img_name + '/' + rnd +
                                    '.jpg', unknow_img)


                        os.mkdir(rnd_name+'/images/'+unknow_img_name)
                        cv2.imwrite(rnd_name+'/images/' + unknow_img_name + '/' + rnd +
                                        '.jpg', unknow_img)




                        img_name = upload('./unknow/images/'+unknow_img_name+'/' + rnd+'.jpg')

                        if count1 > 5:
                            flag = False
                            os.remove('./unknow/images/'+unknow_img_name+'/' + rnd+'.jpg')
                            # shutil.rmtree(rnd_name+'/images/'+unknow_img_name)
                        resualt['url'] = img_name

                    else:
                        flag = True
                        resualt['status'] = '2'

                        str1 = 'rlsb1000'
                        rnd = ''.join(random.choice("0123456789") for i in range(8))
                        rnd = str1 + rnd
                        os.mkdir('unknow/images/'+rnd)
                        unknow_img = unknow_image[:, :, ::-1]
                        cv2.imwrite('./unknow/images/' + rnd + '/' + rnd +
                                    '.jpg', unknow_img)


                        os.mkdir(rnd_name+'/images/'+rnd)
                        cv2.imwrite(rnd_name+'/images/' + rnd + '/' + rnd +
                                    '.jpg', unknow_img)



                        resualt['id'] = rnd
                        img_name = upload('./unknow/images/' + rnd + '/' + rnd + '.jpg')
                        resualt['url'] = img_name
            except:
                flag = True
                print('出错')
                resualt['status'] = '2'

                str1 = 'rlsb1000'
                rnd = ''.join(random.choice("0123456789") for i in range(8))
                rnd = str1 + rnd
                os.mkdir('unknow/images/' + rnd)
                unknow_img = unknow_image[:, :, ::-1]
                cv2.imwrite('./unknow/images/' + rnd + '/' + rnd +
                            '.jpg', unknow_img)

                os.mkdir(rnd_name+'/images/' + rnd)
                cv2.imwrite(rnd_name+'/images/' + rnd + '/' + rnd +
                            '.jpg', unknow_img)


                resualt['id'] = rnd
                img_name = upload('./unknow/images/' + rnd + '/' + rnd + '.jpg')
                resualt['url'] = img_name
        images.append(resualt)
        resualt = {}
    if flag:
        create_face_embedding1(face_net, face_detect, cache_zairu_path, cache_out_emb_path,
                               cache_out_filename)
        file1 = 'unknow/emb/name.txt'
        file2 = rnd_name+'/emb/name.txt'

        def merge(file1, file2):
            f1 = open(file1, 'a+', encoding='utf-8')
            with open(file2, 'r', encoding='utf-8') as f2:
                for i in f2:
                    f1.write(i)

        merge(file1, file2)

        a = np.load('unknow/emb/faceEmbedding.npy')
        b = np.load(rnd_name+'/emb/faceEmbedding.npy')
        c = []
        c = np.append(a, b, axis=0)
        np.save('unknow/emb/faceEmbedding.npy', c)

        unknow_dataset_emb, unknow_names_list = load_dataset(unknow_dataset, unknow_filename)
    shutil.rmtree(rnd_name)
    return json.dumps(images)


@app.route('/faceLogin/', methods=('POST', 'GET'))
def face_Login():
    detect = 'True'
    global dataset_emb, names_list, unknow_dataset_emb, unknow_names_list
    urls = request.form['urls']
    know = request.form['face']
    update = request.form['update']

    urls = urls.strip().split('&&&')
    print(urls)
    for url in urls:
        resp = urllib.request.urlopen(url)
        img = resp.read()
        b64 = base64.b64encode(img)
        b64 = str(b64, 'utf-8')



        if know == 'know':
            path = 'dataset/'
        else:
            path = 'unknow/'

        img = base64.b64decode(b64)
        nparr = np.frombuffer(img, np.uint8)
        img_cv2 = cv2.imdecode(nparr, cv2.COLOR_RGB2BGR)
        image = img_cv2[:, :, ::-1]

        str1 = 'rlsb1000'
        rnd_name = ''.join(random.choice("0123456789") for i in range(8))
        rnd_name = str1 + rnd_name

        pred_names, bboxes = face_recognition_image1(dataset_emb, names_list, face_detect, face_net, image)
        for img_name, bbox in zip(pred_names, bboxes):
            if img_name != 'unknow':
                know_path = 'dataset/images/'+img_name+'/'
                str1 = 'rlsb1000'
                rnd = ''.join(random.choice("0123456789") for i in range(8))
                rnd = str1 + rnd
                cv2.imwrite(know_path+rnd+'.jpg', img_cv2)
                return img_name

        try:
            img_names, bboxes = face_recognition_image1(unknow_dataset_emb, unknow_names_list, face_detect, face_net, image)
            for img_name, bbox in zip(img_names, bboxes):
                if img_name != 'unknow':
                    shutil.rmtree('unknow/images/'+img_name)
                    detect = 'unknow/images/'+img_name
                    rnd_name = img_name
                    create_face_embedding1(face_net, face_detect, unknow_zairu_path, unknow_out_emb_path, unknow_out_filename)
                    unknow_dataset_emb, unknow_names_list = load_dataset(unknow_dataset, unknow_filename)

        except:
            print('unknow not have person')


        try:
            os.mkdir(path+'images/'+rnd_name)
        except:
            print('以存在')
        mk_path = path+'images/'+rnd_name+'/'
        str1 = 'rlsb1000'
        rnd = ''.join(random.choice("0123456789") for i in range(8))
        rnd = str1 + rnd
        cv2.imwrite(mk_path+rnd+'.jpg', img_cv2)

    if update == 'True':
        if know == 'know':
            create_face_embedding1(face_net, face_detect, dataset_zairu_path, out_emb_path, out_filename)
            dataset_emb, names_list = load_dataset(dataset_path, filename)
        else:
            create_face_embedding1(face_net, face_detect, unknow_zairu_path, unknow_out_emb_path, unknow_out_filename)
            unknow_dataset_emb, unknow_names_list = load_dataset(unknow_dataset, unknow_filename)
    return rnd_name

@app.route('/faceUpdate/', methods=('POST', 'GET'))
def face_update():
    global dataset_emb, names_list, unknow_dataset_emb, unknow_names_list
    create_face_embedding1(face_net, face_detect, dataset_zairu_path, out_emb_path, out_filename)
    dataset_emb, names_list = load_dataset(dataset_path, filename)
    create_face_embedding1(face_net, face_detect, unknow_zairu_path, unknow_out_emb_path, unknow_out_filename)
    unknow_dataset_emb, unknow_names_list = load_dataset(unknow_dataset, unknow_filename)
    return 'True'




@app.route('/faceRemove/', methods=('POST', 'GET'))
def face_Remove():
    global dataset_emb, names_list, unknow_dataset_emb, unknow_names_list
    name_path = request.form['namePath']
    name = request.form['name']
    if name == 'unknow':
        path = 'unknow/images/'
        try:
            shutil.rmtree(path + name_path + '/')
            create_face_embedding1(face_net, face_detect, unknow_zairu_path, unknow_out_emb_path, unknow_out_filename)
            unknow_dataset_emb, unknow_names_list = load_dataset(unknow_dataset, unknow_filename)
            return '删除成功'
        except:
            return '没有这个人'

    else:
        path = 'dataset/images/'
        try:
            shutil.rmtree(path + name_path + '/')
            create_face_embedding1(face_net, face_detect, dataset_zairu_path, out_emb_path, out_filename)
            dataset_emb, names_list = load_dataset(dataset_path, filename)
            return '删除成功'
        except:
            return '没有这个人'





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8061)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--gpus', type=str, default='cpu')
    opt = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus  # 使用第二块GPU（从0开始）


    # model_path = 'models/20180408-102900'
    model_path = 'models/20180402-114759'
    dataset_path = 'dataset/emb/faceEmbedding.npy'
    dataset_zairu_path = 'dataset/images'
    filename = 'dataset/emb/name.txt'
    out_emb_path = 'dataset/emb/faceEmbedding.npy'
    out_filename = 'dataset/emb/name.txt'


    dataset_emb, names_list = load_dataset(dataset_path, filename)
    face_detect = face_recognition.Facedetection()
    face_net = face_recognition.facenetEmbedding(model_path)


    create_face_embedding1(face_net, face_detect, dataset_zairu_path, out_emb_path, out_filename)



    unknow_dataset = 'unknow/emb/faceEmbedding.npy'
    unknow_zairu_path = 'unknow/images'
    unknow_filename = 'unknow/emb/name.txt'
    unknow_out_emb_path = 'unknow/emb/faceEmbedding.npy'
    unknow_out_filename = 'unknow/emb/name.txt'





    create_face_embedding1(face_net, face_detect, unknow_zairu_path, unknow_out_emb_path, unknow_out_filename)

    unknow_dataset_emb, unknow_names_list = load_dataset(unknow_dataset, unknow_filename)

    app.run(host='0.0.0.0', port=opt.port)


