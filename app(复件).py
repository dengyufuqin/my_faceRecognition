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

app = Flask(__name__)





def remove_cache_image(cache_path):
    fileList = os.listdir(cache_path)

    for cache_name in fileList:
        image_path = cache_path+cache_name
        os.remove(image_path)


@app.route('/faceRecognition/', methods=('POST', 'GET'))
def face_rec():
    global unknow_dataset_emb, unknow_names_list
    url = request.form['url']
    resp = urllib.request.urlopen(url)
    img = resp.read()
    b64 = base64.b64encode(img)
    b64 = str(b64, 'utf-8')

    resualt = []
    img = base64.b64decode(b64)
    nparr = np.frombuffer(img, np.uint8)
    img_cv2 = cv2.imdecode(nparr, cv2.COLOR_RGB2BGR)

    img_names = face_recognition_image1(dataset_emb, names_list, face_detect, face_net, img_cv2)
    img_name = img_names[0]
    if img_name == 'unknow':
        resualt.append(0)
    else:
        resualt.append(1)
    if img_name == 'unknow':
        try:
            unknow_img_names = face_recognition_image1(unknow_dataset_emb, unknow_names_list, face_detect, face_net, img_cv2)
            unknow_img_name = unknow_img_names[0]
            if unknow_img_name != 'unknow':
                rnd = ''.join(random.sample('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', 32))
                unkonw_path = 'unknow/images/'+unknow_img_name+'/'+rnd+'.jpg'
                cv2.imwrite(unkonw_path, img_cv2)
                img_name = unknow_img_name
            else:
                rnd_path = ''.join(random.sample('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', 32))
                os.mkdir('unknow/images/'+rnd_path)
                unknow_path = 'unknow/images/'+rnd_path+'/'+rnd_path+'.jpg'
                cv2.imwrite(unknow_path, img_cv2)
                img_name = rnd_path
                create_face_embedding(model_path, unknow_zairu_path, unknow_out_emb_path, unknow_out_filename)
                unknow_dataset_emb, unknow_names_list = load_dataset(unknow_dataset, unknow_filename)
        except:
            rnd_path = ''.join(random.sample('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', 32))
            os.mkdir('unknow/images/' + rnd_path)
            unknow_path = 'unknow/images/' + rnd_path + '/' + rnd_path + '.jpg'
            cv2.imwrite(unknow_path, img_cv2)
            img_name = rnd_path
            create_face_embedding(model_path, unknow_zairu_path, unknow_out_emb_path, unknow_out_filename)
            unknow_dataset_emb, unknow_names_list = load_dataset(unknow_dataset, unknow_filename)
    resualt.append()
    return img_name


@app.route('/faceLogin/', methods=('POST', 'GET'))
def face_Login():

    global dataset_emb, names_list, unknow_dataset_emb, unknow_names_list
    url = request.form['url']
    name_path = request.form['namePath']
    resp = urllib.request.urlopen(url)
    img = resp.read()
    b64 = base64.b64encode(img)
    b64 = str(b64, 'utf-8')


    img = base64.b64decode(b64)
    nparr = np.frombuffer(img, np.uint8)
    img_cv2 = cv2.imdecode(nparr, cv2.COLOR_RGB2BGR)


    try:
        os.mkdir('dataset/images/'+name_path)
    except:
        print('以存在')
    mk_path = 'dataset/images/'+name_path+'/'
    rnd = ''.join(random.sample('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', 32))
    cv2.imwrite(mk_path+rnd+'.jpg', img_cv2)
    try:
        img_names = face_recognition_image1(unknow_dataset_emb, unknow_names_list, face_detect, face_net, img_cv2)
        img_name = img_names[0]
        if img_name != 'unknow':
            shutil.rmtree('unknow/images/'+img_name)
            create_face_embedding(model_path, unknow_zairu_path, unknow_out_emb_path, unknow_out_filename)
            unknow_dataset_emb, unknow_names_list = load_dataset(unknow_dataset, unknow_filename)
    except:
        print('unknow not have person')
    create_face_embedding(model_path, dataset_zairu_path, out_emb_path, out_filename)
    dataset_emb, names_list = load_dataset(dataset_path, filename)
    return name_path


@app.route('/faceRemove/', methods=('POST', 'GET'))
def face_Remove():
    global dataset_emb, names_list, unknow_dataset_emb, unknow_names_list
    name_path = request.form['namePath']
    name = request.form['name']
    if name == 'unknow':
        path = 'unknow/images/'
        try:
            shutil.rmtree(path + name_path + '/')
            create_face_embedding(model_path, unknow_zairu_path, unknow_out_emb_path, unknow_out_filename)
            unknow_dataset_emb, unknow_names_list = load_dataset(unknow_dataset, unknow_filename)
            return '删除成功'
        except:
            return '没有这个人'

    else:
        path = 'dataset/images/'
        try:
            shutil.rmtree(path + name_path + '/')
            create_face_embedding(model_path, dataset_zairu_path, out_emb_path, out_filename)
            dataset_emb, names_list = load_dataset(dataset_path, filename)
            return '删除成功'
        except:
            return '没有这个人'



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8061)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--gpus', type=str, default='0')
    opt = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus  # 使用第二块GPU（从0开始）


    model_path = 'models/20180408-102900'
    dataset_path = 'dataset/emb/faceEmbedding.npy'
    dataset_zairu_path = 'dataset/images'
    filename = 'dataset/emb/name.txt'
    out_emb_path = 'dataset/emb/faceEmbedding.npy'
    out_filename = 'dataset/emb/name.txt'

    create_face_embedding(model_path, dataset_zairu_path, out_emb_path, out_filename)

    dataset_emb, names_list = load_dataset(dataset_path, filename)
    face_detect = face_recognition.Facedetection()
    face_net = face_recognition.facenetEmbedding(model_path)

    unknow_dataset = 'unknow/emb/faceEmbedding.npy'
    unknow_zairu_path = 'unknow/images'
    unknow_filename = 'unknow/emb/name.txt'
    unknow_out_emb_path = 'unknow/emb/faceEmbedding.npy'
    unknow_out_filename = 'unknow/emb/name.txt'

    create_face_embedding(model_path, unknow_zairu_path, unknow_out_emb_path, unknow_out_filename)

    unknow_dataset_emb, unknow_names_list = load_dataset(unknow_dataset, unknow_filename)

    app.run(host='0.0.0.0', port=opt.port)


