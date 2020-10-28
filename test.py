# import requests  # python发网络请求的模块(好用)
#
#
# # get请求=================
#
# url = 'http://test.aihw.club/jeecg-boot/fastdfs/fastDFS/fastDFSUpload' # 接口地址
# img_path = 'u=2925049010,1907642622&fm=26&gp=0.jpg'
#
# def upload(img_path):
#     with open(img_path, 'rb') as f_abs:
#         body = {
#             'file': f_abs
#         }
#         rep = requests.post(url=url,files=body) # 发送get请求
#         print(rep.json()['message']) # 返回字典类型
#         print(rep.text) # 返回字符串类型
#         return rep.json()['message']
# upload(img_path)

# import glob
#
# path_file_number = glob.glob('unknow/images/uzi/*.jpg')  # 或者指定文件下个数
#
# print(path_file_number)
# print(len(path_file_number))
#
# map = {}
#
# map['status'] = '0'
# map['True'] = 'True'
# print(map)

# import numpy as np
# import os, sys
#
# file1 = 'unknow/emb/name.txt'
# file2 = 'dataset/emb/name.txt'
#
# def merge(file1, file2):
#     f1 = open(file1, 'a+', encoding='utf-8')
#     with open(file2, 'r', encoding='utf-8') as f2:
#         for i in f2:
#             f1.write(i)
#
# merge(file1, file2)
#
#
# b = np.load('dataset/emb/faceEmbedding.npy')
# a = np.load('unknow/emb/faceEmbedding.npy')
#
# c = []
#
# c = np.append(a, b, axis=0)
#
# np.save('unknow/emb/faceEmbedding.npy', c)
#

#
# li = [1, 2, 3, 4, 4, 5, 6, 7]
# # print(li.index(4))
#
# import random
# str = 'rlsb'
# rnd = ''.join(random.choice("0123456789") for i in range(12))
# rnd = str + rnd
# print(rnd)


string = [123, 2144]
list1 = list(string)
print(list1)