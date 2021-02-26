import pickle
from os import path, mkdir
import os
from shutil import copy, move
from string import Template
import numpy as np
import cv2

base = pickle.load(open("all.pkl", "rb"))
images = base['images']
classes = base['targets']

#transformando de matriz para imagens e as salvando do disco
for i in range(0,len(images)):
    nome = str(i)+'.jpg'
    a = images[i]
    c1 = a[0][0:51][0:51]
    c2 = a[1][0:51][0:51]
    c3 = a[2][0:51][0:51]
    matriz = np.dstack((c3, c2, c1))
    cv2.imwrite('data/'+nome, matriz)

#função que copia as imagens para as pastas supernovas (sn) e não supernovas (nsn)
def copy_images(df, dest):
    src_path = Template('data/$name.jpg')
    dest_path = Template('data/$folder/').substitute(folder=dest)
    print("to aqui")
    if path.isdir(dest_path) is False:
        print("saindo")
        return

    for i in range (0,len(nsn)):
        if(nsn[i] == True):
            copy(src_path.substitute(name=i), dest_path)

sn = classes >= 0.9
nsn = classes <= 0.2


(3,51,51)
(51,51,3)






