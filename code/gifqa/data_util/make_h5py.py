import numpy as np
import h5py
import pandas as pd
import glob
import ipdb
import cPickle as pkl
import hickle as hkl

DATA_PATH = '/data1/yj/tgif/vid_c3d_concat/'
data_list = glob.glob(DATA_PATH + '*/*.c3d')
dataframe = pd.DataFrame().from_csv('/data1/yj/tgif/DataFrame/tgif-new-group.tsv', sep='\t')
#ipdb.set_trace()
'''
with h5py.File('/data1/yj/tgif/features/POOL5.h5','w') as hf:
    for items in dataframe.iterrows():
        try:
            dat =np.squeeze(np.load(open('/data1/yj/tgif/vid_resnet2/'+ items[1]['web_url'].split('/')[-1].split('.')[0]+'/pool5.npy')))
        except:
            print items[1]['web_url'].split('/')[-1]
            continue
        if len(dat) > 1000:
            #ipdb.set_trace()
            dat = dat[:10]
            print dat.shape
            print items[1]['web_url'].split('/')[-1]
        hf.create_dataset(str(items[0]), data=dat)
'''
'''
with h5py.File('/data1/yj/tgif/features/TGIF_C3D.h5','w') as hf:
    for items in dataframe.iterrows():
        try:
            dat =np.squeeze(pkl.load(open('/data1/yj/tgif/vid_c3d_concat/'+ items[1]['web_url'].split('/')[-1].split('.')[0]+'/fc.c3d')))
        except:
            print items[1]['web_url'].split('/')[-1]
            continue
        if len(list(dat.shape)) > 1 and len(dat) > 1000:
            print dat.shape
            dat = dat[:10]
            #print dat.shape
            print items[1]['web_url'].split('/')[-1]
        hf.create_dataset(str(items[0]), data=dat)
'''
with h5py.File('/data1/yj/tgif/features/TGIF_VGG_fc7.h5','w') as hf:
    for items in dataframe.iterrows():
        try:
            dat =np.squeeze(np.load('/data1/yj/tgif/vgg_feat/'+ items[1]['web_url'].split('/')[-1].split('.')[0]+'/vgg_fc7.npy'))
        except:
            print items[1]['web_url'].split('/')[-1]
            continue
        if len(list(dat.shape)) > 1 and len(dat) > 3000:
            print dat.shape
            dat = dat[:10]
            #print dat.shape
            print items[1]['web_url'].split('/')[-1]
        hf.create_dataset(str(items[0]), data=dat)

