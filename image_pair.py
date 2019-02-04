import pandas as pd
import numpy as np
import cv2
import os
from abc import ABC, abstractmethod

class ImagePair(ABC):
    '''
    Class used to generate Image pair datasets.
    '''
    def __init__(self, img_dir='dataset'):
        """
        Parameters
        -----------
        img_dir : directory path containing images 
        
        """
        assert os.path.isdir(img_dir), "Image directory doesn't exist"
        self.img_dir = img_dir
        self.pairs_fp = os.path.join(img_dir,'file_pair.csv')
        self.tr_fp = os.path.join(img_dir,'train_pair.csv')
        self.va_fp = os.path.join(img_dir,'validation_pair.csv')
        self.ts_fp = os.path.join(img_dir,'test_pair.csv')
        self.tr_li = None
        self.va_li = None
        self.ts_li = None
        
    def gen_pair_list(self, save_list=False, validation_size=1500, test_size=2000):
        """
        Generate file pair dataset based on expression in implemented _key_gen method
        
        Parameters
        -----------
        save_list : Save the pair files. boolean
        validation_size : Size of validation set in positive and negative list
        test_size : Size of test set in positive and negative list
        
        Returns:
        ---------
        Dataframe : train, validation and test set
        """
        flist = pd.DataFrame(self._get_file_list(), columns=['file']) 
        flist['key'] = 0
        print("Generating image pair list, ensure there's sufficient free memory")
        flist = pd.merge(flist, flist, how='inner', on='key', suffixes=['1','2'], )
        flist['key'] = self._key_gen(flist)
        flist = flist[['file1', 'file2', 'key']]
        
        pos_li = self._pd_reshuffle(flist[ flist['key']==1 ])
        neg_li = self._pd_reshuffle(flist[ flist['key']==0 ].sample(pos_li.shape[0]))

        tr_end_index = pos_li.shape[0] - (validation_size + test_size)
        va_end_index = tr_end_index + validation_size

        self.tr_li = pos_li[ :tr_end_index].append(neg_li[ :tr_end_index])
        self.tr_li = self._pd_reshuffle(self.tr_li)

        self.va_li = pos_li[tr_end_index : va_end_index].append(neg_li[tr_end_index : va_end_index])
        self.va_li = self._pd_reshuffle(self.va_li)

        self.ts_li = pos_li[va_end_index: ].append(neg_li[va_end_index: ])
        self.ts_li = self._pd_reshuffle(self.ts_li)
        ds = self._pd_reshuffle(pos_li.append(neg_li))
        if save_list:
            ds.to_csv(self.pairs_fp, index=False)
            self.tr_li.to_csv(self.tr_fp, index=False)
            print('saved training set pair to %s' % self.tr_fp)
            self.va_li.to_csv(self.va_fp, index=False)
            print('saved validation set pair to %s' % self.va_fp)
            self.ts_li.to_csv(self.ts_fp, index=False)
            print('saved training set pair to %s' % self.ts_fp)
        else:            
            return self.tr_li, self.va_li, self.tr_li


    @abstractmethod
    def _key_gen(self, df):
        '''
        Expression to generate label for two images eg. Both images are of same type = 1 else 0
        '''
        pass


    def _pd_reshuffle(self, df):
        return df.sample(frac=1).reset_index(drop=True)

    
    def _get_file_list(self):
        fl = os.listdir(self.img_dir)
        # filter png files
        
        return [i for i in fl if 'png' in i]
    
    
    def load_ds(self):
        '''
        load dataset from csv files.
        '''
        assert os.path.isfile(self.tr_fp), "CSV files doesn't exist"
        self.tr_li = pd.read_csv(self.tr_fp, )
        self.va_li = pd.read_csv(self.va_fp, )
        self.ts_li = pd.read_csv(self.ts_fp, )
        return self.tr_li, self.va_li, self.ts_li
    
    
    @staticmethod
    def get_merged_image( img1_fp, img2_fp, axis=0, h=150, w=200, cv2_imread_flag=0):
        '''
        decode image pair from dataset and return a single merged image.
        By default return image of 3:4 aspect ratio image dim(150,200)
        Parameters
        -----------
        index : index of image pair in dataset
        df : dataset dataframe
        axis : merge two image on axis
        cv2_imread_flag : one of -1 (default decode), 0 (grayscale decode), or 1 (BGR decode)
        
        Returns
        '''
        img1 = cv2.imread(img1_fp, cv2_imread_flag)
        img2 = cv2.imread(img2_fp, cv2_imread_flag)
        if axis==0:
            hi = h//2
            wi = w
        elif axis==1:
            hi = h
            wi = w//2
        elif axis==2:
            hi=h
            wi=w
        else:
            raise ValueError('Axis should be 0 or 1')
            
        img1 = ImagePair.padded_reshape(img1, h = hi, w= wi)
        img2 = ImagePair.padded_reshape(img2, h = hi, w= wi)
        img1 = np.expand_dims(img1,axis=2)
        img2 = np.expand_dims(img2,axis=2)
            
        return np.append(img1, img2, axis = axis)
    
    
    def get_batch(self, ds='train', batch_size=32, img_shape=[60,80,1], merge_axis=0, reshuffle=True, flat=False):
        '''
        Yields batches of dataset of batch size

        Parameters:
        ds : dataset to be batched one of [train, valid, test]
        batch_size : size of batch
        img_shape : shape of images in batch
        merge_axis : merge axis of two images in pair
        reshuffle : reshuffle after each epoch
        flat : flat image to vector

        Returns:
        Tuple : [images, labels]  
        '''
        
        assert isinstance(ds, str), 'ds needs to be string'
        assert ds.lower() in ['train', 'validation', 'valid', 'test', 'tr','va','ts'], 'ds needs to be one of %s' %(['train', 'validation', 'valid', 'test', 'tr','va','ts'])
        
        if ds.lower() in ['train', 'tr']:
            df = self.tr_li
        elif ds.lower() in ['validation', 'valid', 'va']:
            df = self.va_li
        else:
            df = self.ts_li
        
                
        while True:
            if reshuffle:
                indexes = np.random.permutation( df.index.tolist() )
                
            else:
                indexes = df.index.tolist()
           
            for i in range(len(indexes)//batch_size+1):
                x = np.zeros([batch_size]+img_shape)
                y = np.zeros([batch_size, 2])
                for j, row in enumerate(df.iloc[indexes[i*batch_size:(i+1)*batch_size]].as_matrix()):
                    #print('row',row)
                    img = self.get_merged_image(os.path.join(self.img_dir,row[0]),os.path.join(self.img_dir,row[1]),h=img_shape[0],w=img_shape[1], axis=merge_axis, )
                    lbl = row[2]
                    x[j] = img
                    y[j] = np.eye(2)[lbl].tolist()
                    
                if flat:
                    x = np.reshape(x, [batch_size,-1])
                yield x,y
                    
            
            
            
    @staticmethod           
    def padded_reshape(img, h=60,w=200, inter=cv2.INTER_AREA, border_type= cv2.BORDER_WRAP):
        '''
        Reshape image to size h,w with padding to keep aspect ratio.
        
        Parameters
        -----------
        img : buffer image (numpy matrix)
        h : reshape height
        w : reshape height
        inter : interpolation method cv2.INTER_*
        border_type : border type cv2.BORDER_*
        
        Returns
        --------
        image buffer : numpy array of image
        '''
        assert h is not None and w is not None, "h/w cannot be none"
        assert h > 0 and w > 0 , "h/w cannot be < 0"

        hi,wi = img.shape[:2]

        if (h - hi)/(h + hi) < (w - wi)/(w + wi):
            #"match h to hi and scale wi with h padding remaining to match w to wi"
            w_re = int(h/hi*wi)
            h_re = h
            pad_l = int((w - w_re)/2)
            pad_r = w - (w_re + pad_l)
            pad_t = 0
            pad_b = 0        
        else:
            'other way around'
            w_re = w
            h_re = int(w/wi*hi)
            pad_l = 0
            pad_r = 0
            pad_t = int((h - h_re)/2)
            pad_b = h - (h_re + pad_t)
            pass

        img = cv2.resize(img, dsize=(w_re,h_re), interpolation = inter)
        img = cv2.copyMakeBorder(img, top=pad_t, bottom=pad_b, left = pad_l, right=pad_r, borderType=border_type)
        return img/255
    
    
    def pair_img(self, img1, img2, axis=0):
        return np.append(img1, img2, axis)
  