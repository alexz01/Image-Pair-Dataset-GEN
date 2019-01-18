from image_pair import ImagePair

class AndDS(ImagePair):

    def _key_gen(self, df):
        '''
        Return expression to build label for two images being of same type
        '''
        return (df['file1'].str[0:4] == df['file2'].str[0:4]) * 1


if __name__ == '__main__':
    andDS = AndDS(img_dir='./testdata')
    andDS.gen_pair_list(save_list=True,validation_size=5,test_size=5)
    'print first 10 training set items'
    print(andDS.tr_fp[:10])
    