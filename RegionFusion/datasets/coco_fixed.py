"""COCO inputs from the configured complete task split lists. Pixel GT is inaccessible to training workers."""
from pathlib import Path
import os,sys
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from . import transforms
from .coco import class_list

def no_training_gt(event,values):
    if event=='open' and isinstance(values[0],(str,bytes,os.PathLike)) and {'SegmentationClass','SegmentationClassAug'}&set(Path(os.fsdecode(values[0])).parts):
        raise RuntimeError('COCO training/teacher worker attempted pixel GT read')

class FixedCoco(Dataset):
    def __init__(self,data,lists,split='train',teacher_dir=None,augment=False):
        self.data=Path(data);self.lists=Path(lists);self.split=split;self.augment=augment
        self.names=(self.lists/(split+'.txt')).read_text().split();self.name_list=self.names
        self.tags=np.load(self.lists/'cls_labels_onehot.npy',allow_pickle=True).item()
        self.teacher=Path(teacher_dir) if teacher_dir else None;self._guard_pid=None
        assert len(self.names)==len(set(self.names))
        if self.teacher:assert not {'SegmentationClass','SegmentationClassAug'}&set(self.teacher.parts)
    def __len__(self):return len(self.names)
    def __getitem__(self,index):
        train=self.split=='train';name=self.names[index]
        if train and self._guard_pid!=os.getpid():sys.addaudithook(no_training_gt);self._guard_pid=os.getpid()
        image=np.array(Image.open(self.data/'JPEGImages'/('train' if train else 'val')/(name+'.jpg')).convert('RGB'))
        tags=np.asarray(self.tags[name],dtype=np.float32);assert tags.shape==(80,)
        if train:
            target=np.array(Image.open(self.teacher/(name+'.png'))) if self.teacher else np.full(image.shape[:2],255,np.uint8)
            assert target.shape==image.shape[:2] and np.isin(target,np.r_[0,np.flatnonzero(tags)+1,255]).all()
            if self.augment:
                image,target=transforms.random_scaling(image,label=target,scale_range=[.5,2.])
                image,target=transforms.random_fliplr(image,label=target)
                image,target,box=transforms.random_crop(image,label=target,crop_size=320,mean_rgb=[0,0,0],ignore_index=255,category_guided=False)
            else:box=np.array([0,image.shape[0],0,image.shape[1]])
            image=np.transpose(transforms.normalize_img(image),(2,0,1)).copy()
            return name,image,tags,np.asarray(box),target.astype(np.int64)
        target=np.array(Image.open(self.data/'SegmentationClass/val'/(name+'.png')))
        assert target.shape==image.shape[:2] and np.isin(target,np.r_[np.arange(81),255]).all()
        return name,np.transpose(transforms.normalize_img(image),(2,0,1)).copy(),target.astype(np.int64),tags
