"""Offline model pseudo labels aligned to exactly the strict baseline augmentation."""
from pathlib import Path
import os,sys
import numpy as np
import imageio.v2 as imageio
from .voc import VOC12ClsDataset
from . import transforms

def training_gt_guard(event,values):
    # DataLoader workers read training images/pseudo labels only. Validation runs
    # in separate workers and legitimately reads GT for evaluation.
    if event=='open' and isinstance(values[0],(str,bytes,os.PathLike)):
        if {'SegmentationClass','SegmentationClassAug'} & set(Path(os.fsdecode(values[0])).parts):
            raise RuntimeError('Training worker attempted to read pixel ground truth')

class VOC12TeacherDataset(VOC12ClsDataset):
    def __init__(self,*args,teacher_dir,**kwargs):
        super().__init__(*args,**kwargs)
        self.teacher_dir=Path(teacher_dir).resolve()
        assert not {'SegmentationClass','SegmentationClassAug'} & set(self.teacher_dir.parts)
        missing=[str(n) for n in self.name_list if not (self.teacher_dir/(str(n)+'.png')).is_file()]
        if missing:raise FileNotFoundError(f'{len(missing)} missing teacher maps, first: {missing[:3]}')
        self._guard_pid=None

    def __getitem__(self,index):
        if self._guard_pid!=os.getpid():sys.addaudithook(training_gt_guard);self._guard_pid=os.getpid()
        name=str(self.name_list[index]);image=np.asarray(imageio.imread(os.path.join(self.img_dir,name+'.jpg')))
        target=np.asarray(imageio.imread(self.teacher_dir/(name+'.png')));cls_label=self.label_list[name]
        allowed=np.r_[0,np.flatnonzero(cls_label)+1,255]
        assert target.shape==image.shape[:2] and np.isin(target,allowed).all()
        image,target=transforms.random_scaling(image,label=target,scale_range=self.rescale_range)
        image,target=transforms.random_fliplr(image,label=target)
        image,target,box=transforms.random_crop(image,label=target,crop_size=self.crop_size,
            mean_rgb=[0,0,0],ignore_index=self.ignore_index,category_guided=False)
        image=np.transpose(transforms.normalize_img(image),(2,0,1))
        return name,image,cls_label,box,target.astype(np.int64)
