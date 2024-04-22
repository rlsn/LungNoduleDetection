"""
rlsn 2024
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import glob, os
from PIL import Image

def read_image(image_file, meta=False):
    if meta:
        import SimpleITK as sitk
        # Read the MetaImage file
        image = sitk.ReadImage(image_file, imageIO="MetaImageIO")
        image_array = sitk.GetArrayFromImage(image)

        # print the image's dimensions
        return image_array, np.array(image.GetOrigin()), np.array(image.GetSpacing())
    else:
        # npy file
        re = np.load(image_file, allow_pickle=True).item()
        return re["img"], re["origin"], re["space"]

def preprocess(datadir):
    for i in range(10):
        filenames = glob.glob(f"{datadir}/subset{i}/*mhd")
        target_dir=f"{datadir}/subset{i}_npy"
        os.makedirs(target_dir, exist_ok=True)
        for fn in filenames:
            print("processing",fn)
            img, origin, space = read_image(fn,meta=True)
            bn = os.path.basename(fn)
            obj = dict(img=img,origin=origin,space=space)
            np.save(f"{target_dir}/{bn[:-3]}npy",obj)

def read_csv(fn):
    with open(fn,"r") as f:
        lines = [l.strip().split(",") for l in f.readlines()]
    return lines
  
def survey_dataset(datadir=".",npy=True):
    data_split = dict()
    for i in range(10):
        if npy:
            files = glob.glob(f"{datadir}/subset{i}_npy/*npy")
        else:
            files = glob.glob(f"{datadir}/subset{i}/*mhd")
        data_split[i]=files
    return data_split

def convert_loc(coord, origin, space):
    displacement = np.array(coord[:3]).astype(float)-origin
    loc = (displacement/space)[::-1]
    return loc

def convert_radius(coord, space):
    r = (float(coord[-1])/2/space)[::-1]
    return r

def convert_bounding_box(coord, origin, space):
    center = convert_loc(coord, origin, space)
    rad = convert_radius(coord, space)
    low = np.round(center-rad)
    high = np.round(center+rad)
    return low, high

def mark_bbox(img, bbox):
    img_size = np.array(img.shape)
    low, high = bbox[:3], bbox[3:]
    low=np.clip((low*img_size).astype(int), 0, img_size-1)    
    high=np.clip((high*img_size).astype(int), 0, img_size-1)
    bbox_imgs = np.zeros_like(img)
    zl,xl,yl = low
    zh,xh,yh = high

    for z in range(zl,zh+1):
        bbox_imgs[z,xl:xh+1,yl]=1
        bbox_imgs[z,xl:xh+1,yh]=1
        bbox_imgs[z,xl,yl:yh+1]=1
        bbox_imgs[z,xh,yl:yh+1]=1

    return bbox_imgs

def export_as_gif(filename, image_array, mark=None, frames_per_second=10, rubber_band=False):
    images = []
    image_array = (image_array-image_array.min())/(image_array.max()-image_array.min())

    for i, arr in enumerate(image_array):
        im = arr*255
        im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
        if mark is not None:
            im[:,:,0] += mark[i]*255
            im = np.clip(im,0,255)
        im = Image.fromarray(im.astype(np.uint8))
        images.append(im)
    if rubber_band:
        images += images[2:-1][::-1]
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=1000 // frames_per_second,
        loop=0,
    )

# compute mean and std
def compute_stats(dataset):
    N = 0
    sum = 0
    for fn in dataset.filenames:
        image,_,_=read_image(fn)
        sum += np.sum(image)
        N+=np.prod(image.shape)
    mean = sum/N
    N = 0
    sum = 0
    for fn in dataset.filenames:
        image,_,_=read_image(fn)
        sum += np.sum((image-mean)**2)
        N+=np.prod(image.shape)
    std = np.sqrt(sum/N)
    return mean, std

def getUID(filename):
    return os.path.basename(filename)[:-4]

def random_crop_3D(img, crop_size):
    size = np.array(img.shape)
    high = size-crop_size
    start = [np.random.randint(0, high=high[0]),
           np.random.randint(0, high=high[1]),
           np.random.randint(0, high=high[2])]
    return img[start[0]:start[0]+crop_size[0],
               start[1]:start[1]+crop_size[1],
               start[2]:start[2]+crop_size[2]], start

def random_crop_around_3D(img, bbox, crop_size, margin=[5,20,20]):
    im_size = np.array(img.shape)
    blow, bhigh = bbox
    blow = blow.astype(int)
    bhigh = bhigh.astype(int)
    margin = np.array(margin)
    low = np.minimum(np.maximum(bhigh+margin-crop_size, 0), im_size-crop_size)
    high = np.minimum(np.maximum(blow-margin, low), im_size-crop_size)+1
    offset = [np.random.randint(low[0], high=high[0]),
           np.random.randint(low[1], high=high[1]),
           np.random.randint(low[2], high=high[2])]
    return img[offset[0]:offset[0]+crop_size[0],
               offset[1]:offset[1]+crop_size[1],
               offset[2]:offset[2]+crop_size[2]], np.array(offset)

def random_flip(img, bbox, axis=0):
    if np.random.rand()<0.5:
        tmp=1-bbox[axis+3]
        bbox[axis+3]=1-bbox[axis]
        bbox[axis]=tmp
        return np.flip(img, axis=axis), bbox
    else:
        return img, bbox

def iou_3d(bbox_pred,bbox):
    if len(bbox_pred.shape)==1:
        bbox_pred = np.expand_dims(bbox_pred,0)
        bbox = np.expand_dims(bbox,0)
    ilow = np.maximum(bbox_pred,bbox)[:,:3]
    ihigh = np.minimum(bbox_pred,bbox)[:,3:]
    i_sides = np.maximum(ihigh-ilow,0)
    i_vol = np.prod(i_sides,-1)
    o_vol = np.prod(bbox_pred[:,3:]-bbox_pred[:,:3],-1)+np.prod(bbox[:,3:]-bbox[:,:3],-1)-i_vol
    return (i_vol/o_vol).mean()

def sliding_window_3d(x, window_size, stride_size):
    """
    x: [d,w,h]
    window_size: [d,w,h]
    stride_size: [d,w,h]
    return: [b,d,w,h]
    """
    
    window_offsets = [list(np.arange(x.shape[i]-window_size[i])[::stride_size[i]])+[x.shape[i]-window_size[i]] for i in range(3)]
    offsets = []
    outputs = []
    for i in window_offsets[0]:
        for j in window_offsets[1]:
            for k in window_offsets[2]:
                offsets.append([i,j,k])
                outputs.append(x[i:i+window_size[0],j:j+window_size[1],k:k+window_size[2]])
    
    return np.array(offsets), np.array(outputs)

def collate_fn(examples):
    pixel_values = torch.cat([example["pixel_values"] for example in examples], 0)
    labels = torch.cat([example["labels"] for example in examples], 0)
    bbox = torch.cat([example["bbox"] for example in examples], 0)

    return {"pixel_values": pixel_values, "labels": labels, "bbox":bbox}

class LUNA16_Dataset(Dataset):
    mean = -775.657161489884
    std = 962.3208802005623
    max_sampling_times = 64
    """
    https://luna16.grand-challenge.org/
    """
    def __init__(self, split=None, data_dir=".", crop_size=[40,128,128], patch_size=[4,16,16], samples_per_img = 8):
        annotations_csv = read_csv(f"{data_dir}/annotations.csv")[1:]
        data_subsets = survey_dataset(data_dir)
        # to filenames
        if split is None:
            split = np.arange(10) # all subsets
        self.filenames = []
        for s in split:
            self.filenames+=data_subsets[s]
        # annotation to dict
        self.annotations = dict([(getUID(k),[]) for k in self.filenames])
        for entry in annotations_csv:
            self.annotations.setdefault(entry[0], [])
            self.annotations[entry[0]]+=[entry[1:]]
        
        self.crop_size = np.array(crop_size)
        self.patch_size = np.array(patch_size)

        self.samples_per_img = samples_per_img
        self.max_sampling_times = max(LUNA16_Dataset.max_sampling_times, self.samples_per_img)
        self.train = True

    def train(self):
        self.train = True
        return self
        
    def eval(self):
        self.train = False
        return self
        
    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, idx):
        if self.train:
            return self._get_train_samples(idx)
        else:
            return self._get_eval_samples(idx)
            
    def _get_eval_samples(self, idx):
        fn = self.filenames[idx]
        uid = getUID(fn)
        image, origin, space = read_image(fn)
        coords = self.annotations[uid]
        patch_size_mm = self.patch_size * space[::-1]
        result = dict(pixel_values=[],labels=[],bbox=[])

        bboxes = []
        for coord in coords:
            bboxes.append(np.concatenate(convert_bounding_box(coord, origin, space),0))
        bboxes = np.array(bboxes)
        
        # get patches with sliding window
        offsets, pixel_values=sliding_window_3d(image,self.crop_size,(self.crop_size*0.75).astype(int))

        # normalize        
        pixel_values = (pixel_values-LUNA16_Dataset.mean)/LUNA16_Dataset.std
        
        result["pixel_values"] = torch.tensor(pixel_values,dtype=torch.float32).unsqueeze(1)
        result["offsets"] = torch.tensor(offsets,dtype=torch.int32)
        result["bbox"] = torch.tensor(bboxes,dtype=torch.int32)        
        result["coords"] = np.array(coords).astype(float)
        result["origin"] = origin
        result["space"] = space
        result["uid"] = uid
                
        return result
        
    def _get_train_samples(self, idx):
        fn = self.filenames[idx]
        uid = getUID(fn)
        image, origin, space = read_image(fn)
        coords = self.annotations[uid]
        patch_size_mm = self.patch_size * space[::-1]
        
        result = dict(pixel_values=[],labels=[],bbox=[])
        
        bboxes = []
        for coord in coords:
            bboxes.append(convert_bounding_box(coord, origin, space))

        i = 0
        while i<self.samples_per_img:
            if i>self.max_sampling_times:
                break
            if len(bboxes)>0 and np.random.rand()<0.5:
                # crop a patch with a random nodule
                # TODO: needs to account for the possibility that multiple nodules are contained
                bbox = bboxes[np.random.randint(len(bboxes))]

                cropped_img, offset = random_crop_around_3D(image, bbox, self.crop_size)
                offset_bbox = bbox[0] - offset, bbox[1] - offset
                target = np.concatenate([offset_bbox[0]/self.crop_size, offset_bbox[1]/self.crop_size])
                
                result["labels"].append(torch.tensor(1))
                bbox = torch.tensor(target).to(torch.float32)
                i+=1
            else:
                # random crop a negative patch
                cropped_img, offset = random_crop_3D(image, self.crop_size)
                img_bbox = np.concatenate([offset, offset+self.crop_size],0)
                img_bbox = np.expand_dims(img_bbox, 0)
                if len(bboxes)>0:
                    # account for the possibility that a positive is contained
                    iou = [iou_3d(img_bbox, np.expand_dims(np.concatenate(bbox,0),0)) for bbox in bboxes]
                    if np.sum(iou)>0:
                        continue
                result["labels"].append(torch.tensor(0))
                bbox = torch.zeros(6)
                i+=1
                
            # random flip (also flip the bbox)
            pixel_values, bbox = random_flip(cropped_img, bbox, 0)
            pixel_values, bbox = random_flip(pixel_values, bbox, 1)
            pixel_values, bbox = random_flip(pixel_values, bbox, 2)

            # normalize
            pixel_values = (pixel_values-LUNA16_Dataset.mean)/LUNA16_Dataset.std

            # to tensor
            pixel_values = torch.tensor(pixel_values.copy()).to(torch.float32)
            # add channel dim
            pixel_values = pixel_values.unsqueeze(0)
            result["pixel_values"].append(pixel_values)

            result["bbox"].append(bbox)


        result["pixel_values"] = torch.stack(result["pixel_values"])
        result["labels"] = torch.stack(result["labels"])
        result["bbox"] = torch.stack(result["bbox"])

        return result