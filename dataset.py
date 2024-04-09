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

def add_marker(img, bbox):
    low, high = bbox
    center = ((low+high)/2).astype(int)
    mark = np.zeros_like(img)
    new_img = np.copy(img)
    value = img.max() if new_img[center[0],center[1]]<(img.max()-img.min())/2 else img.min()
    new_img[low[0]:high[0]+1,low[1]]=value
    new_img[low[0]:high[0]+1,high[1]]=value
    new_img[low[0],low[1]:high[1]+1]=value
    new_img[high[0],low[1]:high[1]+1]=value
    return new_img

def convert_loc(coord, origin, space):
    displacement = np.array(coord[:3]).astype(float)-origin
    loc = np.round(displacement/space)[::-1]
    return loc

def convert_radius(coord, space):
    r = np.round(float(coord[-1])/2/space)[::-1]
    return r

def convert_bounding_box(coord, origin, space):
    center = convert_loc(coord, origin, space)
    rad = convert_radius(coord, space)
    low = np.round(center-rad)
    high = np.round(center+rad)
    return low, high

def mark_bbox(img, bbox):
    low, high = bbox
    low=low.astype(int)
    high=high.astype(int)

    marked_imgs = np.copy(img)
    for z in range(low[0],high[0]+1):
        marked_imgs[z] = add_marker(img[z],(low[1:],high[1:]))
    return marked_imgs

def export_as_gif(filename, image_array, frames_per_second=10, rubber_band=False):
    images = []
    image_array = (image_array-image_array.min())/(image_array.max()-image_array.min())
    for arr in image_array:
        im = Image.fromarray(np.uint8(arr*255))
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


from torch.utils.data import Dataset
import torch

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
               start[2]:start[2]+crop_size[2]]

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
               offset[2]:offset[2]+crop_size[2]], offset

def random_flip(img, axis):
    if np.random.rand()<0.5:
        return np.flip(img, axis=axis)
    else:
        return img

def collate_fn(examples):
    pixel_values = torch.cat([example["pixel_values"] for example in examples], 0)
    labels = torch.cat([example["labels"] for example in examples], 0)
    bbox = torch.cat([example["bbox"] for example in examples], 0)

    return {"pixel_values": pixel_values, "labels": labels, "bbox":bbox}

class LUNA16_Dataset(Dataset):
    mean = -718.0491779355748
    std = 889.6629126452339
    """
    https://luna16.grand-challenge.org/
    """
    def __init__(self, split=None, data_dir=".", crop_size=[40,128,128], patch_size=[4,16,16], return_bbox=False, samples_per_img = 4):
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

        self.return_bbox = return_bbox
        self.samples_per_img = samples_per_img
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fn = self.filenames[idx]
        uid = getUID(fn)
        image, origin, space = read_image(fn)
        coords = self.annotations[uid]
        patch_size_mm = self.patch_size * space[::-1]
        
        result = dict(pixel_values=[],labels=[],bbox=[],bbox_imgs=[])
        
        for i in range(self.samples_per_img):
            if len(coords)>0 and np.random.rand()<0.5:
                # crop with a nodule
                target_idx = np.random.randint(len(coords))
                coord = coords[target_idx]

                bbox = convert_bounding_box(coord, origin, space)
                cropped_img, offset = random_crop_around_3D(image, bbox, self.crop_size)
                offset_bbox = bbox[0] - offset, bbox[1] - offset
                target = np.concatenate([offset_bbox[0]/self.crop_size, offset_bbox[1]/self.crop_size])
                
                result["labels"].append(torch.tensor(1))
                result["bbox"].append(torch.tensor(target).to(torch.float32))
                
                # for debugging
                if self.return_bbox:
                    marked_imgs = mark_bbox(cropped_img, offset_bbox)
                    result["bbox_imgs"].append(marked_imgs)
            else:
                # random crop
                cropped_img = random_crop_3D(image, self.crop_size)
                result["labels"].append(torch.tensor(0))
                result["bbox"].append(torch.zeros(6))

                
            # random flip
            pixel_values = random_flip(cropped_img, 0)
            pixel_values = random_flip(pixel_values, 1)
            pixel_values = random_flip(pixel_values, 2)

            # normalize
            pixel_values = (pixel_values-LUNA16_Dataset.mean)/LUNA16_Dataset.std

            # to tensor
            pixel_values = torch.tensor(pixel_values.copy()).to(torch.float32)
            # add channel dim
            pixel_values = pixel_values.unsqueeze(0)
            result["pixel_values"].append(pixel_values)

        result["pixel_values"] = torch.stack(result["pixel_values"])
        result["labels"] = torch.stack(result["labels"])
        result["bbox"] = torch.stack(result["bbox"])

        return result
    