import torch
from torch.utils.data import Dataset
import os
import numpy as np
from glob import glob
import copy
import augmentator as aug

class DentalModelGenerator(Dataset):
    def __init__(self, data_dir=None, split_with_txt_path=None, scaling_range=None, rotation_range=None, rotation_axis=None, translation_range=None):
        self.data_dir = data_dir
        self.mesh_paths = glob(os.path.join(data_dir,"*_sampled_points.npy"))
        
        if split_with_txt_path:
            self.split_base_name_ls = []
            f = open(split_with_txt_path, 'r')
            while True:
                line = f.readline()
                if not line: break
                self.split_base_name_ls.append(line.strip())
            f.close()

            temp_ls = []
            for i in range(len(self.mesh_paths)):
                p_id = os.path.basename(self.mesh_paths[i]).split("_")[0]
                if p_id in self.split_base_name_ls:
                    temp_ls.append(self.mesh_paths[i])
            self.mesh_paths = temp_ls

        aug_list = []
        if scaling_range:
            aug_list.append(aug.Scaling(scaling_range))
        if rotation_range and rotation_axis:
            aug_list.append(aug.Rotation(rotation_range, rotation_axis))
        if translation_range:
            aug_list.append(aug.Translation(translation_range))
            
        if len(aug_list) > 0:
            self.aug_obj = aug.Augmentator(aug_list)
        else:
            self.aug_obj = None        


    def __len__(self):
        return len(self.mesh_paths)

    def __getitem__(self, idx):
        mesh_arr = np.load(self.mesh_paths[idx].strip())
        output = {}

        low_feat = mesh_arr.copy()[:,:6].astype("float32")
        
        seg_label = mesh_arr.copy()[:,6:].astype("int")
        seg_label -= 1 # -1 means gingiva, 0 means first incisor...
        
        if self.aug_obj:
            self.aug_obj.reload_vals()
            """
            if aug.Flip == type(self.aug_obj.augmentation_list[0]) and \
               self.aug_obj.augmentation_list[0].do_aug:
                    seg_label[seg_label>=8] = seg_label[seg_label>=8] - 805
                    seg_label[seg_label>=0] = seg_label[seg_label>=0] + 8
                    seg_label[seg_label<-500] = seg_label[seg_label<-500] + 805 - 8 
            """
            low_feat = self.aug_obj.run(low_feat)

        low_feat = torch.from_numpy(low_feat)
        low_feat = low_feat.permute(1,0)
        output["feat"] = low_feat

        seg_label = torch.from_numpy(seg_label)
        seg_label = seg_label.permute(1,0)
        output["gt_seg_label"] = seg_label

        output["aug_obj"] = copy.deepcopy(self.aug_obj)
        output["mesh_path"] = self.mesh_paths[idx] 

        return output

#for test
if __name__ == "__main__":
    import gen_utils as gu
    data_generator = DentalModelGenerator(
        data_dir="data_preprocessed_path",
        split_with_txt_path="base_name_train_fold.txt",
        scaling_range=[0.85, 1.15],
        rotation_range=[-30, 30],
        rotation_axis="fixed",
        translation_range=[-0.2, 0.2]
    )
    for batch in data_generator:
        for key in batch.keys():
            if type(batch[key]) == torch.Tensor:
                print(key, batch[key].shape)
            else:
                print(key, batch[key])
        gu.print_3d(gu.np_to_pcd_with_label(gu.torch_to_numpy(batch["feat"].T), gu.torch_to_numpy(batch["gt_seg_label"])))
        gu.print_3d(gu.torch_to_numpy(batch["feat"].T))
