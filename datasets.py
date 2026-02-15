import os
import numpy as np

from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import random
from sklearn.model_selection import train_test_split

import augment4sig as aug

class CustomTransform:
    def __init__(self, signal_length, is_training=True, jitter_strength=0.1, flip_prob=0.5):
        self.signal_length = signal_length
        self.is_training = is_training
        self.jitter_strength = jitter_strength
        self.flip_prob = flip_prob

    def __call__(self, x):
        if self.is_training:
            # Add random jitter
            from scipy import signal
            y = np.zeros((x.shape[0], self.signal_length))
            for i in range(x.shape[0]):
                y[i] = signal.resample(x[i], self.signal_length)
            y = torch.from_numpy(y.copy()).float()
            jitter = torch.randn_like(y) * self.jitter_strength
            y = y + jitter

            # Random flip
            if random.random() < self.flip_prob:
                y = torch.flip(y, dims=[-1])

            # Ensure the signal length is correct
            if y.shape[-1] != self.signal_length:
                y = nn.functional.interpolate(x.unsqueeze(0), size=self.signal_length, mode='linear', align_corners=False).squeeze(0)

        return y

def create_transform(signal_length, is_training=True, jitter_strength=0.1, flip_prob=0.5):
    return CustomTransform(
        signal_length=signal_length,
        is_training=is_training,
        jitter_strength=jitter_strength,
        flip_prob=flip_prob
    )

class TriaxialSignalDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data :  containing triaxial signal data of shape (num_samples, 3, signal_length)
            labels :  containing labels of shape (num_samples,)
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        signal = self.data[idx]
        label = self.labels[idx]
        if self.transform is not None:
            signal = self.transform(signal)
        return signal, label



def build_dataset(args):
    if args.data == "SHL2024":   
        acc_x = np.loadtxt("../../SHL/SHL_2024/train/Hips/Acc_x.txt")
        acc_y = np.loadtxt("../../SHL/SHL_2024/train/Hips/Acc_y.txt")
        acc_z = np.loadtxt("../../SHL/SHL_2024/train/Hips/Acc_z.txt")
        label = np.loadtxt("../../SHL/SHL_2024/train/Hips/Label.txt", dtype=np.int64)
        label = label[:,250]
        label = np.array([i-1 for i in label])
    
        acc_xyz = np.stack([acc_x, acc_y, acc_z], axis=1)
        nb_classes = 8
    elif args.data == "SHL2023":   
        acc_xyz = np.load("SHL_2023/SHL/100hz_5.0s_overlap0.0s/SHL/train/Hips_Acc.npy")
        label = np.load("SHL_2023/SHL/100hz_5.0s_overlap0.0s/SHL/train/Hips_Label.npy")

        label= np.array([i-1 for i in label])
        nb_classes = 8
    elif args.data == "SHL2023_test":   
        #500サンプル(ウィンドウ幅), 100Hz(1秒で100個)*5秒, 196104フレーム
        acc_xyz = np.load("../../SHL/SHL_2023/100hz_5.0s_overlap0.0s/test_clear/Hips_Acc.npy")
        label = np.load("../../SHL/SHL_2023/100hz_5.0s_overlap0.0s/test_clear/Hips_Label.npy")

        label= np.array([i-1 for i in label])

        nb_classes = 8
    elif args.data == "ADL":
        acc = np.load("/home/masaharu/remote/ssl-data/downstream/adl_30hz_clean/X.npy")
        label = np.load("/home/masaharu/remote/ssl-data/downstream/adl_30hz_clean/Y.npy")
        acc_xyz = np.transpose(acc, (0, 2, 1))
        unique_label = np.unique(label)
        label2int = {label: idx for idx, label in enumerate(unique_label)}
        label = np.array([label2int[a] for a in label])
        nb_classes = 5
    elif args.data == "PAMAP":
        acc = np.load("/home/SHL/ssl-data/downstream/pamap_100hz_w10_o5/X.npy")
        label = np.load("/home/SHL/ssl-data/downstream/pamap_100hz_w10_o5/Y.npy")
        acc_xyz = np.transpose(acc, (0, 2, 1))
        unique_label = np.unique(label)
        label2int = {label: idx for idx, label in enumerate(unique_label)}
        label = np.array([label2int[a] for a in label])
        nb_classes = 8


    elif args.data == "PAMAP2":
            root = args.pamap2_root
            X = np.load(os.path.join(root, "X.npy"))         # (N,3,496)
            Y = np.load(os.path.join(root, "Y.npy"))         # (N,)
            G = np.load(os.path.join(root, "groups.npy"))    # (N,) subject id
        
            # drop subject (e.g., 109)
            if args.pamap2_drop_subj is not None:
                mask = (G != args.pamap2_drop_subj)
                X, Y, G = X[mask], Y[mask], G[mask]
        
            # label -> 0..C-1 (stable mapping by sorted unique labels)
            unique_label = np.unique(Y)
            label2int = {int(l): i for i, l in enumerate(unique_label.tolist())}
            Y = np.array([label2int[int(a)] for a in Y], dtype=np.int64)
            nb_classes = len(unique_label)
        
            # -------- Pattern B split --------
            test_subj = args.fold_test_subj
            test_idx = (G == test_subj)
            trainval_idx = (G != test_subj)
        
            X_test, Y_test, G_test = X[test_idx], Y[test_idx], G[test_idx]
            X_tv,   Y_tv,   G_tv   = X[trainval_idx], Y[trainval_idx], G[trainval_idx]
        
            # val is random windows from trainval (subject-mixed, but no test leakage)
            n_tv = len(Y_tv)
            idx = np.arange(n_tv)
            rng = np.random.RandomState(args.seed)  # reproducible
            rng.shuffle(idx)
        
            n_val = int(round(n_tv * args.val_ratio))
            val_ids = idx[:n_val]
            train_ids = idx[n_val:]
        
            train_data = TriaxialSignalDataset(
                data=X_tv[train_ids],
                labels=Y_tv[train_ids],
                transform=build_transform(is_train=True, args=args),
            )
            val_data = TriaxialSignalDataset(
                data=X_tv[val_ids],
                labels=Y_tv[val_ids],
                transform=build_transform(is_train=False, args=args),
            )
            test_data = TriaxialSignalDataset(
                data=X_test,
                labels=Y_test,
                transform=build_transform(is_train=False, args=args),
            )
        
            return train_data, val_data, test_data, nb_classes

    

    elif args.data == "REALWORLD":
        is_first_file = True
        acc_xyz = None # 初期値はNoneにしておく
        label = None
        
        data_path = "../../SHL/RealWorld_processed_6_3_parts/proband"

        for i in range(1, 11):
            st = data_path + str(i) + "/Data.npy"
            st2 = data_path + str(i) + "/Label.npy"
            a = np.load(st)
            b = np.load(st2)
            if is_first_file:
                acc_xyz = a # 最初の配列をそのまま代入
                label = b
                is_first_file = False
            else:
                acc_xyz = np.concatenate([acc_xyz, a], axis=0) # 2回目以降は結合
                label = np.concatenate([label, b], axis=0)
                
        #norm_obj = aug.SubjectNormalize(acc_xyz)
        
        unique_label = np.unique(label)
        label2int = {label: idx for idx, label in enumerate(unique_label)}
        label = np.array([label2int[a] for a in label])
        
        is_first_file = True
        acc_xyz2 = None # 初期値はNoneにしておく
        label2 = None
        
        for i in range(11, 13):
            st = data_path + str(i) + "/Data.npy"
            st2 = data_path + str(i) + "/Label.npy"
            a = np.load(st)
            b = np.load(st2)
            if is_first_file:
                acc_xyz2 = a # 最初の配列をそのまま代入
                label2 = b
                is_first_file = False
            else:
                acc_xyz2 = np.concatenate([acc_xyz2, a], axis=0) # 2回目以降は結合
                label2 = np.concatenate([label2, b], axis=0)
        
        unique_label = np.unique(label2)
        label2int = {label2: idx for idx, label2 in enumerate(unique_label)}
        label2 = np.array([label2int[a] for a in label2])
        
        args.num_chans = 15
        nb_classes = 8
        
        train_data = TriaxialSignalDataset(data=acc_xyz, labels=label,transform=build_transform(is_train=True, args=args))
        val_data = TriaxialSignalDataset(data=acc_xyz2, labels=label2, transform=build_transform(is_train=False, args=args))
        
        #train_data = TriaxialSignalDataset(data=acc_xyz, labels=label,transform=build_transform(is_train=True, args=args, norm_obj=norm_obj))
        #val_data = TriaxialSignalDataset(data=acc_xyz2, labels=label2, transform=build_transform(is_train=False, args=args, norm_obj=norm_obj))
        
        return train_data, val_data, nb_classes
    elif args.data == "WISDM":
        acc = np.load("/home/SHL/ssl-data/downstream/wisdm_30hz_clean/X.npy")
        label = np.load("/home/SHL/ssl-data/downstream/wisdm_30hz_clean/Y.npy")
        acc_xyz = np.transpose(acc, (0, 2, 1))
        unique_label = np.unique(label)
        label2int = {label: idx for idx, label in enumerate(unique_label)}
        label = np.array([label2int[a] for a in label])
        nb_classes = 18
    elif args.data == "CAPTURE":
        acc = np.load("/home/SHL/ssl-data/downstream/capture24_30hz_full/X.npy")
        label = np.load("/home/SHL/ssl-data/downstream/capture24_30hz_full/Y.npy")
        acc_xyz = np.transpose(acc, (0, 2, 1))
        unique_label = np.unique(label)
        label2int = {label: idx for idx, label in enumerate(unique_label)}
        label = np.array([label2int[a] for a in label])
        nb_classes = 4
    
    if not args.data == "SHL2023_test": 
        index = np.array(range(label.shape[0]))
        train_label, val_label, train_index, val_index = train_test_split(label, index, train_size=0.9, random_state=0)
    else:
        index = np.array(range(label.shape[0]))
        train_label, val_label, train_index, val_index = train_test_split(label, index, train_size=0.8, random_state=0)
        val_label, _, val_index, _ = train_test_split(val_label, val_index, train_size=0.5, random_state=0)
        
    train_data = TriaxialSignalDataset(data=acc_xyz[train_index], labels=label[train_index],transform=build_transform(is_train=True, args=args))
    val_data = TriaxialSignalDataset(data=acc_xyz[val_index], labels=label[val_index], transform=build_transform(is_train=False, args=args))
    

    return train_data, val_data, nb_classes


def build_transform(is_train=True, args=None, norm_obj=None):
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            signal_length=args.input_size,
            is_training=True,
            jitter_strength=0.1,
            flip_prob = 0.5 if args.flip_on else 0.0
            )
        
        if args.student == "DeepConvLSTM":
            if norm_obj is not None:
                return transforms.Compose([transform, norm_obj, aug.Signal4DeepConvLSTM()])
            return transforms.Compose([transform, aug.Signal4DeepConvLSTM()])
        
        if norm_obj is not None:
            return transforms.Compose([transform, norm_obj])

        return transform

    t = []
    
    t.append(aug.Resample(num_sample=496))
    
    if norm_obj is not None:
        t.append(norm_obj)
    
    t.append(aug.Signal2Tensor())
    
    if args.student == "DeepConvLSTM":
        t.append(aug.Signal4DeepConvLSTM())
        return transforms.Compose(t)

    
    return transforms.Compose(t)