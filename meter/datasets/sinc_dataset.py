import os
import random
import pickle
import pyarrow as pa
import numpy as np
from scipy.stats import rankdata
import torch
from transformers import RobertaTokenizerFast
from .base_dataset import BaseDataset

class SINCDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        test_names = [
            "vqav2_val",
            #"coco_caption_karpathy_test",
            #"coco_caption_karpathy_test_yesno",
        ]
        train_names = [
            "coco_caption_karpathy_train",
            "coco_caption_karpathy_train_yesno",
            "conceptual_caption_train_0",
            "conceptual_caption_train_0_yesno",
            "sbu_0",
            "sbu_0_yesno",
            "vg",
        ]

        if kwargs["config"]["pretrain_scale"] == "medium" or \
           kwargs["config"]["pretrain_scale"] == "large":
            train_names += [
                "conceptual_caption_train_1",
                "conceptual_caption_train_1_yesno",
                "sbu_1",
                "sbu_1_yesno",
            ]

        if kwargs["config"]["pretrain_scale"] == "large":
            train_names += [
                "conceptual_caption_train_2",
                "conceptual_caption_train_2_yesno",
                "sbu_2",
                "sbu_2_yesno",
            ]

        if split == "train":
            names = train_names
        elif split == "val":
            names = test_names
        elif split == "test":
            names = test_names

        super().__init__(*args, **kwargs, names=names, text_column_name="caption_mask")

        self.use_meta_encoder = kwargs["config"]["use_meta_encoder"]
        self.no_example_baseline = kwargs["config"]["no_example_baseline"]
        self.example_num = kwargs["config"]["example_num"]
        self.feature_source = kwargs["config"]["feature_source"]
        self.only_vl_feats = kwargs["config"]["only_vl_feats"]
        self.only_v_and_l_feats = kwargs["config"]["only_v_and_l_feats"]

        self.num_eval_ways = kwargs["config"]["num_eval_ways"]
        assert self.example_num % self.num_eval_ways == 0
        self.nway_recur_num = int(self.example_num/self.num_eval_ways)
        self.bursty_recur_num = int((self.example_num*1.0)/self.num_eval_ways)
        self.bursty_rest_num = int(self.example_num-self.bursty_recur_num*self.num_eval_ways)

        db_names = train_names

        # Load features
        feat_root = kwargs["config"]["data_root"].replace("arrows", "features")
        if self.feature_source is not None:
            feat_root = feat_root.replace("features", f"features/{self.feature_source}")

        feats = [
            pa.ipc.RecordBatchFileReader(
                pa.memory_map(f"{feat_root}/merged/{name}.arrow", "r")
            ).read_all()
            for name in names
        ]
        self.feats = pa.concat_tables(feats, promote=True)

        db_feats = [
            pa.ipc.RecordBatchFileReader(
                pa.memory_map(f"{feat_root}/merged/{db_name}.arrow", "r")
            ).read_all()
            for db_name in db_names
        ]
        self.db_feats = pa.concat_tables(db_feats, promote=True)

        # Load retrieves
        retrieve_root = kwargs["config"]["data_root"].replace("arrows", "retrieves")
        if self.feature_source is not None:
            retrieve_root = retrieve_root.replace("retrieves", f"retrieves/{self.feature_source}")
        retrieves = []
        for n in names:
            if kwargs["config"]["pretrain_scale"] == "large":
                retrieves.append(
                    np.load(f"{retrieve_root}/merged/retrieves_{n}_db_large.npy")
                )
            else:
                retrieves.append(
                    np.load(f"{retrieve_root}/merged/retrieves_{n}_db_{'_'.join(db_names)}.npy")
                )
        self.retrieves = np.concatenate(retrieves)
        assert len(self.feats) == len(self.retrieves)

        # Load db tables
        db_tables = [
            pa.ipc.RecordBatchFileReader(
                pa.memory_map(f"{kwargs['config']['data_root']}/{db_name}.arrow", "r")
            ).read_all()
            for db_name in db_names
        ]
        db_table = pa.concat_tables(db_tables, promote=True)

        # Get labels
        self.labels = np.concatenate(self.table["labels"].to_pandas().tolist())
        self.db_labels = np.concatenate(db_table["labels"].to_pandas().tolist())
        self.train_classes, self.test_classes = np.arange(2926), np.arange(2926,3926)

        concept2data_root = "../Datasets/concept2data/CAPTIONS"
        with open(f"{concept2data_root}/concept2id.pkl", "rb") as f:
            class_names = list(pickle.load(f).keys())

        assert len(self.labels) == len(self.feats)
        assert len(self.db_labels) == len(self.db_feats)

        # Get class tokens
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        encoding = tokenizer(class_names, padding="longest", add_special_tokens=False)

        self.class_tokens = np.array(encoding["input_ids"])
        self.class_token_masks = np.array(encoding["attention_mask"])

        self.train_class_tokens = self.class_tokens[self.train_classes]
        self.train_class_token_masks = self.class_token_masks[self.train_classes]

        self.test_class_tokens = self.class_tokens[self.test_classes]
        self.test_class_token_masks = self.class_token_masks[self.test_classes]

        # Prepare db sampling information
        self.db_class2data = {c: [] for c in range(kwargs["config"]["pcls_label_size"])}
        for i, l in enumerate(self.db_labels):
            self.db_class2data[l].append(i)
        self.db_class2data = {k: np.array(v) for k, v in self.db_class2data.items()}

        self.valid_train_classes = np.array(
            [c for c in self.train_classes if len(self.db_class2data[c])>=self.bursty_recur_num]
        )
        self.valid_test_classes = np.array(
            [c for c in self.test_classes if len(self.db_class2data[c])>=self.nway_recur_num+1]
        )

        self.db_train_sample_index = np.array(
            [i for i, l in enumerate(self.db_labels) if l in self.valid_train_classes]
        )
        self.db_test_sample_index = np.array(
            [i for i, l in enumerate(self.db_labels) if l in self.valid_test_classes]
        )
        if kwargs["config"]["max_train_samples"] is not None:
            np.random.seed(kwargs["config"]["sample_seed"])
            self.db_train_sample_index = np.random.choice(
                self.db_train_sample_index, kwargs["config"]["max_train_samples"], replace=False,
            )

        # Resample data with specific classes
        self.index_mapper = dict()
        k = 0
        for i in range(len(self.table[self.text_column_name])):
            for j in range(len(self.table[self.text_column_name][i])):
                self.index_mapper[k] = (i, j)
                k += 1

        if self.split == "train":
            max_samples = kwargs["config"]["max_train_samples"]
        elif self.split == "val":
            max_samples = kwargs["config"]["max_val_samples"]
        elif self.split == "test":
            max_samples = kwargs["config"]["max_test_samples"]

        self.sample_index = np.array(
            [i for i, l in enumerate(self.labels) if l in self.valid_train_classes]
        )
        if max_samples is not None:
            np.random.seed(kwargs["config"]["sample_seed"]) if self.split=="train" else np.random.seed(0)
            self.sample_index = np.random.choice(self.sample_index, max_samples, replace=False)

        self.index_mapper = {i: self.index_mapper[si] for i, si in enumerate(self.sample_index)}
        self.index_mapper = np.array([list(v) for v in self.index_mapper.values()])

    def __getitem__(self, index):
        suite = {}
        if self.split == "train":
            main_feats = self.get_feats(self.feats, self.sample_index[index])
            main_label = self.labels[self.sample_index[index]]
            suite["main_bursty_feats"] = main_feats
            suite["main_bursty_labels"] = main_label
            suite["main_nonbursty_feats"] = main_feats
            suite["main_nonbursty_labels"] = main_label

            if not self.no_example_baseline:
                self.get_examples(suite, index=self.sample_index[index],
                                  main_label=main_label, mode="bursty")
                self.get_examples(suite, index=self.sample_index[index],
                                  main_label=main_label, mode="nonbursty")
        else:
            main_feats = self.get_feats(self.feats, self.sample_index[index])
            main_label = self.labels[self.sample_index[index]]
            suite["main_nway_feats"] = main_feats
            suite["main_nway_labels"] = main_label # NOTE: will be overwrite
            suite["main_nonbursty_feats"] = main_feats
            suite["main_nonbursty_labels"] = main_label

            if not self.no_example_baseline:
                self.get_examples(suite, index=self.sample_index[index],
                                  main_label=main_label, mode="nway")
                self.get_examples(suite, index=self.sample_index[index],
                                  main_label=main_label, mode="nonbursty")

        return suite

    def get_examples(self, suite, index, main_label, mode):
        if mode == "nway":
            rand_classes = np.random.choice(self.valid_test_classes, self.num_eval_ways, replace=False)
            map_classes = np.random.permutation(self.num_eval_ways)
            recur_nums = [self.nway_recur_num+1] + [self.nway_recur_num]*(self.num_eval_ways-1)

            indexes, labels = [], []
            for n, rc, mc in zip(recur_nums, rand_classes, map_classes):
                indexes.append(self.db_class2data[rc][np.random.randint(0, len(self.db_class2data[rc]), n)])
                labels.append(np.array([mc]*n))
            indexes = np.concatenate(indexes)
            labels = np.concatenate(labels)

            main_index, exp_indexes = indexes[0], indexes[1:]
            main_label, exp_labels = labels[0], labels[1:]

            rand_perm = np.random.permutation(len(exp_indexes))
            exp_indexes = exp_indexes[rand_perm]
            exp_labels = exp_labels[rand_perm]

            suite[f"main_{mode}_feats"] = self.get_feats(self.db_feats, main_index)
            suite[f"main_{mode}_labels"] = main_label
            suite[f"exp_{mode}_feats"] = self.get_feats(self.db_feats, exp_indexes)
            suite[f"exp_{mode}_labels"] = exp_labels

        if mode == "bursty":
            if np.random.randint(2) == 0:
                rand_classes = np.random.choice(self.valid_train_classes, size=self.num_eval_ways-1+self.bursty_rest_num)
                classes = np.concatenate([np.atleast_1d(main_label), rand_classes])
                recur_nums = [self.bursty_recur_num]*self.num_eval_ways + [1]*self.bursty_rest_num

                exp_indexes = []
                for n, c in zip(recur_nums, classes):
                    exp_indexes.append(self.db_class2data[c][np.random.randint(0, len(self.db_class2data[c]), n)])

                exp_indexes = np.concatenate(exp_indexes)
                np.random.shuffle(exp_indexes)
                #if self.split == "train" and \
                #   main_label not in [1289, 863] and \
                #   len(np.unique(exp_indexes.tolist() + [index])) == 9:
                #    print("LID")
                #    import pdb; pdb.set_trace()
            else:
                exp_indexes = self.retrieves[index][
                    np.random.randint(0, self.retrieves.shape[1], self.example_num)
                ]
                #if self.split == "train" and main_label not in [1289, 863]:
                #    print("DID")
                #    import pdb; pdb.set_trace()

            suite[f"exp_{mode}_feats"] = self.get_feats(self.db_feats, exp_indexes)
            suite[f"exp_{mode}_labels"] = self.db_labels[exp_indexes]

        if mode == "nonbursty":
            exp_indexes = self.db_train_sample_index[
                np.random.randint(0, len(self.db_train_sample_index), self.example_num)
            ]

            suite[f"exp_{mode}_feats"] = self.get_feats(self.db_feats, exp_indexes)
            suite[f"exp_{mode}_labels"] = self.db_labels[exp_indexes]
            #if self.split == "train" and \
            #   1289 not in suite[f"exp_{mode}_labels"].tolist()+[main_label] and \
            #   863 not in suite[f"exp_{mode}_labels"].tolist()+[main_label]:
            #    print("OD")
            #    import pdb; pdb.set_trace()

    def get_feats(self, feats, indexes):
        # Organize features
        index_masks = np.zeros(len(feats))
        index_masks[indexes] = 1
        index_masks = index_masks.astype(bool)
        select_feats = feats.filter(index_masks)

        if "vl_v_cls_feat" in select_feats.column_names:
            vl_feats = np.stack([
                np.stack(select_feats["vl_v_cls_feat"].to_numpy()),
                np.stack(select_feats["vl_v_avg_feat"].to_numpy()),
                np.stack(select_feats["vl_l_cls_feat"].to_numpy()),
                np.stack(select_feats["vl_l_avg_feat"].to_numpy()),
                np.stack(select_feats["vl_l_mask_feat"].to_numpy()),
            ], axis=1)
        else:
            vl_feats = None

        if "v_cls_feat" in select_feats.column_names:
            v_l_feats = np.stack([
                np.stack(select_feats["v_cls_feat"].to_numpy()),
                np.stack(select_feats["v_avg_feat"].to_numpy()),
                np.stack(select_feats["l_cls_feat"].to_numpy()),
                np.stack(select_feats["l_avg_feat"].to_numpy()),
                np.stack(select_feats["l_mask_feat"].to_numpy()),
            ], axis=1)
        else:
            v_l_feats = None

        if self.only_vl_feats:
            merged_feats = vl_feats.astype(np.float32)
        elif self.only_v_and_l_feats:
            merged_feats = v_l_feats.astype(np.float32)
        else:
            merged_feats = np.concatenate([vl_feats, v_l_feats], axis=2).astype(np.float32)

        # Recover order
        ranks = rankdata(indexes, method="dense")-1
        merged_feats = merged_feats[ranks]
        if np.isscalar(indexes):
            merged_feats = merged_feats.squeeze(0)

        return merged_feats

    def collate(self, batch, mlm_collator):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        img_sizes = list()

        for img_key in img_keys:
            img = dict_batch[img_key]
            img_sizes += [ii.shape for i in img if i is not None for ii in i]

        for size in img_sizes:
            assert (
                len(size) == 3
            ), f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"

        if len(img_keys) != 0:
            max_height = max([i[1] for i in img_sizes])
            max_width = max([i[2] for i in img_sizes])

        for img_key in img_keys:
            img = dict_batch[img_key]
            view_size = len(img[0])

            new_images = [
                torch.zeros(batch_size, 3, max_height, max_width)
                for _ in range(view_size)
            ]

            for bi in range(batch_size):
                orig_batch = img[bi]
                for vi in range(view_size):
                    if orig_batch is None:
                        new_images[vi][bi] = None
                    else:
                        orig = img[bi][vi]
                        new_images[vi][bi, :, : orig.shape[1], : orig.shape[2]] = orig

            dict_batch[img_key] = new_images

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

        if len(txt_keys) != 0:
            texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            draw_text_len = len(encodings)
            flatten_encodings = [e for encoding in encodings for e in encoding]
            flatten_mlms = mlm_collator(flatten_encodings)

            for i, txt_key in enumerate(txt_keys):
                texts, encodings = (
                    [d[0] for d in dict_batch[txt_key]],
                    [d[1] for d in dict_batch[txt_key]],
                )

                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask

        # NEW
        if "main_bursty_feats" in dict_batch.keys():
            dict_batch["main_bursty_feats"] = torch.tensor(np.stack(dict_batch["main_bursty_feats"]))
        if "main_nonbursty_feats" in dict_batch.keys():
            dict_batch["main_nonbursty_feats"] = torch.tensor(np.stack(dict_batch["main_nonbursty_feats"]))
        if "main_nway_feats" in dict_batch.keys():
            dict_batch["main_nway_feats"] = torch.tensor(np.stack(dict_batch["main_nway_feats"]))

        if "exp_bursty_feats" in dict_batch.keys():
            dict_batch["exp_bursty_feats"] = torch.tensor(np.stack(dict_batch["exp_bursty_feats"]))
        if "exp_nonbursty_feats" in dict_batch.keys():
            dict_batch["exp_nonbursty_feats"] = torch.tensor(np.stack(dict_batch["exp_nonbursty_feats"]))
        if "exp_nway_feats" in dict_batch.keys():
            dict_batch["exp_nway_feats"] = torch.tensor(np.stack(dict_batch["exp_nway_feats"]))

        if "main_bursty_labels" in dict_batch.keys():
            dict_batch["main_bursty_labels"] = torch.tensor(dict_batch["main_bursty_labels"])
        if "main_nonbursty_labels" in dict_batch.keys():
            dict_batch["main_nonbursty_labels"] = torch.tensor(dict_batch["main_nonbursty_labels"])
        if "main_nway_labels" in dict_batch.keys():
            dict_batch["main_nway_labels"] = torch.tensor(dict_batch["main_nway_labels"])

        if "exp_bursty_labels" in dict_batch.keys():
            dict_batch["exp_bursty_labels"] = torch.tensor(np.stack(dict_batch["exp_bursty_labels"]))
        if "exp_nonbursty_labels" in dict_batch.keys():
            dict_batch["exp_nonbursty_labels"] = torch.tensor(np.stack(dict_batch["exp_nonbursty_labels"]))
        if "exp_nway_labels" in dict_batch.keys():
            dict_batch["exp_nway_labels"] = torch.tensor(np.stack(dict_batch["exp_nway_labels"]))

        return dict_batch
