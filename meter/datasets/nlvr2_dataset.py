from .base_dataset import BaseDataset
import sys
import random

# NEW
import os
import h5py
import pyarrow as pa
import numpy as np
from scipy.stats import rankdata
import torch
from transformers import RobertaTokenizerFast

class NLVR2Dataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["nlvr2_dev", "nlvr2_test1"]
        elif split == "val":
            names = ["nlvr2_dev", "nlvr2_test1"]
        elif split == "test":
            names = ["nlvr2_dev", "nlvr2_test1"]

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="prompt", # NOTE
            remove_duplicate=False,
        )

        # NEW
        if kwargs["config"]["use_meta_encoder"]:
            self.use_meta_encoder = kwargs["config"]["use_meta_encoder"]
            self.no_example_baseline = kwargs["config"]["no_example_baseline"]
            self.example_num = kwargs["config"]["example_num"]
            self.feature_source = kwargs["config"]["feature_source"]
            self.only_vl_feats = kwargs["config"]["only_vl_feats"]
            self.only_v_and_l_feats = kwargs["config"]["only_v_and_l_feats"]

            self.select_demonstrations = kwargs["config"]["select_demonstrations"]

            db_names = ["nlvr2_dev", "nlvr2_test1"]

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
            self.labels = (np.concatenate(self.table["answers"].to_pandas().tolist())=="True").astype(np.int)
            self.db_labels = (np.concatenate(db_table["answers"].to_pandas().tolist())=="True").astype(np.int)

            assert len(self.labels) == len(self.feats)
            assert len(self.db_labels) == len(self.db_feats)

            # Get class tokens
            tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
            encoding = tokenizer(["no", "yes"], padding="longest", add_special_tokens=False)
            self.class_tokens = np.array(encoding["input_ids"])
            self.class_token_masks = np.array(encoding["attention_mask"])

            # Prepare db sampling information
            self.db_sample_index = np.arange(len(self.db_labels))
            if kwargs["config"]["max_train_samples"] is not None:
                np.random.seed(kwargs["config"]["sample_seed"])
                self.db_sample_index = np.random.randint(
                    len(self.db_labels), size=kwargs["config"]["max_train_samples"]
                )

    def __getitem__(self, index):
        # NEW
        if self.use_meta_encoder:
            suite = {}
            suite["main_feats"] = self.get_feats(self.feats, self.sample_index[index])
            suite["main_labels"] = self.labels[self.sample_index[index]]

            if not self.no_example_baseline:
                self.get_examples(suite, self.sample_index[index])

            index, question_index = self.index_mapper[index]
            suite["table_name"] = self.table_names[index]

            return suite
        else:
            result = None
            while result is None:
                try:
                    image_tensor_0 = self.get_image(index, image_key="image_0")["image"]
                    image_tensor_1 = self.get_image(index, image_key="image_1")["image"]
                    text = self.get_text(index)["text"]
                    result = True
                except:
                    print(
                        f"error while read file idx {index} in {self.names[0]}",
                        file=sys.stderr,
                    )
                    index = random.randint(0, len(self.index_mapper) - 1)

            index, question_index = self.index_mapper[index]
            answers = self.table["answers"][index][question_index].as_py()
            answers = answers == "True"

            return {
                "image_0": image_tensor_0,
                "image_1": image_tensor_1,
                "text": text,
                "answers": answers,
                "table_name": self.table_names[index],
            }

    # NEW
    def get_examples(self, suite, index):
        if self.select_demonstrations:
            exp_indexes = self.retrieves[index][:self.example_num]
        else:
            exp_indexes = self.db_sample_index[
                np.random.randint(0, len(self.db_sample_index), size=self.example_num)
            ]

        #if suite["main_labels"] == 1:
        #    print(exp_indexes.tolist() + [index])
        #    import pdb; pdb.set_trace()

        suite["exp_feats"] = self.get_feats(self.db_feats, exp_indexes)
        suite["exp_labels"] = self.db_labels[exp_indexes]

    # NEW
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
        if "main_feats" in dict_batch.keys():
            dict_batch["main_feats"] = torch.tensor(np.stack(dict_batch["main_feats"]))

        if "exp_feats" in dict_batch.keys():
            dict_batch["exp_feats"] = torch.tensor(np.stack(dict_batch["exp_feats"]))

        if "main_labels" in dict_batch.keys():
            dict_batch["main_labels"] = torch.tensor(dict_batch["main_labels"])

        if "exp_labels" in dict_batch.keys():
            dict_batch["exp_labels"] = torch.tensor(np.stack(dict_batch["exp_labels"]))

        return dict_batch
