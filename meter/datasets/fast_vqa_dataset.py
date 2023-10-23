from .base_dataset import BaseDataset

# NEW
import os
import pickle
import pyarrow as pa
import numpy as np
from scipy.stats import rankdata
import torch
from transformers import RobertaTokenizerFast

class FASTVQADataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["fast_vqa_shots_5_ways_2_all_questions"]
        elif split == "val":
            names = ["fast_vqa_shots_5_ways_2_all_questions"]
        elif split == "test":
            names = ["fast_vqa_shots_5_ways_2_all_questions"]

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="prompt", # NOTE
            remove_duplicate=False,
        )

        assert kwargs["config"]["use_meta_encoder"]
        self.use_meta_encoder = kwargs["config"]["use_meta_encoder"]
        ways = int(names[0].split("_")[-3])
        shots = int(names[0].split("_")[-5])
        self.example_num = int(ways * shots) # NOTE: overwrite example number
        self.no_example_baseline = kwargs["config"]["no_example_baseline"]

        # Load features
        feat_root = kwargs["config"]["data_root"].replace("arrows", "features")

        feats = [
            pa.ipc.RecordBatchFileReader(
                pa.memory_map(f"{feat_root}/merged/{name}.arrow", "r")
            ).read_all()
            for name in names
        ]
        self.feats = pa.concat_tables(feats, promote=True)

        # Get labels
        concept2data_root = "../Datasets/concept2data/FAST-VQA"
        with open(f"{concept2data_root}/concept2id.pkl", "rb") as f:
            concept2data = pickle.load(f)
            class_names = list(concept2data.keys())

        answers = np.concatenate(self.table["answer"].to_pandas().tolist())
        self.labels = [concept2data[a] for a in answers]

        assert len(self.labels) == len(self.feats)

        # Get class tokens
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        encoding = tokenizer(class_names, padding="longest", add_special_tokens=False)
        self.class_tokens = np.array(encoding["input_ids"])
        self.class_token_masks = np.array(encoding["attention_mask"])

        # Force sample index
        self.sample_index = self.sample_index[
            [i for i in range(0, len(self.sample_index), self.example_num+1)]
        ]
        self.index_mapper = {i: self.index_mapper[si] for i, si in enumerate(self.sample_index)}
        self.index_mapper = np.array([list(v) for v in self.index_mapper.values()])

    def __getitem__(self, index):
        suite = {}
        suite["main_feats"] = self.get_feats(self.feats, self.sample_index[index])
        suite["main_labels"] = self.labels[self.sample_index[index]]

        if not self.no_example_baseline:
            exp_feats, exp_labels = [], []
            for i in range(1, self.example_num+1):
                exp_feats.append(self.get_feats(self.feats, self.sample_index[index]+i))
                exp_labels.append(self.labels[self.sample_index[index]+i])

        # Use other query data
        #if not self.no_example_baseline:
        #    exp_feats, exp_labels = [], []
        #    for i in range(1, self.example_num+1):
        #        shift = np.random.choice(self.sample_index)
        #        exp_feats.append(self.get_feats(self.feats, shift))
        #        exp_labels.append(self.labels[shift])

            suite["exp_feats"] = np.stack(exp_feats)
            suite["exp_labels"] = np.stack(exp_labels)

        return suite

    def get_feats(self, feats, indexes):
        # Organize features
        index_masks = np.zeros(len(feats))
        index_masks[indexes] = 1
        index_masks = index_masks.astype(bool)

        vl_feats = np.stack([
            np.stack(feats["vl_v_cls_feat"].filter(index_masks).to_numpy()),
            np.stack(feats["vl_v_avg_feat"].filter(index_masks).to_numpy()),
            np.stack(feats["vl_l_cls_feat"].filter(index_masks).to_numpy()),
            np.stack(feats["vl_l_avg_feat"].filter(index_masks).to_numpy()),
            np.stack(feats["vl_l_mask_feat"].filter(index_masks).to_numpy()),
        ], axis=1)

        v_feats = np.stack([
            np.stack(feats["v_cls_feat"].filter(index_masks).to_numpy()),
            np.stack(feats["v_avg_feat"].filter(index_masks).to_numpy()),
        ], axis=1)

        l_feats = np.stack([
            np.stack(feats["l_cls_feat"].filter(index_masks).to_numpy()),
            np.stack(feats["l_avg_feat"].filter(index_masks).to_numpy()),
            np.stack(feats["l_mask_feat"].filter(index_masks).to_numpy()),
        ], axis=1)

        merged_feats = np.concatenate([vl_feats, np.concatenate([v_feats, l_feats], axis=1)], axis=2).astype(np.float32)

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
