import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools

from torch.utils.data.distributed import DistributedSampler
from einops import rearrange

from .dist_utils import all_gather

# NEW
import numpy as np
from copy import deepcopy

# NEW
def meta_infer(pl_module, batch, main_label_embeds, exp_label_embeds):
    if pl_module.hparams.config["no_example_baseline"]:
        batch_size = batch["main_feats"].shape[0]

        # For main features
        qry = pl_module.fuser_qry.unsqueeze(0).expand(batch_size, -1, -1)
        qry_masks = torch.ones(qry.shape[:2], dtype=torch.long, device=pl_module.device)
        extend_qry_masks = pl_module.text_transformer.get_extended_attention_mask(
            qry_masks, qry_masks.size(), pl_module.device
        )

        main_feats = batch["main_feats"]
        main_feat_masks = torch.ones(main_feats.shape[:2], dtype=torch.long, device=pl_module.device)
        extend_main_feat_masks = pl_module.text_transformer.get_extended_attention_mask(
            main_feat_masks, main_feat_masks.size(), pl_module.device
        )
        main_embeds = pl_module.fuser(qry, main_feats, extend_qry_masks, extend_main_feat_masks).mean(1)
        sim_embeds = pl_module.meta_sim_proj(torch.mul(main_embeds, main_embeds))

        meta_embeds = torch.stack([sim_embeds, main_embeds], dim=-2)
    else:
        batch_size = batch["main_feats"].shape[0]
        exp_num = batch["exp_feats"].shape[1]

        # For example features
        qry = pl_module.fuser_qry.unsqueeze(0).expand(int(batch_size*exp_num), -1, -1)
        qry_masks = torch.ones(qry.shape[:2], dtype=torch.long, device=pl_module.device)
        extend_qry_masks = pl_module.text_transformer.get_extended_attention_mask(
            qry_masks, qry_masks.size(), pl_module.device
        )

        exp_feats = batch["exp_feats"]
        exp_feats = rearrange(exp_feats, "b e l h -> (b e) l h")
        exp_feat_masks = torch.ones(exp_feats.shape[:2], dtype=torch.long, device=pl_module.device)
        extend_exp_feat_masks = pl_module.text_transformer.get_extended_attention_mask(
            exp_feat_masks, exp_feat_masks.size(), pl_module.device
        )
        exp_embeds = pl_module.fuser(qry, exp_feats, extend_qry_masks, extend_exp_feat_masks).mean(1)
        exp_embeds = rearrange(exp_embeds, "(b e) h -> b e h", b=batch_size)

        # For main features
        qry = pl_module.fuser_qry.unsqueeze(0).expand(batch_size, -1, -1)
        qry_masks = torch.ones(qry.shape[:2], dtype=torch.long, device=pl_module.device)
        extend_qry_masks = pl_module.text_transformer.get_extended_attention_mask(
            qry_masks, qry_masks.size(), pl_module.device
        )

        main_feats = batch["main_feats"]
        main_feat_masks = torch.ones(main_feats.shape[:2], dtype=torch.long, device=pl_module.device)
        extend_main_feat_masks = pl_module.text_transformer.get_extended_attention_mask(
            main_feat_masks, main_feat_masks.size(), pl_module.device
        )
        main_embeds = pl_module.fuser(qry, main_feats, extend_qry_masks, extend_main_feat_masks).mean(1)

        if pl_module.hparams.config["correlation_embeddings"]:
            sim_embeds = pl_module.meta_sim_proj(
                torch.mul(
                    main_embeds.unsqueeze(1).expand(-1, exp_embeds.shape[1],-1),
                    exp_embeds
                ).mean(1)
            )

            meta_embeds = torch.cat([
                sim_embeds.unsqueeze(1),
                torch.flatten(torch.stack([exp_embeds, exp_label_embeds], dim=-2), 1, 2),
                main_embeds.unsqueeze(1),
            ], dim=-2)
        else:
            meta_embeds = torch.cat([
                torch.flatten(torch.stack([exp_embeds, exp_label_embeds], dim=-2), 1, 2),
                main_embeds.unsqueeze(1),
            ], dim=-2)

    meta_feats = pl_module.meta_encoder(
        inputs_embeds=pl_module.meta_in_transform(meta_embeds)
    )[0]
    if pl_module.hparams.config["correlation_embeddings"]:
        meta_target_feats = torch.concat([meta_feats[:,1:-1,:][:,::2], meta_feats[:,-1:,:]], dim=1)
    else:
        meta_target_feats = torch.concat([meta_feats[:,:-1,:][:,::2], meta_feats[:,-1:,:]], dim=1)

    return {
        "cls_feats": meta_target_feats,
    }

# NEW
def compute_pcls(pl_module, batch):
    if pl_module.training:
        bursty_ratio = pl_module.hparams.config["bursty_ratio"]
        exp_mode = "bursty" if np.random.choice([0,1], p=[1-bursty_ratio, bursty_ratio])==1 else "nonbursty"
        ret = compute_pcls_step(pl_module, batch, exp_mode=exp_mode)

        phase = "train" if pl_module.training else "val"
        loss = getattr(pl_module, f"{phase}_pcls_loss")(ret["pcls_loss"])
        main_acc = getattr(pl_module, f"{phase}_pcls_main_accuracy")(
            ret["pcls_main_logits"], ret["pcls_main_labels"]
        )
        pl_module.log(f"pcls/{phase}/loss", loss)
        pl_module.log(f"pcls/{phase}/main_accuracy", main_acc)

        if not pl_module.hparams.config["no_example_baseline"]:
            exp_acc = getattr(pl_module, f"{phase}_pcls_exp_accuracy")(
                ret["pcls_exp_logits"], ret["pcls_exp_labels"]
            )
            pl_module.log(f"pcls/{phase}/exp_accuracy", exp_acc)
    else:
        ret = {}

        # N-way
        if not pl_module.hparams.config["no_example_baseline"]:
            num_eval_ways = pl_module.hparams.config["num_eval_ways"]
            ret_1 = compute_pcls_step(pl_module, batch, exp_mode="nway", pred_labels=range(num_eval_ways))

            phase = "train" if pl_module.training else "val"
            loss = getattr(pl_module, f"{phase}_pcls_nway_loss")(ret_1["pcls_loss"])
            main_acc = getattr(pl_module, f"{phase}_pcls_nway_main_accuracy")(
                ret_1["pcls_main_logits"], ret_1["pcls_main_labels"]
            )
            pl_module.log(f"pcls/{phase}/nway_loss", loss)
            pl_module.log(f"pcls/{phase}/nway_main_accuracy", main_acc)

            if not pl_module.hparams.config["no_example_baseline"]:
                exp_acc = getattr(pl_module, f"{phase}_pcls_nway_exp_accuracy")(
                    ret_1["pcls_exp_logits"], ret_1["pcls_exp_labels"]
                )
                pl_module.log(f"pcls/{phase}/nway_exp_accuracy", exp_acc)

            ret["pcls_nway_loss"] = ret_1["pcls_loss"]

        # Nonbursty
        ret_2 = compute_pcls_step(pl_module, batch, exp_mode="nonbursty")

        phase = "train" if pl_module.training else "val"
        loss = getattr(pl_module, f"{phase}_pcls_nonbursty_loss")(ret_2["pcls_loss"])
        main_acc = getattr(pl_module, f"{phase}_pcls_nonbursty_main_accuracy")(
            ret_2["pcls_main_logits"], ret_2["pcls_main_labels"]
        )
        pl_module.log(f"pcls/{phase}/nonbursty_loss", loss)
        pl_module.log(f"pcls/{phase}/nonbursty_main_accuracy", main_acc)

        if not pl_module.hparams.config["no_example_baseline"]:
            exp_acc = getattr(pl_module, f"{phase}_pcls_nonbursty_exp_accuracy")(
                ret_2["pcls_exp_logits"], ret_2["pcls_exp_labels"]
            )
            pl_module.log(f"pcls/{phase}/nonbursty_exp_accuracy", exp_acc)

        ret["pcls_nonbursty_loss"] = ret_2["pcls_loss"]

    return ret

# NEW
def compute_pcls_step(pl_module, batch, exp_mode, pred_labels=None):
    exp_num = pl_module.hparams.config["example_num"]
    label_multiplicity = pl_module.hparams.config["label_multiplicity"]

    # Get label embeddings
    class_tokens = torch.tensor(
        pl_module.trainer.datamodule.dms[0].train_dataset.class_tokens
    ).to(pl_module.device)
    class_token_masks = torch.tensor(
        pl_module.trainer.datamodule.dms[0].train_dataset.class_token_masks
    ).to(pl_module.device)
    task_label_embeddings = pl_module.meta_label_embeddings(class_tokens)
    task_label_embeddings = (task_label_embeddings*class_token_masks.unsqueeze(-1)).sum(1) / \
                            class_token_masks.sum(-1).unsqueeze(-1)

    # Assign main features
    batch["main_feats"] = batch[f"main_{exp_mode}_feats"]

    # Prepare main label embeddings
    main_labels = batch[f"main_{exp_mode}_labels"]
    main_label_embeds = pl_module.meta_label_transforms(task_label_embeddings[main_labels])
    if label_multiplicity is not None:
        rand_idx = np.random.randint(label_multiplicity)
        main_label_embeds = pl_module.meta_label_projs[rand_idx](main_label_embeds)

    if pl_module.hparams.config["no_example_baseline"]:
        exp_label_embeds = None
        pcls_labels = main_labels.unsqueeze(-1)
    else:
        # Assign example features
        batch["exp_feats"] = batch[f"exp_{exp_mode}_feats"]

        # Prepare example label embeddings
        exp_labels = batch[f"exp_{exp_mode}_labels"]
        exp_label_embeds = pl_module.meta_label_transforms(task_label_embeddings[exp_labels])
        if label_multiplicity is not None:
            exp_label_embeds = pl_module.meta_label_projs[rand_idx](exp_label_embeds)
        pcls_labels = torch.cat([exp_labels, main_labels.unsqueeze(-1)], dim=-1)

    # Meta infer
    infer = meta_infer(pl_module, batch, main_label_embeds, exp_label_embeds)

    # Compute loss
    if pred_labels is not None:
        pcls_logits = pl_module.meta_classifier(
            infer["cls_feats"],
            pl_module.meta_label_transforms(task_label_embeddings)
        )[:,:,pred_labels]

        # NOTE: important!!!
        pcls_labels_for_loss = deepcopy(pcls_labels)
        if not pl_module.hparams.config["no_example_baseline"]:
            pcls_labels_for_loss[:,:exp_num] = -100

        pcls_loss = F.cross_entropy(
            pcls_logits.view(-1, len(pred_labels)),
            pcls_labels_for_loss.view(-1),
        )
    else:
        pcls_logits = pl_module.meta_classifier(
            infer["cls_feats"],
            pl_module.meta_label_transforms(task_label_embeddings)
        )

        # NOTE: important!!!
        pcls_labels_for_loss = deepcopy(pcls_labels)
        if not pl_module.hparams.config["no_example_baseline"]:
            pcls_labels_for_loss[:,:exp_num] = -100

        pcls_loss = F.cross_entropy(
            pcls_logits.view(-1, pl_module.hparams.config["pcls_label_size"]),
            pcls_labels_for_loss.view(-1),
        )

    if pl_module.hparams.config["no_example_baseline"]:
        return {
            "pcls_loss": pcls_loss,
            "pcls_main_logits": pcls_logits[:,-1],
            "pcls_main_labels": pcls_labels[:,-1],
        }
    else:
        return {
            "pcls_loss": pcls_loss,
            "pcls_main_logits": pcls_logits[:,-1],
            "pcls_main_labels": pcls_labels[:,-1],
            "pcls_exp_logits": pcls_logits[:,:-1],
            "pcls_exp_labels": pcls_labels[:,:-1],
        }


def compute_mlm(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=True, mask_image=False)
    mlm_logits = pl_module.mlm_score(infer["text_feats"])
    mlm_labels = infer["text_labels"]

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mlm_loss")(ret["mlm_loss"])
    acc = getattr(pl_module, f"{phase}_mlm_accuracy")(
        ret["mlm_logits"], ret["mlm_labels"]
    )
    pl_module.log(f"mlm/{phase}/loss", loss)
    pl_module.log(f"mlm/{phase}/accuracy", acc)

    return ret

def compute_itm(pl_module, batch):
    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )
    itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]

    itm_images = [
        torch.stack(
            [
                ti if itm_labels[i] == 1 else fi
                for i, (ti, fi) in enumerate(zip(bti, bfi))
            ]
        )
        for bti, bfi in zip(batch["image"], batch["false_image_0"])
    ]

    batch = {k: v for k, v in batch.items()}
    batch["image"] = itm_images

    infer = pl_module.infer(batch, mask_text=False, mask_image=False)

    itm_logits = pl_module.itm_score(infer["cls_feats"])
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itm_loss")(ret["itm_loss"])
    acc = getattr(pl_module, f"{phase}_itm_accuracy")(
        ret["itm_logits"], ret["itm_labels"]
    )
    pl_module.log(f"itm/{phase}/loss", loss)
    pl_module.log(f"itm/{phase}/accuracy", acc)

    return ret

def compute_snli(pl_module, batch):
    # NEW
    if pl_module.hparams.config["use_meta_encoder"]:
        exp_num = pl_module.hparams.config["example_num"]
        label_multiplicity = pl_module.hparams.config["label_multiplicity"]

        # Get label embeddings
        class_tokens = torch.tensor(
            pl_module.trainer.datamodule.dms[0].train_dataset.class_tokens
        ).to(pl_module.device)
        class_token_masks = torch.tensor(
            pl_module.trainer.datamodule.dms[0].train_dataset.class_token_masks
        ).to(pl_module.device)
        task_label_embeddings = pl_module.meta_label_embeddings(class_tokens)
        task_label_embeddings = (task_label_embeddings*class_token_masks.unsqueeze(-1)).sum(1) / \
                                class_token_masks.sum(-1).unsqueeze(-1)

        # Prepare main label embeddings
        main_labels = batch["main_labels"]
        main_label_embeds = pl_module.meta_label_transforms(task_label_embeddings[main_labels])
        if label_multiplicity is not None:
            rand_idx = np.random.randint(label_multiplicity)
            main_label_embeds = pl_module.meta_label_projs[rand_idx](main_label_embeds)

        if pl_module.hparams.config["no_example_baseline"]:
            exp_label_embeds = None
            snli_labels = main_labels.unsqueeze(-1)
        else:
            # Prepare example label embeddings
            exp_labels = batch["exp_labels"]
            exp_label_embeds = pl_module.meta_label_transforms(task_label_embeddings[exp_labels])
            if label_multiplicity is not None:
                exp_label_embeds = pl_module.meta_label_projs[rand_idx](exp_label_embeds)
            snli_labels = torch.cat([exp_labels, main_labels.unsqueeze(-1)], dim=-1)

        # Meta infer
        infer = meta_infer(pl_module, batch, main_label_embeds, exp_label_embeds)

        # Compute loss
        snli_logits = pl_module.meta_classifier(
            infer["cls_feats"],
            pl_module.meta_label_transforms(task_label_embeddings)
        )

        # NOTE: important!!!
        snli_labels_for_loss = deepcopy(snli_labels)
        if not pl_module.hparams.config["no_example_baseline"]:
            snli_labels_for_loss[:,:exp_num] = -100

        snli_loss = F.cross_entropy(
            snli_logits.view(-1, 3),
            snli_labels_for_loss.view(-1),
        )

        ret = {
            "snli_loss": snli_loss,
            "snli_logits": snli_logits[:,-1],
            "snli_labels": snli_labels[:,-1],
        }
    else:
        infer = pl_module.infer(
            batch, mask_text=False, mask_image=False, 
        )
        snli_logits = pl_module.snli_classifier(infer["cls_feats"])

        snli_labels = batch["labels"]
        snli_labels = torch.tensor(snli_labels).to(pl_module.device).long()
        snli_loss = F.cross_entropy(snli_logits, snli_labels.view(-1))

        ret = {
            "snli_loss": snli_loss,
            "snli_logits": snli_logits,
            "snli_labels": snli_labels,
        }

    phase = "train" if pl_module.training else "val"

    if phase == "train":
        loss = getattr(pl_module, f"{phase}_snli_loss")(ret["snli_loss"])
        acc = getattr(pl_module, f"{phase}_snli_accuracy")(
            ret["snli_logits"], ret["snli_labels"]
        )
        pl_module.log(f"snli/{phase}/loss", loss)
        pl_module.log(f"snli/{phase}/accuracy", acc)
    else:
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            dev_loss = getattr(pl_module, f"dev_snli_loss")(
                F.cross_entropy(
                    ret["snli_logits"][dev_batches], ret["snli_labels"][dev_batches]
                )
            )
            dev_acc = getattr(pl_module, f"dev_snli_accuracy")(
                ret["snli_logits"][dev_batches], ret["snli_labels"][dev_batches]
            )
            pl_module.log(f"snli/dev/loss", dev_loss)
            pl_module.log(f"snli/dev/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_snli_loss")(
                F.cross_entropy(
                    ret["snli_logits"][test_batches], ret["snli_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_snli_accuracy")(
                ret["snli_logits"][test_batches], ret["snli_labels"][test_batches]
            )
            pl_module.log(f"snli/test/loss", test_loss)
            pl_module.log(f"snli/test/accuracy", test_acc)

    return ret

def compute_vqa(pl_module, batch):
    # NEW
    if pl_module.hparams.config["use_meta_encoder"]:
        exp_num = pl_module.hparams.config["example_num"]
        label_multiplicity = pl_module.hparams.config["label_multiplicity"]

        # Get label embeddings
        class_tokens = torch.tensor(
            pl_module.trainer.datamodule.dms[0].train_dataset.class_tokens
        ).to(pl_module.device)
        class_token_masks = torch.tensor(
            pl_module.trainer.datamodule.dms[0].train_dataset.class_token_masks
        ).to(pl_module.device)
        task_label_embeddings = pl_module.meta_label_embeddings(class_tokens)
        task_label_embeddings = (task_label_embeddings*class_token_masks.unsqueeze(-1)).sum(1) / \
                                class_token_masks.sum(-1).unsqueeze(-1)

        # Prepare main label embeddings
        main_labels = batch["main_labels"]
        main_label_embeds = pl_module.meta_label_transforms(task_label_embeddings[main_labels])
        if label_multiplicity is not None:
            rand_idx = np.random.randint(label_multiplicity)
            main_label_embeds = pl_module.meta_label_projs[rand_idx](main_label_embeds)

        if pl_module.hparams.config["no_example_baseline"]:
            exp_label_embeds = None
            vqa_labels = main_labels.unsqueeze(-1)
        else:
            # Prepare example label embeddings
            exp_labels = batch["exp_labels"]
            exp_label_embeds = pl_module.meta_label_transforms(task_label_embeddings[exp_labels])
            if label_multiplicity is not None:
                exp_label_embeds = pl_module.meta_label_projs[rand_idx](exp_label_embeds)
            vqa_labels = torch.cat([exp_labels, main_labels.unsqueeze(-1)], dim=-1)

        # Meta infer
        infer = meta_infer(pl_module, batch, main_label_embeds, exp_label_embeds)

        # Compute loss
        vqa_logits = pl_module.meta_classifier(
            infer["cls_feats"],
            pl_module.meta_label_transforms(task_label_embeddings)
        )[:,-1]
        vqa_targets = torch.zeros(
            len(vqa_logits), pl_module.hparams.config["vqav2_label_size"]
        ).to(pl_module.device)

        vqa_labels = batch["vqa_labels"]
        vqa_scores = batch["vqa_scores"]

        for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
            for l, s in zip(_label, _score):
                vqa_targets[i, l] = s

        vqa_loss = (
            F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets)
            * vqa_targets.shape[1]
        )  # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

        ret = {
            "vqa_loss": vqa_loss,
            "vqa_logits": vqa_logits,
            "vqa_targets": vqa_targets,
            "vqa_labels": vqa_labels,
            "vqa_scores": vqa_scores,
        }
    else:
        infer = pl_module.infer(batch, mask_text=False, mask_image=False)
        vqa_logits = pl_module.vqa_classifier(infer["cls_feats"])
        vqa_targets = torch.zeros(
            len(vqa_logits), pl_module.hparams.config["vqav2_label_size"]
        ).to(pl_module.device)

        vqa_labels = batch["vqa_labels"]
        vqa_scores = batch["vqa_scores"]

        for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
            for l, s in zip(_label, _score):
                vqa_targets[i, l] = s

        vqa_loss = (
            F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets)
            * vqa_targets.shape[1]
        )  # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

        ret = {
            "vqa_loss": vqa_loss,
            "vqa_logits": vqa_logits,
            "vqa_targets": vqa_targets,
            "vqa_labels": vqa_labels,
            "vqa_scores": vqa_scores,
        }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vqa_loss")(ret["vqa_loss"])
    score = getattr(pl_module, f"{phase}_vqa_score")(
        ret["vqa_logits"], ret["vqa_targets"]
    )
    pl_module.log(f"vqa/{phase}/loss", loss)
    pl_module.log(f"vqa/{phase}/score", score)

    return ret


def compute_nlvr2(pl_module, batch):
    # NEW
    if pl_module.hparams.config["use_meta_encoder"]:
        exp_num = pl_module.hparams.config["example_num"]
        label_multiplicity = pl_module.hparams.config["label_multiplicity"]

        # Get label embeddings
        class_tokens = torch.tensor(
            pl_module.trainer.datamodule.dms[0].train_dataset.class_tokens
        ).to(pl_module.device)
        class_token_masks = torch.tensor(
            pl_module.trainer.datamodule.dms[0].train_dataset.class_token_masks
        ).to(pl_module.device)
        task_label_embeddings = pl_module.meta_label_embeddings(class_tokens)
        task_label_embeddings = (task_label_embeddings*class_token_masks.unsqueeze(-1)).sum(1) / \
                                class_token_masks.sum(-1).unsqueeze(-1)

        # Prepare main label embeddings
        main_labels = batch["main_labels"]
        main_label_embeds = pl_module.meta_label_transforms(task_label_embeddings[main_labels])
        if label_multiplicity is not None:
            rand_idx = np.random.randint(label_multiplicity)
            main_label_embeds = pl_module.meta_label_projs[rand_idx](main_label_embeds)

        if pl_module.hparams.config["no_example_baseline"]:
            exp_label_embeds = None
            nlvr2_labels = main_labels.unsqueeze(-1)
        else:
            # Prepare example label embeddings
            exp_labels = batch["exp_labels"]
            exp_label_embeds = pl_module.meta_label_transforms(task_label_embeddings[exp_labels])
            if label_multiplicity is not None:
                exp_label_embeds = pl_module.meta_label_projs[rand_idx](exp_label_embeds)
            nlvr2_labels = torch.cat([exp_labels, main_labels.unsqueeze(-1)], dim=-1)

        # Meta infer
        infer = meta_infer(pl_module, batch, main_label_embeds, exp_label_embeds)

        # Compute loss
        nlvr2_logits = pl_module.meta_classifier(
            infer["cls_feats"],
            pl_module.meta_label_transforms(task_label_embeddings)
        )

        # NOTE: important!!!
        nlvr2_labels_for_loss = deepcopy(nlvr2_labels)
        if not pl_module.hparams.config["no_example_baseline"]:
            nlvr2_labels_for_loss[:,:exp_num] = -100

        nlvr2_loss = F.cross_entropy(
            nlvr2_logits.view(-1, 2),
            nlvr2_labels_for_loss.view(-1),
        )

        ret = {
            "nlvr2_loss": nlvr2_loss,
            "nlvr2_logits": nlvr2_logits[:,-1],
            "nlvr2_labels": nlvr2_labels[:,-1],
        }
    else:
        infer1 = pl_module.infer(
            batch, mask_text=False, mask_image=False, image_token_type_idx=1
        )
        infer2 = pl_module.infer(
            batch, mask_text=False, mask_image=False, image_token_type_idx=2
        )

        cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
        nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)

        nlvr2_labels = batch["answers"]
        nlvr2_labels = torch.tensor(nlvr2_labels).to(pl_module.device).long()
        nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels.view(-1))

        ret = {
            "nlvr2_loss": nlvr2_loss,
            "nlvr2_logits": nlvr2_logits,
            "nlvr2_labels": nlvr2_labels,
        }

    phase = "train" if pl_module.training else "val"

    if phase == "train":
        loss = getattr(pl_module, f"{phase}_nlvr2_loss")(ret["nlvr2_loss"])
        acc = getattr(pl_module, f"{phase}_nlvr2_accuracy")(
            ret["nlvr2_logits"], ret["nlvr2_labels"]
        )
        pl_module.log(f"nlvr2/{phase}/loss", loss)
        pl_module.log(f"nlvr2/{phase}/accuracy", acc)
    else:
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            dev_loss = getattr(pl_module, f"dev_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
                )
            )
            dev_acc = getattr(pl_module, f"dev_nlvr2_accuracy")(
                ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
            )
            pl_module.log(f"nlvr2/dev/loss", dev_loss)
            pl_module.log(f"nlvr2/dev/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_nlvr2_accuracy")(
                ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
            )
            pl_module.log(f"nlvr2/test/loss", test_loss)
            pl_module.log(f"nlvr2/test/accuracy", test_acc)

    return ret


# NEW
def compute_open_mi(pl_module, batch):
    exp_num = pl_module.hparams.config["example_num"]
    label_multiplicity = pl_module.hparams.config["label_multiplicity"]

    # Get label embeddings
    class_tokens = torch.tensor(
        pl_module.trainer.datamodule.dms[0].train_dataset.class_tokens
    ).to(pl_module.device)
    class_token_masks = torch.tensor(
        pl_module.trainer.datamodule.dms[0].train_dataset.class_token_masks
    ).to(pl_module.device)
    task_label_embeddings = pl_module.meta_label_embeddings(class_tokens)
    task_label_embeddings = (task_label_embeddings*class_token_masks.unsqueeze(-1)).sum(1) / \
                            class_token_masks.sum(-1).unsqueeze(-1)

    # Prepare main label embeddings
    main_labels = batch["main_labels"]
    main_label_embeds = pl_module.meta_label_transforms(task_label_embeddings[main_labels])
    if label_multiplicity is not None:
        rand_idx = np.random.randint(label_multiplicity)
        main_label_embeds = pl_module.meta_label_projs[rand_idx](main_label_embeds)

    if pl_module.hparams.config["no_example_baseline"]:
        exp_label_embeds = None
        open_mi_labels = main_labels.unsqueeze(-1)
    else:
        # Prepare example label embeddings
        exp_labels = batch["exp_labels"]
        exp_label_embeds = pl_module.meta_label_transforms(task_label_embeddings[exp_labels])
        if label_multiplicity is not None:
            exp_label_embeds = pl_module.meta_label_projs[rand_idx](exp_label_embeds)
        open_mi_labels = torch.cat([exp_labels, main_labels.unsqueeze(-1)], dim=-1)

    # Meta infer
    infer = meta_infer(pl_module, batch, main_label_embeds, exp_label_embeds)

    # Compute loss
    open_mi_logits = pl_module.meta_classifier(
        infer["cls_feats"],
        pl_module.meta_label_transforms(task_label_embeddings)
    )

    # NOTE: important!!!
    open_mi_labels_for_loss = deepcopy(open_mi_labels)
    if not pl_module.hparams.config["no_example_baseline"]:
        open_mi_labels_for_loss[:,:exp_num] = -100

    open_mi_loss = F.cross_entropy(
        open_mi_logits.view(-1, 2),
        open_mi_labels_for_loss.view(-1),
    )

    ret = {
        "open_mi_loss": open_mi_loss,
        "open_mi_logits": open_mi_logits[:,-1],
        "open_mi_labels": open_mi_labels[:,-1],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_open_mi_loss")(ret["open_mi_loss"])
    acc = getattr(pl_module, f"{phase}_open_mi_accuracy")(
        ret["open_mi_logits"], ret["open_mi_labels"]
    )
    pl_module.log(f"open_mi/{phase}/loss", loss)
    pl_module.log(f"open_mi/{phase}/accuracy", acc)

    return ret


# NEW
def compute_fast_vqa(pl_module, batch):
    exp_num = pl_module.hparams.config["example_num"]
    label_multiplicity = pl_module.hparams.config["label_multiplicity"]

    # Get label embeddings
    class_tokens = torch.tensor(
        pl_module.trainer.datamodule.dms[0].train_dataset.class_tokens
    ).to(pl_module.device)
    class_token_masks = torch.tensor(
        pl_module.trainer.datamodule.dms[0].train_dataset.class_token_masks
    ).to(pl_module.device)
    task_label_embeddings = pl_module.meta_label_embeddings(class_tokens)
    task_label_embeddings = (task_label_embeddings*class_token_masks.unsqueeze(-1)).sum(1) / \
                            class_token_masks.sum(-1).unsqueeze(-1)

    # Prepare main label embeddings
    main_labels = batch["main_labels"]
    main_label_embeds = pl_module.meta_label_transforms(task_label_embeddings[main_labels])
    if label_multiplicity is not None:
        rand_idx = np.random.randint(label_multiplicity)
        main_label_embeds = pl_module.meta_label_projs[rand_idx](main_label_embeds)

    if pl_module.hparams.config["no_example_baseline"]:
        exp_label_embeds = None
        fast_vqa_labels = main_labels.unsqueeze(-1)
    else:
        # Prepare example label embeddings
        exp_labels = batch["exp_labels"]
        exp_label_embeds = pl_module.meta_label_transforms(task_label_embeddings[exp_labels])
        if label_multiplicity is not None:
            exp_label_embeds = pl_module.meta_label_projs[rand_idx](exp_label_embeds)
        fast_vqa_labels = torch.cat([exp_labels, main_labels.unsqueeze(-1)], dim=-1)

    # Meta infer
    infer = meta_infer(pl_module, batch, main_label_embeds, exp_label_embeds)

    # Compute loss
    fast_vqa_logits = pl_module.meta_classifier(
        infer["cls_feats"],
        pl_module.meta_label_transforms(task_label_embeddings)
    )

    # NOTE: important!!!
    fast_vqa_labels_for_loss = deepcopy(fast_vqa_labels)
    if not pl_module.hparams.config["no_example_baseline"]:
        fast_vqa_labels_for_loss[:,:exp_num] = -100

    fast_vqa_loss = F.cross_entropy(
        fast_vqa_logits.view(-1, 1509),
        fast_vqa_labels_for_loss.view(-1),
    )

    ret = {
        "fast_vqa_loss": fast_vqa_loss,
        "fast_vqa_logits": fast_vqa_logits[:,-1],
        "fast_vqa_labels": fast_vqa_labels[:,-1],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_fast_vqa_loss")(ret["fast_vqa_loss"])
    acc = getattr(pl_module, f"{phase}_fast_vqa_accuracy")(
        ret["fast_vqa_logits"], ret["fast_vqa_labels"]
    )
    pl_module.log(f"fast_vqa/{phase}/loss", loss)
    pl_module.log(f"fast_vqa/{phase}/accuracy", acc)

    return ret


def compute_irtr(pl_module, batch):
    is_training_phase = pl_module.training

    _bs, _c, _h, _w = batch["image"][0].shape
    false_len = pl_module.hparams.config["draw_false_text"]
    text_ids = torch.stack(
        [batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1
    )
    text_masks = torch.stack(
        [batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1
    )
    text_labels = torch.stack(
        [batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1
    )

    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)
    images = batch["image"][0].unsqueeze(1).expand(_bs, false_len + 1, _c, _h, _w)

    infer = pl_module.infer(
        {
            "image": [rearrange(images, "bs fs c h w -> (bs fs) c h w")],
            "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
            "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
            "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
        }
    )
    score = pl_module.rank_output(infer["cls_feats"])[:, 0]
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    answer = torch.zeros(_bs).to(score).long()
    irtr_loss = F.cross_entropy(score, answer)

    ret = {
        "irtr_loss": irtr_loss,
    }

    phase = "train" if pl_module.training else "val"
    irtr_loss = getattr(pl_module, f"{phase}_irtr_loss")(ret["irtr_loss"])

    pl_module.log(f"irtr/{phase}/irtr_loss", irtr_loss)

    return ret


@torch.no_grad()
def compute_irtr_recall(pl_module):
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=64,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
        image_only=True
    )
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    #TODO: speed up the process by caching text/image features
    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        text_preload.append(
            {
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
                "img_index": _b["img_index"],
            }
        )

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)

    image_preload = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        image_preload.append((_b['image'][0], _b["img_index"][0]))

    rank_scores = list()
    rank_iids = list()

    for img_batch in tqdm.tqdm(image_preload, desc="rank loop"):
        _im, _iid = img_batch

        img_batch_score = list()
        for txt_batch in text_preload:
            fblen = len(txt_batch["text_ids"])
            im = _im.repeat(fblen, 1, 1, 1).to(device=txt_batch['text_ids'].device)

            with torch.cuda.amp.autocast():
                score = pl_module.rank_output(
                    pl_module.infer(
                        {
                            "text_ids": txt_batch["text_ids"],
                            "text_masks": txt_batch["text_masks"],
                            "text_labels": txt_batch["text_labels"],
                        },
                        img=im,
                    )["cls_feats"]
                )[:, 0]

            img_batch_score.append(score)

        img_batch_score = torch.cat(img_batch_score)
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids.append(_iid)

    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def vqa_test_step(pl_module, batch, output):
    try:
        id2answer = (
            pl_module.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
            if "vqa_trainval" in pl_module.trainer.datamodule.dm_dicts
            else pl_module.trainer.datamodule.dm_dicts["vqa"].id2answer
        )
    except:
        id2answer = (
            pl_module.trainer.datamodule.dm_dicts["gqa_test"].id2answer
            if "gqa_test" in pl_module.trainer.datamodule.dm_dicts
            else pl_module.trainer.datamodule.dm_dicts["gqa"].id2answer
        )
        vqa_logits = output["vqa_logits"]
        vqa_preds = vqa_logits.argmax(dim=-1)
        vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
        #questions = batch["text"] # NEW
        qids = batch["qid"]
        return {"qids": qids, "preds": vqa_preds, "gqa": True}
    vqa_logits = output["vqa_logits"]
    vqa_preds = vqa_logits.argmax(dim=-1)
    vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
    #questions = batch["text"] # NEW
    qids = batch["qid"]
    return {"qids": qids, "preds": vqa_preds, "gqa": False}


def arc_test_step(pl_module, batch, output):
    return output


def vqa_test_wrapup(outs, model_name):
    rank = torch.distributed.get_rank()
    qids, preds = list(), list()
    gqa = False
    for out in outs:
        qids += out["qids"]
        preds += out["preds"]
        gqa = out['gqa']

    rets = list()
    for qid, pred in zip(qids, preds):
        if gqa:
            rets.append({"questionId": qid, "prediction": pred})
        else:
            rets.append({"question_id": qid, "answer": pred})
    with open(f"vqa_submit_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob("vqa_submit_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result", exist_ok=True)
        with open(f"result/vqa_submit_{model_name}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"vqa_submit_{rank}.json")


def arc_test_wrapup(outs, caplen, model_name):
    rank = torch.distributed.get_rank()
    iids, captions = list(), list()
    for out in outs:
        iids += out["iid"]
        captions += out["captions"]

    rets = list()
    for iid, caption in zip(iids, captions):
        rets.append({"image_id": iid, "caption": caption})
    with open(f"coco_cap_len{caplen}_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob(f"coco_cap_len{caplen}_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result/arc", exist_ok=True)
        jsons = sorted(jsons, key=lambda x: x["image_id"])
        with open(f"result/arc/coco_cap_{model_name}_len{caplen}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"coco_cap_len{caplen}_{rank}.json")
