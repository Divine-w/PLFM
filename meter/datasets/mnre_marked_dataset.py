from .base_dataset import BaseDataset
import torch


class MNREMarkedDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        names = ["mnre_marked"]

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="texts",
            remove_duplicate=False,
        )

    def get_text(self, raw_index):
        index, caption_index = self.index_mapper[raw_index]

        words = self.all_texts[index][caption_index]

        ntokens = [self.tokenizer.cls_token]
        for word in words:  # iterate every word
            tokens = self.tokenizer.tokenize(word)  # one word may be split into several tokens
            ntokens.extend(tokens)
        ntokens = ntokens[:self.max_text_len - 1]
        ntokens.append(self.tokenizer.sep_token)

        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
        mask = [1] * len(input_ids)

        pad_len = self.max_text_len - len(input_ids)
        rest_pad = [0] * pad_len  # pad to max_len
        input_ids.extend([self.tokenizer.pad_token_id] * pad_len)
        mask.extend(rest_pad)

        ntokens.extend([self.tokenizer.pad_token] * pad_len)

        encoding = {"input_ids": input_ids, "attention_mask": mask}

        return {
            "text": (words, encoding),
            "img_index": index,
            "cap_index": caption_index,
            "raw_index": raw_index,
        }

    def collate(self, batch):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}  # {k：[batch_size]}

        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        img_sizes = list()

        for img_key in img_keys:
            img = dict_batch[img_key]
            img_sizes += [ii.shape for i in img if i is not None for ii in i]  # ii是经过self.get_image转换过的图像

        for size in img_sizes:
            assert (
                len(size) == 3
            ), f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"

        if len(img_keys) != 0:
            max_height = max([i[1] for i in img_sizes])
            max_width = max([i[2] for i in img_sizes])

        for img_key in img_keys:
            img = dict_batch[img_key]
            view_size = len(img[0])  # transform_keys列表长度

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

            dict_batch[img_key] = new_images  # [batch_size * tensor] tensor: [view_size, 3, H, W]

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

        if len(txt_keys) != 0:
            for i, txt_key in enumerate(txt_keys):
                texts, encodings = (
                    [d[0] for d in dict_batch[txt_key]],
                    [d[1] for d in dict_batch[txt_key]],  # [batch_size]
                )

                input_ids = []
                attention_mask = []  # tensor:[batch_size, encode_size]
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        encoding["input_ids"],  # tensor:[encode_size]
                        encoding["attention_mask"],
                    )
                    input_ids.append(_input_ids)
                    attention_mask.append(_attention_mask)

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = torch.tensor(input_ids).to(self.device)
                dict_batch[f"{txt_key}_masks"] = torch.tensor(attention_mask).to(self.device)
                dict_batch["image"] = dict_batch["image"][0].to(self.device)

        return dict_batch

    def __getitem__(self, index):
        image_tensor = self.get_image(index)["image"]
        text = self.get_text(index)["text"]

        index, tweet_index = self.index_mapper[index]

        labels = self.table["labels"][index][tweet_index].as_py()
        labels = self.rel2id[labels]

        return {
            "image": image_tensor,
            "text": text,
            "labels": labels,
            "table_name": self.table_names[index],
        }
