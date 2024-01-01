from .base_dataset import BaseDataset
import torch


class MNREAuxiliaryDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["mnre_train_refine"]
        elif split == "val":
            names = ["mnre_test_refine"]
        elif split == "test":
            names = ["mnre_test_refine"]

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="texts",
            remove_duplicate=False,
        )

    def get_text(self, raw_index):
        index, caption_index = self.index_mapper[raw_index]

        words = self.all_texts[index][caption_index].tolist()
        auxiliary = self.table["auxiliary"][index][caption_index].as_py().split()

        head_d, tail_d = self.table["heads"][index][caption_index].as_py(), self.table["tails"][index][
            caption_index].as_py()
        head_pos, tail_pos = head_d['pos'], tail_d['pos']
        words.insert(head_pos[0], '<h>')
        words.insert(head_pos[1], '</h>')
        words.insert(tail_pos[0], '<t>')
        words.insert(tail_pos[1], '</t>')

        ntokens = [self.tokenizer.cls_token]
        for word in words:  # iterate every word
            tokens = self.tokenizer.tokenize(word)  # one word may be split into several tokens
            ntokens.extend(tokens)
        if self.with_auxiliary:
            ntokens.append(self.tokenizer.sep_token)
            for word in auxiliary:
                tokens = self.tokenizer.tokenize(word)
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
        # if raw_index < 1:
        #     print(self.all_texts[index])
        #     print(type(self.all_texts))
        #     print(type(self.all_texts[index]))
        #     print(type(words))
        #     print(words, encoding)

        return {
            "text": (words, encoding),
            "img_index": index,
            "cap_index": caption_index,
            "raw_index": raw_index,
        }

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
