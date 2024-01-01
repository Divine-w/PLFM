from sacred import Experiment

ex = Experiment("METER")


def _loss_names(d):
    ret = {
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "vqa": 0,
        "vcr": 0,
        "vcr_qar": 0,
        "nlvr2": 0,
        "irtr": 0,
        "contras": 0,
        "snli": 0,
        "mnre": 0,
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "meter"
    seed = 0
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    image_size = 224
    patch_size = 32
    draw_false_image = 1
    image_only = False
    resolution_before = 224

    # Text Setting
    vqav2_label_size = 3129
    max_text_len = 40
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False  # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    draw_false_text = 0

    # Transformer Setting
    num_top_layer = 6
    input_image_embed_size = 768
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    hidden_size = 768
    num_heads = 12
    num_layers = 6
    mlp_ratio = 4
    drop_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-5
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 100000
    warmup_steps = 10000
    end_lr = 0
    lr_mult_head = 5  # multiply lr for downstream heads
    lr_mult_cross_modal = 5  # multiply lr for the cross-modal module

    # Downstream Setting
    get_recall_metric = False

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False
    model_type = None

    # below params varies with the environment
    data_root = "./data"
    log_dir = "result"
    per_gpu_batchsize = 32  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    load_path = ""
    num_workers = 0
    precision = 32

    checkpoint_callback = False

    rel2id = {"None": 0, "/per/per/parent": 1, "/per/per/siblings": 2, "/per/per/couple": 3, "/per/per/neighbor": 4,
              "/per/per/peer": 5, "/per/per/charges": 6, "/per/per/alumi": 7, "/per/per/alternate_names": 8,
              "/per/org/member_of": 9, "/per/loc/place_of_residence": 10, "/per/loc/place_of_birth": 11,
              "/org/org/alternate_names": 12, "/org/org/subsidiary": 13, "/org/loc/locate_at": 14,
              "/loc/loc/contain": 15, "/per/misc/present_in": 16, "/per/misc/awarded": 17, "/per/misc/race": 18,
              "/per/misc/religion": 19, "/per/misc/nationality": 20, "/misc/misc/part_of": 21, "/misc/loc/held_on": 22}
    with_auxiliary = True
    modal_type = None


@ex.named_config
def meter_clip16_roberta_pretrain():
    batch_size = 256
    draw_false_image = 0
    max_text_len = 50
    load_path = "./models/meter_clip16_224_roberta_pretrain.ckpt"

    vit = 'ViT-B/16'
    image_size = 224
    patch_size = 16
    input_image_embed_size = 768

    tokenizer = "roberta-base"
    vocab_size = 50265
    input_text_embed_size = 768
    train_transform_keys = ["clip_randaug"]


@ex.named_config
def task_finetune_mnre_clip16_roberta():
    exp_name = "finetune_mnre"
    datasets = ["mnre"]
    loss_names = _loss_names({"mnre": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0
    draw_false_image = 0
    learning_rate = 1e-5
    lr_mult_head = 10
    lr_mult_cross_modal = 10
    max_text_len = 200
    per_gpu_batchsize = 16
    # optim_type = "only_classifier"

    vit = 'ViT-B/16'
    image_size = 224
    patch_size = 16
    input_image_embed_size = 768
    train_transform_keys = ["clip_randaug"]

    tokenizer = "roberta-base"
    vocab_size = 50265
    input_text_embed_size = 768


@ex.named_config
def task_finetune_mnre_clip16_bert():
    exp_name = "finetune_mnre"
    datasets = ["mnre"]
    loss_names = _loss_names({"mnre": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0
    draw_false_image = 0
    learning_rate = 1e-5
    lr_mult_head = 10
    lr_mult_cross_modal = 10
    max_text_len = 200

    # optim_type = "only_classifier"

    vit = 'ViT-B/16'
    image_size = 224
    patch_size = 16
    input_image_embed_size = 768
    train_transform_keys = ["clip_randaug"]


@ex.named_config
def task_finetune_mnre_bert_large():
    exp_name = "finetune_mnre"
    datasets = ["mnre"]
    loss_names = _loss_names({"mnre": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0
    draw_false_image = 0
    learning_rate = 1e-5
    lr_mult_head = 10
    lr_mult_cross_modal = 10
    max_text_len = 200
    hidden_size = 1024
    modal_type = "text"

    tokenizer = "bert-large-uncased"
    vocab_size = 30522
    input_text_embed_size = 1024


@ex.named_config
def task_finetune_mnre_bert():
    exp_name = "finetune_mnre"
    datasets = ["mnre"]
    loss_names = _loss_names({"mnre": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0
    draw_false_image = 0
    learning_rate = 1e-5
    lr_mult_head = 10
    lr_mult_cross_modal = 10
    max_text_len = 200
    modal_type = "text"
    tokenizer = r"huggingface\bert-base-uncased"

@ex.named_config
def task_finetune_mnre_deberta():
    exp_name = "finetune_mnre"
    datasets = ["mnre"]
    loss_names = _loss_names({"mnre": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0
    draw_false_image = 0
    learning_rate = 1e-5
    lr_mult_head = 10
    lr_mult_cross_modal = 10
    max_text_len = 200
    modal_type = "text"

    tokenizer = "microsoft/deberta-base"
    vocab_size = 50265
    input_text_embed_size = 768


@ex.named_config
def task_finetune_mnre_roberta():
    exp_name = "finetune_mnre"
    datasets = ["mnre"]
    loss_names = _loss_names({"mnre": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0
    draw_false_image = 0
    learning_rate = 1e-5
    lr_mult_head = 10
    lr_mult_cross_modal = 10
    max_text_len = 200
    modal_type = "text"

    tokenizer = "roberta-base"
    vocab_size = 50265
    input_text_embed_size = 768


# Named configs for "etc" which are orthogonal to "env" and "task", need to be added at the end

@ex.named_config
def full_train():
    max_epoch = 30
    warmup_steps = 0.1
    seed = 0
    learning_rate = 5e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 10
    checkpoint_callback = True


@ex.named_config
def without_auxiliary():
    with_auxiliary = False
    max_text_len = 50


@ex.named_config
def test_only():
    test_only = True
    load_path = ""