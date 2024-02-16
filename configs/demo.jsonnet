##################
# Model settings #
##################
local pretrained_model = "distilbert-base-cased";

####################
# Trainer settings #
####################

# Trainer settings, adjust to your use-case.
local training_steps = 4000;  # total number of optimization steps to train for
local validate_every = 400;  # how often to validate and save checkpoints

# This is the batch size per GPU, ignoring gradient accumulation:
local batch_size = 64;

local amp = false;  # use PyTorch's native automatic mixed precision

######################
# Optimizer settings #
######################

local warmup_steps = 400;
local learning_rate = 3e-5;  # you can probably use a higher LR for a small model like "gpt2"

local training_engine = {
    type: "torch",
    optimizer: {
        type: "torch::AdamW",
        lr: learning_rate,
        betas: [0.9, 0.95],
        eps: 1e-5,
    },
    lr_scheduler: {
        type: "transformers::linear",
        num_warmup_steps: warmup_steps,
        num_training_steps: training_steps,
    },
    amp: amp
};

local collate_fn = {
    type: "transformers::DataCollatorWithPadding",
    tokenizer: {
        pretrained_model_name_or_path: pretrained_model
    }
};
local train_dataloader = {
    shuffle: true,
    batch_size: batch_size,
    collate_fn: collate_fn,
};
local val_dataloader = {
    shuffle: false,
    batch_size: batch_size,
    collate_fn: collate_fn
};

{
    steps: {
        stype_instances: {
            type: "construct_stype_instances",
            train_conllu: "data/en_gum-ud-train.conllu",
            dev_conllu: "data/en_gum-ud-dev.conllu",
            test_conllu: "data/en_gum-ud-test.conllu",
            tokenizer: { pretrained_model_name_or_path: pretrained_model }
        },
        label_count: {
            type: "label_count",
            dataset: { type: "ref", ref: "stype_instances" }
        },
        trained_model: {
            type: "torch::train",
            model: {
                //type: "transformers::AutoModelForSequenceClassification::from_config",
                type: "demo_auto_model_wrapper::from_config",
                config: {
                    pretrained_model_name_or_path: pretrained_model,
                    num_labels: { type: "ref", ref: "label_count" },
                    problem_type: "single_label_classification"
                },
            },
            dataset_dict: { type: "ref", ref: "stype_instances" },
            training_engine: training_engine,
            log_every: 1,
            train_dataloader: train_dataloader,
            train_steps: training_steps,
            validate_every: validate_every,
            checkpoint_every: validate_every,
            validation_split: "dev",
            validation_dataloader: val_dataloader,
            val_metric_name: "accuracy",
            minimize_val_metric: false,
        },
        final_metrics: {
            type: "torch::eval",
            model: { type: "ref", ref: "trained_model" },
            dataset_dict: { type: "ref", ref: "stype_instances" },
            dataloader: val_dataloader,
            metric_names: ["loss", "accuracy"],
            test_split: "test",
        },
    }
}
