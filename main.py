#Import the training data for English and Spanish and
# import pandas as pd
from utils.datareader import en_train_df, es_train_df
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.fine_tuning import training, validate
# from utils.logger import Logger
# import logging
from utils.utils import set_seed, product_dict
from sklearn.model_selection import KFold
import wandb
from datetime import datetime

# Get current date and time
current_datetime = datetime.now()

# Format it to include hours, minutes, and seconds
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")


wandb.login()

SEED=1234
set_seed(SEED)

preconfig = {
    0: {
        "lang": "english",
        "model_name": "roberta-base",
    },
    1: {
        "lang": "english",
        "model_name": "microsoft/deberta-base",
    },
    # 2: {
    #     "lang": "spanish",
    #     "model_name": "dccuchile/bert-base-spanish-wwm-uncased",
    # },
    # 3: {
    #     "lang": "spanish",
    #     "model_name": "PlanTL-GOB-ES/roberta-base-bne",
    # },
    # 4: {
    #     "lang": "spanish",
    #     "model_name": "bert-base-multilingual-uncased"
    # }
}

hyperparams = {
    "optimizer_name": ["adam", "rmsprop"], # ["adam", "rmsprop", "sgd"]
    "learning": [0.5e-5, 1e-6], # [0.5e-5, 1e-5, 0.5e-6, 1e-6
    "schedule": ["linear", "cosine"], # ["linear", "cosine", "constant"]
    "patience": [5, 10], # [3, 5, 10]
    "epochs": [5, 20], # [5, 10, 20]
    "measure": ["mcc"],
    "batch_size": [32], # [16, 32, 64, 128]
    "max_length": [128]
}

# epochs = 5 #[5, 10, 20]
# batch_size = 32 #[16, 32, 64, 128]
# measure = "mcc"
# patience = 3 #[5, 10]
# max_length = 128 #[This value can be estimated on the training set]
# Define KFold cross-validation
kf = KFold(n_splits=5)

# For each preconfiguration
for i, preconfig in preconfig.items():
    lang = preconfig["lang"]
    model_name = preconfig["model_name"]
    
    if lang == "spanish":
        X= es_train_df
    elif lang == "english":
        X= en_train_df
    
    # Start a parent run for this preconfiguration
    # parent_run = wandb.init(project='lnr_oppositional_thinking',
    #                         entity='davidandreuroqueta',
    #                         group=f'{lang}_{model_name}',
    #                         job_type='model')
    # parent_run.config.update(preconfig)
    # parent_run.config.update({"SEED":SEED})

    print("Loading Tokenizer " + model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading Transformer Model " + model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


    # Initialize a counter for the runs
    run_counter = 0

    # For each hyperparameter configuration
    for config in product_dict(**hyperparams):
        run_counter += 1
        # Start a child run for this hyperparameter configuration
        # with wandb.init(project='lnr_oppositional_thinking',
        #                 entity='davidandreuroqueta',
        #                 group=f'{lang}_{model_name}',
        #                 job_type='hyperparam-tuning',
        #                 name=f'{lang}_{model_name}_{run_counter}',
        #                 ) as run:
        #     # Log hyperparameters
        #     run.config.update(config)
            
        # For each fold
        for fold, (train_index, val_index) in enumerate(kf.split(X)):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]

            # Start a child run for this fold
            with wandb.init(project=f'lnr_oppositional_thinking_{formatted_datetime}',
                            entity='davidandreuroqueta',
                            group=f'{lang}_{model_name}',
                            job_type=f'hyperparam-tuning-{run_counter}',
                            name=f'{lang}_{model_name}_{run_counter}_fold_{fold}'
                            ) as fold_run:
                fold_run.config.update(preconfig)
                fold_run.config.update(config)
                fold_run.config.update({"SEED":SEED})

                # Log the fold number
                fold_run.config.update({"fold": fold + 1})

                # Train and validate your model, log metrics, etc.
                # ...
                # FINE-TUNING the model and obtaining the best model across all epochs
                fineTmodel = training(_wandb=fold_run, _model=model, _train_data=X_train, _val_data=X_val,
                                    _learning_rate=config["learning"], _optimizer_name=config["optimizer_name"],
                                    _schedule=config["schedule"], _epochs=config["epochs"], _tokenizer=tokenizer,
                                    _batch_size=config["batch_size"], _padding="max_length", _max_length=config["max_length"],
                                    _truncation=True, _patience=config["patience"], _measure=config["measure"], _out="./out")

                # VALIDATING OR PREDICTING on the test partition, this time I'm using the validation set, but you have to use the test set.
                preds = validate(_wandb=fold_run, _model=fineTmodel, _test_data=X_val, _tokenizer=tokenizer,
                                _batch_size=config["batch_size"], _padding="max_length", _max_length=config["max_length"],
                                _truncation=True, _measure=config["measure"], evaltype=True)

    # End the parent run
    # parent_run.finish()
