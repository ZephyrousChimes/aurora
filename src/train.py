
from tqdm import tqdm

from functools import partial
from transformers import PretrainedConfig

from data import load_dataset
from src.dataset import create_backtest_dataloader, create_train_dataloader, transform_start_field
from gluonts.dataset.multivariate_grouper import MultivariateGrouper

from gluonts.time_feature import time_features_from_frequency_str


from transformers import InformerConfig, InformerForPrediction

from accelerate import Accelerator
from torch.optim import AdamW

import torch

def get_dataloaders():
    dataset = load_dataset()
    freq="1D"
    prediction_length = 30

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    validation_dataset = dataset["validation"]

    train_dataset.set_transform(partial(transform_start_field, freq=freq))
    test_dataset.set_transform(partial(transform_start_field, freq=freq))


    num_of_variates = len(train_dataset)

    time_features = time_features_from_frequency_str(freq)

    config = InformerConfig(
        input_size=num_of_variates,
        prediction_length=prediction_length,
        context_length=prediction_length * 2,
        lags_sequence=[1, 2, 3],
        num_time_features=len(time_features) + 1,
        
        dropout=0.1,
        encoder_layers=6,
        decoder_layers=4,
        d_model=64,
    )

    train_grouper = MultivariateGrouper(max_target_dim=num_of_variates)
    test_grouper = MultivariateGrouper(
        max_target_dim=num_of_variates,
        num_test_dates=len(test_dataset) // num_of_variates,
    )

    multi_variate_train_dataset = train_grouper(train_dataset)
    multi_variate_test_dataset = test_grouper(test_dataset)

    train_dataloader = create_train_dataloader(
        config=config,
        freq=freq,
        data=multi_variate_train_dataset,
        batch_size=256,
        num_batches_per_epoch=100,
        num_workers=2,
    )

    test_dataloader = create_backtest_dataloader(
        config=config,
        freq=freq,
        data=multi_variate_test_dataset,
        batch_size=32,
    )

    return train_dataloader, test_dataloader, config


def train():
    train_dataloader, test_dataloader, config = get_dataloaders()

    model = InformerForPrediction(config)

    epochs = 100
    loss_history = []

    accelerator = Accelerator()
    device = accelerator.device

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=1e-1)

    model, optimizer, train_dataloader = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
    )

    model.train()
    pbar = tqdm(train_dataloader, total=25557)
    for epoch in range(epochs):
        for idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            outputs = model(
                static_categorical_features=batch["static_categorical_features"].to(device)
                if config.num_static_categorical_features > 0
                else None,
                static_real_features=batch["static_real_features"].to(device)
                if config.num_static_real_features > 0
                else None,
                past_time_features=batch["past_time_features"].to(device),
                past_values=batch["past_values"].to(device),
                future_time_features=batch["future_time_features"].to(device),
                future_values=batch["future_values"].to(device),
                past_observed_mask=batch["past_observed_mask"].to(device),
                future_observed_mask=batch["future_observed_mask"].to(device),
            )
            loss = outputs.loss
            
            accelerator.backward(loss)
            optimizer.step()

            

            loss_history.append(loss.item())
            pbar.set_postfix(loss=loss.item())

            if idx % 1000 == 0:
                torch.save(model, f"checkpoints/model_{idx}.pt")


