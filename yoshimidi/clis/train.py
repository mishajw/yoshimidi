import dotenv
import fire
import toml
import torch
import torch.utils.data
import tqdm
import wandb
from loguru import logger
from pydantic import BaseModel
from torch.utils.data import DataLoader

from yoshimidi.data.midi_dataset import MidiDataset
from yoshimidi.output_config import OutputConfig
from yoshimidi.train import checkpoints, evals
from yoshimidi.train.checkpoints import CheckpointConfig, load_checkpoint_
from yoshimidi.train.determinism import determinism
from yoshimidi.train.evals import EvalConfig
from yoshimidi.train.flops import calculate_flops, calculate_num_parameters
from yoshimidi.train.midi_loss import autoregressive_midi_loss
from yoshimidi.train.model import transformer
from yoshimidi.train.model.transformer_config import TransformerConfig
from yoshimidi.train.timer import Timer
from yoshimidi.train.training_config import TrainingConfig

dotenv.load_dotenv()


class Config(BaseModel, extra="forbid"):
    tag: str
    output: OutputConfig = OutputConfig()
    transformer: TransformerConfig
    training: TrainingConfig
    checkpoint: CheckpointConfig
    eval: EvalConfig
    use_wandb: bool = True


@determinism("main")
def main(config_path: str, *, resume: bool = False) -> None:
    with open(config_path) as f:
        config = Config.model_validate(toml.load(f))
    logger.info("Starting training")
    logger.info("Config: {}", config.model_dump_json(indent=2))
    logger.info(f"Num parameters: {calculate_num_parameters(config.transformer):.2E}")
    assert resume or not config.output.has_checkpoints(
        tag=config.tag
    ), f"Checkpoints already exist for tag: {config.tag}"

    if config.use_wandb:
        logger.info("Setting up WandB")
        wandb.login()
        wandb.init(
            project="yoshimidi",
            name=config.tag,
            dir=".wandb",
            config=config.model_dump(),
        )

    logger.debug("Loading dataset")
    with determinism("dataset-load"):
        dataset = MidiDataset.from_path(
            config.output.dataset_tokenized,
            context_window=config.transformer.context_window,
            device=config.training.torch_device(),
            dtype=config.training.torch_dtype(),
        )
    # TODO: We shouldn't split on the batch-level, as the same song could be split
    # between eval and train. Instead, we should split while parsing.
    with determinism("dataset-split"):
        dataset_eval, dataset_train = torch.utils.data.random_split(
            dataset, [config.eval.split, 1 - config.eval.split]
        )
    with determinism("data-loader-train"):
        data_loader_train = DataLoader(
            dataset_train,
            batch_size=config.training.batch_size,
            shuffle=True,
        )
    with determinism("data-loader-eval"):
        data_loader_eval = DataLoader(
            dataset_eval, batch_size=config.eval.batch_size, shuffle=True
        )
    logger.debug(f"Num tokens: {len(dataset) * config.transformer.context_window:.2E}")
    logger.debug(f"Num rows: {len(dataset):.2E}")
    logger.debug(f"Num batches: {len(data_loader_train):.2E}")
    logger.debug(f"Num batches (eval): {len(data_loader_eval):.2E}")

    logger.debug("Loading model")
    model = transformer.load_model(config.transformer).to(
        device=config.training.torch_device(), dtype=config.training.torch_dtype()
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    logger.debug(
        f"Num loaded parameters: {sum(p.numel() for p in model.parameters()):.2E}"
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=len(data_loader_train),
    )
    bar = tqdm.tqdm(data_loader_train, desc="Training")
    dataloader_iterator = enumerate(bar)

    if resume:
        logger.info("Resuming from checkpoint")
        checkpoint_info = load_checkpoint_(
            config.tag,
            step="latest",
            output_config=config.output,
            device=config.training.torch_device(),
            model=model,
            optimizer=optimizer,
        )
        assert checkpoint_info.step != "rolling", checkpoint_info

        logger.debug(f"Fast-forwarding {checkpoint_info.step} steps")
        # We add one: the checkpoint is saved after the step is taken.
        for step in range(checkpoint_info.step + 1):
            _, batch = next(dataloader_iterator)
            lr_scheduler.step()

    timer = Timer()
    timer.register("load-batch")
    for step, batch in dataloader_iterator:
        timer.register("zero-grad")
        model.train()
        optimizer.zero_grad()

        timer.register("forward")
        logits = model(batch)
        timer.register("forward-loss")
        loss_and_stats = autoregressive_midi_loss(batch=batch, logits=logits)

        timer.register("backward")
        loss_and_stats.loss.backward()

        timer.register("step")
        optimizer.step()
        (lr,) = lr_scheduler.get_last_lr()
        lr_scheduler.step()

        timer.register("metrics")
        metrics: dict[str, float] = {
            "loss/loss": loss_and_stats.loss.item(),
            "lr": lr,
        }
        timer.register("metrics-loss")
        for loss_name, loss_values in loss_and_stats.range_stats.items():
            metrics[f"loss/values/{loss_name}"] = loss_values.value.item()
            metrics[f"loss/entropy/{loss_name}"] = loss_values.entropy.item()
            metrics[
                f"loss/target_entropy/{loss_name}"
            ] = loss_values.target_entropy.item()
            metrics[
                f"loss/num_predicted/{loss_name}"
            ] = loss_values.num_predicted.item()
            metrics[f"loss/num_target/{loss_name}"] = loss_values.num_target.item()
        timer.register("metrics-norms")
        if config.training.metrics_schedule.should_run(
            step=step, max_steps=len(data_loader_train)
        ):
            metrics.update(
                {
                    **{
                        f"norms/{name}": param.norm().item()
                        for name, param in model.named_parameters()
                    },
                    **{
                        f"grad_norms/{name}": param.grad.norm().item()
                        if param.grad is not None
                        else -1
                        for name, param in model.named_parameters()
                    },
                }
            )

        timer.register("checkpoints")
        checkpoints.maybe_save_checkpoints(
            tag=config.tag,
            step=step,
            max_steps=len(data_loader_train),
            model=model,
            optimizer=optimizer,
            transformer_config=config.transformer,
            training_config=config.training,
            checkpoint_config=config.checkpoint,
            output_config=config.output,
        )

        timer.register("evals")
        if config.eval.schedule.should_run(step=step, max_steps=len(data_loader_train)):
            eval_loss = evals.evaluate(
                model=model,
                data_loader_eval=data_loader_eval,
            )
            bar.set_postfix(eval=eval_loss.loss.item())
            if config.use_wandb:
                eval_metrics = {
                    "evals/loss/loss": eval_loss.loss.item(),
                }
                for loss_name, loss_values in eval_loss.range_stats.items():
                    eval_metrics.update(
                        {
                            f"evals/loss/values/{loss_name}": loss_values.value.item(),
                            f"evals/loss/entropy/{loss_name}": (
                                loss_values.entropy.item()
                            ),
                            f"evals/loss/target_entropy/{loss_name}": (
                                loss_values.target_entropy.item()
                            ),
                            f"evals/loss/num_predicted/{loss_name}": (
                                loss_values.num_predicted.item()
                            ),
                            f"evals/loss/num_target/{loss_name}": (
                                loss_values.num_target.item()
                            ),
                        }
                    )
                wandb.log(eval_metrics)

        timer.register("end")
        flops = calculate_flops(
            config.transformer, config.training, timer.since_start().total_seconds()
        )
        metrics["perf/flops"] = flops
        metrics.update(timer.timing_dict())
        bar.set_postfix(loss=metrics["loss/loss"], flops=metrics["perf/flops"])
        if config.use_wandb:
            wandb.log(metrics, step=step)
        timer = Timer()
        timer.register("load-batch")


if __name__ == "__main__":
    fire.Fire(main)
