#!/usr/bin/env python3
"""Training recipe with BoundaryPredictor compression between CNN and Transformer.

Identical to train.py except the encoder forward pass is:
    feats -> CNN -> reshape(4D->3D) -> BoundaryPredictor -> Transformer
and the objective adds a binomial BP loss:
    loss = loss_asr + boundary_predictor_loss_weight * bp_loss

Run with:
    python train_bp.py hparams/conformer_small_bp.yaml
"""

import os
import sys
from pathlib import Path

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos

        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        augment_warmup = 0
        if hasattr(self.hparams, "augment_warmup"):
            augment_warmup = self.hparams.augment_warmup
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "fea_augment"):
            if self.optimizer_step > augment_warmup:
                feats, fea_lens = self.hparams.fea_augment(feats, wav_lens)
                tokens_bos = self.hparams.fea_augment.replicate_labels(
                    tokens_bos
                )

        # CNN frontend
        enc = self.modules.CNN(feats)
        enc_lens = wav_lens

        # CNN outputs (B, T, C1, C2); BP expects (B, T, D)
        if enc.dim() == 4:
            bz, t, ch1, ch2 = enc.shape
            enc = enc.reshape(bz, t, ch1 * ch2)

        # BoundaryPredictor: compresses sequence, produces binomial prior loss
        (enc, bp_loss, num_boundaries, total_positions,
         enc_lens, boundary_cv, boundary_adjacent_pct,
         _bp_probs, _bp_n_per_sample,
         _bp_total_per_sample) = self.modules.BoundaryPredictor(
            hidden=enc,
            lengths=enc_lens,
            target_boundary_counts=None,
            return_unreduced_boundary_loss=False,
        )

        self.boundary_predictor_loss = bp_loss
        self.num_boundaries = num_boundaries
        self.total_positions = total_positions
        self.boundary_cv = boundary_cv
        self.boundary_adjacent_pct = boundary_adjacent_pct

        enc_out, pred = self.modules.Transformer(
            enc, tokens_bos, enc_lens, pad_idx=self.hparams.pad_index
        )

        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        hyps = None
        current_epoch = self.hparams.epoch_counter.current
        is_valid_search = (
            stage == sb.Stage.VALID
            and current_epoch % self.hparams.valid_search_interval == 0
        )
        is_test_search = stage == sb.Stage.TEST

        if any([is_valid_search, is_test_search]):
            if stage == sb.Stage.VALID:
                hyps, _, _, _ = self.hparams.valid_search(
                    enc_out.detach(), enc_lens
                )
            else:
                hyps, _, _, _ = self.hparams.test_search(
                    enc_out.detach(), enc_lens
                )

        return p_ctc, p_seq, enc_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        (p_ctc, p_seq, enc_lens, hyps) = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if stage == sb.Stage.TRAIN:
            augment_warmup = 0
            if hasattr(self.hparams, "augment_warmup"):
                augment_warmup = self.hparams.augment_warmup
            if (
                hasattr(self.hparams, "fea_augment")
                and self.optimizer_step > augment_warmup
            ):
                (
                    tokens,
                    tokens_lens,
                    tokens_eos,
                    tokens_eos_lens,
                ) = self.hparams.fea_augment.replicate_multiple_labels(
                    tokens, tokens_lens, tokens_eos, tokens_eos_lens
                )

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        ).sum()

        loss_ctc = self.hparams.ctc_cost(
            p_ctc, tokens, enc_lens, tokens_lens
        ).sum()

        loss_asr = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_seq
        )

        if stage == sb.Stage.TRAIN:
            loss_bp = self.boundary_predictor_loss
        else:
            loss_bp = torch.tensor(0.0, device=p_ctc.device)

        loss = loss_asr + self.hparams.boundary_predictor_loss_weight * loss_bp

        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or (
                stage == sb.Stage.TEST
            ):
                predicted_words = [
                    tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in hyps
                ]
                target_words = [wrd.split(" ") for wrd in batch.wrd]
                self.wer_metric.append(ids, predicted_words, target_words)

            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)
        return loss

    def on_evaluate_start(self, max_key=None, min_key=None):
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model"
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()
        print("Loaded the average")

    def on_stage_start(self, stage, epoch):
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.wer_metric = self.hparams.error_rate_computer()
        else:
            # Linear temperature anneal 1.0 -> ~0 across training, clamped 1 step short.
            total_epochs = self.hparams.number_of_epochs
            if total_epochs > 1:
                effective_epoch = min(epoch, total_epochs - 2)
                temperature = 1.0 - (effective_epoch / (total_epochs - 1))
            else:
                temperature = 1.0
            self.modules.BoundaryPredictor.set_temperature(temperature)
            logger.info(
                f"Epoch {epoch}: BoundaryPredictor temperature = {temperature:.4f}"
            )

    def on_stage_end(self, stage, stage_loss, epoch):
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
            if getattr(self, "total_positions", 0) > 0:
                stage_stats["boundary_rate"] = (
                    self.num_boundaries / self.total_positions
                )
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            if getattr(self, "total_positions", 0) > 0:
                stage_stats["boundary_rate"] = (
                    self.num_boundaries / self.total_positions
                )
            if self.boundary_cv is not None:
                stage_stats["boundary_cv"] = self.boundary_cv
            if self.boundary_adjacent_pct is not None:
                stage_stats["boundary_adjacent_pct"] = self.boundary_adjacent_pct

        if stage == sb.Stage.VALID:
            lr = self.hparams.noam_annealing.current_lr
            steps = self.optimizer_step
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            if hasattr(self.hparams, "wandb_logger"):
                self.hparams.wandb_logger.log_stats(
                    stats_meta=epoch_stats,
                    train_stats=self.train_stats,
                    valid_stats=stage_stats,
                )
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=self.hparams.avg_checkpoints,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if hasattr(self.hparams, "wandb_logger"):
                self.hparams.wandb_logger.log_stats(
                    stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                    test_stats=stage_stats,
                )
            if if_main_process():
                with open(
                    self.hparams.test_wer_file, "w", encoding="utf-8"
                ) as w:
                    self.wer_metric.write_stats(w)

            self.checkpointer.save_and_keep_only(
                meta={"ACC": 1.1, "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=1,
            )

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        if should_step:
            self.hparams.noam_annealing(self.optimizer)


def dataio_prepare(hparams):
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        train_data = train_data.filtered_sorted(sort_key="duration")
        hparams["train_dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        hparams["train_dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] == "random":
        pass
    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]
    valtest_datasets = [valid_data] + [i for k, i in test_datasets.items()]

    tokenizer = hparams["tokenizer"]

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(valtest_datasets, audio_pipeline)

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline_train(wav):
        if "speed_perturb" in hparams:
            sig = sb.dataio.dataio.read_audio(wav)
            sig = hparams["speed_perturb"](sig.unsqueeze(0)).squeeze(0)
        else:
            sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item([train_data], audio_pipeline_train)

    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
    )

    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams_train = hparams["dynamic_batch_sampler_train"]
        dynamic_hparams_valid = hparams["dynamic_batch_sampler_valid"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            length_func=lambda x: x["duration"],
            **dynamic_hparams_train,
        )
        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            length_func=lambda x: x["duration"],
            **dynamic_hparams_valid,
        )

    return (
        train_data,
        valid_data,
        test_datasets,
        tokenizer,
        train_batch_sampler,
        valid_batch_sampler,
    )


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    sb.utils.distributed.ddp_init_group(run_opts)

    from librispeech_prepare import prepare_librispeech  # noqa

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )

    (
        train_data,
        valid_data,
        test_datasets,
        tokenizer,
        train_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams)

    hparams["pretrainer"].collect_files()
    hparams["pretrainer"].load_collected()

    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    asr_brain.tokenizer = hparams["tokenizer"]
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        collate_fn = None
        if "collate_fn" in train_dataloader_opts:
            collate_fn = train_dataloader_opts["collate_fn"]

        train_dataloader_opts = {
            "batch_sampler": train_bsampler,
            "num_workers": hparams["num_workers"],
        }

        if collate_fn is not None:
            train_dataloader_opts["collate_fn"] = collate_fn

    if valid_bsampler is not None:
        collate_fn = None
        if "collate_fn" in valid_dataloader_opts:
            collate_fn = valid_dataloader_opts["collate_fn"]

        valid_dataloader_opts = {"batch_sampler": valid_bsampler}

        if collate_fn is not None:
            valid_dataloader_opts["collate_fn"] = collate_fn

    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )

    if not os.path.exists(hparams["output_wer_folder"]):
        os.makedirs(hparams["output_wer_folder"])

    for k in test_datasets.keys():
        asr_brain.hparams.test_wer_file = os.path.join(
            hparams["output_wer_folder"], f"wer_{k}.txt"
        )
        asr_brain.evaluate(
            test_datasets[k],
            max_key="ACC",
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )
