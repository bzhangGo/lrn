# coding: utf-8

from lrs import vanillalr, noamlr, scorelr, gnmtplr, epochlr


def get_lr(params):

    strategy = params.lrate_strategy.lower()

    if strategy == "noam":
        return noamlr.NoamDecayLr(
            params.lrate,
            params.warmup_steps,
            params.hidden_size
        )
    elif strategy == "gnmt+":
        return gnmtplr.GNMTPDecayLr(
            params.lrate,
            params.warmup_steps,
            params.nstable,
            params.lrdecay_start,
            params.lrdecay_end
        )
    elif strategy == "epoch":
        return epochlr.EpochDecayLr(
            params.lrate,
            params.lrate_decay,
        )
    elif strategy == "score":
        return scorelr.ScoreDecayLr(
            params.lrate,
            history_scores=[v[1] for v in params.recorder.valid_script_scores],
            decay=params.lrate_decay,
            patience=params.lrate_patience,
        )
    elif strategy == "vanilla":
        return vanillalr.VanillaLR(
            params.lrate,
        )
    else:
        raise NotImplementedError(
            "{} is not supported".format(strategy))