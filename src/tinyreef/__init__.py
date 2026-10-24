from .model import TinyReefNet


def build_model(cfg):
    m, d = cfg["model"], cfg["data"]
    return TinyReefNet(resolution=tuple(d["resolution"]), num_classes=d["num_classes"],
                       encoder_channels=tuple(m["encoder_channels"]),
                       seg_decoder_channels=tuple(m["seg_decoder_channels"]),
                       rest_decoder_channels=tuple(m["rest_decoder_channels"]))


__all__ = ["TinyReefNet", "build_model"]
