from scania_outliers.model_factory import ModelFactory


def test_model_factory_resolves_all_models():
    cfg = {"modeling": {"models": ["lstm_autoencoder", "cnn_lstm_autoencoder", "transformer_encoder"]}}
    assert ModelFactory.resolve_requested_models("all", cfg) == [
        "lstm_autoencoder",
        "cnn_lstm_autoencoder",
        "transformer_encoder",
    ]


def test_model_factory_accepts_old_transformer_alias():
    cfg = {"modeling": {"models": ["transformer_encoder_simplified"]}}
    assert ModelFactory.resolve_requested_models("all", cfg) == ["transformer_encoder"]
    assert ModelFactory.resolve_requested_models("transformer_encoder_simplified", cfg) == ["transformer_encoder"]


def test_model_factory_creates_lstm():
    cfg = {"modeling": {"hidden_dim": 8, "latent_dim": 4}}
    model = ModelFactory.create("lstm_autoencoder", n_features=3, window_size=5, config=cfg)
    assert model is not None
