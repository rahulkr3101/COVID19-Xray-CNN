from covidcnn.model import ModelConfig, build_model


def test_build_model_output_shape():
    model = build_model(ModelConfig(input_shape=(64, 64, 3)))
    assert model.output_shape == (None, 1)


def test_build_model_is_binary_classifier():
    model = build_model(ModelConfig())
    assert model.loss == "binary_crossentropy"
