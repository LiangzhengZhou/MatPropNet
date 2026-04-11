from matpropnet.config.loader import validate_config


def test_validate_config_injects_static_loss_weighting_default():
    config = {
        "trainer": "property",
        "model": {"name": "property_model"},
        "task": {"tasks": {"target1": {"type": "regression"}}},
        "optim": {"lr_initial": 1.0e-3},
    }

    validated = validate_config(config)

    assert validated["loss_weighting"]["mode"] == "static"
