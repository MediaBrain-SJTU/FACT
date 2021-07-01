from models import ResNet
from models import classifier

encoders_map = {
    'resnet18': ResNet.resnet18,
    'resnet50': ResNet.resnet50,
}

classifiers_map = {
    'base': classifier.Classifier,
}

def get_encoder(name):
    if name not in encoders_map:
        raise ValueError('Name of network unknown %s' % name)

    def get_network_fn(**kwargs):
        return encoders_map[name](**kwargs)

    return get_network_fn


def get_encoder_from_config(config):
    return get_encoder(config["name"])()


def get_classifier(name):
    if name not in classifiers_map:
        raise ValueError('Name of network unknown %s' % name)

    def get_network_fn(**kwargs):
        return classifiers_map[name](**kwargs)

    return get_network_fn

def get_classifier_from_config(config):
    cls_type = config.get("cls_type", False)
    bias = config.get("bias", False)
    scale = config.get("scale", 1.0)
    learn_scale = config.get("learn_scale", False)

    if cls_type:
        return get_classifier(config["name"])(
            in_dim=config["in_dim"],
            num_classes=config["num_classes"],
            bias=bias,
            scale=scale,
            learn_scale=learn_scale,
            cls_type=cls_type
        )
    else:
        return get_classifier(config["name"])(
            in_dim=config["in_dim"],
            num_classes=config["num_classes"],
            group=config["group"],
            bias=bias,
            scale=scale,
            learn_scale=learn_scale
        )