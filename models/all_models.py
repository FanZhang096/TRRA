import functools
from torchvision import models
import pretrainedmodels
from .efficientnet import efficientnet_b0
from .efficientnet import efficientnet_b1
from .efficientnet import efficientnet_b2
from .efficientnet import efficientnet_b3
from .efficientnet import efficientnet_b4
from .efficientnet import efficientnet_b5
from .efficientnet import efficientnet_b6
from .efficientnet import efficientnet_b7

# To save loading time, the unused models are commented out here.
# If needed, you can uncomment.
# You can also add models that are not in the list below.
model_map = {
    # 'Dense121': models.densenet121(pretrained=True),
    # 'Dense169': models.densenet169(pretrained=True),
    # 'Dense161': models.densenet161(pretrained=True),
    # 'Dense201': models.densenet201(pretrained=True),
    #
    # 'vgg11': models.vgg11(pretrained=True),
    # 'vgg13': models.vgg13(pretrained=True),
    # 'vgg16': models.vgg16(pretrained=True),
    # 'vgg19': models.vgg19(pretrained=True),
    #
    'Resnet18': models.resnet18(pretrained=True),
    # 'Resnet34': models.resnet34(pretrained=True),
    # 'Resnet50': models.resnet50(pretrained=True),
    # 'Resnet101': models.resnet101(pretrained=True),
    # 'Resnet152': models.resnet152(pretrained=True),
    #
    # 'se-resnet50': pretrainedmodels.se_resnet50(num_classes=1000, pretrained='imagenet'),
    # 'se-resnet101': pretrainedmodels.se_resnet101(num_classes=1000, pretrained='imagenet'),
    # 'se-resnet152': pretrainedmodels.se_resnet152(num_classes=1000, pretrained='imagenet'),
    # 'senet154': pretrainedmodels.senet154(num_classes=1000, pretrained='imagenet'),

    # for efficientnet
    # 'efficientnet_b0': efficientnet_b0(num_classes=1000, pretrained='imagenet'),
    'efficientnet_b1': efficientnet_b1(num_classes=1000, pretrained='imagenet'),
    # 'efficientnet_b2': efficientnet_b2(num_classes=1000, pretrained='imagenet'),
    # 'efficientnet_b3': efficientnet_b3(num_classes=1000, pretrained='imagenet'),
    # 'efficientnet_b4': efficientnet_b4(num_classes=1000, pretrained='imagenet'),
    # 'efficientnet_b5': efficientnet_b5(num_classes=1000, pretrained='imagenet'),
    # 'efficientnet_b6': efficientnet_b6(num_classes=1000, pretrained='imagenet'),
    # 'efficientnet_b7': efficientnet_b7(num_classes=1000, pretrained='imagenet'),
}


def getModel(model_name):
    """Returns a function for a model
    Args:
      mdlParams: dictionary, contains configuration
      is_training: bool, indicates whether training is active
    Returns:
      model: A function that builds the desired model
    Raises:
      ValueError: If model name is not recognized.
    """
    if model_name not in model_map:
        raise ValueError('Name of model unknown %s' % model_name)
    func = model_map[model_name]

    @functools.wraps(func)
    def model():
        return func

    return model
