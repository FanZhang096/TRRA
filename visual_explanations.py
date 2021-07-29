from tools.torchcam.cams.gradcam import GradCAMpp
import os
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
import torch.nn
from models import all_models
import matplotlib.pyplot as plt
from tools.torchcam.utils import overlay_mask

"""
Reference : https://github.com/frgfm/torch-cam
"""


def get_model(path):
    # load trained model
    model = all_models.getModel('efficientnet_b1')()
    num_ftrs = model._fc.in_features
    model._fc = torch.nn.Linear(num_ftrs, 2)
    model = model.cuda()
    model = torch.nn.DataParallel(model, [0, 1])
    param_dict = torch.load(path)
    model.load_state_dict(param_dict['model'])
    return model


def input_image(subject, slice):
    transformations1 = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transformations2 = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor()])
    path = '/home/zxt/4T/ADNI_AD_CN_CAPS1/subjects/sub-ADNI%s/ses-M00/deeplearning_prepare_data/slice_based/t1_linear/sub-ADNI%s_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_axis-sag_channel-rgb_slice-%s_T1w.pt' % (subject, subject, slice)
    img = torch.load(path)
    input = transformations1(img)
    show_input = transformations2(img)
    return input, show_input


os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5'
model_path = 'results/efficientnet_b1_learning_rate_5e-05_weight_decay_0.001_labels_1/fold-0/models/best_accuracy/model_best.pth.tar'
model = get_model(model_path)
model.eval()
cam_extractor = GradCAMpp(model, target_layer='module._bn1')

subject_number = '002S0955'
slice_number = 20
for i in range(42, 43):
    # Get your input
    slice_number = i
    input_tensor, img = input_image(subject_number, slice_number)
    # Preprocess your data and feed it to the model
    out = model(input_tensor.unsqueeze(0))
    print(out, out.squeeze(0).argmax().item())
    # Retrieve the CAM by passing the class index and the model output
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    # Visualize the raw CAMtemplate
    plt.imshow(activation_map.cpu().numpy()); plt.axis('off'); plt.tight_layout(); plt.show()
    # Resize the CAM and overlay ittemplate
    result = overlay_mask(to_pil_image(img), to_pil_image(activation_map.cpu(), mode='F'), alpha=0.5)
    # Display it
    plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()
    plt.imshow(to_pil_image(img)); plt.axis('off'); plt.tight_layout(); plt.show()
    # to_pil_image(img).save('/home/zxt/4T/codes/Alzheimer_disease_AD_CN_with_test1/036S5063_1/original/%s.jpeg' % str(i), quality=90)
    # result.save('/home/zxt/4T/codes/Alzheimer_disease_AD_CN_with_test1/036S5063_1/overlay/%s.jpeg' % str(i), quality=90)
