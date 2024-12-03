import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import os

class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False)

    def forward(self, x):
        return self.conv(x)
    
    
class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()
        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")

class fSEModule(nn.Module):
    def __init__(self, high_feature_channel, low_feature_channels, output_channel=None):
        super(fSEModule, self).__init__()
        in_channel = high_feature_channel + low_feature_channels
        out_channel = high_feature_channel
        if output_channel is not None:
            out_channel = output_channel
        reduction = 16
        channel = in_channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

        self.conv_se = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, high_features, low_features):
        features = [upsample(high_features)]
        features += low_features
        features = torch.cat(features, 1)

        b, c, _, _ = features.size()
        y = self.avg_pool(features).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        y = self.sigmoid(y)
        features = features * y.expand_as(features)

        return self.relu(self.conv_se(features))



class ResNetMultiImageInput(models.ResNet):
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        features.append(self.encoder.relu(x))
        features.append(self.encoder.layer1(self.encoder.maxpool(features[-1])))
        features.append(self.encoder.layer2(features[-1]))
        features.append(self.encoder.layer3(features[-1]))
        features.append(self.encoder.layer4(features[-1]))

        return features


class HRDepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, mobile_encoder=False):
        super(HRDepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.num_ch_enc = num_ch_enc
        self.scales = scales
        self.mobile_encoder = mobile_encoder
        if mobile_encoder:
            self.num_ch_dec = np.array([4, 12, 20, 40, 80])
        else:
            self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.all_position = ["01", "11", "21", "31", "02", "12", "22", "03", "13", "04"]
        self.attention_position = ["31", "22", "13", "04"]
        self.non_attention_position = ["01", "11", "21", "02", "12", "03"]
            
        self.convs = nn.ModuleDict()
        for j in range(5):
            for i in range(5 - j):
                # upconv 0
                num_ch_in = num_ch_enc[i]
                if i == 0 and j != 0:
                    num_ch_in /= 2
                num_ch_out = num_ch_in / 2
                self.convs["X_{}{}_Conv_0".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)

                # X_04 upconv 1, only add X_04 convolution
                if i == 0 and j == 4:
                    num_ch_in = num_ch_out
                    num_ch_out = self.num_ch_dec[i]
                    self.convs["X_{}{}_Conv_1".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)

        # declare fSEModule and original module
        for index in self.attention_position:
            row = int(index[0])
            col = int(index[1])
            if mobile_encoder:
                self.convs["X_" + index + "_attention"] = fSEModule(num_ch_enc[row + 1] // 2, self.num_ch_enc[row]
                                                                          + self.num_ch_dec[row]*2*(col-1),
                                                                         output_channel=self.num_ch_dec[row] * 2)
            else:
                self.convs["X_" + index + "_attention"] = fSEModule(num_ch_enc[row + 1] // 2, self.num_ch_enc[row]
                                                                         + self.num_ch_dec[row + 1] * (col - 1))
        for index in self.non_attention_position:
            row = int(index[0])
            col = int(index[1])
            if mobile_encoder:
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(
                    self.num_ch_enc[row]+ self.num_ch_enc[row + 1] // 2 +
                    self.num_ch_dec[row]*2*(col-1), self.num_ch_dec[row] * 2)
            else:
                if col == 1:
                    self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(num_ch_enc[row + 1] // 2 +
                                                                            self.num_ch_enc[row], self.num_ch_dec[row + 1])
                else:
                    self.convs["X_"+index+"_downsample"] = Conv1x1(num_ch_enc[row+1] // 2 + self.num_ch_enc[row]
                                                                          + self.num_ch_dec[row+1]*(col-1), self.num_ch_dec[row + 1] * 2)
                    self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(self.num_ch_dec[row + 1] * 2, self.num_ch_dec[row + 1])

        if self.mobile_encoder:
            self.convs["dispConvScale0"] = Conv3x3(4, self.num_output_channels)
            self.convs["dispConvScale1"] = Conv3x3(8, self.num_output_channels)
            self.convs["dispConvScale2"] = Conv3x3(24, self.num_output_channels)
            self.convs["dispConvScale3"] = Conv3x3(40, self.num_output_channels)
        else:
            for i in range(4):
                self.convs["dispConvScale{}".format(i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def nestConv(self, conv, high_feature, low_features):
        conv_0 = conv[0]
        conv_1 = conv[1]
        assert isinstance(low_features, list)
        high_features = [upsample(conv_0(high_feature))]
        for feature in low_features:
            high_features.append(feature)
        high_features = torch.cat(high_features, 1)
        if len(conv) == 3:
            high_features = conv[2](high_features)
        return conv_1(high_features)

    def forward(self, input_features):
        outputs = {}
        features = {}
        for i in range(5):
            features["X_{}0".format(i)] = input_features[i]
        # Network architecture
        for index in self.all_position:
            row = int(index[0])
            col = int(index[1])
            low_features = []
            for i in range(col):
                low_features.append(features["X_{}{}".format(row, i)])
            # add fSE block to decoder
            if index in self.attention_position:
                features["X_"+index] = self.convs["X_" + index + "_attention"](
                    self.convs["X_{}{}_Conv_0".format(row+1, col-1)](features["X_{}{}".format(row+1, col-1)]), low_features)
            elif index in self.non_attention_position:
                conv = [self.convs["X_{}{}_Conv_0".format(row + 1, col - 1)],
                        self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)]]
                if col != 1 and not self.mobile_encoder:
                    conv.append(self.convs["X_" + index + "_downsample"])
                features["X_" + index] = self.nestConv(conv, features["X_{}{}".format(row+1, col-1)], low_features)

        x = features["X_04"]
        x = self.convs["X_04_Conv_0"](x)
        x = self.convs["X_04_Conv_1"](upsample(x))
        outputs[("disparity", "Scale0")] = self.sigmoid(self.convs["dispConvScale0"](x))
        outputs[("disparity", "Scale1")] = self.sigmoid(self.convs["dispConvScale1"](features["X_04"]))
        outputs[("disparity", "Scale2")] = self.sigmoid(self.convs["dispConvScale2"](features["X_13"]))
        outputs[("disparity", "Scale3")] = self.sigmoid(self.convs["dispConvScale3"](features["X_22"]))
        return outputs



import cv2
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm


def disp_rescale(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


"""
Define and Load Model
"""
depth_encoder = ResnetEncoder(18, False)
depth_decoder = HRDepthDecoder(depth_encoder.num_ch_enc)

depth_encoder_path = 'ckpts/encoder.pth'
depth_decoder_path = 'ckpts/depth.pth'

encoder_dict = torch.load(depth_encoder_path)
img_height = encoder_dict["height"]
img_width = encoder_dict["width"]
print("Test image height is:", img_height)
print("Test image width is:", img_width)
load_dict = {k: v for k, v in encoder_dict.items() if k in depth_encoder.state_dict()}

decoder_dict = torch.load(depth_decoder_path)

depth_encoder.load_state_dict(load_dict)
depth_decoder.load_state_dict(decoder_dict)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
depth_encoder = depth_encoder.to(device)
depth_decoder = depth_decoder.to(device)

"""
Settings for Inference
"""
MEAN = [0.45734706, 0.43338275, 0.40058118]
STD = [0.23965294, 0.23532275, 0.2398498]
H_org = 1024
W_org = 2048


"""
Prepare Inference Dataset
"""
toTensor = transforms.ToTensor()
normTensor = transforms.Normalize(MEAN,STD)

root_dir = "." 
result_dir = os.path.join(root_dir,"results")

os.makedirs(result_dir,exist_ok=True)


depth_pred_dir = os.path.join(result_dir,"depth")
os.makedirs(depth_pred_dir,exist_ok=True)

images = glob.glob(os.path.join(root_dir,'student_dataset/student_test/current_image/*.png'))


##  define MAX and MIN depth
MIN_DEPTH = 1e-3
MAX_DEPTH = 4 #80 

for image in tqdm(images):
    name = os.path.basename(image)
    image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_RGB2BGR)

    imageT = normTensor(toTensor(image)).unsqueeze(0).cuda()
    

    ########################################
    ###########Your Implementation##########
    ########################################


    ## predict the disparity of monocular input
    output = depth_decoder(depth_encoder(imageT))
    
    pred_disp, _ = disp_rescale(output[("disparity", "Scale0")], 0.1, 100.0)
    pred_disp = pred_disp.detach()

    pred_disp = pred_disp.cpu()[:, 0].numpy()

    ## Resize to the original resolution
    HH, WW = image.shape[:2]
    #new_pred_disp = cv2.resize(pred_disp, (WW, HH))

    # Convert disparity to depth
    pred_depth = 1 / pred_disp
    
    MAX_DEPTH = np.max(pred_depth)
    ## cap the range of the depth estimation
    pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
    #pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
    
    # Normalize the depth map for saving as an image
    norm_depth = (pred_depth - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
    norm_depth = (norm_depth * 255).astype(np.uint8)
    
    # Save the mono-depth prediction result
    depth_save_path = os.path.join(depth_pred_dir, f"{name}")
    norm_depth = norm_depth[0]
    cv2.imwrite(depth_save_path, norm_depth)
