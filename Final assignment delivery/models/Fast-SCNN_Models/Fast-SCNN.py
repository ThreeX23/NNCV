import torch
import torch.nn as nn
import torch.nn.functional as F

    """
    Fast-SCNN model based on [1]:
    Rudra PK Poudel, Stephan Liwicki, Roberto Cipolla,
    "Fast-SCNN: Fast Semantic Segmentation Network"
    
    Help with code implementation from:
    https://github.com/Tramac/Fast-SCNN-pytorch/blob/master/models/fast_scnn.py
    
    CityScapes data, size 1024x2048x3
    
    Two possible research questions, depending on the Fast-SCNN accuracy.
    Can the Fast-SCNN be reduced in size in order to get 80% of the UNet performance?
    Can the Fast-SCNN be improved to reach 80% of UNet performance?
    """


########################################### Blocks ###########################################   
class DWConv(nn.Module):
    """Depthwise Convolution"""
    def __init__(self, in_channels, out_channels, stride):
        super(DWConv, self).__init__()
        self.conv = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x)
        
            
class DSConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_channels, out_channels, stride):
        super(DSConv, self).__init__()
        self.conv = nn.Sequential(
          nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
          nn.BatchNorm2d(in_channels),
          nn.ReLU(True),
          nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv(x)
        
class Bottleneck(nn.Module):
    """The setup can be seen in Table 2 of [1]"""
    def __init__(self, in_channels, out_channels, t, stride):
        super(Bottleneck, self).__init__()
        self.residual_shortcut = (stride==1 and in_channels==out_channels)
        self.bottleneck = nn.Sequential(
          nn.Conv2d(in_channels, t*in_channels, kernel_size=1, stride=1, padding=0, bias=False),
          nn.BatchNorm2d(t*in_channels),
          nn.ReLU(True),
          DWConv(t*in_channels, t*in_channels, stride=stride),
          nn.Conv2d(in_channels * t, out_channels, kernel_size=1, bias=False),
          nn.BatchNorm2d(out_channels)
        )
      
    def forward(self, x):
        out = self.bottleneck(x)
        # In [1] in 3.2.2 residual connection is described
        if self.residual_shortcut:
            out = x + out 
        return out
        
class PPM(nn.Module):
    """Pyramid Pooling Module (PPM)"""
    def __init__(self, in_channels, out_channels):
        super(PPM, self).__init__()
        channels = int(in_channels/4)
        self.conv1 = nn.Sequential( 
          nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
          nn.BatchNorm2d(channels),
          nn.ReLU(True)
        )
        self.conv2 = nn.Sequential( 
          nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
          nn.BatchNorm2d(channels),
          nn.ReLU(True)
        )
        self.conv3 = nn.Sequential( 
          nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
          nn.BatchNorm2d(channels),
          nn.ReLU(True)
        )
        self.conv4 = nn.Sequential( 
          nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
          nn.BatchNorm2d(channels),
          nn.ReLU(True)
        )
        self.conv_out = nn.Sequential( 
          nn.Conv2d(in_channels*2, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(True)
        )
    
    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)
        
    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.conv_out(x)
        return x
        


########################################## Functions ##########################################         
class LearningToDownsample(nn.Module):
    """First stage: Lightweight downsampling"""
    def __init__(self, in_channels, channel_1, channel_2, out_channels):
        super(LearningToDownsample, self).__init__()
        self.conv1 = nn.Sequential( 
          nn.Conv2d(in_channels, channel_1, kernel_size=3, stride=2, padding=0, bias=False),
          nn.BatchNorm2d(channel_1),
          nn.ReLU(True)
        )
        self.conv2 = DSConv(channel_1, channel_2, 2)
        self.conv3 = DSConv(channel_2, out_channels, 2)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        return x3
        
class GlobalFeatureExtractor(nn.Module):
    """Second stage: Capture long-range dependencies"""
    def __init__(self, in_channels, out_channels, t):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = nn.Sequential(
          Bottleneck(in_channels, 64, t, stride=2),
          Bottleneck(64, 64, t, stride=1),
          Bottleneck(64, 64, t, stride=1)
        )
        self.bottleneck2 = nn.Sequential(
          Bottleneck(64, 96, t, stride=2),
          Bottleneck(96, 96, t, stride=1),
          Bottleneck(96, 96, t, stride=1)
        )
        self.bottleneck3 = nn.Sequential(
          Bottleneck(96, 128, t, stride=1),
          Bottleneck(128, 128, t, stride=1),
          Bottleneck(128, 128, t, stride=1)
        )
        self.ppm = PPM(128, out_channels)
        
    def forward(self, x):
        x1 = self.bottleneck1(x)
        x2 = self.bottleneck2(x1)
        x3 = self.bottleneck3(x2)
        x4 = self.ppm(x3)
        return x4
        
        
class FeatureFusion(nn.Module):
    """Third stage: Merge low-level and high-level features"""
    def __init__(self, high_res_in_channels, low_res_in_channels, out_channels, scale_factor):
        super(FeatureFusion, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = DWConv(low_res_in_channels, out_channels, stride=1)
        self.conv_low_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size = 1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_high_res = nn.Sequential(
            nn.Conv2d(high_res_in_channels, out_channels, kernel_size = 1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(True)

    def forward(self, high_res_feature, low_res_feature):
        low_res_feature = F.interpolate(low_res_feature, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        low_res_feature = self.dwconv(low_res_feature)
        low_res_feature = self.conv_low_res(low_res_feature)

        high_res_feature = self.conv_high_res(high_res_feature)
        out = high_res_feature + low_res_feature
        return self.relu(out)
        
        
class Classifier(nn.Module):
    """Final segmentation prediction"""
    def __init__(self, in_channels, num_classes, stride):
        super(Classifier, self).__init__()
        self.dsconv1 = DSConv(in_channels, in_channels, stride)
        self.dsconv2 = DSConv(in_channels, in_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(in_channels, num_classes, kernel_size = 1)
        )

    def forward(self, x, size):
        x1 = self.dsconv1(x)
        x2 = self.dsconv2(x1)
        x3 = self.conv(x2)
        x4 = F.interpolate(x3, size, mode='bilinear', align_corners=True)
        return x4
     
 
########################################### Model ###########################################       
class Model(nn.Module):
    """Full Fast-SCNN Model"""
    def __init__(self, in_channels = 3, n_classes=19):
        super(Model, self).__init__()
        self.downsample = LearningToDownsample(in_channels=in_channels, channel_1 = 32, channel_2 = 48, out_channels=64)
        self.feature_extractor = GlobalFeatureExtractor(in_channels=64, out_channels=128, t=6)
        self.fusion = FeatureFusion(high_res_in_channels = 64, low_res_in_channels = 128, out_channels = 128, scale_factor = 4)
        self.classifier = Classifier(in_channels=128, num_classes=n_classes, stride=1)
    
    def forward(self, x):
        size = x.size()[2:]
        low_level_features = self.downsample(x)
        high_level_features = self.feature_extractor(low_level_features)
        fused_features = self.fusion(low_level_features, high_level_features)
        return self.classifier(fused_features, size)
