import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.downsample = downsample
        if downsample:
            self.conv3 = ConvBlock(out_channels, out_channels, stride=2)
        else:
            self.conv3 = ConvBlock(out_channels, out_channels)
            
    def forward(self, x):
        x = self.conv1(x)
        x_encode = self.conv2(x)
        x = self.conv3(x_encode)
        return x, x_encode

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_kernel=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=up_kernel, stride=2)
        self.conv1 = ConvBlock(out_channels * 2, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBlock(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvBlock(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_res = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        res = self.bn_res(self.residual(x))
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + res
        return out
    
class RegressionHead(nn.Module):
    def __init__(self, in_channels, hidden_dims=[128, 64]):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        layers = []
        prev_dim = in_channels
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            # layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1)) 
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class ClassificationHead(nn.Module):
    def __init__(self, in_channels, hidden_dims=[128, 64]):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        layers = []
        prev_dim = in_channels
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1)) 
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=4, hidden_dims_reg=[128, 64], hidden_dims_cls=[128, 64], use_cls_head=True, use_reg_head=True):
        super().__init__()
        
        # Encoder
        self.enc1 = EncoderBlock(in_channels, 32, downsample=False)
        self.enc2 = EncoderBlock(32, 64, downsample=True)
        self.enc3 = EncoderBlock(64, 128, downsample=True)
        
        # Bottleneck
        self.bottleneck1 = BottleNeck(128, 256, 64)
        self.bottleneck2 = BottleNeck(256, 256, 64)
        
        # Regression head
        self.use_reg_head = use_reg_head
        if use_reg_head:
            self.regression_head = RegressionHead(256, hidden_dims=hidden_dims_reg)
            
        # Classification head
        self.use_cls_head = use_cls_head
        if use_cls_head:
            self.classification_head = ClassificationHead(256, hidden_dims=hidden_dims_cls)
        
        # Decoder
        self.dec3 = DecoderBlock(256, 128)
        self.dec2 = DecoderBlock(128, 64)
        self.dec1 = nn.Sequential(
            ConvBlock(64 + 32, 32),
            ConvBlock(32, 32),
            ConvBlock(32, 32)
        )
        
        # Output layers
        self.conv_final = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        # Encoder
        x, x1 = self.enc1(x)
        x, x2 = self.enc2(x)
        x, x3 = self.enc3(x)

        # Bottleneck
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)

        # Classification head
        cost_cls = self.classification_head(x) if self.use_cls_head else None
        
        # Regression head
        cost_reg = self.regression_head(x) if self.use_reg_head else None

        # Decoder
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(torch.cat([x1, x], dim=1))

        # Output
        out = torch.sigmoid(self.conv_final(x))
        
        return out, cost_cls, cost_reg

class UNetOccupancy(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()
        
        # Encoder
        self.enc1 = EncoderBlock(in_channels, 32, downsample=False)
        self.enc2 = EncoderBlock(32, 64, downsample=True)
        self.enc3 = EncoderBlock(64, 128, downsample=True)
        
        # Bottleneck
        self.bottleneck1 = BottleNeck(128, 256, 64)
        self.bottleneck2 = BottleNeck(256, 256, 64)

        # Decoder
        self.dec3 = DecoderBlock(256, 128)
        self.dec2 = DecoderBlock(128, 64)
        self.dec1 = nn.Sequential(
            ConvBlock(64 + 32, 32),
            ConvBlock(32, 32),
            ConvBlock(32, 32)
        )
        
        # Output layers
        self.conv_final = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        # Encoder
        x, x1 = self.enc1(x)
        x, x2 = self.enc2(x)
        x, x3 = self.enc3(x)

        # Bottleneck
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)

        # Decoder
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(torch.cat([x1, x], dim=1))

        # Output
        out = torch.sigmoid(self.conv_final(x))

        return out

if __name__ == "__main__":
    model = UNet()
    x = torch.randn(2, 6, 80, 80)
    output, cost_cls, cost_reg = model(x)
    print("Output shape:", output.shape)
    print("Classification cost shape:", cost_cls.shape)
    print("Regression cost shape:", cost_reg.shape)