import torch.nn as nn
import torch.nn.functional as F
import torch



class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(EncoderBlock, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        if dropout:
            self.encode.add_module('dropout', nn.Dropout())
        self.encode.add_module('maxpool', nn.MaxPool2d(2, stride=2))

    def forward(self, x):
        return self.encode(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, 3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, 2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)

class CenterBlock(nn.Module):
    def __init__(self, in_channels,middle_channels, out_channels, dropout=False):
        super(CenterBlock, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, middle_channels, 3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),

            # nn.ConvTranspose2d(middle_channels, out_channels, 2, stride=2),
        )

    def forward(self, x):
        return self.encode(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, drop=0.2):
        super(UNet, self).__init__()

        self.encoder1 = EncoderBlock(in_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        self.center = CenterBlock(512, 512, 512)

        self.decoder4 = DecoderBlock(1024, 512, 256)
        self.decoder3 = DecoderBlock(512, 256, 128)
        self.decoder2 = DecoderBlock(256, 128, 64)
        self.decoder1 = DecoderBlock(128, 64, out_channels)
        # self.decoder1 = nn.Sequential(
        #     nn.Conv2d(128, 64, 3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, 3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, out_channels, 1),
        # )
        self.final = nn.Hardtanh()


    def forward(self, x):
        enc1 = self.encoder1(x)#1,64,256,256
        enc2 = self.encoder2(enc1)#1,128,128,128
        enc3 = self.encoder3(enc2)#    256, 64
        enc4 = self.encoder4(enc3)#   512,32
        x = self.center(enc4)#     512,32
        x = self.decoder4(torch.cat((enc4, x), dim=1))
        x = self.decoder3(torch.cat((enc3, x), dim=1))
        x = self.decoder2(torch.cat((enc2, x), dim=1))
        x = self.decoder1(torch.cat((enc1, x), dim=1))
        x = self.final(x)
        return x


class UNetSoft(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, drop=0.2):
        super(UNetSoft, self).__init__()

        self.encoder1 = EncoderBlock(in_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        self.center = CenterBlock(512, 512, 512)

        self.decoder4 = DecoderBlock(1024, 512, 256)
        self.decoder3 = DecoderBlock(512, 256, 128)
        self.decoder2 = DecoderBlock(256, 128, 64)
        self.decoder1 = DecoderBlock(128, 64, out_channels)
        # self.decoder1 = nn.Sequential(
        #     nn.Conv2d(128, 64, 3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, 3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, out_channels, 1),
        # )
        self.final = nn.Tanhshrink()


    def forward(self, x):
        enc1 = self.encoder1(x)#1,64,256,256
        enc2 = self.encoder2(enc1)#1,128,128,128
        enc3 = self.encoder3(enc2)#    256, 64
        enc4 = self.encoder4(enc3)#   512,32
        x = self.center(enc4)#     512,32
        x = self.decoder4(torch.cat((enc4, x), dim=1))
        x = self.decoder3(torch.cat((enc3, x), dim=1))
        x = self.decoder2(torch.cat((enc2, x), dim=1))
        x = self.decoder1(torch.cat((enc1, x), dim=1))
        x = self.final(x)
        return x