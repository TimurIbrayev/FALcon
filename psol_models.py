import torchvision
import torch.nn as nn
import torch

def copy_parameters(model, pretrained_dict):
    model_dict = model.state_dict()

    if 'module' in list(pretrained_dict.keys())[0]:
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict and pretrained_dict[k].size()==model_dict[k[7:]].size()}
        for k, v in pretrained_dict.items():
            print(k)
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and pretrained_dict[k].size() == model_dict[k].size()}
        for k, v in pretrained_dict.items():
            print(k)

    model_dict.update(pretrained_dict)
    missing, unexpected = model.load_state_dict(model_dict)
    return model

class VGGGAP(nn.Module):
    def __init__(self, pretrained=True, num_classes=200):
        super(VGGGAP,self).__init__()
        self.features = torchvision.models.vgg16(pretrained=pretrained).features
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential((nn.Linear(512,512),nn.ReLU(),nn.Linear(512,num_classes),nn.Sigmoid()))
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x
class VGG16(nn.Module):
    def __init__(self, pretrained=True, num_classes=200):
        super(VGG16,self).__init__()
        self.features = torchvision.models.vgg16(pretrained=pretrained).features
        temp_classifier = torchvision.models.vgg16(pretrained=pretrained).classifier
        removed = list(temp_classifier.children())
        removed = removed[:-1]
        temp_layer = nn.Sequential(nn.Linear(4096,512),nn.ReLU(),nn.Linear(512,num_classes),nn.Sigmoid())
        removed.append(temp_layer)
        self.classifier = nn.Sequential(*removed)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x

def choose_locmodel(model_name, pretrained=False):
    if model_name == 'densenet161':
        model = torchvision.models.densenet161(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(2208, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Sigmoid()
        )
        if pretrained:
            model = copy_parameters(model, torch.load('./psol/densenet161loc.pth.tar'))
    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True, num_classes=1000)
        model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Sigmoid()
        )
        if pretrained:
            model = copy_parameters(model, torch.load('resnet50loc.pth.tar'))
    elif model_name == 'vgggap':
        model = VGGGAP(pretrained=True, num_classes=4)
        if pretrained:
            model = copy_parameters(model, torch.load('vgggaploc.pth.tar'))
    elif model_name == 'vgg16':
        model = VGG16(pretrained=True, num_classes=4)
        if pretrained:
            model = copy_parameters(model, torch.load('./psol/vgg16loc.pth.tar'))
    elif model_name == 'inceptionv3':
        #need for rollback inceptionv3 official code
        pass
    else:
        raise ValueError('Do not have this model currently!')
    return model


def choose_clsmodel(model_name, pretrained=False, num_classes=1000):
    # for ImageNet dataset
    if num_classes == 1000:
        if model_name == 'vgg16':
            cls_model = torchvision.models.vgg16(pretrained=True)
        elif model_name == 'inceptionv3':
            cls_model = torchvision.models.inception_v3(pretrained=True, aux_logits=True, transform_input=True)
        elif model_name == 'resnet50':
            cls_model = torchvision.models.resnet50(pretrained=True)
        elif model_name == 'densenet161':
            cls_model = torchvision.models.densenet161(pretrained=True)
        elif model_name == 'dpn131':
            cls_model = torch.hub.load('rwightman/pytorch-dpn-pretrained', 'dpn131', pretrained=True,test_time_pool=True)
        elif model_name == 'efficientnetb7':
            from efficientnet_pytorch import EfficientNet
            cls_model = EfficientNet.from_pretrained('efficientnet-b7')
    # for datasets other than ImageNet
    else: 
        if model_name == 'vgg16':
            cls_model = torchvision.models.vgg16(pretrained=True)
            ### replace classifier
            # temp_classifier = cls_model.classifier
            # removed = list(temp_classifier.children())
            # removed = removed[:-1]
            # temp_layer = nn.Sequential(nn.Linear(4096,512),nn.ReLU(),nn.Linear(512, num_classes))
            # removed.append(temp_layer)
            # cls_model.classifier = nn.Sequential(*removed)
            cls_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
            cls_model.classifier = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, num_classes)
            )
            # load pretrained for inference
            if pretrained:
                cls_model = copy_parameters(cls_model, torch.load('vgg16cls.pth.tar'))
        elif model_name == 'resnet50':
            cls_model = torchvision.models.resnet50(pretrained=True)
            # replace classifier
            cls_model.fc = nn.Linear(2048, num_classes)
            for m in cls_model.fc.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
            # load pretrained for inference
            if pretrained:
                cls_model = copy_parameters(cls_model, torch.load('resnet50cls.pth.tar'))
        elif model_name == 'densenet161':
            cls_model = torchvision.models.densenet161(pretrained=True)
            # replace classifier
            cls_model.classifier = nn.Sequential(
                nn.Linear(2208, 512),
                nn.ReLU(),
                nn.Linear(512, num_classes)
            )
            # load pretrained for inference
            if pretrained:
                cls_model = copy_parameters(cls_model, torch.load('densenet161cls.pth.tar'))
    return cls_model
