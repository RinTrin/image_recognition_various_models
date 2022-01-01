import torchvision.models as models
    
def make_model_dic(pretrained):
    model_dic = {'resnet18':models.resnet18(pretrained),
                 'vgg16'   :models.vgg16(pretrained),
                #  'lenet'   :models.lenet
                 }
    return model_dic