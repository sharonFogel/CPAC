import torch
import torchvision.models
from torch.autograd import Variable

def return_deep_fetures(dataset):
    vgg19_trained=torchvision.models.vgg19(pretrained=True).cuda()
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=50, shuffle=False)
    deep_features = torch.FloatTensor()
    for i_batch, (x, y) in enumerate(dataloader):
        print('passing batch number %f through vgg19'%i_batch)
        x = Variable(x).cuda()
        current_deep_features = vgg19_trained.features(x)
        deep_features = torch.cat([deep_features, current_deep_features.data.cpu()])
    dataset.train_data = deep_features.view(deep_features.shape[0],-1)
    dataset.transform = None
    return dataset


