import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.utils.data
import torchvision.utils
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import os
import argparse
import visdom

from model import NetD, NetG

parser = argparse.ArgumentParser()
parser.add_argument('--nlayer', type=int, default=5, help='number of layers')
parser.add_argument('--type', type=str, default='shuffle', help='deconv or shuffle')
parser.add_argument('--ep', type=int, default=-1, help='')
parser.add_argument('--cls', type=str, default='natural_obj', help='')
parser.add_argument('--opath', type=str, default='.', help='path to training images')
parser.add_argument('--ipath', type=str, default='.', help='path to save checkpoints')
opt = parser.parse_args()
print(opt)

vis = visdom.Visdom()
win = vis.image(torch.zeros(3, 100, 100))

"""parameters"""
imgRoot = opt.ipath
iterNum = 30
ndf = 64
ngf = 64
nz = 100
genLayerNum = opt.nlayer
deLayerNum = opt.nlayer
imgSize = 2 ** (genLayerNum + 2)
batchSize = 64

checkRoot = opt.opath
if not os.path.exists(checkRoot):
    os.makedirs(checkRoot)

"""dataset"""
dataSet = dset.ImageFolder(root=imgRoot,
                           transform=transforms.Compose([transforms.Scale(imgSize),
                                                         transforms.CenterCrop(imgSize),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
dataLoader = torch.utils.data.DataLoader(dataSet, batch_size=batchSize, shuffle=True, num_workers=2)
dataIter = iter(dataLoader)

"""model"""
netG = NetG(genLayerNum, ngf, nz, up_type=opt.type)
print netG
netD = NetD(deLayerNum, ndf)
print netD
z = torch.FloatTensor(batchSize, nz, 1, 1)
realData = torch.FloatTensor(batchSize, 3, imgSize, imgSize)
label = torch.FloatTensor(batchSize)
criterion = nn.BCELoss()

realData = Variable(realData)
z = Variable(z)
label = Variable(label)

netG = netG.cuda()
netD = netD.cuda()
z = torch.FloatTensor(batchSize, nz, 8, 8)
realData = torch.FloatTensor(batchSize, 3, imgSize, imgSize)
one = torch.FloatTensor([1])
mone = one * -1

realData = Variable(realData)
z = Variable(z)

netG = netG.cuda()
netD = netD.cuda()
z = z.cuda()
realData = realData.cuda()
one = one.cuda()
mone = mone.cuda()

# setup optimizer
optimizerD = optim.RMSprop(netD.parameters(), lr=0.00005)  # 0.00005
optimizerG = optim.RMSprop(netG.parameters(), lr=0.00005)
if opt.ep != -1:
    netD.load_state_dict(torch.load(checkRoot + '/netD_epoch_' + str(opt.ep) + '.pth'))
    netG.load_state_dict(torch.load(checkRoot + '/netG_epoch_' + str(opt.ep) + '.pth'))
# train
ig = 0
for it in np.arange(iterNum) + opt.ep + 1:
    dataIter = iter(dataLoader)
    ib = 0
    while ib < len(dataLoader):
        ############################
        # (1) Update D network
        ###########################
        # train the discriminator Diters times
        if ig < 25 or ig % 500 == 0:
            Diters = 10
        else:
            Diters = 5
        id = 0
        while id < Diters and ib < len(dataLoader):
            # clamp parameters to a cube
            for p in netD.parameters():
                p.data.clamp_(-0.01, 0.01)

            data = dataIter.next()
            ib += 1
            # train with real
            batchSize_now = data[0].size(0)
            netD.zero_grad()
            realData.data.resize_(data[0].size()).copy_(data[0])
            errD_real = netD(realData)
            errD_real = errD_real.mean()
            errD_real.backward(one)
            # train with fake
            z.data.resize_(batchSize_now, nz, 1, 1).normal_()
            fakeData = netG(z)
            # pdb.set_trace()
            errD_fake = netD(fakeData.detach())
            errD_fake = errD_fake.mean()
            errD_fake.backward(mone)
            optimizerD.step()
            id += 1
        ############################
        # (2) Update G network
        ###########################
        netG.zero_grad()
        errG = netD(fakeData)
        errG = errG.mean()
        errG.backward(one)
        optimizerG.step()
        ig += 1
        hhh = fakeData.data.cpu()
        hhh = hhh / 2 + 0.5
        vis.image(torchvision.utils.make_grid(hhh), win=win)
        print('epoch %d, batch %d, Dreal: %.4f, Dfake: %.4f, errG: %.4f'
              % (it, ib, errD_real.data[0], errD_fake.data[0], errG.data[0]))
    # do checkpointing
    hhh = netG(z).data.cpu()
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (checkRoot, it))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (checkRoot, it))
