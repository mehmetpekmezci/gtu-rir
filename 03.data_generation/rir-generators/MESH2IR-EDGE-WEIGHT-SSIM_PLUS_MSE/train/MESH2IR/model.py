import torch
import torch.nn as nn
import torch.nn.parallel
from miscc.config import cfg
from torch.autograd import Variable

from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

#torch.backends.cudnn.benchmark = True

def conv3x1(in_planes, out_planes, stride=1):
    "3x1 convolution with padding"
    kernel_length  = 41
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_length, stride=stride,
                     padding=20, bias=False)

def old_conv3x1(in_planes, out_planes, stride=1):
    "3x1 convolution with padding"
    kernel_length  = 3
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_length, stride=stride,
                     padding=1, bias=False)
# def convn3x1(in_planes, out_planes, stride=1):
#     "3x1 convolution with padding"
#     return nn.Conv1d(in_planes, out_planes, kernel_size=9, stride=stride,
#                      padding=4, bias=False)


# Upsale the spatial size by a factor of 2
def upBlock4(in_planes, out_planes):
    kernel_length  = 41
    stride = 4
    block = nn.Sequential(
        # nn.Upsample(scale_factor=4, mode='nearest'),
        # conv3x1(in_planes, out_planes),
        nn.ConvTranspose1d(in_planes,out_planes,kernel_size=kernel_length,stride=stride, padding=19,output_padding=1),
        nn.BatchNorm1d(out_planes),
        # nn.ReLU(True)
        nn.PReLU())
    return block
def upBlock2(in_planes, out_planes):
    kernel_length  = 41
    stride = 2
    block = nn.Sequential(
        # nn.Upsample(scale_factor=4, mode='nearest'),
        # conv3x1(in_planes, out_planes),
        nn.ConvTranspose1d(in_planes,out_planes,kernel_size=kernel_length,stride=stride, padding=20,output_padding=1),
        nn.BatchNorm1d(out_planes),
        # nn.ReLU(True)
        nn.PReLU())
    return block

def sameBlock(in_planes, out_planes):
    block = nn.Sequential(
        # nn.Upsample(scale_factor=4, mode='nearest'),
        conv3x1(in_planes, out_planes),
        nn.BatchNorm1d(out_planes),
        # nn.ReLU(True)
        nn.PReLU())
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x1(channel_num, channel_num),
            nn.BatchNorm1d(channel_num),
            # nn.ReLU(True),
            nn.PReLU(),
            conv3x1(channel_num, channel_num),
            nn.BatchNorm1d(channel_num))
        self.relu = nn.PReLU()#nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class COND_NET(nn.Module): #not chnaged yet
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(COND_NET, self).__init__()
        self.t_dim = cfg.TEXT.DIMENSION
        self.c_dim = cfg.GAN.CONDITION_DIM
        print(self.t_dim)
        print(self.c_dim)
        self.fc = nn.Linear(self.t_dim, self.c_dim, bias=True)
        self.relu = nn.PReLU()#nn.ReLU()

    def encode(self, full_embed):
        x = self.relu(self.fc(full_embed))
        # mu = x[:, :self.c_dim]
        # logvar = x[:, self.c_dim:]
        return x

    # def reparametrize(self, mu, logvar):
    #     std = logvar.mul(0.5).exp_()
    #     if cfg.CUDA:
    #         eps = torch.cuda.FloatTensor(std.size()).normal_()
    #     else:
    #         eps = torch.FloatTensor(std.size()).normal_()
    #     eps = Variable(eps)
    #     return eps.mul(std).add_(mu)

    def forward(self, full_embed):
        c_code = self.encode(full_embed)
        # c_code = self.reparametrize(mu, logvar)
        return c_code #, mu, logvar

class MESH_NET(nn.Module):
    def __init__(self):
        super(MESH_NET,self).__init__()
        self.feature_dim = 3
        self.conv1 = GCNConv(self.feature_dim, 32)
        self.pool1 = TopKPooling(32, ratio=0.6)
        self.conv2 = GCNConv(32, 32) #(32, 64)
        self.pool2 = TopKPooling(32, ratio=0.6) #64, ratio=0.6)
        self.conv3 = GCNConv(32, 32) #(64, 128)
        self.pool3 = TopKPooling(32, ratio=0.6) #(128, ratio=0.6)
        # self.item_embedding = torch.nn.Embedding(num_embeddings=df.item_id.max() +1, embedding_dim=self.feature_dim)
        self.lin1 = torch.nn.Linear(64, 16) #(256, 128)
        self.lin2 = torch.nn.Linear(16, 8) #(128, 64)
        # self.lin3 = torch.nn.Linear(8, 1) #(64, 1)
        self.bn1 = torch.nn.BatchNorm1d(16) #(128)
        self.bn2 = torch.nn.BatchNorm1d(8) #(64)
        self.act1 = torch.nn.ReLU()
        # self.act2 = torch.nn.ReLU()        
  
    def forward(self, data):
        x, edge_index,edge_weight, batch = data.pos, data.edge_index, data.edge_weights, data.batch
      
        ## MP : edge_weight >=0 , generates nan value if negative.
        ## edge_weight=edge_weight.abs()
        ## MP: no need to do the above line, we already normalized the cos -1,1  to 0,1 in miscc/datasets.py

        # x = self.item_embedding(x)
        # x = x.squeeze(1)        
      
        #print("---#################################################")
        #for i in x:
        #    print(f"x.0={i}")
        #print("000#################################################")
        #print("batch ",batch.shape)
        #print("edge_weight ",edge_weight.shape)
        #print("edge_index ",edge_index.shape)
        #print("x ",x.shape)
        
        x = F.relu(self.conv1(x, edge_index,edge_weight))
        ## MP: We can only use edge_weight in the first layer, because, while pooling, we dont recalculate edge weight according to source speakers cos_theta
       
        #x, edge_index, edge_weight, batch, _ ,_= self.pool1(x, edge_index, edge_weight, batch)
        x, edge_index, _, batch, _ ,_= self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        #x = F.relu(self.conv2(x, edge_index,edge_weight))
        x = F.relu(self.conv2(x, edge_index,None))
        
        #x, edge_index, edge_weight, batch, _,_ = self.pool2(x, edge_index, edge_weight, batch)
        x, edge_index, _, batch, _,_ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        #x = F.relu(self.conv3(x, edge_index,edge_weight))
        x = F.relu(self.conv3(x, edge_index,None))

        #x, edge_index, edge_weight, batch, _,_ = self.pool3(x, edge_index, edge_weight, batch)
        x, edge_index, _, batch, _,_ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        # print("x1 shape ", x1.shape)
        # print("x2 shape ", x2.shape)
        # print("x3 shape ", x3.shape)
        x = x1 + x2 + x3

        x = self.lin1(x)
        x = self.act1(x)
        # print("x shape1 ", x.shape)
        # x = self.lin2(x)
        # x = self.act2(x)      
        x = F.dropout(x, p=0.5, training=self.training)
        # print("x shape2 ", x.shape)
        x = torch.sigmoid(self.lin2(x)).squeeze(1)
        # print("x shape3 ", x.shape)
        #print("x return ",x)
        return x



class D_GET_LOGITS(nn.Module): #not chnaged yet
    def __init__(self, ndf, nef, bcondition=True):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        kernel_length =41
        if bcondition:
            self.convd1d =  nn.ConvTranspose1d(ndf*8,ndf //2,kernel_size=kernel_length,stride=1, padding=20)
            # self.outlogits = nn.Sequential(
            #     old_conv3x1(ndf * 8 + nef, ndf * 8),
            #     nn.BatchNorm1d(ndf * 8),
            #     nn.LeakyReLU(0.2, inplace=True),
            #     nn.Conv1d(ndf * 8, 1, kernel_size=16, stride=4),
            #     # nn.Conv1d(1, 1, kernel_size=16, stride=4),
            #     nn.Sigmoid()
            #     )
            self.outlogits = nn.Sequential(
                old_conv3x1(ndf //2 + nef, ndf //2 ),
                nn.BatchNorm1d(ndf //2 ),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(ndf //2 , 1, kernel_size=16, stride=4),
                # nn.Conv1d(1, 1, kernel_size=16, stride=4),
                nn.Sigmoid()
                )
        else:
            # self.outlogits = nn.Sequential(
            #     nn.Conv1d(ndf * 8, 1, kernel_size=16, stride=4),
            #     # nn.Conv1d(1, 1, kernel_size=16, stride=4),
            #     nn.Sigmoid())
            self.convd1d =  nn.ConvTranspose1d(ndf*8,ndf //2,kernel_size=kernel_length,stride=1, padding=20)
            self.outlogits = nn.Sequential(
                nn.Conv1d(ndf // 2 , 1, kernel_size=16, stride=4),
                # nn.Conv1d(1, 1, kernel_size=16, stride=4),
                nn.Sigmoid())

    def forward(self, h_code, c_code=None):
        # conditioning output
        h_code = self.convd1d(h_code)
        if self.bcondition and c_code is not None:
            #print("mode c_code1 ",c_code.size())
            c_code = c_code.view(-1, self.ef_dim, 1)
            #print("mode c_code2 ",c_code.size())

            c_code = c_code.repeat(1, 1, 16)
            # state size (ngf+egf) x 16
            #print("mode c_code ",c_code.size())
            #print("mode h_code ",h_code.size())

            h_c_code = torch.cat((h_code, c_code), 1)
        else:
            h_c_code = h_code

        output = self.outlogits(h_c_code)

        return output.view(-1)


# ############# Networks for stageI GAN #############


class STAGE1_G(nn.Module):
    def __init__(self):
        super(STAGE1_G, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM * 8
        self.ef_dim = cfg.GAN.CONDITION_DIM
        print(f"self.gf_dim={self.gf_dim}")
        print(f"self.ef_dim={self.ef_dim}")
        # self.z_dim = cfg.Z_DIM
        self.define_module()

    def define_module(self):
        kernel_length  = 41
        ninput = self.ef_dim #self.z_dim + self.ef_dim
        ngf = self.gf_dim
        # TEXT.DIMENSION -> GAN.CONDITION_DIM
        # self.ca_net = CA_NET()
        self.cond_net = COND_NET()
        # self.mesh_net = MESH_NET()
        # -> ngf x 16
        self.fc = nn.Sequential(
            nn.Linear(ninput, ngf * 16, bias=False),
            nn.BatchNorm1d(ngf * 16),
            # nn.ReLU(True)
            nn.PReLU())

        # ngf x 16 -> ngf/2 x 64
        self.upsample1 = upBlock4(ngf, ngf // 2)
        # -> ngf/4 x 256
        self.upsample2 = upBlock4(ngf // 2, ngf // 4)
        # -> ngf/8 x 1024
        self.upsample3 = upBlock4(ngf // 4, ngf // 8)
        # -> ngf/16 x 4096
        self.upsample4 = upBlock2(ngf // 8, ngf // 16)
        self.upsample5 = upBlock2(ngf // 16, ngf // 16)
        # -> 1 x 4096
        self.RIR = nn.Sequential(
            nn.ConvTranspose1d(ngf // 16,1,kernel_size=kernel_length,stride=1, padding=20),
            # old_conv3x1(ngf // 16, 1), # conv3x3(ngf // 16, 3),
            nn.Tanh())

    def forward(self, text_embedding,mesh_embed):
        # mesh_embed = self.mesh_net(data)
        

        # c_code, mu, logvar = self.ca_net(text_embedding)
        # c_code = self.cond_net(text_embedding)
        # print("mesh_embed ", mesh_embed.shape)
        # print("text_embedding ", text_embedding.shape)
        full_embed= torch.cat((mesh_embed, text_embedding), 1)
        c_code = self.cond_net(full_embed)

        h_code = self.fc(c_code)

        h_code = h_code.view(-1, self.gf_dim, 16)
        # print("h_code 1 ",h_code.size())
        h_code = self.upsample1(h_code)
        # print("h_code 2 ",h_code.size())
        h_code = self.upsample2(h_code)
        # print("h_code 3 ",h_code.size())
        h_code = self.upsample3(h_code)
        # print("h_code 4 ",h_code.size())
        h_code = self.upsample4(h_code)
        h_code = self.upsample5(h_code)
        # print("h_code 5 ",h_code.size())
        # # state size 3 x 64 x 64
        fake_RIR = self.RIR(h_code)
        # print("fake_RIR ",fake_RIR.size())
        # # return None, fake_RIR, mu, logvar
        # #print("generator ", text_embedding.size())
        # return None, fake_RIR, text_embedding #c_code
        return None, fake_RIR, c_code


class STAGE1_D(nn.Module):
    def __init__(self):
        super(STAGE1_D, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        print(f"self.df_dim={self.df_dim}")
        print(f"self.ef_dim={self.ef_dim}")
        self.define_module()

    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim
        kernel_length =41
        self.encode_RIR = nn.Sequential(
            nn.Conv1d(1, ndf, kernel_length, 4, 20, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 1024
            nn.Conv1d(ndf, ndf * 2, kernel_length, 4, 20, bias=False),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) x 256
            nn.Conv1d(ndf*2, ndf * 4, kernel_length, 4, 20, bias=False),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # # state size (ndf*4) x 64
            nn.Conv1d(ndf*4, ndf * 8, kernel_length, 4, 20, bias=False),
            nn.BatchNorm1d(ndf * 8),
            # state size (ndf * 8) x 16)
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.get_cond_logits = D_GET_LOGITS(ndf, nef)
        self.get_uncond_logits = None

    def forward(self, RIRs):
        #print("model RIRs ",RIRs.size())
        RIR_embedding = self.encode_RIR(RIRs)
        #print("models RIR_embedding ",RIR_embedding.size())

        return RIR_embedding


# ############# Networks for stageII GAN #############
class STAGE2_G(nn.Module):
    def __init__(self, STAGE1_G):
        super(STAGE2_G, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        # self.z_dim = cfg.Z_DIM
        self.STAGE1_G = STAGE1_G
        # fix parameters of stageI GAN
        for param in self.STAGE1_G.parameters():
            param.requires_grad = False
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        # TEXT.DIMENSION -> GAN.CONDITION_DIM
        # self.ca_net = CA_NET()
        self.cond_net = COND_NET()
        # --> 4ngf x 16 x 16
        self.encoder = nn.Sequential(
            conv3x1(1, ngf),
            nn.ReLU(True),
            nn.Conv1d(ngf, ngf * 2, 16, 4, 6, bias=False),
            nn.BatchNorm1d(ngf * 2),
            nn.ReLU(True),
            nn.Conv1d(ngf * 2, ngf * 4, 16, 4, 6, bias=False),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True))
        self.hr_joint = nn.Sequential(
            conv3x1(self.ef_dim + ngf * 4, ngf * 4),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True))
        self.residual = self._make_layer(ResBlock, ngf * 4)
        # --> 2ngf x 1024
        self.upsample1 = upBlock4(ngf * 4, ngf * 2)
        # --> ngf x 4096
        self.upsample2 = upBlock4(ngf * 2, ngf)
        # --> ngf // 2 x 16384
        self.upsample3 = upBlock4(ngf, ngf // 2)
        # --> ngf // 4 x 16384
        self.upsample4 = sameBlock(ngf // 2, ngf // 4)
        # --> 1 x 16384
        self.RIR = nn.Sequential(
            conv3x1(ngf // 4, 1),
            nn.Tanh())

    def forward(self, text_embedding):
        _, stage1_RIR, _= self.STAGE1_G(text_embedding)
        stage1_RIR = stage1_RIR.detach()
        encoded_RIR = self.encoder(stage1_RIR)

        # c_code, mu, logvar = self.ca_net(text_embedding)
        c_code1 = self.cond_net(text_embedding)
        c_code = c_code1.view(-1, self.ef_dim, 1)
        c_code = c_code.repeat(1, 1, 256) # c_code.repeat(1, 1, 16, 16)
        i_c_code = torch.cat([encoded_RIR, c_code], 1)
        h_code = self.hr_joint(i_c_code)
        h_code = self.residual(h_code)

        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)

        fake_RIR = self.RIR(h_code)
        return stage1_RIR, fake_RIR, c_code1 #mu, logvar


class STAGE2_D(nn.Module):
    def __init__(self):
        super(STAGE2_D, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.define_module()

    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_RIR = nn.Sequential(
            nn.Conv1d(1, ndf, 3, 1, 1, bias=False),  # 16384 * ndf
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf, ndf * 2, 16, 4, 6, bias=False),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),  # 4096 * ndf * 2
            nn.Conv1d(ndf * 2, ndf * 4, 16, 4, 6, bias=False),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),  # 1024 * ndf * 4
            nn.Conv1d(ndf * 4, ndf * 8, 16, 4, 6, bias=False),
            nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),  # 256 * ndf * 8
            nn.Conv1d(ndf * 8, ndf * 16, 16, 4, 6, bias=False),
            nn.BatchNorm1d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),  # 64 * ndf * 16
            nn.Conv1d(ndf * 16, ndf * 32, 16, 4, 6, bias=False),
            nn.BatchNorm1d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),  # 16 * ndf * 32
            conv3x1(ndf * 32, ndf * 16),
            nn.BatchNorm1d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),   # 16 * ndf * 16
            conv3x1(ndf * 16, ndf * 8),
            nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)   # 16 * ndf * 8
        )

        self.get_cond_logits = D_GET_LOGITS(ndf, nef, bcondition=True)
        self.get_uncond_logits = D_GET_LOGITS(ndf, nef, bcondition=False)

    def forward(self, RIRs):
        RIR_embedding = self.encode_RIR(RIRs)
        return RIR_embedding
