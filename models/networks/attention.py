import torch
import torch.nn as nn
import torch.nn.functional as F

class Self_Attention(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Self_Attention,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x, valid):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        
        m_batchsize,C,width ,height = x.size()
        valid = F.interpolate(valid, (width,height), mode='nearest')
        valid = valid.view(m_batchsize, 1, width*height)

        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)

        energy =  torch.bmm(proj_query,proj_key) # transpose check
        
        #indices= torch.nonzero(valid, as_tuple=True)
        #valid_energy = energy[indices[0], indices[1], indices[2]]
        #valid_attention = self.softmax(valid_energy)
        #attention = torch.zeros(m_batchsize, width*height, width*height).to(torch.cuda.current_device())
        #attention[:,:,valid==1] = valid_attention
        attention = self.softmax(energy) # BX (N) X (N) 

        valid_attention = attention * valid # B x N x N 
        attention_sums = torch.sum(valid_attention, dim=2) # B x N
        attention_scale = 1 / attention_sums # B x N
        scaled_attention = valid_attention * attention_scale

        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N


        out = torch.bmm(proj_value,scaled_attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,scaled_attention
