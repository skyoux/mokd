from operator import is_
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np

from utils.utils import trunc_normal_


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class CNNStudentWrapper(nn.Module):
    def __init__(self, backbone, head, transhead):
        super(CNNStudentWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head
        self.transhead = transhead

    def forward(self, x):
        fea, fea4 = self.backbone(torch.cat(x[:]))

        fea_trans, fea_reduce_dim = self.transhead(fea4)

        return self.head(fea), fea_trans, fea_reduce_dim


class CNNTeacherWrapper(nn.Module):
    def __init__(self, backbone, head, transhead, num_crops):
        super(CNNTeacherWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head
        self.transhead = transhead
        self.num_crops = num_crops
        

    def forward(self, x, local_token=None, global_token=None):
        fea, fea4 = self.backbone(torch.cat(x[:]))

        fea4_trans, _ = self.transhead(fea4)
        local_search_fea = None
        global_search_fea = None
        if local_token != None:
            local_search_fea, _ = self.transhead(fea4.chunk(2)[0].repeat(self.num_crops, 1, 1, 1), local_token)
        if global_token != None:
            global_search_fea, _ = self.transhead(fea4, global_token)
        return self.head(fea), fea4_trans, local_search_fea, global_search_fea


class ViTStudentWrapper(nn.Module):
    def __init__(self, backbone, head, transhead):
        super(ViTStudentWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head
        self.transhead = transhead

    def forward(self, x):
        tokens = self.backbone(torch.cat(x[:]), return_all_tokens=True)
        if isinstance(tokens, tuple):
            tokens = tokens[0]

        return self.head(tokens[:, 0]), self.transhead(tokens[:, 1:]), tokens[:, 0]


class ViTTeacherWrapper(nn.Module):
    def __init__(self, backbone, head, transhead, num_crops):
        super(ViTTeacherWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head
        self.transhead = transhead
        self.num_crops = num_crops

    def forward(self, x, local_token=None, global_token=None):
        tokens = self.backbone(torch.cat(x[:]), return_all_tokens=True)
        if isinstance(tokens, tuple):
            tokens = tokens[0]

        cls_token = tokens[:, 0]
        pth_token = tokens[:, 1:] # B,N,C
        
        local_search_fea = None
        global_search_fea = None
        if local_token != None:
            local_search_fea = self.transhead(pth_token.chunk(2)[0].repeat(self.num_crops, 1, 1), local_token)
        if global_token != None:
            global_search_fea = self.transhead(pth_token, global_token)
        return self.head(cls_token), self.transhead(pth_token), local_search_fea, global_search_fea


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class CTSearchLoss(nn.Module):
    def __init__(self, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student, teacher, epoch):

        temp = self.teacher_temp_schedule[epoch]

        student_out = student / self.student_temp
        teacher_out = F.softmax(teacher / temp, dim=-1)
        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1).mean()

        return loss

