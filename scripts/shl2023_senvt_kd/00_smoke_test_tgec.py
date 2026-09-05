import torch
from torch import nn
from models.TeacherGuidedEvidenceCondensation import PatchReservoir
from loss_func import DistillationLoss

class FakeTeacher(nn.Module):
    def forward(self, x): return torch.randn(len(x), 8, device=x.device)
    def forward_with_temporal_evidence(self, x, teacher_layer, patch_size):
        evidence=torch.randn(len(x),31,768,device=x.device)
        return self(x), torch.softmax(torch.log(evidence.norm(dim=-1)),dim=-1), evidence

x=torch.randn(2,3,496); y=torch.tensor([0,1]); teacher=FakeTeacher()
for mode in ("aps","teacher_skip","tgec"):
    model=PatchReservoir(3,16,16,32,8,.5,mode).train()
    output=model(x)
    criterion=DistillationLoss(nn.CrossEntropyLoss(),teacher,
        "none" if mode=="aps" else "soft",.7,2.5,1,1,1,16)
    loss=criterion(x,output,y); loss.backward()
    assert torch.isfinite(loss)
    assert model.router[-1].weight.grad is not None
    print("[OK]",mode,"loss=",float(loss),"metrics=",criterion.last)

