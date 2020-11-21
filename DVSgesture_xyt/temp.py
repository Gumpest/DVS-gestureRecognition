import torch
import torchvision
import time
# 获取模型实例
model = torchvision.models.resnet18()

data = torch.rand(1, 3, 224, 224)

start = time.clock()

model.eval()
output = model(data)

end = time.clock()
print(end - start)