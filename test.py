import torch
import torch.nn.functional as F


# label = torch.rand(4, 3, 3)
#
# label_list = []
# bg = torch.ones_like(label[0].unsqueeze(dim=0))
# for i in range(4):
#     label_i = label[i].unsqueeze(dim=0)
#     bg -= label_i
#     label_list.append(label_i)
# label_list = [bg] + label_list
# stacked_label = torch.cat(label_list, dim=0)
# gt = torch.argmax(stacked_label, dim=0)
# print(stacked_label.shape)
# print(gt.shape)

def channel_threshold(input_tensor, threshold=0.2):
    # 채널별 최솟값 계산
    channel_min = torch.min(input_tensor, dim=0)
    print('min', channel_min)
    # 0.2보다 작으면 1, 아니면 0으로 채운 텐서 생성
    channel_min[channel_min < threshold] = 0
    channel_min[channel_min >= threshold] = 1
    print(channel_min.shape)
    return channel_min


# 예시
input_tensor = torch.randn(3, 2, 2)
print('input', input_tensor)
output_tensor = channel_threshold(input_tensor)
# print(output_tensor.shape)
