import socket
import sys
import threading
import json

from PIL import Image
import numpy as np

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
print(torch.cuda.is_available())

torch.cuda.current_device()
torch.cuda._initialized = True
# nn=network.getNetWork()
# cnn = conv.main(False)
# 深度学习训练的神经网络,使用TensorFlow训练的神经网络模型，保存在文件中

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.inc = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1)
                                 , nn.ReLU(inplace=True))
        self.conv = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1)
                                  , nn.ReLU(inplace=True))
        self.outc = nn.Sequential(nn.Conv2d(32, 3, 3, padding=1)
                                  , nn.ReLU(inplace=True))

    def forward(self, x):
        conv1 = self.inc(x)
        conv2 = self.conv(conv1)
        conv3 = self.conv(conv2)
        conv4 = self.conv(conv3)
        conv5 = self.outc(conv4)
        return conv5


class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        # forward 需要两个输入，x1 是需要上采样的小尺寸 feature map
        # x2 是以前的大尺寸 feature map，因为中间的 pooling 可能损失了边缘像素，
        # 所以上采样以后的 x1 可能会比 x2 尺寸小

    def forward(self, x1, x2):
        # x1 上采样
        x1 = self.up(x1)
        # 输入数据是四维的，第一个维度是样本数，剩下的三个维度是 CHW
        # 所以 Y 方向上的悄寸差别在 [2],  X 方向上的尺寸差别在 [3]
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # 给 x1 进行 padding 操作
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        # 把 x2 加到反卷积后的 feature map
        x = x2 + x1
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.inc = nn.Sequential(
            single_conv(6, 64),
            single_conv(64, 64))

        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(
            single_conv(64, 128),
            single_conv(128, 128),
            single_conv(128, 128))

        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(
            single_conv(128, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256))

        self.up1 = up(256)
        self.conv3 = nn.Sequential(
            single_conv(128, 128),
            single_conv(128, 128),
            single_conv(128, 128))

        self.up2 = up(128)
        self.conv4 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64))

        self.outc = outconv(64, 3)

    def forward(self, x):
        # input conv : 6 ==> 64 ==> 64
        inx = self.inc(x)

        # 均值 pooling, 然后 conv1 : 64 ==> 128 ==> 128 ==> 128
        down1 = self.down1(inx)
        conv1 = self.conv1(down1)

        # 均值 pooling，然后 conv2 : 128 ==> 256 ==> 256 ==> 256 ==> 256 ==> 256 ==> 256
        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)

        # up1 : conv2 反卷积，和 conv1 的结果相加，输入256，输出128
        up1 = self.up1(conv2, conv1)
        # conv3 : 128 ==> 128 ==> 128 ==> 128
        conv3 = self.conv3(up1)

        # up2 : conv3 反卷积，和 input conv 的结果相加，输入128，输出64
        up2 = self.up2(conv3, inx)
        # conv4 : 64 ==> 64 ==> 64
        conv4 = self.conv4(up2)

        # output conv: 65 ==> 3，用1x1的卷积降维，得到降噪结果
        out = self.outc(conv4)
        return out


class CBDNet(nn.Module):
    def __init__(self):
        super(CBDNet, self).__init__()
        self.fcn = FCN()
        self.unet = UNet()

    def forward(self, x):
        noise_level = self.fcn(x)
        concat_img = torch.cat([x, noise_level], dim=1)
        out = self.unet(concat_img) + x
        return noise_level, out


class fixed_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out_image, gt_image, est_noise, gt_noise, if_asym):
        # 分别得到图像的高度和宽度
        h_x = est_noise.size()[2]
        w_x = est_noise.size()[3]
        # 每个样本为 CHW ，把 H 方向第一行的数据去掉，统计一下一共多少元素
        count_h = self._tensor_size(est_noise[:, :, 1:, :])
        # 每个样本为 CHW ，把 W 方向第一列的数据去掉，统计一下一共多少元素
        count_w = self._tensor_size(est_noise[:, :, :, 1:])
        # H 方向，第一行去掉得后的矩阵，减去最后一行去掉后的矩阵，即下方像素减去上方像素，平方，然后求和
        h_tv = torch.pow((est_noise[:, :, 1:, :] - est_noise[:, :, :h_x - 1, :]), 2).sum()
        # W 方向，第一列去掉得后的矩阵，减去最后一列去掉后的矩阵，即右方像素减去左方像素，平方，然后求和
        w_tv = torch.pow((est_noise[:, :, :, 1:] - est_noise[:, :, :, :w_x - 1]), 2).sum()
        # 求平均，得到平均每个像素上的 tvloss
        tvloss = h_tv / count_h + w_tv / count_w
        loss = torch.mean( \
            # 第三部分：重建损失
            torch.pow((out_image - gt_image), 2)) + \
               if_asym * 0.5 * torch.mean(
            torch.mul(torch.abs(0.3 - F.relu(gt_noise - est_noise)), torch.pow(est_noise - gt_noise, 2))) + \
               0.05 * tvloss
        return loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


# 这个类用于存储 loss，观察结果时使用
# 每轮训练一张图像，就计算一下 loss 的均值存储在 self.avg 里，用于输出观察变化
# 同时，把当前 loss 的值存储在 self.val 里
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 图像矩阵由 hwc 转换为 chw ，这个就不多解释了
def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])


# 图像矩阵由 chw 转换为 hwc ，这个也不多解释
def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0])





nnservice = torch.load(r'D:/code/NEW_CBDNet.pth')


def main():
    # 创建服务器套接字
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 获取本地主机名称
    host = socket.gethostname()
    # 设置一个端口
    port = 12345
    # 将套接字与本地主机和端口绑定
    serversocket.bind((host, port))
    # 设置监听最大连接数
    serversocket.listen(5)
    # 获取本地服务器的连接信息
    myaddr = serversocket.getsockname()
    print("服务器地址:%s" % str(myaddr))
    # 循环等待接受客户端信息
    while True:
        # 获取一个客户端连接
        clientsocket, addr = serversocket.accept()
        print("连接地址:%s" % str(addr))
        try:
            t = ServerThreading(clientsocket)  # 为每一个请求开启一个处理线程
            t.start()
            pass
        except Exception as identifier:
            print(identifier)
            pass
        pass
    serversocket.close()
    pass


class ServerThreading(threading.Thread):
    # words = text2vec.load_lexicon()
    def __init__(self, clientsocket, recvsize=1024 * 1024, encoding="utf-8"):
        threading.Thread.__init__(self)
        self._socket = clientsocket
        self._recvsize = recvsize
        self._encoding = encoding
        self.cnt=1
        pass

    def run(self):
        print("开启线程.....")
        try:
            # 接受数据
            msg = ''
            while True:
                # 读取recvsize个字节
                rec = self._socket.recv(self._recvsize)
                # 解码
                msg += rec.decode(self._encoding)
                # 文本接受是否完毕，因为python socket不能自己判断接收数据是否完毕，
                # 所以需要自定义协议标志数据接受完毕
                if msg.strip().endswith('over'):
                    msg = msg[:-4]
                    break


            # 解析json格式的数据
            re = json.loads(msg)
            print(re['content'])
            # 调用神经网络模型处理请求
            device = torch.device("cuda:0")

            to_dir='D:/code/Java_web/ImageRepair/src/main/webapp/image/'+str(self.cnt)+'.bmp'#改这个路径
            img = str(self.cnt) + '.bmp'
            self.cnt= self.cnt + 1
            noisy_img = cv2.imread(re['content'])

            noisy_img = noisy_img[:, :, ::-1] / 255.0
            noisy_img = np.array(noisy_img).astype('float32')
            temp_noisy_img_chw = hwc_to_chw(noisy_img)
            # 图像放到 gpu 上
            input_var = torch.from_numpy(temp_noisy_img_chw.copy()).type(torch.FloatTensor).unsqueeze(0).to(device)
            # 输入模型得到结果
            _, output = nnservice(input_var)
            output_np = output.squeeze().cpu().detach().numpy()
            output_np = chw_to_hwc(np.clip(output_np, 0, 1))
            tempImg = np.concatenate((noisy_img, output_np), axis=1) * 255.0
            Image.fromarray(np.uint8(tempImg)).save(fp=to_dir, format='JPEG')



            # sendmsg = json.dumps(to_dir)
            # 发送数据
            self._socket.send(("%s" % img).encode(self._encoding))
            print(("%s" % img).encode(self._encoding))
            pass
        except Exception as identifier:
            self._socket.send("500".encode(self._encoding))
            print(identifier)
            pass
        finally:
            self._socket.close()
        print("任务结束.....")

        pass

    def __del__(self):

        pass


if __name__ == "__main__":
    main()
