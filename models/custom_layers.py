import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomConv2d(nn.Module):
    """Кастомный сверточный слой с обучаемым масштабированием"""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(CustomConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.padding = padding
        # Обучаемые параметры: веса и bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        # Масштабирующий коэффициент
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        # Кастомная свертка: стандартная свертка + масштабирование
        out = F.conv2d(x, self.weight, self.bias, padding=self.padding)
        return out * self.scale

    @staticmethod
    def test():
        """Тест кастомного сверточного слоя"""
        x = torch.randn(1, 3, 32, 32)
        conv = CustomConv2d(3, 16, kernel_size=3, padding=1)
        out = conv(x)
        assert out.shape == (1, 16, 32, 32), "Неправильный размер выхода"
        # Сравнение с nn.Conv2d
        conv_std = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        conv_std.weight.data = conv.weight.data
        conv_std.bias.data = conv.bias.data
        out_std = conv_std(x)
        print("CustomConv2d тест пройден, разница с nn.Conv2d:", torch.abs(out - out_std).mean().item())

class CustomAttention(nn.Module):
    """Кастомный attention-механизм для CNN"""
    def __init__(self, in_channels):
        super(CustomAttention, self).__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Масштабирующий коэффициент
    
    def forward(self, x):
        batch_size, C, H, W = x.size()
        # Вычисление query, key, value
        query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # B x HW x C'
        key = self.key_conv(x).view(batch_size, -1, H * W)  # B x C' x HW
        attention = torch.bmm(query, key)  # B x HW x HW
        attention = F.softmax(attention, dim=-1)
        value = self.value_conv(x).view(batch_size, -1, H * W)  # B x C x HW
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x HW
        out = out.view(batch_size, C, H, W)
        return self.gamma * out + x

    @staticmethod
    def test():
        """Тест attention-механизма"""
        x = torch.randn(1, 64, 32, 32)
        attn = CustomAttention(64)
        out = attn(x)
        assert out.shape == x.shape, "Неправильный размер выхода"
        print("CustomAttention тест пройден")

class CustomActivation(nn.Module):
    """Кастомная функция активации: LeakyReLU с обучаемым параметром наклона"""
    def __init__(self, negative_slope=0.1):
        super(CustomActivation, self).__init__()
        self.negative_slope = nn.Parameter(torch.tensor(negative_slope))
    
    def forward(self, x):
        return torch.where(x > 0, x, x * self.negative_slope)

    @staticmethod
    def test():
        """Тест кастомной функции активации"""
        x = torch.tensor([-1.0, 0.0, 1.0])
        act = CustomActivation(0.1)
        out = act(x)
        expected = torch.tensor([-0.1, 0.0, 1.0])
        assert torch.allclose(out, expected), "Неправильный выход активации"
        print("CustomActivation тест пройден")

class CustomPooling(nn.Module):
    """Кастомный пулинговый слой: адаптивный средний пуллинг"""
    def __init__(self, output_size):
        super(CustomPooling, self).__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)
    
    def forward(self, x):
        # Адаптивный пуллинг: вычисляем stride и kernel_size
        B, C, H, W = x.size()
        stride_h = H // self.output_size[0]
        stride_w = W // self.output_size[1]
        kernel_h = H - (self.output_size[0] - 1) * stride_h
        kernel_w = W - (self.output_size[1] - 1) * stride_w
        return F.avg_pool2d(x, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w))

    @staticmethod
    def test():
        """Тест кастомного пулингового слоя"""
        x = torch.randn(1, 16, 32, 32)
        pool = CustomPooling(output_size=8)
        out = pool(x)
        assert out.shape == (1, 16, 8, 8), "Неправильный размер выхода"
        print("CustomPooling тест пройден")

if __name__ == "__main__":
    # Запуск тестов для всех кастомных слоев
    CustomConv2d.test()
    CustomAttention.test()
    CustomActivation.test()
    CustomPooling.test()