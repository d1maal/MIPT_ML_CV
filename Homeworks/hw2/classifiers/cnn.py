from builtins import object
import numpy as np

from .layers import *
from .layer_utils import *


class ThreeLayerConvNet(object):
    """
    трехслойная сверточная сеть с архитектурой:
    conv - relu - 2x2 max pool - linear - relu - linear - softmax

    Сеть работает с пакетами данных формой (N, C, H, W),
состоящими из N изображений, каждое высотой H и шириной W и с C входными
каналами.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=16,
        filter_size=3,
        hidden_dim=50,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Инициализация.
        Входные параметры:
            - input_dim: кортеж (C, H, W), указывающий размер входных данных
            - num_filters: количество фильтров сверточного слоя
            - filter_size: ширина/высота фильтров сверточного слоя
            - hidden_dim: количество нейронов для использования в полносвязном скрытом слое
            - num_classes: количество оценок, получаемых из финального линейного слоя.
            - weight_scale: скаляр, указывающий стандартное отклонение для случайной инициализации
            весов.
            - reg: скаляр, указывающий силу L2-регуляризации
            - dtype: тип данных numpy для вычислений.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Инициализируйте веса и смещения для трехслойной сверточной сети
        # сети. Веса должны быть инициализированы гауссовым распределением с центром в 0,0
        # со стандартным отклонением, равным weight_scale; смещения должны быть
        # инициализированы нулем. Все веса и смещения должны храниться в
        # словаре self.params. Сохраняйте веса и смещения для сверточного
        # слоя, используя ключи 'W1' и 'b1'; используйте ключи 'W2' и 'b2' для
        # весов и смещений скрытого слоя, и ключи 'W3' и 'b3'
        # для весов и смещений выходного слоя. 
        # #
        # ВАЖНО: паддинг и страйды
        # первого сверточного слоя выбраны таким образом, чтобы #
        # **ширина и высота входных данных сохранялись**. Взгляните на 
        # начало функции loss() #
        ############################################################################
        F, (C, H, W) = num_filters, input_dim
        self.params.update({
            'W1': np.random.randn(F, C, filter_size, filter_size) * weight_scale,
            'b1': np.zeros(F),
            'W2': np.random.randn(F * (H // 2) * (W // 2), hidden_dim) * weight_scale,
            'b2': np.zeros(hidden_dim),
            'W3': np.random.randn(hidden_dim, num_classes) * weight_scale,
            'b3': np.zeros(num_classes),
        })
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # conv_param проводим в сверточнйй слой
        # паддинг и страйд выбраны для сохранения размера
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        #  pool_param проводим в слой max-pooling
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO: Реализовать прямой проход для трехслойной сверточной сети, #
        # вычисляя оценки классов для X и сохраняя их в переменной scores #
        ############################################################################

        # conv - relu - pool
        h1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        # affine - relu
        h2, cache2 = affine_relu_forward(h1, W2, b2)
        # affine
        scores, cache3 = affine_forward(h2, W3, b3)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: обратный проход для трехслойной сверточной сети #
        ############################################################################

        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))

        dh2, dW3, db3 = affine_backward(dscores, cache3)
        dh1, dW2, db2 = affine_relu_backward(dh2, cache2)
        _, dW1, db1 = conv_relu_pool_backward(dh1, cache1)

        dW1 += self.reg * W1
        dW2 += self.reg * W2
        dW3 += self.reg * W3

        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
