# -*- coding: utf-8 -*-
# <autor>Кныш Александр</autor>

from Training import *
from module.Dynamic.Dynamic import Dynamic
from Test import *


def main(arg):
    '''
    Старотовый метод приложения.
    :param arg: Объект с параметрами запуска проложения.
    :return: 0 в случае успеха иначе код ошибки
    '''
    print '---start---'
    if (arg.isTrain):
        train()
    else:
        test()
    print('---end---\n')
    pass


# отправная точка приложения.
if __name__ == '__main__':
    arg = Dynamic()
    arg.isTrain = False
    main(arg)
