'''
Base classes for all filters with parameters
'''

from abc import ABCMeta, abstractmethod


class BaseFilter(metaclass=ABCMeta):
    @abstractmethod
    def sample_parameters(self):
        pass

    @abstractmethod
    def test_range(self, i_start, i_end):
        pass

    @abstractmethod
    def get_result(self, i):
        pass

    @abstractmethod
    def export_para(self, i):
        pass
