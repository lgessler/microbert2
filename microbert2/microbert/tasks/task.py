from tango.common import Registrable


class MicroBERTTask(Registrable):
    """
    A combination of a
    """

    @property
    def data_keys(self):
        raise NotImplemented()

    @property
    def head(self):
        raise NotImplemented()

    def collate_data(self, key, values):
        raise NotImplemented()
