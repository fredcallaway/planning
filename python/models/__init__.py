from utils import dict_product

class MouselabModel(object):
    """Model of the mouselab-MDP task."""
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

    def preference(self, state, action):
        raise NotImplementedError()

    @staticmethod
    def param_choices(env):
        return {}

    @classmethod
    def all_models(cls, env):
        choices = cls.param_choices(env)
        if choices:
            for prm in dict_product(choices):
                yield cls(**prm)
        else:
            yield cls()

    def __repr__(self):
        cls = type(self).__name__
        params = ', '.join(f'{k}={v}' for k, v in self.__dict__.items())
        return f'{cls}({params})'