__all__ = ('FitnessClipper', )

import sklearn.base


class FitnessClipper(sklearn.base.MetaEstimatorMixin,
                     sklearn.base.RegressorMixin, sklearn.base.BaseEstimator):

    def __init__(self, model):
        super().__init__()
        self._model = model

    def fit(self, *args, **kargs):
        self._model.fit(*args, **kargs)

    def predict(self, *args, **kargs):
        prediction = self.model.predict(*args, **kargs)
        return prediction.clip(min=0.5)



