class CoresetMethod(object):
    def __init__(self, dst_train, args, fraction, random_seed=None, **kwargs):
        self.dst_train = dst_train
        self.num_classes = len(dst_train.dataset.classes)
        self.fraction = fraction
        if fraction < 0 or fraction > 1:
            raise ValueError("Illegal Coreset Size.")
        self.random_seed = random_seed
        self.index = []
        self.args = args

        self.n_train = len(dst_train)
        self.coreset_size = round(fraction * self.n_train)

    def select(self, **kwargs):
        return

