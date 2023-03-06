
class EarlyStopping():
    """Simple early stopping class. Only works for minimization."""
    def __init__(self, patience=5):
        self.cnt = 0
        self.patience = patience
        self.best_score = None

    def update(self, val):
        if self.best_score is None:
            self.best_score = val
        elif val < self.best_score:
            self.cnt = 0
            self.best_score = val
        else:
            self.cnt += 1

    @property
    def stop(self):
        return self.cnt >= self.patience