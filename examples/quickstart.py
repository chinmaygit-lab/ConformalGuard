"""Minimal ConformalGuard quickstart."""

from sklearn.datasets import load_breast_cancer

from conformalguard import ConformalGuard

data = load_breast_cancer(as_frame=True)
guard = ConformalGuard()
guard.run(data.data, data.target)

print(guard.summary().to_string(index=False))
guard.save_report("artifacts")
