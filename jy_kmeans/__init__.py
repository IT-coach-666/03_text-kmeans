from .main import calculator
from .kmeans import KmeansCalculator

cal = calculator()

cossim = cal.cossim
minhash = cal.minhash
simhash = cal.simhash
ngram = cal.ngram


__all__ = [
    'calculator',
    'KmeansCalculator',
]
