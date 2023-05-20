from pysenal import read_lines
from jy_kmeans.kmeans import KmeansCalculator
from jy_kmeans.main import calculator
from jy_kmeans.func import functions


ls_en = ['eureka query intent', 'eureka roadmap', 'eureka roadmap techfield', 'event drug news cn',
           'event drug news cn com rel', 'event drug news cn ner', 'event drug news cn rel',
           'event drug news cn trigger', 'event drug news en', 'event drug news en ner',
           'event drug news en rel', 'event drug news en trigger', 'evolutional potential en',
           'extendedfamilydetector', 'extr datafix en', 'extr datafix jp', 'extractor datafix',
           'fairseq jp2en', 'faiss', 'field chart en', 'fig cn crf', 'fig desc extraction en',
           'fig desc extraction jp', 'figure claim cn', 'figure claim en', 'figure en',
           'figure extractor claim jp', 'figure extractor cn']

ls_zh = [
    "我曾经失落失望失掉所有方向",
    "直到看见平凡才是唯一的答案",
    "我曾经跨过山和大海",
    "也穿过人山人海",
    "转眼都飘散如烟",
    "我曾经拥有着的一切",
    "也哭也笑平凡着",
    "那片笑声让我想起我的那些花儿",
]

# ================================ kmeans 聚类 ===============================
"""
# jy: 注意: kmeans 自动 TFIDF 加权, 且用欧几里得距离算出文本之间的距离
# jy: 初始化 kmeans 聚类器;
km = KmeansCalculator(n_init=2, max_iter=10)

# jy: 中文文本聚类:
res = km.kmeans(3, ls_zh, WithKeys=True)
print(res)

print("=" * 66)

# jy: 英文文本聚类:
res = km.kmeans(3, ls_en, WithKeys=True)
print(res)

print("=" * 66)

# 计算 Kmeans 的 K 值
# 注意: calk() 从 0 到 MaxNum 算出每个 Kmeans 并找出最优 K 值, 因为算量较大所以会导致时间较长
# (可通过以 C 为内核的多线程优化提速)
#res = km.calk(ls_zh)
res = km.calk(ls_en)
print(res)
"""


# ================================ 文本相似度 ================================
"""
cal = calculator()

# jy: 余弦相似度
res_1 = cal.cossim(["我曾经失落失望失掉所有方向", "直到看见平凡才是唯一的答案"])
print(res_1)
res_2 = cal.cossim(["我曾经失落失望失掉所有方向", "我曾经跨过山和大海"])
print(res_2)


# Simhash & Minhash & Ngram 相似度示例
res_3 = cal.simhash(["我曾经失落失望失掉所有方向", "直到看见平凡才是唯一的答案"])
print(res_3)
res_4 = cal.minhash(["我曾经失落失望失掉所有方向", "直到看见平凡才是唯一的答案"])
print(res_4)
# Ngram (适用于英文文本)
res_5 = cal.ngram(["When life gives you lemons", "eat watermelons"])
print(res_5)
"""


# ================================= 计算 TF, IDF, TFIDF =======================
#"""
ls_word = ["文本", "相似度", "应该", "怎么", "衡量", "？", "衡量", "方法", "多种多样"]
func_ = functions()
res_1 = func_.GetTF(ls_word)          # corpus 为文本, 必须先分词好
print(res_1)
#res_2 = func_.GetIDF(corpus, lists)   # corpus 和 lists 为文本, 必须先分词好
#res_3 = func_.GetTFIDF(corpus, lists) # corpus 和 lists 为文本, 不用分词
#"""

# ================================= 其它加权方法 ==============================
"""
cal = calculator()
cal.weight = "tf"
#cal.weight = "tfidf"


# 如果是 TFIDF 加权, 将用于 IDF 的文本加在原来的 list 后面; 假如原来是: 
res_1 = cal.cossim(["我曾经失落失望失掉所有方向", "我曾经拥有着的一切"])
# 则 TFIDF 加权为: 
ls_ = [
    "我曾经失落失望失掉所有方向",
    "我曾经拥有着的一切",
    "我曾经跨过山和大海",           # 用于 TFIDF
    "也穿过人山人海",               # 用于 TFIDF
    "转眼都飘散如烟",               # 用于 TFIDF
    "我曾经拥有着的一切",           # 用于 TFIDF
    "也哭也笑平凡着",               # 用于 TFIDF
    "那片笑声让我想起我的那些花儿", # 用于 TFIDF
]
res_2 = cal.cossim(ls_)
print(res_1)
print(res_2)
"""



