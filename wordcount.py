import re
import time
from collections import Counter
from pyspark.sql import SparkSession

FILE = "Data/Big_Data.txt"          # ganti kalau nama/path file kamu beda
PARTITIONS = [1, 2, 4, 8, 16, 32]
ULANGI = 3                    # tiap partition dijalankan 3x, lalu dirata-rata

spark = SparkSession.builder.appName("WordCount").getOrCreate()
sc = spark.sparkContext


def tokenisasi(baris):
    # huruf kecil, ambil hanya kata (huruf/angka)
    return re.findall(r"\w+", baris.lower())

# test push ke github shabrina

def wordcount_spark(n_partition):
    mulai = time.time()
    teks = sc.textFile(FILE, minPartitions=n_partition)   # data dibagi ke partition
    kata = teks.flatMap(tokenisasi)                       # transformation
    pasangan = kata.map(lambda w: (w, 1))                 # transformation
    hitung = pasangan.reduceByKey(lambda a, b: a + b)     # transformation (shuffle)
    top10 = hitung.sortBy(lambda x: -x[1]).take(10)       # ACTION (baru jalan di sini)
    return time.time() - mulai, top10


def wordcount_python():
    mulai = time.time()
    c = Counter()
    with open(FILE, encoding="utf-8") as f:
        for baris in f:
            c.update(tokenisasi(baris))
    return time.time() - mulai, c.most_common(10)


# Pemanasan: run pertama selalu lebih lambat (JVM baru start), jadi dibuang
wordcount_spark(4)

print("\n=== Eksperimen Partition (Spark) ===")
print(f"{'Partition':<10}{'Rata-rata (detik)':<20}{'Tiap percobaan'}")
for p in PARTITIONS:
    waktu = []
    for _ in range(ULANGI):
        t, top10 = wordcount_spark(p)
        waktu.append(t)
    rata = sum(waktu) / len(waktu)
    print(f"{p:<10}{rata:<20.3f}{[round(x, 3) for x in waktu]}")

print("\n=== Pembanding: Python biasa (single machine) ===")
waktu = [wordcount_python()[0] for _ in range(ULANGI)]
print(f"Rata-rata: {sum(waktu) / len(waktu):.3f} detik")

print("\n=== 10 kata terbanyak ===")
for kata, jumlah in top10:
    print(f"{kata}\t{jumlah}")

print("\nBuka http://localhost:4040 sekarang untuk screenshot Spark UI.")
input("Tekan Enter kalau sudah selesai screenshot...")
spark.stop()