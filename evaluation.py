import csv
import math
import time


for n in range(1, 4):
    with open("tf_idf_"+str(n)+"_output.csv", "r") as file:
        output = open("eval_"+str(n)+"_output.csv", "w")
        writer = csv.writer(output)
        writer.writerow(["term", "count", "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11", "c12", "c13", "c14"])
        reader = csv.reader(file)
        reader.__next__()
        for row in reader:
            term = row[0]
            count = int(row[1])
            tf_idf_str = row[2:]
            tf_idf = []
            for val in tf_idf_str:
                tf_idf.append(str(float(val)/math.sqrt(count)))

            writer.writerow([term, count] + tf_idf)
