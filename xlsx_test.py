# coding=utf-8
import csv
csv_path = '/home/gdh-95/data/CT/Ⅲ、Ⅳ期肺癌预后已随访1.csv'
csv_reader = csv.reader(open(csv_path))
line_num = -2
for item in csv_reader:
    line_num += 1
    if line_num>0 and len(item[27]):
        print(item[1],item[2],item[27])



