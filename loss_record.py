import os
import csv

class RECORDER():
    def __init__(self,path):
        self.path = path
        self.file = open(self.path, 'a')
        self.writer = csv.writer(self.file)
    def write_date(self,data_list):
        self.writer.writerow(data_list)