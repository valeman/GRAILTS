from Causal_Test import test_only_grail
import csv

csvfile = open('lag5scaletune.csv', 'w')
csvwriter = csv.writer(csvfile)
m = 128
ns = [200,1000,5000, 10000]
lag = 5

for n in ns:
    print(n, lag)
    result_by_neighbor = test_only_grail(n, lag, m, neighbor_param = [10])
    for n_num in result_by_neighbor:
        csvwriter.writerow([n] + [lag] + [n_num] + list(result_by_neighbor[n_num].values()))

for n in ns:
    print(n, lag)
    result_by_neighbor = test_only_grail(n, lag, m, neighbor_param=[100])
    for n_num in result_by_neighbor:
        csvwriter.writerow([n] + [lag] + [n_num] + list(result_by_neighbor[n_num].values()))

    csvfile.flush()
csvfile.close()