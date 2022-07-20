set -ex

# papermill -p result_name 100-rr  -p results_dir results/1648827940 prod.ipynb plots/100-rr.out.ipynb
# papermill -p result_name 100-ce  -p results_dir results/1648828073 prod.ipynb plots/100-ce.out.ipynb
# papermill -p result_name 1000-rr -p results_dir results/1648829259 prod.ipynb plots/1000-rr.out.ipynb
# papermill -p result_name 1000-ce -p results_dir results/1648829770 prod.ipynb plots/1000-ce.out.ipynb
# papermill -p result_name 5000-rr -p results_dir results/1648835886 prod.ipynb plots/5000-rr.out.ipynb
# papermill -p result_name 5000-ce -p results_dir results/1648838189 prod.ipynb plots/5000-ce.out.ipynb
# papermill -p result_name 10000-ce-32 -p results_dir results/sigmod/1649918865/1649918865 prod.ipynb plots/10000-ce-32.out.ipynb
# papermill -p result_name 10000-rr-32 -p results_dir results/sigmod/1649927869/1649927869 prod.ipynb plots/10000-rr-32.out.ipynb

# for workers in 32 24 8 #20 16 24 #10 8 4 1
# do
# for algo in ce rr
# do

# ['1650068801', ce-32
#  '1650069533', rr-32
#  '1650069954', ce-24
#  '1650070343', rr-24
#  '1650070730', ce-8
#  '1650071058'] rr-8

papermill -p result_name 2000-ce-32 -p results_dir results/sigmod-scaling/1650068801 prod.ipynb plots/2000-ce-32.out.ipynb
papermill -p result_name 2000-rr-32 -p results_dir results/sigmod-scaling/1650069533 prod.ipynb plots/2000-rr-32.out.ipynb
papermill -p result_name 2000-ce-24 -p results_dir results/sigmod-scaling/1650069954 prod.ipynb plots/2000-ce-24.out.ipynb
papermill -p result_name 2000-rr-24 -p results_dir results/sigmod-scaling/1650070343 prod.ipynb plots/2000-rr-24.out.ipynb
papermill -p result_name 2000-ce-8 -p results_dir results/sigmod-scaling/1650070730 prod.ipynb plots/2000-ce-8.out.ipynb
papermill -p result_name 2000-rr-8 -p results_dir results/sigmod-scaling/1650071058 prod.ipynb plots/2000-rr-8.out.ipynb
