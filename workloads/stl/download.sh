N=195
(
for i in {1..195}
do

    FILE=/home/ubuntu/workspace/experiments/datasets/azure/vm_cpu_readings-file-${i}-of-195.csv
    if [ -f "$FILE" ]; then
        echo "$FILE exists."
    else 
        #wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-${i}-of-195.csv.gz & 
	aws s3 cp s3://feature-store-datasets/vldb/datasets/azure/vm_cpu_readings-file-${i}-of-195.csv ${FILE}
    fi
done
)



#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/Azure 2019 Public Dataset V2 - Trace Analysis.ipynb
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/schema.csv
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/azure2019_data/category.txt
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/azure2019_data/cores.txt
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/azure2019_data/cpu.txt
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/azure2019_data/deployment.txt
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/azure2019_data/lifetime.txt
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/azure2019_data/memory.txt
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vmtable/vmtable.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/deployments/deployments.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/subscriptions/subscriptions.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-1-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-2-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-3-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-4-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-5-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-6-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-7-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-8-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-9-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-10-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-11-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-12-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-13-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-14-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-15-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-16-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-17-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-18-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-19-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-20-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-21-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-22-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-23-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-24-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-25-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-26-of-195.csv.gz
wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-27-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-28-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-29-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-30-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-31-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-32-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-33-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-34-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-35-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-36-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-37-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-38-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-39-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-40-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-41-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-42-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-43-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-44-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-45-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-46-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-47-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-48-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-49-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-50-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-51-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-52-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-53-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-54-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-55-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-56-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-57-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-58-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-59-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-60-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-61-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-62-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-63-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-64-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-65-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-66-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-67-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-68-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-69-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-70-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-71-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-72-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-73-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-74-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-75-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-76-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-77-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-78-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-79-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-80-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-81-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-82-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-83-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-84-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-85-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-86-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-87-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-88-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-89-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-90-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-91-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-92-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-93-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-94-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-95-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-96-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-97-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-98-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-99-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-100-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-101-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-102-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-103-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-104-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-105-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-106-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-107-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-108-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-109-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-110-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-111-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-112-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-113-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-114-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-115-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-116-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-117-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-118-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-119-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-120-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-121-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-122-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-123-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-124-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-125-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-126-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-127-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-128-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-129-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-130-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-131-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-132-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-133-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-134-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-135-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-136-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-137-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-138-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-139-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-140-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-141-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-142-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-143-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-144-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-145-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-146-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-147-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-148-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-149-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-150-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-151-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-152-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-153-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-154-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-155-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-156-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-157-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-158-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-159-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-160-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-161-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-162-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-163-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-164-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-165-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-166-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-167-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-168-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-169-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-170-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-171-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-172-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-173-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-174-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-175-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-176-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-177-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-178-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-179-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-180-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-181-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-182-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-183-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-184-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-185-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-186-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-187-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-188-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-189-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-190-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-191-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-192-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-193-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-194-of-195.csv.gz
#wget https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-195-of-195.csv.gz