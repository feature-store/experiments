for keys in 60 70 80 90; do
for number in 72; do
/usr/local/kafka/bin/kafka-topics.sh --delete --topic records --bootstrap-server localhost:9092
/usr/local/kafka/bin/kafka-topics.sh --create --topic records --bootstrap-server localhost:9092
python workloads/stl/flink_client.py --slide $number --keys $keys
done
done
for keys in 5 20 10 50 60 70 80 90 100 500; do
for number in 288; do
/usr/local/kafka/bin/kafka-topics.sh --delete --topic records --bootstrap-server localhost:9092
/usr/local/kafka/bin/kafka-topics.sh --create --topic records --bootstrap-server localhost:9092
python workloads/stl/flink_client.py --slide $number --keys $keys
done
done
