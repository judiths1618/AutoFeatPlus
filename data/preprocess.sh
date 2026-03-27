# 先用 --min-rows 0 跑一遍，只看 gap 统计，不用担心输出
python split_by_gaps.py --input ./6g_testbed_dataset/EUR/6907619/amf-performance.csv --gap-multiplier 100 --min-rows 10

# python split_by_gaps.py --input ./6g_testbed_dataset/EUR/6907619/python-web-server-performance.csv --gap-multiplier 100 --min-rows 10

# python split_by_gaps.py --input ./6g_testbed_dataset/EUR/6907619/golang-web-server-performance.csv --gap-multiplier 50 --min-rows 10

# python split_by_gaps.py --input ./6g_testbed_dataset/EUR/6907619/rabbitmq-performance.csv --gap-multiplier 50 --min-rows 0