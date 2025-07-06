import baostock as bs
import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
import threading

# 检查是否存在 data 文件夹，如果不存在则创建
if not os.path.exists('./data'):
    os.makedirs('./data')

# 登录baostock系统
print("登录baostock系统...")
lg = bs.login()
if lg.error_code != '0':
    print(f'登录失败: {lg.error_msg}')
    exit()

# 获取股票代码列表
print("获取股票代码列表...")
rs = bs.query_all_stock(day='2024-08-01')
stock_list = []
while (rs.error_code == '0') & rs.next():
    stock_list.append(rs.get_row_data())

stock_df = pd.DataFrame(stock_list, columns=rs.fields)
# 过滤A股股票（沪深两市）
stock_df = stock_df[stock_df['code'].str.startswith(('sh.6', 'sz.0', 'sz.3'))]
codes_names = stock_df['code'].values

start_date = '2010-01-01'
end_date = '2025-07-01'
length = len(codes_names)
all_data = pd.DataFrame([])

print(f"共找到 {length} 只股票，开始批量获取历史数据...")

# 优化批次大小和处理逻辑
batch_size = 20  # 增加批次大小
failed_stocks = []
processed_count = 0
total_records = 0

# 创建双重进度条：批次进度和股票进度
batch_progress = tqdm(range(0, length, batch_size), desc="批次进度", position=0, leave=True)
stock_progress = tqdm(total=length, desc="股票进度", position=1, leave=True)

for batch_start in batch_progress:
    batch_end = min(batch_start + batch_size, length)
    batch_codes = codes_names[batch_start:batch_end]
    
    batch_num = batch_start // batch_size + 1
    batch_progress.set_description(f"批次 {batch_num}/{(length-1)//batch_size + 1}")
    
    # 处理当前批次的每只股票
    batch_records = 0
    batch_success = 0
    
    for i, stock_code in enumerate(batch_codes):
        try:
            # 获取单只股票的历史数据
            rs = bs.query_history_k_data_plus(
                stock_code,
                "date,open,close,high,low,volume,amount,turn,pctChg",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2"  # 后复权
            )
            
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            if data_list:
                data_df = pd.DataFrame(data_list, columns=rs.fields)
                # 转换数据类型
                numeric_cols = ['open', 'close', 'high', 'low', 'volume', 'amount', 'turn', 'pctChg']
                for col in numeric_cols:
                    data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
                
                data_df['stock_id'] = stock_code.split('.')[1]  # 去掉前缀
                all_data = pd.concat([all_data, data_df], ignore_index=True)
                batch_records += len(data_df)
                batch_success += 1
                
            processed_count += 1
            # 更新股票进度条
            stock_progress.set_description(f"股票 {processed_count}/{length} | 成功: {processed_count-len(failed_stocks)} | 失败: {len(failed_stocks)}")
            stock_progress.update(1)
                
        except Exception as e:
            failed_stocks.append(stock_code)
            processed_count += 1
            stock_progress.set_description(f"股票 {processed_count}/{length} | 成功: {processed_count-len(failed_stocks)} | 失败: {len(failed_stocks)}")
            stock_progress.update(1)
    
    total_records += batch_records
    
    # 更新批次进度条描述
    batch_progress.set_postfix({
        '本批成功': f"{batch_success}/{len(batch_codes)}",
        '本批记录': batch_records,
        '总记录': total_records
    })
    
    # 减少暂停时间
    time.sleep(0.5)  # 从1秒减少到0.5秒
    
    # 每处理3批次（60只股票）保存一次临时数据
    if batch_num % 3 == 0 and not all_data.empty:
        temp_file = f'./data/temp_{end_date.replace("-", "")}_batch_{batch_num}.csv'
        all_data.to_csv(temp_file, encoding='utf_8_sig', index=False)
        tqdm.write(f"临时保存: {temp_file} (已处理 {processed_count} 只股票)")

# 关闭进度条
stock_progress.close()
batch_progress.close()

# 保存最终数据
if not all_data.empty:
    final_file = f'./data/{end_date.replace("-", "")}.csv'
    all_data.to_csv(final_file, encoding='utf_8_sig', index=False)
    print(f"\n数据保存完成！")
    print(f"文件: {final_file}")
    print(f"总记录数: {len(all_data)}")
    print(f"股票数量: {all_data['stock_id'].nunique()}")
    print(f"成功率: {((processed_count-len(failed_stocks))/processed_count*100):.1f}%")
else:
    print("\n未获取到任何数据")

# 输出失败统计
if failed_stocks:
    print(f"\n失败股票数: {len(failed_stocks)}/{length}")
    print("失败股票代码:", failed_stocks[:10], "..." if len(failed_stocks) > 10 else "")
else:
    print(f"\n全部成功！共处理 {length} 只股票")

# 登出baostock系统
bs.logout()
