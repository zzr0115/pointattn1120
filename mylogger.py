import logging
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime


class CSVLogger:
    """CSV格式的训练指标记录器，支持按列增量更新"""
    
    def __init__(self, csv_file, columns, add_timestamp=True):
        """
        Args:
            csv_file: CSV文件路径
            columns: 列名列表，如 ['epoch', 'train/loss', 'val/cd_p', 'train/lr']
            add_timestamp: 是否在文件名中添加时间戳
        """
        # 处理时间戳
        if add_timestamp:
            csv_path = Path(csv_file)
            timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
            new_filename = f"{csv_path.stem}_{timestamp}{csv_path.suffix}"
            csv_file = str(csv_path.parent / new_filename)
        
        self.csv_file = Path(csv_file)
        self.columns = columns
        
        # 创建CSV文件并写入表头
        self.csv_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化空的DataFrame
        self.data = pd.DataFrame(columns=self.columns)
        self._save_to_file()
    
    def add_scalar(self, tag, value, step):
        """
        添加一个标量值到指定的epoch行和tag列
        
        Args:
            tag: 列名（指标名称），如 'train/loss', 'val/cd_p'
            value: 指标值
            step: 行索引（通常是epoch）
        
        Example:
            csv_logger.add_scalar('train/loss', 0.123, epoch=0)
            csv_logger.add_scalar('val/cd_p', 0.045, epoch=0)
        """
        # 确保tag在columns中
        if tag not in self.columns:
            raise ValueError(f"Tag '{tag}' not in predefined columns: {self.columns}")
        
        # 如果该step的行不存在，创建新行
        if step not in self.data.index:
            # 创建新行，epoch列填step值，其他列填NaN
            new_row = pd.Series(index=self.columns, dtype=object)
            new_row['epoch'] = step
            self.data.loc[step] = new_row
        
        # 更新指定列的值
        self.data.at[step, tag] = value
        
        # 保存到文件
        self._save_to_file()
    
    def _save_to_file(self):
        """将DataFrame保存到CSV文件"""
        # 按epoch排序
        self.data = self.data.sort_index()
        self.data.to_csv(self.csv_file, index=False)
    
    def get_filepath(self):
        """返回CSV文件的完整路径"""
        return str(self.csv_file)


def get_logger(name="app", log_file="train.log", level="INFO", add_timestamp=True):
   """
   获取配置好的 logger
 
   Args:
      name: logger 名称
      log_file: 日志文件路径
      level: 日志级别，可选值: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
      add_timestamp: 是否自动在文件名中添加时间戳，默认 True
    
   Returns:
      配置好的 logger 实例
   """
   # 如果需要添加时间戳，自动在文件名前加上时间戳
   if add_timestamp:
      log_path = Path(log_file)
      timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
      # 在原文件名前添加时间戳
      new_filename = f"{log_path.stem}_{timestamp}{log_path.suffix}"
      log_file = str(log_path.parent / new_filename)
   # 将字符串级别转换为 logging 常量
   level_map = {
       "DEBUG": logging.DEBUG,
       "INFO": logging.INFO,
       "WARNING": logging.WARNING,
       "ERROR": logging.ERROR,
       "CRITICAL": logging.CRITICAL
   }
   # 把字符串转为level的整数，第二个参数是default值
   log_level = level_map.get(level.upper(), logging.INFO) 
   
   logger = logging.getLogger(name)
   logger.setLevel(log_level)
   # 避免重复添加 handler
   if logger.handlers:
      return logger
   
   # 文件 handler
   file_handler = logging.FileHandler(filename=log_file, mode='w')
   file_handler.setLevel(log_level)
   file_handler.setFormatter(logging.Formatter(
     '%(asctime)s %(levelname)s %(name)s: %(pathname)s line %(lineno)d\n%(message)s',
     datefmt='%y-%m-%d %H:%M:%S'
   ))
   
   # 控制台 handler
   console_handler = logging.StreamHandler()
   console_handler.setLevel(log_level)
   console_handler.setFormatter(logging.Formatter(
      '%(levelname)s %(name)s - %(module)s - line %(lineno)d: %(message)s',
      datefmt='%y-%m-%d %H:%M:%S'
   ))

   # 添加 handler 到 logger
   logger.addHandler(file_handler)
   logger.addHandler(console_handler)

   return logger