import logging
from pathlib import Path
from datetime import datetime

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